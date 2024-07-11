import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.droppath import DropPath
from common.model.transformer import Mlp


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000**(torch.arange(0, dim) / dim))
    sinusoid_inp = (torch.einsum("i, j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x))
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):

    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer("scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim))

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = - (length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale**torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x


class SimpleRetention(nn.Module):

    def __init__(self, hidden_size, gamma, seq_len, chunk_size, head_size=None, double_v_dim=False, trainable=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=trainable)

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)    #.to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size).to(x_i.device)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1).to(x_i.device)

        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma**(_i + 1)

        cross_chunk = (Q @ r_i_1) * e

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1).to(self.gamma.device)
        m = torch.arange(sequence_length).unsqueeze(0).to(self.gamma.device)
        D = torch.pow(self.gamma, n - m) * (n >= m)
        D[D != D] = 0
        return D


class JointRetention(nn.Module):

    def __init__(self, hidden_size, gamma, seq_len, chunk_size, head_size=None, double_v_dim=False, trainable=False, num_joints=17):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(JointRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.num_joints = num_joints
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=trainable)

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)).view(-1, self.num_joints, sequence_length, sequence_length)
        
        ret = ret * D.unsqueeze(0)
        ret = ret.view(-1, sequence_length, sequence_length)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n_1 = s_n_1.view(-1, self.num_joints, self.head_size, self.head_size)
        s_n = self.gamma.unsqueeze(-1).unsqueeze(-1) * s_n_1
        s_n = s_n.view(-1, self.head_size, self.head_size) + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = torch.pow(self.gamma, chunk_size).unsqueeze(-1).unsqueeze(-1) * r_i_1.view(-1, self.num_joints, self.head_size, self.head_size)
        VD = V.view(-1, self.num_joints, *V.shape[1:]) * D[:, -1].view(1, self.num_joints, chunk_size, 1)
        VD = VD.view(-1, *VD.shape[2:])
        r_i = (K.transpose(-1, -2) @ VD) + r_i.view(-1, *r_i.shape[2:])

        inner_chunk = ((Q @ K.transpose(-1, -2)).view(-1, self.num_joints, chunk_size, chunk_size) * D.unsqueeze(0)).view(-1, chunk_size, chunk_size) @ V
        
        e = torch.pow(self.gamma.unsqueeze(-1), torch.arange(1, chunk_size + 1).unsqueeze(0).to(x_i.device))
        cross_chunk = Q @ r_i_1
        cross_chunk = cross_chunk.view(-1, self.num_joints, *cross_chunk.shape[1:]) * e.unsqueeze(-1)
        cross_chunk = cross_chunk.view(-1, *cross_chunk.shape[2:])

        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1).to(self.gamma.device)
        m = torch.arange(sequence_length).unsqueeze(0).to(self.gamma.device)
        D = torch.pow(self.gamma.unsqueeze(-1).unsqueeze(-1), (n - m).unsqueeze(0)) * (n >= m)
        D[D != D] = 0
        return D


class MultiScaleRetention(nn.Module):

    def __init__(self, hidden_size, heads, seq_len, chunk_size, gamma_divider=8, double_v_dim=False, joint_related=False, trainable=False, dataset='h36m', num_joints=17):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        if joint_related:
            if dataset == 'h36m':
                if num_joints == 17:
                    joints_dividers = [4, 4, 1, 1, 4, 1, 1, 4, 4, 4, 2, 2, 0.5, 0.5, 2, 0.5, 0.5]
                elif num_joints == 16:
                    joints_dividers = [2, 2, 1, 0.5, 2, 1, 0.5, 2, 1.5, 1.5, 2, 1, 0.5, 2, 1, 0.5]
            elif dataset == 'mpii3d_univ':
                joints_dividers = [1.5, 1.5, 2, 1, 0.5, 2, 1, 0.5, 2, 1, 0.5, 2, 1, 0.5, 2, 2, 1.5]
            elif dataset == 'mpii3d':
                joints_dividers = [0.5, 1, 2, 2, 1, 0.5, 0.5, 1, 2, 2, 1, 0.5, 1.5, 1.5]
            
            self.gamma = torch.zeros(heads, len(joints_dividers))
            for i, jd in enumerate(joints_dividers):
                self.gamma[:, i] = 1 - torch.exp(torch.linspace(math.log(1 / (jd * gamma_divider)), math.log(1 / (jd * gamma_divider * 16)), heads))
            self.gamma = self.gamma.detach().cpu().tolist()
        else:
            self.gamma = (1 - torch.exp(torch.linspace(math.log(1 / gamma_divider), math.log(1 / (gamma_divider * 16)), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)
        
        if joint_related:
            self.retentions = nn.ModuleList([
                JointRetention(self.hidden_size, gamma, seq_len, chunk_size, self.head_size,
                                double_v_dim, trainable, num_joints) for gamma in self.gamma
            ])
        else:
            self.retentions = nn.ModuleList([
                SimpleRetention(self.hidden_size, gamma, seq_len, chunk_size, self.head_size,
                                double_v_dim, trainable) for gamma in self.gamma
            ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (heads, batch_size, head_size, head_size)

        """
        Y = []
        s_ns = []
        batch_size = x_n.shape[0]

        for i in range(self.heads):
            s_n_input = torch.zeros(batch_size, self.head_size, self.head_size).to(x_n.device) if s_n_1s is None else s_n_1s[i]
            y, s_n = self.retentions[i].forward_recurrent(x_n, s_n_input, n)
            Y.append(y)
            s_ns.append(s_n.detach())

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (heads, batch_size, head_size, head_size)
        """
        Y = []
        r_is = []
        batch_size = x_i.shape[0]

        for j in range(self.heads):
            r_i_input = torch.zeros(batch_size, self.head_size, self.head_size).to(x_i.device) if r_i_1s is None else r_i_1s[j]
            y, r_i = self.retentions[j].forward_chunkwise(x_i, r_i_input, i)
            Y.append(y)
            r_is.append(r_i.detach())

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is


class RetentionBlock(nn.Module):
    def __init__(self, dim, num_heads, gamma_divider=8, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 joint_related=False, trainable=False, chunk_size=None, seq_len=None, dataset='h36m', num_joints=17) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleRetention(dim, num_heads, seq_len, chunk_size, gamma_divider, joint_related=joint_related, trainable=trainable, dataset=dataset, num_joints=num_joints)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.chunk_size = chunk_size

    def forward(self, x, s_n=None, n=None):
        if n is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            b, f, _ = x.shape
            x_norm_1 = self.norm1(x)

            if f < self.chunk_size:
                pad_len = self.chunk_size - f
                x_norm_1 = F.pad(x_norm_1, (0, 0, 0, pad_len))
            
            o_n, s_n = self.attn.forward_chunkwise(x_norm_1, s_n, n)
            x = x + self.drop_path(o_n[:, :f, :])
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x, s_n