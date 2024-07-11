import math
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn

# import sys
# sys.path.append('/home/zhengkl/mixste-retnet')
from common.model.retention import RetentionBlock
from common.model.retention_uncausal import RetentionBlockUncausal
from common.model.transformer import TransformerBlock


class RetMixSTE(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,  norm_layer=None, 
                 gamma_divider=8, joint_related=False, trainable=False, chunk_size=None, 
                 causal=True, dataset='h36m'):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            TransformerBlock(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        block = RetentionBlock if causal else RetentionBlockUncausal
        self.TTEblocks = nn.ModuleList([
            block(
                dim=embed_dim, num_heads=num_heads, gamma_divider=gamma_divider, mlp_ratio=mlp_ratio, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                joint_related=joint_related, trainable=trainable, chunk_size=chunk_size, seq_len=num_frame, dataset=dataset, num_joints=num_joints)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
    
    def forward(self, x, S_prev_list=None, n=None):
        b, f, j, c = x.shape
        
        S_list = []
        for i in range(self.block_depth):
            x = rearrange(x, 'b f j cw -> (b f) j cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            if i == 0:
                x = self.Spatial_patch_to_embedding(x)
                x += self.Spatial_pos_embed
                x = self.pos_drop(x)
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            if i == 0:
                x = self.pos_drop(x)
            
            if n is None:
                x = tteblock(x)
            else:
                s_i = None if S_prev_list is None else S_prev_list[i]
                x, s_i = tteblock(x, s_i, n)
                S_list.append(s_i)
            
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b j) f cw -> b f j cw', j=j)
        
        x = self.head(x)
        x = x.view(b, f, j, -1)

        if n is None:
            return x
        else:
            return x, S_list


if __name__ == '__main__':
    model = RetMixSTE(num_frame=81, embed_dim_ratio=32, drop_path_rate=0, chunk_size=2, causal=True)
    model.eval()
    dummy_input = torch.randn(2, 81, 17, 2) * 1000
    out1 = model(dummy_input)
    
    chunk_size = 2

    num_chunks = math.ceil(81 / chunk_size)
    S_prev = None
    out_list = []
    for n in range(num_chunks):
        start = chunk_size * n
        end = min(chunk_size * (n + 1), 81)
        input_chunk = dummy_input[:, start: end]
        out_n, S_prev = model(input_chunk, S_prev, n)
        out_list.append(out_n)
    
    out2 = torch.cat(out_list, dim=1)
    print(out1.shape, out2.shape)
    error = torch.mean(torch.norm(out1 - out2, dim=-1), dim=(0, -1)).tolist()
    print(error)