import math
import torch
from copy import deepcopy

from common.model.loss import *
from common.dataset.data_utils import *


def train_one_iter(args, model_train, cameras_train, batch_2d, batch_3d, optimizer):
    if cameras_train is not None:
        cameras_train = torch.from_numpy(cameras_train.astype('float32'))
    
    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
    if inputs_2d.shape[-1] > 2:
        inputs_2d, valid = inputs_2d[..., :2], inputs_2d[..., 2].bool()
    else:
        valid = torch.ones(*inputs_2d.shape[:-1], dtype=torch.bool)

    if torch.cuda.is_available():
        inputs_3d = inputs_3d.cuda()
        inputs_2d = inputs_2d.cuda()
        valid = valid.cuda()
        if cameras_train is not None:
            cameras_train = cameras_train.cuda()
    
    if not isinstance(args.root_idx, list):
        inputs_3d[:, :, args.root_idx] = 0

    optimizer.zero_grad()

    # Predict 3D poses
    if args.train_mode == "chunkwise":
        num_chunks = math.ceil(args.number_of_frames / args.chunk_size)
        predicted_3d_pos = []
        S_prev_list = None
        for chunk_id in range(num_chunks):
            start = chunk_id * args.chunk_size
            end = min((chunk_id + 1) * args.chunk_size, args.number_of_frames)
            predicted_3d_pos_chunk, S_prev_list = model_train(inputs_2d[:, start: end], S_prev_list=S_prev_list, n=chunk_id)
            predicted_3d_pos.append(predicted_3d_pos_chunk)
        predicted_3d_pos = torch.cat(predicted_3d_pos, dim=1)
    else:
        predicted_3d_pos = model_train(inputs_2d)

    if args.dataset == 'h36m':
        if args.keypoints == 'hr':
            w_mpjpe = torch.tensor([1, 1, 2.5, 4, 1, 2.5, 4, 1, 1.5, 1.5, 1, 2.5, 4, 1, 2.5, 4]).cuda()
        else:
            w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
    elif args.dataset == 'mpii3d_univ':
        w_mpjpe = torch.tensor([1.5, 1.5, 1, 2.5, 4, 1, 2.5, 4, 1, 2.5, 4, 1, 2.5, 4, 1, 1, 1.5]).cuda()
    elif args.dataset == 'mpii3d':
        w_mpjpe = torch.tensor([4, 2.5, 1, 1, 2.5, 4, 4, 2.5, 1, 1, 2.5, 4, 1.5, 1.5]).cuda()
    
    loss_3d_pos = weighted_mpjpe(predicted_3d_pos, inputs_3d, w_mpjpe, valid)
    loss_diff = 0.5 * TCLoss(predicted_3d_pos, w_mpjpe, valid) + 2.0 * mean_velocity_error_train(predicted_3d_pos, inputs_3d, valid)

    loss_total = loss_3d_pos + loss_diff
    loss_total.backward(loss_total.clone().detach())
    loss_total = torch.mean(loss_total)
    optimizer.step()

    return inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item(), inputs_3d.shape[0] * inputs_3d.shape[1]


def eval_one_iter(args, model_pos, batch, batch_2d, kps_left, kps_right, joints_left, joints_right):
    inputs_3d = torch.from_numpy(batch.astype('float32'))
    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
    if inputs_2d.shape[-1] > 2:
        inputs_2d, valid = inputs_2d[..., :2], inputs_2d[..., 2].bool()
    else:
        valid = torch.ones(*inputs_2d.shape[:-1], dtype=torch.bool)
    
    ##### apply test-time-augmentation (following Videopose3d)
    inputs_2d_flip = inputs_2d.clone()
    inputs_2d_flip[:, :, :, 0] *= -1
    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

    ##### convert size
    inputs_3d_p = deepcopy(inputs_3d)
    orig_seq_len = inputs_3d_p.shape[1]
    inputs_2d, inputs_3d = eval_data_prepare(args.number_of_frames, inputs_2d, inputs_3d_p)
    inputs_2d_flip, _ = eval_data_prepare(args.number_of_frames, inputs_2d_flip, inputs_3d_p)

    if torch.cuda.is_available():
        inputs_3d_p = inputs_3d_p.cuda()
        inputs_3d = inputs_3d.cuda()
        inputs_2d = inputs_2d.cuda()
        valid = valid.cuda()
        inputs_2d_flip = inputs_2d_flip.cuda()
    
    if not isinstance(args.root_idx, list):
        inputs_3d_p[:, :, args.root_idx] = 0

    predicted_3d_pos = model_pos(inputs_2d)
    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
    
    predicted_3d_pos_flip[:, :, :, 0] *= -1
    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]
    predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2

    # evaluate on original frames
    predicted_3d_pos = get_original_frames(predicted_3d_pos, orig_seq_len)
    
    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d_p, valid)
    n = (torch.sum(valid) / valid.shape[-1]).item()
    loss_3d_valid = n * loss_3d_pos.item()

    loss_3d_vel = mean_velocity_error_train(predicted_3d_pos, inputs_3d_p, valid)
    loss_3d_vel += n * loss_3d_vel.item()

    return loss_3d_valid, loss_3d_vel, n


def eval_one_iter_retnet(args, model_pos, batch, batch_2d, kps_left, kps_right, joints_left, joints_right):
    inputs_3d = torch.from_numpy(batch.astype('float32'))
    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
    if inputs_2d.shape[-1] > 2:
        inputs_2d, valid = inputs_2d[..., :2], inputs_2d[..., 2].bool()
    else:
        valid = torch.ones(*inputs_2d.shape[:-1], dtype=torch.bool)

    inputs_2d_flip = inputs_2d.clone()
    inputs_2d_flip[:, :, :, 0] *= -1
    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]
    inputs_2d = torch.cat([inputs_2d, inputs_2d_flip], dim=0)

    if torch.cuda.is_available():
        inputs_3d = inputs_3d.cuda()
        inputs_2d = inputs_2d.cuda()
        valid = valid.cuda()
    
    if not isinstance(args.root_idx, list):
        inputs_3d[:, :, args.root_idx] = 0
    
    num_chunks = math.ceil(inputs_3d.shape[1] / args.chunk_size)
    predicted_3d_pos = []
    S_prev_list = None
    for chunk_id in range(num_chunks):
        start = chunk_id * args.chunk_size
        end = min((chunk_id + 1) * args.chunk_size, inputs_3d.shape[1])
        predicted_3d_pos_chunk, S_prev_list = model_pos(inputs_2d[:, start: end], S_prev_list=S_prev_list, n=chunk_id)
        predicted_3d_pos.append(predicted_3d_pos_chunk)
    predicted_3d_pos = torch.cat(predicted_3d_pos, dim=1)
    
    predicted_3d_pos[1, :, :, 0] *= -1
    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d, valid)
    n = (torch.sum(valid) / valid.shape[-1]).item()
    loss_3d_valid = n * loss_3d_pos.item()

    loss_3d_vel = mean_velocity_error_train(predicted_3d_pos, inputs_3d, valid)
    loss_3d_vel += n * loss_3d_vel.item()

    return loss_3d_valid, loss_3d_vel, n
