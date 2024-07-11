import math
import numpy as np
from einops import rearrange

import torch
from torch.nn import functional as F

from common.utils import deterministic_random
from common.pose_utils.camera import world_to_camera, normalize_screen_coordinates


def load_dataset(args):
    print('Loading dataset...')
    from common.dataset.h36m_dataset import Human36mDataset
    dataset = Human36mDataset('data/data_3d_h36m.npz', remove_nose=(args.keypoints == 'hr'))

    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    if args.dataset == 'h36m':
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    else:
                        pos_3d = anim['positions']
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
    
    return dataset


def load_2D_detections(args, dataset):
    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)

    if args.keypoints == 'hr':
        keypoints_metadata = {
            'layout_name': 'h36m', 
            'num_joints': 16, 
            'keypoints_symmetry': [[4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]]
        }
    else:
        keypoints_metadata = keypoints['metadata'].item()
    
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
    
    return keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right, cam


def fetch(args, dataset, keypoints, subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


def fetch_actions(args, dataset, keypoints, actions):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)): # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

    stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d


def fetch_3dhp_univ(is_train=True, test_subject=None):
    out_poses_3d = []
    out_poses_2d = []

    if is_train == True:
        data = np.load('data/data_train_3dhp.npz', allow_pickle=True)['data'].item()
        for seq in data.keys():
            for cam in data[seq][0].keys():
                anim = data[seq][0][cam]
                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d.append(data_3d / 1000.)
                data_2d = anim['data_2d']
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                out_poses_2d.append(data_2d)

    else:
        data = np.load('data/data_test_3dhp.npz', allow_pickle=True)['data'].item()
        for seq in data.keys():
            if seq not in test_subject:
                continue
            
            anim = data[seq]
            data_3d = anim['data_3d']
            data_3d[:, :14] -= data_3d[:, 14:15]
            data_3d[:, 15:] -= data_3d[:, 14:15]
            out_poses_3d.append(data_3d / 1000.)
            data_2d = anim['data_2d']
            if seq == "TS5" or seq == "TS6":
                width, height = 1920, 1080
            else:
                width, height = 2048, 2048
            
            data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
            valid = np.repeat(anim["valid"].reshape(-1, 1, 1), repeats=data_2d.shape[1], axis=1)
            data_2d = np.concatenate([data_2d[..., :2], valid], axis=-1)
            out_poses_2d.append(data_2d)

    return out_poses_3d, out_poses_2d


def fetch_3dhp(subjects):
    out_poses_3d = []
    out_poses_2d = []

    data_2d = np.load('data/data_2d_mpii3d_gt.npz', allow_pickle=True)['positions_2d'].item()
    data_3d = np.load('data/data_3d_mpii3d.npz', allow_pickle=True)['positions_3d'].item()
    
    for subject in subjects:
        data_2d_sub = data_2d[subject]
        data_3d_sub = data_3d[subject]
        for act in data_2d_sub.keys():
            data_2d_item = data_2d_sub[act][0]
            try:
                data_3d_item = data_3d_sub[act]
            except KeyError:
                print('Seq {} of subject {} does not have 3D annotations'.format(act, subject))
            
            if subject in ['TS5', 'TS6']:
                width, height = 1920, 1080
            else:
                width, height = 2048, 2048
            
            data_2d_item[..., :2] = normalize_screen_coordinates(data_2d_item[..., :2], w=width, h=height)
            
            out_poses_2d.append(data_2d_item)
            out_poses_3d.append(data_3d_item)
    
    return out_poses_3d, out_poses_2d


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d, dim=0)
    inputs_3d_p = torch.squeeze(inputs_3d, dim=0)

    out_num = math.ceil(inputs_2d_p.shape[0] / receptive_field)
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    
    # 若序列长度不足一个receptive field，则重复最后一帧
    if inputs_2d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'f j c -> j c f')
        inputs_2d_p = F.pad(inputs_2d_p, (0, pad_right), mode='replicate')
        inputs_2d_p = rearrange(inputs_2d_p, 'j c f -> f j c')

    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'f j c -> j c f')
        inputs_3d_p = F.pad(inputs_3d_p, (0, pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'j c f -> f j c')
    
    # 若序列长度不能够被receptive field整除，则batch的最后一个维度用最后receptive field帧
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


def get_original_frames(predicted_pos, seq_len):
    num_batch, chunk_size, num_joints, n_chans = predicted_pos.shape

    if seq_len <= chunk_size:
        assert num_batch == 1
        pos_original_frames = predicted_pos[0, :seq_len]
    else:
        last_batch_num = seq_len % chunk_size
        pos_original_frames = torch.cat([predicted_pos[:-1].view(-1, num_joints, n_chans), predicted_pos[-1, -last_batch_num:]], dim=0)
    
    return pos_original_frames.unsqueeze(0)


def eval_data_prepare_seq2frame(receptive_field, inputs_2d, inputs_3d, tds=3):
    # For the test of STCFormer
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d, dim=0)
    inputs_3d_p = torch.squeeze(inputs_3d, dim=0)

    seq_len = inputs_2d_p.shape[0]
    pad_len = (receptive_field - 1) // 2
    eval_input_2d = torch.empty(seq_len, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(seq_len, 1, inputs_3d_p.shape[1], inputs_3d_p.shape[2])
    
    for i in range(seq_len):
        start, end = i - pad_len * tds, i + pad_len * tds + 1
        start_ = max(0, start)
        end_ = min(seq_len, end)
        pad_left = start_ - start
        pad_right = end - end_
        current_2d = inputs_2d_p[start_: end_]
        current_3d = inputs_3d_p[i]
        
        if pad_left != 0 or pad_right != 0:
            current_2d = rearrange(current_2d, 'f j c -> j c f')
            current_2d = F.pad(current_2d, (pad_left, pad_right), mode='replicate')
            current_2d = rearrange(current_2d, 'j c f -> f j c')
        
        eval_input_2d[i] = current_2d[::tds]
        eval_input_3d[i, 0] = current_3d

    return eval_input_2d, eval_input_3d