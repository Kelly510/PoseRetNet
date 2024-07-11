import os
import math
from copy import deepcopy

import torch

from common.model.loss import *
from common.dataset.data_utils import *


def evaluate_parallel(args, model_eval, test_generator, kps_left, kps_right, joints_left, joints_right, subset=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    
    if args.dataset == 'h36m':
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_vel = 0
    elif args.dataset in ['mpii3d', 'mpii3d_univ']:
        epoch_pck = 0
        epoch_auc = 0

    joints_err_list = []
    with torch.no_grad():
        model_eval.eval()

        N = 0
        seq_idx = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            seq_idx += 1
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if inputs_2d.shape[-1] > 2:
                inputs_2d, valid = inputs_2d[..., :2], inputs_2d[..., 2].bool()
            else:
                valid = torch.ones(*inputs_2d.shape[:-1], dtype=torch.bool)
            
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_3d_p = deepcopy(inputs_3d)
            if not isinstance(args.root_idx, list):
                inputs_3d_p[:, :, args.root_idx] = 0
            
            if torch.cuda.is_available():
                inputs_3d_p = inputs_3d_p.cuda()
                valid = valid.cuda()
            orig_seq_len = inputs_3d_p.shape[1]
            
            if args.model == 'stc':
                inputs_2d, _ = eval_data_prepare_seq2frame(args.number_of_frames, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_seq2frame(args.number_of_frames, inputs_2d_flip, inputs_3d_p)
                middle_frame = (args.number_of_frames - 1) // 2
                
                predicted_3d_pos = []
                for n in range(math.ceil(orig_seq_len / args.batch_size)):
                    batch_2d = inputs_2d[n * args.batch_size: min((n + 1) * args.batch_size, orig_seq_len)]
                    batch_2d_flip = inputs_2d_flip[n * args.batch_size: min((n + 1) * args.batch_size, orig_seq_len)]
                    
                    if torch.cuda.is_available():
                        batch_2d = batch_2d.cuda()
                        batch_2d_flip = batch_2d_flip.cuda()
                    
                    batch_predicted = model_eval(batch_2d)[:, middle_frame: middle_frame + 1]
                    batch_predicted_flip = model_eval(batch_2d_flip)[:, middle_frame: middle_frame + 1]
                    
                    batch_predicted_flip[:, :, :, 0] *= -1
                    batch_predicted_flip[:, :, joints_left + joints_right] = batch_predicted_flip[:, :, joints_right + joints_left]
                    batch_predicted = (batch_predicted + batch_predicted_flip) / 2
                    predicted_3d_pos.append(batch_predicted)
                    
                predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0).permute(1, 0, 2, 3)
                
            else:
                inputs_2d, _ = eval_data_prepare(args.number_of_frames, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(args.number_of_frames, inputs_2d_flip, inputs_3d_p)

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                
                predicted_3d_pos = model_eval(inputs_2d)
                predicted_3d_pos_flip = model_eval(inputs_2d_flip)
                
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]
                predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2
        
                predicted_3d_pos = get_original_frames(predicted_3d_pos, orig_seq_len)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            num_valid_frames = (torch.sum(valid) / valid.shape[-1]).item()
            N += num_valid_frames
            error, joints_err = mpjpe(predicted_3d_pos, inputs_3d_p, valid, return_joints_err=True)
            epoch_loss_3d_pos += num_valid_frames * error.item()
            joints_err_list.append(joints_err)
            
            if args.export_path is not None:
                output_path = os.path.join(args.export_path, '{}_seq{:02d}.npz'.format(subset, seq_idx))
                print('Saving results in {}'.format(output_path))
                np.savez(output_path, gt=inputs_3d_p.squeeze(0).cpu().numpy(), pred=predicted_3d_pos.squeeze(0).cpu().numpy())
            
            if args.dataset == 'h36m':
                inputs = inputs_3d_p.squeeze(0).cpu().numpy()
                predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()
                valid = valid.squeeze(0).cpu().numpy()
                epoch_loss_3d_pos_procrustes += num_valid_frames * p_mpjpe(predicted_3d_pos, inputs, valid)
                epoch_loss_3d_vel += num_valid_frames * mean_velocity_error(predicted_3d_pos, inputs, valid)
            
            elif args.dataset in ['mpii3d', 'mpii3d_univ']:
                pck = PCK(predicted_3d_pos, inputs_3d_p, valid)
                auc = AUC(predicted_3d_pos, inputs_3d_p, valid)
                
                epoch_pck += num_valid_frames * pck.item()
                epoch_auc += num_valid_frames * auc

    if subset is None:
        print('----------')
    else:
        print('----' + subset + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    joints_err_array = np.mean(np.array(joints_err_list), axis=0)
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    
    if args.dataset == 'h36m':
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        ev = (epoch_loss_3d_vel / N) * 1000
        print('Velocity Error (MPJVE):', ev, 'mm')
    elif args.dataset in ['mpii3d', 'mpii3d_univ']:
        e2 = epoch_pck / N
        print('PCK:', e2)
        ev = epoch_auc / N
        print('AUC:', ev)

    return e1, e2, ev, joints_err_array, N


def evaluate_chunkwise(args, model_eval, test_generator, kps_left, kps_right, joints_left, joints_right, subset=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    
    if args.dataset == 'h36m':
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_vel = 0
    elif args.dataset in ['mpii3d', 'mpii3d_univ']:
        epoch_pck = 0
        epoch_auc = 0

    joints_err_list = []
    with torch.no_grad():
        model_eval.eval()

        N = 0
        seq_idx = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            seq_idx += 1
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            orig_seq_len = inputs_3d.shape[1]
            if inputs_2d.shape[-1] > 2:
                inputs_2d, valid = inputs_2d[..., :2], inputs_2d[..., 2].bool()
            else:
                valid = torch.ones(*inputs_2d.shape[:-1], dtype=torch.bool)
            
            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]
            inputs_2d = torch.cat([inputs_2d, inputs_2d_flip], dim=0)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
                valid = valid.cuda()
            
            if not isinstance(args.root_idx, list):
                inputs_3d[:, :, args.root_idx] = 0
            
            num_chunks = math.ceil(inputs_3d.shape[1] / args.chunk_size)
            predicted_3d_pos = []
            S_prev_list = None
            for chunk_id in range(num_chunks):
                start = chunk_id * args.chunk_size
                end = min((chunk_id + 1) * args.chunk_size, inputs_3d.shape[1])
                predicted_3d_pos_chunk, S_prev_list = model_eval(inputs_2d[:, start: end], S_prev_list=S_prev_list, n=chunk_id)
                predicted_3d_pos.append(predicted_3d_pos_chunk)
            predicted_3d_pos = torch.cat(predicted_3d_pos, dim=1)
            
            predicted_3d_pos[1, :, :, 0] *= -1
            predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
            predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            num_valid_frames = (torch.sum(valid) / valid.shape[-1]).item()
            N += num_valid_frames
            error, joints_err = mpjpe(predicted_3d_pos, inputs_3d, valid, return_joints_err=True)
            epoch_loss_3d_pos += num_valid_frames * error.item()
            joints_err_list.append(joints_err)
            
            if args.export_path is not None:
                output_path = os.path.join(args.export_path, '{}_seq{:02d}.npz'.format(subset, seq_idx))
                print('Saving results in {}'.format(output_path))
                np.savez(output_path, gt=inputs_3d.squeeze(0).cpu().numpy(), pred=predicted_3d_pos.squeeze(0).cpu().numpy())

            if args.dataset == 'h36m':
                inputs = inputs_3d.squeeze(0).cpu().numpy()
                predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()
                valid = valid.squeeze(0).cpu().numpy()
                epoch_loss_3d_pos_procrustes += num_valid_frames * p_mpjpe(predicted_3d_pos, inputs, valid)
                epoch_loss_3d_vel += num_valid_frames * mean_velocity_error(predicted_3d_pos, inputs, valid)
            
            elif args.dataset in ['mpii3d', 'mpii3d_univ']:
                pck = PCK(predicted_3d_pos, inputs_3d, valid)
                auc = AUC(predicted_3d_pos, inputs_3d, valid)
                epoch_pck += num_valid_frames * pck.item()
                epoch_auc += num_valid_frames * auc

    if subset is None:
        print('----------')
    else:
        print('----' + subset + '----')
    e1 = (epoch_loss_3d_pos / N)*1000
    joints_err_array = np.mean(np.array(joints_err_list), axis=0)
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    
    if args.dataset == 'h36m':
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        ev = (epoch_loss_3d_vel / N)*1000
        print('Velocity Error (MPJVE):', ev, 'mm')
    elif args.dataset in ['mpii3d', 'mpii3d_univ']:
        e2 = epoch_pck / N
        print('PCK:', e2)
        ev = epoch_auc / N
        print('AUC:', ev)

    return e1, e2, ev, joints_err_array, N


def get_joint_set_error(joints_err_arr, joints_set):
    """
    joints_err_arr: number_of_actions, num_joints
    joints_set: list of tuples
    """
    mean_errs = []
    for set in joints_set:
        mean_errs.append(np.mean(joints_err_arr[:, set]))
    return mean_errs
