# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import random
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from common.my_logging import Logger
from common.model.model_stc import STCModel
from common.model.model_mixste import MixSTE2
from common.model.model_retnet import RetMixSTE
from common.dataset.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq

from common.arguments import parse_args
from common.function.train import train_one_iter, eval_one_iter
from common.dataset.data_utils import load_dataset, load_2D_detections, fetch, fetch_actions
from common.function.evaluate import evaluate_parallel, evaluate_chunkwise, get_joint_set_error

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.evaluate != '':
    description = "Evaluate!"
elif args.evaluate == '':
    description = "Train!"

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log + '_' + TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log + '_' + TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print(description)
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint == '':
    args.checkpoint = args.log + '_' + TIMESTAMP
    print(args.checkpoint)
    os.makedirs(args.checkpoint, exist_ok=True)

dataset = load_dataset(args=args)
keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right, cam = load_2D_detections(args, dataset)
subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

cameras_valid, poses_valid, poses_valid_2d = fetch(args, dataset, keypoints, subjects_test)

print('INFO: Receptive field: {} frames'.format(args.number_of_frames))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Receptive field', str(args.number_of_frames))
min_loss = args.min_loss
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

if args.model == 'mixste':
    model_pos_train =  MixSTE2(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

    model_pos =  MixSTE2(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)
    
elif args.model == 'stc':
    model_pos_train = STCModel(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

    model_pos = STCModel(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)

elif args.model == 'retnet':
    model_pos_train = RetMixSTE(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1, 
                                gamma_divider=args.gamma_divider, joint_related=args.joint_related, trainable=args.trainable, 
                                chunk_size=args.chunk_size, causal=not args.uncausal, dataset=args.dataset)
    
    model_pos = RetMixSTE(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep, 
                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, 
                          gamma_divider=args.gamma_divider, joint_related=args.joint_related, trainable=args.trainable, 
                          chunk_size=args.chunk_size, causal=not args.uncausal, dataset=args.dataset)

else:
    raise NotImplementedError

causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')

# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()

# Training start
if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(args, dataset, keypoints, subjects_train, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_valid = []
    epoch = 0

    train_generator = ChunkedGenerator_Seq(args.batch_size, cameras_train, poses_train, poses_train_2d, args.number_of_frames, args.stride,
                                           shuffle=True, random_seed=args.seed, random_shift=args.random_shift, augment=args.data_augmentation,
                                           kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d)
    
    print('INFO: Total {} batches'.format(train_generator.num_batches))
    print('INFO: Training on {} frames'.format(train_generator.num_frames()))
    if not args.nolog:
        writer.add_text(args.log+'_'+TIMESTAMP + '/Training Frames', str(train_generator.num_frames()))
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
    if not args.nolog:
        writer.add_text(args.log+'_'+TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))
    
    if args.resume:
        chk_filename = os.path.join(args.checkpoint, args.resume)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)

        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        if not args.coverlr:
            lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Pos model only
    train_iters = 0
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()

        for cameras_train, batch_3d, batch_2d in tqdm(train_generator.next_epoch(), desc='Train-epoch {}'.format(epoch)):
            loss, n = train_one_iter(args, model_pos_train, cameras_train, batch_2d, batch_3d, optimizer)
            if train_iters <= args.warmup_step:
                lr = args.init_lr + (args.learning_rate - args.init_lr) * (train_iters / args.warmup_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.init_lr + (args.learning_rate - args.init_lr) * (train_iters / args.warmup_step)
            epoch_loss_3d_train += loss
            N += n
            train_iters += 1
        losses_3d_train.append(epoch_loss_3d_train / N)
        
        # Decay learning rate exponentially
        if train_iters > args.warmup_step:
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
        
        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_3d_vel = 0
            N = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in tqdm(test_generator.next_epoch(), desc='Test-epoch {}'.format(epoch)):
                    loss_3d_valid, loss_3d_vel, n = eval_one_iter(args, model_pos, batch, batch_2d, kps_left, kps_right, joints_left, joints_right)
                    epoch_loss_3d_valid += loss_3d_valid
                    epoch_loss_3d_vel += loss_3d_vel
                    N += n

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                epoch_loss_3d_vel = epoch_loss_3d_vel / N

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f 3d_val_velocity %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1] * 1000,
                epoch_loss_3d_vel * 1000))
            if not args.nolog:
                writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1] * 1000, epoch+1)
        if not args.nolog:
            writer.add_scalar("Loss/3d training loss", losses_3d_train[-1] * 1000, epoch+1)
            writer.add_scalar("Parameters/learing rate", lr, epoch+1)
            writer.add_scalar('Parameters/training time per epoch', elapsed, epoch+1)

        epoch += 1

        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict()
            }, chk_path)

        # save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict()
            }, best_chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')

if args.evaluate == '':
    chk_file_path = os.path.join(args.checkpoint, 'best_epoch.bin')
    print('Loading best checkpoint', chk_file_path)
elif args.evaluate != '':
    chk_file_path = os.path.join(args.checkpoint, args.evaluate)
    print('Loading evaluate checkpoint', chk_file_path)

checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
if 'model_pos' not in checkpoint.keys():
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if not k.startswith('module.'):
            new_k = 'module.' + k
        else:
            new_k = k
        new_checkpoint[new_k] = v
    model_pos.load_state_dict(new_checkpoint, strict=True)
else:
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'], strict=True)


print('Evaluating...')
all_actions = {}
for subject in sorted(subjects_test):
    for action in dataset[subject].keys():
        action_name = action.split(' ')[0]
        if action_name not in all_actions:
            all_actions[action_name] = []
        all_actions[action_name].append((subject, action))

errors_p1 = []
errors_p2 = []
errors_vel = []
errors_p1_joints = []

for action_key in sorted(all_actions.keys()):
    poses_act, poses_2d_act = fetch_actions(args, dataset, keypoints, all_actions[action_key])
    gen = UnchunkedGenerator_Seq(None, poses_act, poses_2d_act)
    if args.model in ['mixste', 'stc']:
        e1, e2, ev, joints_err_array, N = evaluate_parallel(args, model_pos, gen, kps_left, kps_right, joints_left, joints_right, action_key)
    else:
        e1, e2, ev, joints_err_array, N = evaluate_chunkwise(args, model_pos, gen, kps_left, kps_right, joints_left, joints_right, action_key)

    errors_p1.append(e1)
    errors_p2.append(e2)
    errors_vel.append(ev)
    errors_p1_joints.append(joints_err_array)

metric_1 = np.mean(errors_p1)
metric_2 = np.mean(errors_p2)
metric_3 = np.mean(errors_vel)

print('----Average----')
print('Protocol #1   (MPJPE) action-wise average:', round(metric_1, 1), 'mm')
print('Protocol #2 (P-MPJPE) action-wise average:', round(metric_2, 1), 'mm')
print('Velocity      (MPJVE) action-wise average:', round(metric_3, 2), 'mm')

if args.return_joints_err:
    errors_p1_joints = np.array(errors_p1_joints)
    joints_set = [(0, 1, 4, 7, 8, 9), (10, 11), (2, 3, 5, 6), (12, 13, 15, 16)]
    mean_errs = get_joint_set_error(errors_p1_joints, joints_set)
    print('----Per-Joint Error----')
    print('torso, head, middle limb, terminal limb:', mean_errs)

if not args.nolog:
    writer.close()
