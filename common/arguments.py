# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', choices=['h36m', 'mpii3d', 'mpii3d_univ'], type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-l', '--log', default='log/default', type=str, metavar='PATH', help='log file directory')
    parser.add_argument('-cf','--checkpoint-frequency', default=20, type=int, metavar='N', help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('--nolog', action='store_true', help='forbiden log function')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--return-joints-err', action='store_true')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model arguments
    parser.add_argument('--model', default='mixste', type=str, choices=['mixste', 'stc', 'retnet'])
    parser.add_argument('--gamma-divider', default=8, type=int)
    parser.add_argument('--joint-related', action='store_true')
    parser.add_argument('--trainable', action='store_true')
    parser.add_argument('--uncausal', action='store_true')
    parser.add_argument('-s', '--stride', default=32, type=int, metavar='N', help='stride when building dataset')
    parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0., type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.00004, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--init-lr', default=1e-7, type=float)
    parser.add_argument('--warmup-step', default=-1, type=int)
    parser.add_argument('-lrd', '--lr-decay', default=0.99, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('--coverlr', action='store_true', help='cover learning rate with assigned during resuming previous model')
    parser.add_argument('-mloss', '--min_loss', default=100000, type=float, help='assign min loss(best loss) during resuming previous model')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false', help='disable train-time flipping')
    parser.add_argument('-cs', default=512, type=int, help='channel size of model, only for trasformer') 
    parser.add_argument('-dep', default=8, type=int, help='depth of model')    
    parser.add_argument('-f', '--number-of-frames', default='900', type=int, metavar='N', help='how many frames used as input')
    
    # Experimental
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('-gpu', default='0', type=str, help='assign the gpu(s) to use')
    parser.add_argument('--train-mode', default='parallel', type=str, choices=['parallel', 'chunkwise'])
    parser.add_argument('--chunk-size', default=243, type=int)
    parser.add_argument('--random-shift', action='store_true')
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--export-path', default=None, type=str, help='path to save npy files')
    
    # Visualization
    parser.add_argument('--viz-subject', default='S9', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', default='Photo', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', default=0, type=int, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', action='store_true')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()
    
    if args.model == 'stc':
        args.dep = 6
        args.cs = 256
    
    if args.dataset == 'h36m':
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'
        args.by_subject = False
        args.root_idx = 0
    
    elif args.dataset == 'mpii3d_univ':
        args.subjects_train = 'TS0'
        args.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6'
        args.keypoints = 'gt'
        args.by_subject = True
        args.root_idx = 14
        args.num_joints = 17
    
    elif args.dataset == 'mpii3d':
        args.subjects_train = 'TS0'
        args.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6'
        args.keypoints = 'gt'
        args.by_subject = True
        args.root_idx = [2, 3]
        args.num_joints = 14
    
    if args.export_path is not None:
        os.makedirs(args.export_path, exist_ok=True)

    return args