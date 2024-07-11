import os
import sys
import numpy as np

import torch
import torch.nn as nn

from common.model.model_stc import STCModel
from common.model.model_mixste import MixSTE2
from common.model.model_retnet import RetMixSTE
from common.pose_utils.camera import camera_to_world, image_coordinates
from common.function.visualization import render_animation
from common.dataset.generators import UnchunkedGenerator_Seq

from common.arguments import parse_args
from common.dataset.data_utils import load_dataset, load_2D_detections
from common.function.evaluate import evaluate_parallel, evaluate_chunkwise

args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# initial setting
print("Evaluate!")
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

dataset = load_dataset(args=args)
keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right, cam = load_2D_detections(args, dataset)
num_joints = keypoints_metadata['num_joints']

if args.model == 'mixste':
    model_pos =  MixSTE2(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)
    
elif args.model == 'stc':
    model_pos = STCModel(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)

elif args.model == 'retnet':
    model_pos = RetMixSTE(num_frame=args.number_of_frames, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep, 
                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, 
                          gamma_divider=args.gamma_divider, joint_related=args.joint_related, trainable=args.trainable, 
                          chunk_size=args.chunk_size, causal=not args.uncausal, dataset=args.dataset)

else:
    raise NotImplementedError

# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

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
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

print('Rendering...')
# print(keypoints[args.viz_subject].keys())
# ['Directions 1', 'Discussion 1', 'Eating 1', 'Greeting 1', 'Phoning 1', 'Posing 1', 'Purchases 1', 
# 'Sitting 1', 'SittingDown', 'Smoking 1', 'Photo 1', 'Waiting 1', 'Walking 1', 'WalkDog 1', 
# 'WalkTogether 1', 'Directions', 'Discussion 2', 'Eating', 'Greeting', 'Phoning', 'Posing', 
# 'Purchases', 'Sitting', 'SittingDown 1', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
ground_truth = None
if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
    if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
        ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
if ground_truth is None:
    print('INFO: this action is unlabeled. Ground truth will not be rendered.')

gen = UnchunkedGenerator_Seq(None, [ground_truth], [input_keypoints])
if args.model in ['mixste', 'stc']:
    prediction = evaluate_parallel(args, model_pos, gen, kps_left, kps_right, joints_left, joints_right, return_predictions=True)
else:
    prediction = evaluate_chunkwise(args, model_pos, gen, kps_left, kps_right, joints_left, joints_right, return_predictions=True)

if ground_truth is not None:
    trajectory = ground_truth[:, :1]
    ground_truth[:, 1:] += trajectory
    prediction += trajectory

# Invert camera transformation
cam = dataset.cameras()[args.viz_subject][args.viz_camera]
if ground_truth is not None:
    prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
    ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
else:
    for subject in dataset.cameras():
        if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
            rot = dataset.cameras()[subject][args.viz_camera]['orientation']
            break
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    
anim_output = {'Reconstruction': prediction}
if ground_truth is not None and not args.viz_no_ground_truth:
    anim_output['Ground truth'] = ground_truth
input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

args.viz_action = '-'.join(args.viz_action.split(' '))
if args.viz_output is None:
    os.makedirs(os.path.join('output', os.path.basename(args.checkpoint)), exist_ok=True)
    args.viz_output = os.path.join('output', os.path.basename(args.checkpoint), '{}_{}_{}.gif'.format(args.viz_subject, args.viz_action, args.viz_camera))
else:
    os.makedirs(args.viz_output, exist_ok=True)
    args.viz_output = os.path.join(args.viz_output, '{}_{}_{}.gif'.format(args.viz_subject, args.viz_action, args.viz_camera))

if args.viz_export:
    export_path = os.path.join(os.path.dirname(args.viz_output), '{}_{}_{}.npy'.format(args.viz_subject, args.viz_action, args.viz_camera))
    np.save(export_path, anim_output)

render_animation(input_keypoints, keypoints_metadata, anim_output,
                 dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                 limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                 input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                 input_video_skip=args.viz_skip)
