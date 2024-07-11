import joblib
import numpy as np

def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d

train_data = joblib.load("data/mpii3d/mpii3d_train_db.pt")
train_data_joints3D = convert_kps(train_data['joints3D'], src='spin', dst='common')
train_data_joints2D = convert_kps(train_data['joints2D'], src='spin', dst='common')
metadata = {'layout_name': 'mpii3d', 'num_joints': 14, 'keypoints_symmetry': [[3, 4, 5, 9, 10, 11], [0, 1, 2, 6, 7, 8]]}

num_frames = train_data_joints2D.shape[0]
positions_3d, positions_2d = {}, {}

for i in range(num_frames):
    vid_name = train_data['vid_name'][i]

    if vid_name not in positions_3d.keys():
        positions_3d[vid_name] = {}
        positions_2d[vid_name] = [{}]
    
    train_data_joints2D[i, :, -1] = 1.0
    frame_id = train_data['frame_id'][i].split('_')[-1]
    assert frame_id not in positions_3d[vid_name].keys()
    positions_3d[vid_name][frame_id] = train_data_joints3D[i]
    positions_2d[vid_name][0][frame_id] = train_data_joints2D[i]

train_lens = []

for action in positions_3d.keys():
    seq_data_3d = []
    seq_data_2d = []
    for k in sorted(positions_3d[action].keys()):
        seq_data_3d.append(positions_3d[action][k])
        seq_data_2d.append(positions_2d[action][0][k])
    positions_3d[action] = np.array(seq_data_3d)
    positions_2d[action][0] = np.array(seq_data_2d)
    train_lens.append(np.array(seq_data_3d).shape[0])

positions_3d_full = {'TS0': positions_3d}
positions_2d_full = {'TS0': positions_2d}

test_lens = []
for ts_idx in range(1, 7):
    valid_data = joblib.load("data/mpii3d/mpii3d_val_db_TS{}.pt".format(ts_idx))
    valid_data_joints3D = convert_kps(valid_data['joints3D'], src='spin', dst='common')
    valid_data_joints2D = convert_kps(valid_data['joints2D'], src='spin', dst='common')

    num_frames = valid_data_joints2D.shape[0]
    positions_3d, positions_2d = {}, {}

    for i in range(num_frames):
        vid_name = valid_data['vid_name'][i]
        if vid_name not in positions_3d.keys():
            positions_3d[vid_name] = {}
            positions_2d[vid_name] = [{}]
        
        valid_data_joints2D[i, :, -1:] = np.repeat(np.expand_dims(valid_data['valid_i'][i], axis=0), repeats=14, axis=0)
        frame_id = valid_data['frame_id'][i].split('_')[-1]
        positions_3d[vid_name][frame_id] = valid_data_joints3D[i]
        positions_2d[vid_name][0][frame_id] = valid_data_joints2D[i]

    for action in positions_3d.keys():
        seq_data_3d = []
        seq_data_2d = []
        for k in sorted(positions_3d[action].keys()):
            seq_data_3d.append(positions_3d[action][k])
            seq_data_2d.append(positions_2d[action][0][k])
        positions_3d[action] = np.array(seq_data_3d)
        positions_2d[action][0] = np.array(seq_data_2d)
        test_lens.append(np.array(seq_data_3d).shape[0])
    
    positions_3d_full['TS{}'.format(ts_idx)] = positions_3d
    positions_2d_full['TS{}'.format(ts_idx)] = positions_2d

np.savez_compressed('data_2d_mpii3d_gt.npz', metadata=metadata, positions_2d=positions_2d_full)
np.savez_compressed('data_3d_mpii3d.npz', positions_3d=positions_3d_full)
