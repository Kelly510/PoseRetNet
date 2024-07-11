import copy
import numpy as np
from common.pose_utils.skeleton import Skeleton
from common.dataset.mocap_dataset import MocapDataset

"""
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
"""

mpii3d_skeleton = Skeleton(parents=[1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, -1, 12],
       joints_left=[3, 4, 5, 9, 10, 11],
       joints_right=[0, 1, 2, 6, 7, 8])

mpii3d_cameras_extrinsic_params = {
    'TS0':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS1':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS2':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS3':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS4':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS5':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ],
    'TS6':[
        {
            'azimuth': 70, # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
    ]
}

mpii3d_cameras_intrinsic_params = [
    {
        'res_w': 2048,
        'res_h': 2048,
        'azimuth': 70
    }
]


def get_mpii3d_cam(subject):
    _cameras = copy.deepcopy(mpii3d_cameras_extrinsic_params)
    
    for cameras in _cameras.values():
        for i, cam in enumerate(cameras):
            cam.update(mpii3d_cameras_intrinsic_params[i])
            for k, v in cam.items():
                if k not in ['res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')
            if 'translation' in cam:
                cam['translation'] = cam['translation'] / 1000  # mm to meters
    return _cameras[subject]

