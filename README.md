# 3D Human Pose Estimation via Non-Causal Retentive Networks

This is the official implementation of paper 3D Human Pose Estimation via Non-Causal Retentive Networks which has been accepted to ECCV 2024. 

![demo](assets/demo.gif)


## Dependencies

The code is developed and tested with the following dependencies.
```
python=3.7
pytorch=1.10.1
matplotlib
einops
tensorboardX
tqdm
joblib
```

## Data

- The Human3.6M dataset is set up following [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). 
- The original MPII-INF-3DHP dataset is downloaded from [VIBE](https://github.com/mkocabas/VIBE/blob/master/scripts/prepare_data.sh) and preprocessed using [preprocess_mpii3d.py](./common/dataset/preprocess_mpii3d.py). 
- The normalized MPII-INF-3DHP dataset is set up following [P-STMO](https://github.com/paTRICK-swk/P-STMO).

Put all the data in `./data` as following.

```
./data
    |--- data_2d_h36m_cpn_ft_h36m_dbb.npz   # CPN-detected 2D keypoints of the Human3.6M dataset
    |--- data_2d_h36m_gt.npz                # GT 2D keypoints of the Human3.6M dataset
    |--- data_3d_h36m.npz                   # GT 3D keypoints of the Human3.6M dataset
    |--- data_2d_mpii3d_gt.npz              # GT 2D keypoints of the original MPI-INF-3DHP dataset
    |--- data_3d_mpii3d.npz                 # GT 3D keypoints of the original MPI-INF-3DHP dataset
    |--- data_test_3dhp.npz                 # 2D & 3D keypoints of the test set of the normalized MPI-INF-3DHP dataset
    |--- data_train_3dhp.npz                # 2D & 3D keypoints of the training set of the normalized MPI-INF-3DHP dataset
```

## Evaluation

Download pretrained checkpoint from [Google Drive](https://drive.google.com/drive/folders/1Ik3YP3d8eC0zsxWioonqtz6H4ozJqjfJ?usp=drive_link), and put it in `./checkpoint`. Run the following commands to evaluate the model.

```bash
# Evaluate the PoseRetNet on the CPN-detected 2D keypoints of the Human3.6M dataset
python run.py -d h36m -k cpn_ft_h36m_dbb --model retnet --joint-related --uncausal --chunk-size 243 -c checkpoint --evaluate PoseRetNet_h36m_cpn_243f.pth --nolog

# Chunk size can be set to 81 or 27 to get the results of transferred model
python run.py -d h36m -k cpn_ft_h36m_dbb --model retnet --joint-related --uncausal --chunk-size 81 -c checkpoint --evaluate PoseRetNet_h36m_cpn_243f.pth --nolog
python run.py -d h36m -k cpn_ft_h36m_dbb --model retnet --joint-related --uncausal --chunk-size 27 -c checkpoint --evaluate PoseRetNet_h36m_cpn_243f.pth --nolog
```

## Visualization

Run the following command to render the visualization result.

```bash
# Render the results predicted by the PoseRetNet on the CPN-detected 2D keypoints of the Human3.6M dataset
python render.py -d h36m -k cpn_ft_h36m_dbb --model retnet --joint-related --uncausal --chunk-size 243 -c checkpoint --evaluate PoseRetNet_h36m_cpn_243f.pth --viz-output output/retnet_cpn_243f --viz-subject S9 --viz-action 'Photo 1' --viz-limit 500
```

The following is an example of visualization.

![demo](assets/demo_h36m.gif)

## Training

To train from scratch on 2 GPUs, run the following command. 

```bash
# Train the PoseRetNet on the CPN-detected 2D keypoints in the Human3.6M dataset
python run.py -k cpn_ft_h36m_dbb --model retnet --joint-related --uncausal --chunk-size 243 -f 900 -s 450 --random-shift -b 2 -l log/run -gpu 0,1

# Train the PoseRetNet on the original MPI-INF-3DHP dataset
python run_3dhp.py -d mpii3d --model retnet --joint-related --uncausal --chunk-size 9 -f 600 -s 600 -lr 1e-4 --lr-decay 0.98 --random-shift -b 4 -l log/run -gpu 0,1

# Train the PoseRetNet on the normalized MPI-INF-3DHP dataset
python run_3dhp.py -d mpii3d_univ --model retnet --joint-related --uncausal --chunk-size 9 -f 600 -s 600 -lr 1e-4 --lr-decay 0.98 --random-shift -b 4 -l log/run -gpu 0,1
```

# Acknowledgement

The code is constructed based on the following repos. We thank the authors for releasing them.

- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO/tree/main)
- [STCFormer](https://github.com/zhenhuat/STCFormer)
- [RetNet](https://github.com/Jamie-Stirling/RetNet)
