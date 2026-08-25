<div align="center">
<h1>[ICCV 2025] CityGS-X : A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction
</h1>

<a href="https://arxiv.org/abs/2503.23044" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2503.23044"><img src="https://img.shields.io/badge/arXiv-2503.23044-b31b1b" alt="arXiv"></a>
<a href="https://lifuguan.github.io/CityGS-X/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


**Northwestern Polytechnical University**; **Shanghai Artificial Intelligence Laboratory**

<!-- | [ICCV 2025] CityGS-X : A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction -->

[Yuanyuan Gao*](https://scholar.google.com/citations?hl=en&user=1zDq0q8AAAAJ), [Hao Li*](https://lifuguan.github.io/), [Jiaqi Chen*](https://github.com/chenttt2001), [Zhengyu Zou](https://vision-intelligence.com.cn), [Zhihang Zhong†](https://zzh-tech.github.io), [Dingwen Zhang†](https://vision-intelligence.com.cn), [Xiao Sun](https://jimmysuen.github.io), [Junwei Han](https://vision-intelligence.com.cn)<br>(\* indicates equal contribution, † means co-corresponding author)<br>

</div>

![Teaser image](assets/cityx_tease.jpg)

This repo contains official implementations of CityGS-X, ⭐ us if you like it!

## Project Updates
- 🔥🔥 News: ```2025/4/17```: training & inference code is now available! You can try it.
- 🔥🔥 News: ```2025/6/28```: CityGS-X has been accepted to ICCV 2025.
  
## Todo List
- [x] Release the training & inference code of CityGS-X.
- [x] Reproduction scripts, records, and results under different parameter settings are available at [yy456/CityGS-X](https://huggingface.co/datasets/yy456/CityGS-X). Many thanks to [@gongshiqiang02](https://github.com/gongshiqiang02) for providing the reproduction results.


## Installation

We tested CityGS-X on a server configured with Ubuntu 18.04, cuda 11.6 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/gyy456/CityGS-X.git --recursive
cd CityGS-X
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate citygx-x
pip install submodule_cityx/diff-gaussian-rasterization
pip install submodule_cityx/simple-knn
```

### Depth regularization


When training on a synthetic dataset, depth maps can be produced and they do not require further processing to be used in our method. 

For real world datasets depth maps should be generated for each input images, to generate them please do the following:
1. Clone [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#usage) （You can try other depth estimation models）:
    ```
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    ```
2. Download weights from [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it under `Depth-Anything-V2/checkpoints/`
3. Generate depth maps (set the depth image reslution align with the training reslution you want):
   ```
   python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale \
   --img-path <path to input images> --outdir <output path>
   ```
5. Generate a `depth_params.json` file using:
    ```
    python utils/make_depth_scale.py --base_dir <path to colmap> --depths_dir <path to generated depths>
    ```
6. use the multi-view constrains to filter the depth:
    ```
    python  multi_view_precess.py  -s  datasets/<scene_name> --resolution 4 \
    --model_path datasets/<scene_name>/train/mask  --images train/rgbs  --pixel_thred 1
    ```

- pixel_thred: set the threshold of the pixel position loss;
## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

### Get the COLMAP result

Download the datasets following [the Mega-NeRF repository](https://github.com/cmusatyalab/mega-nerf).

After downloading, for Mill-19 and UrbanScene3D, run the following code for each dataset: 

```
python tools/merge_val_train.py -d $DATASET_DIR(data/<scene_name>)
```

```
bash tools/colmap_full.sh  $COLMAP_RESULTS_DIR  $DATASET_ROOT(data/<scene_name>)
```

While for the MatrixCity, CityGS-X follow the preprocess of [CityGaussianV2](https://github.com/Linketic/CityGaussian/blob/main/doc/data_preparation.md)


The data structure will be organised as follows:

```
data/
├── scene_name(Mill-19 and UrbanScene3D)
│   ├── train/
│   │   ├── rgbs
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   │   ├── depths
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   ├── mask
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── ...
│   ├── val/
│   │   ├── rgbs
│   │   │   ├── 000000.jpg
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   ├── sparse/
│   │   └──0/
├── scene_name(MatrixCity)
│   ├── train/block_all
│   │   ├── images
│   │   │   ├── 0000.png
│   │   │   ├── 0001.png
│   │   │   ├── ...
│   │   ├── depth
│   │   │   ├── 0000.png
│   │   │   ├── 0001.png
│   │   │   ├── ...
│   │   ├── mask
│   │   │   ├── 0000.png
│   │   │   ├── 0001.png
│   │   │   ├── ...
│   │   ├── sparse
│   │       └──0/
│   ├── test/block_all_test
│   │   ├── images
│   │   │   ├── 0000.png
│   │   │   ├── 0001.png
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```





## Training

### Training multiple scenes

To train multiple scenes in parallel, we provide batch training scripts: 

 - Mill-19 and UrbanScene3D: ```train_mill19.sh```

 - MatrixCity: ```train_matrix_city.sh```

 run them with 

 ```
bash train_xxx.sh
 ```

### Training a single scene

- not_use_dpt_loss: you can jump Step2 depth supervision;
- not_use_multi_view_loss: you can jump Step3 multi-view geometric constrains;
- not_use_single_view_loss: you can choose not use the single-view geometric loss;
- gpu_num: specify the GPU number to use;
- bsz: set the taining batch size;
- iteration: set the whole training iterations;
- single_view_weight_from_iter: set the start iteration of the single-view geometric loss default `10_000`;
- scale_loss_from_iter:  set the start iteration of the scale loss default `0`;
- dpt_loss_from_iter: set the start iteration of the depth supervision default `10_000`;
- multi_view_weight_from_iter: seet the start iteration of multi-view constrains default `30_000`;
- default_voxel_size: set the mean voxel size of the anchor default `0.001`; (default_voxel_size will influence the final anchors number)
- distributed_dataset_storage: if cpu memory is enough set it `False` (Load all the RGB depth and gray image on every process), if cpu memory is not enough， set it `Ture` (Load RGB depth and gray image on one process and broadcast to other process).
- distributed_save: if Ture load the final model seperately by process, if `False` load the final model in one model(default)
- default_voxel_size: set the default voxel size for initialization.
- dpt_end_iter: set the end iteration of step2 depth supervision.
- multi_view_patch_size: the multi-view patchh loss is calculated by gray image, for the less texture scene or higher reslution, larger patch_size works better, but may caused longer training time.
 #### Multi gpu
```
torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_num>  train.py --bsz <bsz> -s datasets/<scene_name> \
    --resolution 4 --model_path output/<save_path> --iterations 100000 --images train/rgbs \
    --single_view_weight_from_iter 10000  --depth_l1_weight_final 0.01 --depth_l1_weight_init 0.5 \
    --dpt_loss_from_iter 10000  --multi_view_weight_from_iter 30000 --default_voxel_size 0.001 \
    --dpt_end_iter 30_000 --multi_view_patch_size 3
```

 #### Single gpu

```
python train.py --bsz <bsz> -s datasets/<scene_name> --resolution 4 --model_path output/<save_path> \
    --iterations 100000 --images train/rgbs --single_view_weight_from_iter 10000 \
    --depth_l1_weight_final 0.01 --depth_l1_weight_init 0.5 --dpt_loss_from_iter 10000 \
    --multi_view_weight_from_iter 30000 --default_voxel_size 0.001 --dpt_end_iter 30000 \
    --multi_view_patch_size 3
```

The training time may faster than the table provided in our paper, as we have optimize the multi-process dataloader.


## Evaluation
Evalutaion image is saved and PSNR is calcuated during training by default except MartrixCity.
 ### Rendering on multi gpu

```
torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_num>  render.py --bsz <bsz> \
    -s datasets/<scene_name> --resolution 4 --model_path output/<save_path> \
    --images train/rgbs --skip_train
```

 ### Rendering on single gpu

```
python render.py --bsz <bsz> -s datasets/<scene_name> --resolution 4 \
    --model_path output/<save_path> --images train/rgbs --skip_train
```

 ### Metrics
```
python metrics.py -m output/<save_path>
```


 ### Mesh extraction
 #### multi gpu
```
torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_num>  render_mesh.py --bsz <bsz> \
    -s datasets/<scene_name> --resolution 4 --model_path output/<save_path> \
    --images train/rgbs --voxel_size 0.001 --max_depth 5 --use_depth_filter
```

 #### single gpu

```
python render_mesh.py --bsz <bsz> -s datasets/<scene_name> --resolution 4 \
    --model_path output/<save_path> --images train/rgbs --voxel_size 0.001 \
    --max_depth 5 --use_depth_filter
```

- voxel_size: set the mesh voxel size.


 ### Metrics for F1 score

```
python eval_f1.py --ply_path_pred <mesh_path> --ply_path_gt <gt_point_cloud_path> \
    --dtau 0.5 --sample_points 2000000 --output_json geometry_metrics.json
```

`dtau` is expressed in the coordinate unit shared by the aligned prediction and
ground truth. The evaluator samples triangle surfaces (rather than comparing
only mesh vertices) and reports precision, recall, F-score, accuracy,
completeness, 95th-percentile errors, and symmetric Chamfer-L1. Use
`--legacy_half_threshold` only to reproduce the older `dtau / 2` behavior.

For real-world depth supervision, both supported image layouts resolve sibling
directories automatically:

```text
train/images/<name>.jpg -> train/depths/<name>.png
train/rgbs/<name>.jpg   -> train/depths/<name>.png
```

Masks and normals similarly use `train/mask` and `train/normals`. Generate and
inspect these files before enabling DPT loss; missing supervision is not a
substitute for a controlled RGB-only baseline.

### PLY compatibility

The files saved under `point_cloud/iteration_<iteration>/` are CityGS-X model
checkpoints, not standard 3D Gaussian Splatting interchange files.  Their PLY
vertices store octree anchors and CityGS-X-specific fields such as `level`,
`extra_level`, `f_offset_*`, and `f_anchor_feat_*`.  The learned opacity, color,
scale, and rotation of the rendered Gaussians are generated by the accompanying
MLPs for a particular camera view.  Keep the PLY together with
`checkpoints.pth` (or the per-rank MLP files when distributed saving is enabled)
and use `render.py` to view the trained model.

Consequently, viewers that only support the conventional 3DGS PLY schema, such
as SuperSplat, cannot directly open `point_cloud_rk<rank>_ws<world_size>.ply`.
Renaming the file or merely renaming its PLY properties does not convert the
model: a standard, static splat export would first need to evaluate the
view-dependent CityGS-X MLPs from a chosen camera (or bake an explicit
view-independent approximation).

### Spatial block training

For a concrete implementation plan for single-GPU, CityGaussian-style spatial
partitioning—including the manifest schema, camera/point selection, AABB-owned
densification, checkpoint rules, overlap rendering, and required tests—see
[`docs/SPATIAL_BLOCK_TRAINING.md`](docs/SPATIAL_BLOCK_TRAINING.md).

Create a manifest and train one block per process:

```bash
python tools/partition_scene.py -s data/CameraA --images train/images \
    --grid 3 3 --overlap 0.15 --min_cameras 20 \
    --output data/CameraA/blocks

python train.py --bsz 1 -s data/CameraA --images train/images \
    --resolution 4 --iterations 30000 \
    --block_manifest data/CameraA/blocks/manifest.json --block_id 000_000 \
    --model_path output/cameraA_blocks/000_000
```

Use a separate `model_path` for every block.  Add `--block_core_only` when the
saved PLY should contain only uniquely owned core anchors rather than the halo
used during training.

After validating one block, train every manifest block sequentially on one GPU:

```bash
python tools/train_blocks.py \
    --manifest data/CameraA/blocks/manifest.json \
    --model_root output/cameraA_blocks -- \
    --bsz 1 -s data/CameraA --images train/images \
    --resolution 4 --iterations 30000 --not_use_dpt_loss
```


## Acknowledgement
We would like to express our gratitude to the authors of the following algorithms and libraries, which have greatly inspired and supported this project:

- [Grendel-GS: On Scaling Up 3D Gaussian Splatting Training](https://github.com/nyu-systems/Grendel-GS)
- [PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction](https://zju3dv.github.io/pgsr)
- [Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians](https://city-super.github.io/octree-gs/)
- [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](https://dekuliutesla.github.io/CityGaussianV2)
- [Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](https://github.com/Jixuan-Fan/Momentum-GS)



Your contributions to the open-source community have been invaluable and are deeply appreciated.

## BibTeX

```bibtex
@misc{gao2025citygsxscalablearchitectureefficient,
      title={CityGS-X: A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction}, 
      author={Yuanyuan Gao and Hao Li and Jiaqi Chen and Zhengyu Zou and Zhihang Zhong and Dingwen Zhang and Xiao Sun and Junwei Han},
      year={2025},
      eprint={2503.23044},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23044}, 
}
```
