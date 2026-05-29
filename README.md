# DDR：小相机运动视频的动态新视角合成

该工程面向单目动态视频输入。系统会先对视频帧进行预处理，得到相机参数、单目深度、前后向光流和运动区域掩码等先验信息；随后训练动态/静态双分支 NeRF，并联合优化相机位姿和焦距；最终可以进行新视角渲染、慢动作时间插值、空间-时间联合插值以及定量评估。

## 项目特点

- 支持小相机运动下的动态单目视频新视角合成。
- 采用动态 NeRF 与静态/刚体 NeRF 双分支结构。
- 使用 scene flow 建立相邻帧之间的时序一致性。
- 使用 DDR 损失约束 ray 上 rendering weights 的分布形状。
- 支持训练过程中联合优化相机位姿和焦距。
- 提供 MiDaS 深度、RAFT 光流、Mask-RCNN 掩码、COLMAP/RCVD 位姿转换等预处理脚本。
- 提供 Nvidia Dynamic Scene Dataset 风格的评估脚本。

## 代码结构

```text
DDR/
|-- README.md
|-- exp/
|   |-- run_nerf.py                 # 训练与渲染主入口
|   |-- render_utils.py             # 体渲染、动态/静态融合、视角/时间渲染工具
|   |-- run_nerf_helpers.py         # NeRF 网络、ray 生成、NDC 投影、损失函数和光流工具
|   |-- sampling_argmax.py          # DDR 渲染权重分布约束的核心实现
|   |-- load_llff.py                # LLFF/NSFF 风格数据加载
|   |-- poses.py                    # 可学习相机位姿残差模块
|   |-- focal.py                    # 可学习焦距模块
|   |-- evaluation.py               # PSNR、SSIM、LPIPS 等定量评估
|   |-- weights_map.py              # rendering weights 可视化工具
|   |-- softsplat.py                # 基于 CUDA/CuPy 的 soft splatting
|   |-- poseInterpolator.py         # 位姿插值工具
|   |-- Q_Slerp.py                  # 四元数插值工具
|   |-- run.py                      # 批量评估 Nvidia 数据集中的多个场景
|   |-- configs/config_nvidia/      # Nvidia Dynamic Scene Dataset 示例配置
|   `-- models/                     # LPIPS/感知距离评价网络及其权重
`-- scripts/
    |-- save_poses_nerf.py          # 将 COLMAP 输出转换为 poses_bounds.npy
    |-- get_rcvd_poses.py           # 将 RCVD 相机参数转换为 poses_bounds.npy
    |-- run_midas.py                # 生成 MiDaS 单目深度/视差先验
    |-- run_flows_video.py          # 生成 RAFT 前后向光流和一致性 mask
    |-- mask.py                     # 针对指定类别生成 Mask-RCNN 语义 mask
    |-- motion_mask.py              # 针对所有检测目标生成运动/对象 mask
    |-- flow_utils.py               # 光流可视化、缩放、warp、极线距离等工具
    |-- colmap_read_model.py        # COLMAP text/binary 模型读取工具
    |-- trans_pose.py               # 位姿序列重排工具
    |-- download_models.sh          # 下载外部预训练模型
    |-- core/                       # RAFT 网络实现
    |-- alt_cuda_corr/              # RAFT 可选 CUDA correlation 扩展
    `-- models/                     # MiDaS 网络结构
```

## 方法与代码对应关系

| 论文/方法模块 | 主要文件 | 关键函数或变量 |
| --- | --- | --- |
| 动态 NeRF 与 scene flow | `exp/run_nerf_helpers.py`, `exp/render_utils.py` | `NeRF`, `render_rays`, `raw_sf_ref2post`, `raw_sf_ref2prev` |
| 静态/刚体 NeRF 分支 | `exp/run_nerf_helpers.py`, `exp/render_utils.py` | `Rigid_NeRF`, `rgb_map_rig`, `sigma_rigid` |
| 动态/静态结果融合 | `exp/render_utils.py` | `raw2outputs_blending`, `raw_blend_w` |
| DDR 渲染权重分布损失 | `exp/sampling_argmax.py`, `exp/run_nerf.py` | `sampling_argmax_loss`, `arg_loss`, `w_arg` |
| 空空间/密度约束 | `exp/run_nerf_helpers.py`, `exp/run_nerf.py` | `compute_empty_loss`, `empty_loss` |
| 深度梯度平滑约束 | `exp/run_nerf_helpers.py`, `exp/run_nerf.py` | `compute_gradient_loss`, `gradient_loss` |
| 相机位姿联合优化 | `exp/poses.py`, `exp/run_nerf.py` | `LearnPose`, `learn_poses`, `pose_lr` |
| 焦距联合优化 | `exp/focal.py`, `exp/run_nerf.py` | `LearnFocal`, `learn_focal`, `focal_lr` |
| rendering weights 可视化 | `exp/weights_map.py` | `main`, `main2` |

## 环境配置

建议在 Linux + NVIDIA GPU + CUDA 的环境下运行。本项目继承了 NSFF、RAFT、MiDaS 和 soft splatting 等代码组件，部分脚本默认使用 CUDA。

### 1. 创建 Conda 环境

```bash
conda create -n ddr python=3.7 -y
conda activate ddr
```

建议使用 Python 3.7 或 3.8，以兼容较早版本的 PyTorch、torchvision、CuPy、RAFT 和 MiDaS 依赖。

### 2. 安装 PyTorch

根据本机 CUDA 版本选择合适的 PyTorch 安装命令。例如：

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
```

如果使用更新版本 CUDA，请参考 PyTorch 官网生成对应安装命令。

### 3. 安装 Python 依赖

```bash
pip install configargparse matplotlib opencv-python scikit-image scipy imageio tqdm kornia tensorboard pillow
pip install cupy-cuda102
```

其中 `cupy-cuda102` 需要根据 CUDA 版本替换，例如 `cupy-cuda11x` 或 `cupy-cuda12x`。

### 4. 可选：编译 RAFT CUDA correlation 扩展

```bash
cd scripts/alt_cuda_corr
python setup.py install
cd ../..
```

如果编译失败，通常是 CUDA Toolkit、PyTorch CUDA 版本或编译器版本不匹配导致的。也可以先使用默认 correlation 实现运行。

### 5. 下载外部模型权重

预处理阶段通常需要以下模型：

- RAFT 光流模型，例如 `models/raft-things.pth`。
- MiDaS 单目深度模型，`scripts/run_midas.py` 默认读取 `scripts/model.pt`。

可以先尝试运行：

```bash
cd scripts
bash download_models.sh
cd ..
```

如果脚本没有下载到 `model.pt`，请手动下载 MiDaS 权重并放置到：

```text
DDR/scripts/model.pt
```

## 数据目录格式

每个场景建议采用 LLFF/NSFF 风格目录结构：

```text
scene_name/dense/
|-- images/                  # 原始 RGB 视频帧，文件名需保证排序正确
|-- images_512x288/          # run_midas.py 生成的缩放图像
|-- sparse/                  # COLMAP sparse 输出，如果使用 COLMAP
|-- camera_params/           # RCVD 相机参数，如果使用 RCVD
|-- poses_bounds.npy         # NeRF/LLFF 相机参数文件
|-- disp/                    # MiDaS 生成的 .npy 深度/视差文件
|-- motion_masks/            # 运动区域或对象区域 mask
`-- flow_i1/                 # 训练阶段读取的 RAFT 光流 .npz 文件
```

注意：当前训练代码在 `exp/run_nerf_helpers.py` 中默认读取 `flow_i1`。如果 `scripts/run_flows_video.py` 生成的是 `flow_i3`，需要将其改名为 `flow_i1`，或者修改代码中的 `flow_dir`。

## 数据预处理流程

训练前需要依次准备相机参数、深度先验、光流和运动 mask。

### 1. 准备相机参数

如果使用 COLMAP 输出：

```bash
cd scripts
python save_poses_nerf.py --data_path /path/to/scene/dense
cd ..
```

输入目录中需要包含：

```text
images/
sparse/
```

如果使用 RCVD 输出：

```bash
cd scripts
python get_rcvd_poses.py
cd ..
```

当前 `get_rcvd_poses.py` 中的 `basedir` 是硬编码路径。运行前需要将其改为自己的数据目录，或自行将脚本改成支持 `--data_path` 参数。RCVD 输入目录中需要包含：

```text
images/
camera_params/*.npz
```

### 2. 生成 MiDaS 深度先验

```bash
cd scripts
python run_midas.py \
  --data_path /path/to/scene/dense \
  --resize_height 288 \
  --data_name scene_name
cd ..
```

该步骤会生成：

```text
images_*x288/
disp/*.npy
```

`--resize_height` 需要与训练配置文件中的 `final_height` 保持一致。

### 3. 生成 RAFT 光流

```bash
cd scripts
python run_flows_video.py \
  --model models/raft-things.pth \
  --data_path /path/to/scene/dense \
  --data_name scene_name
cd ..
```

该步骤生成相邻帧前后向光流以及光流一致性 mask：

```text
flow_i1/00000_fwd.npz
flow_i1/00001_bwd.npz
...
```

如果脚本输出目录是 `flow_i3`，可以改名：

```bash
mv /path/to/scene/dense/flow_i3 /path/to/scene/dense/flow_i1
```

### 4. 生成运动区域 mask

生成较宽泛的对象/运动区域 mask：

```bash
cd scripts
python motion_mask.py --data_path /path/to/scene/dense
cd ..
```

如果只希望针对指定 COCO 类别生成 mask，可以使用：

```bash
cd scripts
python mask.py --data_path /path/to/scene/dense
cd ..
```

训练阶段默认读取：

```text
/path/to/scene/dense/motion_masks/
```

## 训练流程

首先修改或新建 `exp/configs/config_nvidia/` 下的配置文件。至少需要修改：

```text
expname = MyScene
basedir = ./logs
datadir = /path/to/scene/dense
start_frame = 0
end_frame = 24
final_height = 288
```

启动训练：

```bash
cd exp
python run_nerf.py --config configs/config_nvidia/balloon1.txt
```

训练日志和 checkpoint 会保存到：

```text
exp/logs/{expname}_F{start_frame}-{end_frame}/
```

完整训练开销较大。论文设置中，每个场景大约训练 300k iterations，在单张 NVIDIA RTX A6000 上约需一天。显存不足时，可以降低 `N_rand`、`N_samples`、`chunk` 或 `netchunk`。

## 推理与渲染

训练完成后，使用相同配置文件加载 checkpoint 进行渲染。

### 固定时间的新视角路径渲染

```bash
cd exp
python run_nerf.py \
  --config configs/config_nvidia/balloon1.txt \
  --render_bt \
  --target_idx 10
```

### 固定相机视角的慢动作时间插值

```bash
cd exp
python run_nerf.py \
  --config configs/config_nvidia/balloon1.txt \
  --render_lockcam_slowmo \
  --target_idx 10
```

### 空间-时间联合插值

```bash
cd exp
python run_nerf.py \
  --config configs/config_nvidia/balloon1.txt \
  --render_slowmo_bt \
  --target_idx 10
```

渲染结果会保存在对应实验日志目录下。

## 定量评估

对于 Nvidia Dynamic Scene Dataset 风格的数据，可以运行：

```bash
cd exp
python evaluation.py --config configs/config_nvidia/balloon1.txt
```

批量评估多个示例配置：

```bash
cd exp
python run.py
```

评估脚本会计算 PSNR、SSIM 和 LPIPS 等指标。LPIPS 相关实现与权重位于 `exp/models/`。

## 重要参数说明

| 参数 | 含义 |
| --- | --- |
| `datadir` | 场景数据目录，需要包含图像、位姿、深度、mask 和光流。 |
| `expname` | 实验名称，用于日志和 checkpoint 保存。 |
| `final_height` | 训练图像高度，需要与 MiDaS 预处理高度一致。 |
| `start_frame`, `end_frame` | 训练使用的视频帧范围。 |
| `N_rand` | 每次迭代随机采样的 ray 数量。 |
| `N_samples` | 每条 ray 上采样的三维点数量。 |
| `netwidth`, `netdepth` | MLP 网络宽度和深度。 |
| `use_viewdirs` | 是否将观察方向作为 NeRF 输入，通常开启。 |
| `no_ndc` | 是否关闭 NDC。当前工程主要面向 forward-facing NDC 场景。 |
| `chain_sf` | 是否使用更长时间范围的 scene flow 一致性约束。 |
| `use_motion_mask` | 是否使用运动 mask 对动态区域进行 hard mining。 |
| `w_depth` | 普通 MiDaS 深度期望损失权重。 |
| `w_arg` | DDR 渲染权重分布损失权重。 |
| `num_samples` | DDR 中 Gumbel-Softmax 采样次数，论文常用 30。 |
| `basis_type` | DDR 子分布类型，可选 `tri`、`uni`、`gaussian`。 |
| `tau` | Gumbel-Softmax 温度系数，示例配置常用 2。 |
| `w_gradient` | 深度梯度平滑损失权重。 |
| `learn_poses` | 是否联合优化相机位姿，小相机运动场景建议开启。 |
| `learn_focal` | 是否联合优化焦距，内参不确定时建议开启。 |
| `pose_lr`, `focal_lr` | 位姿和焦距优化学习率。 |

## 常见问题与建议

- 如果训练时报找不到光流文件，优先检查目录是否为 `flow_i1`。
- 如果新视角结果出现明显重影，优先检查 RAFT 光流和 `motion_masks` 质量。
- 如果几何明显塌缩或漂浮，检查 `disp/*.npy` 是否正确生成，并适当调整 `w_arg`、`w_gradient` 和 `empty_loss` 相关参数。
- 对真实小相机运动视频，建议使用 RCVD 初始化相机参数，并开启 `learn_poses=True` 和 `learn_focal=True`。
- 显存不足时，优先降低 `N_rand`、`N_samples`、`chunk` 和 `netchunk`。
- 建议先用 20 到 60 帧视频做小规模实验，确认预处理和训练流程正确后再扩展到更长序列。

## 致谢

本工程基于 [Neural Scene Flow Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields) 搭建，并使用或改写了 NeRF、LLFF、MiDaS、RAFT、softmax splatting、位姿插值和 LPIPS 等相关代码组件。

## 引用

如果使用本仓库，请引用 DDR 论文以及本工程所基于的 NSFF 论文/代码。

```bibtex
@article{sun2025dynamic,
  title={Dynamic View Synthesis From Small Camera Motion Videos},
  author={Sun, Huiqiang and Li, Xingyi and Peng, Juewen and Shen, Liao and Cao, Zhiguo and Xian, Ke and Lin, Guosheng},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025}
}

@inproceedings{li2021neural,
  title={Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes},
  author={Li, Zhengqi and Niklaus, Simon and Snavely, Noah and Wang, Oliver},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
