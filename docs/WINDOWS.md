# Windows 11 / RTX 5060 部署与中性 SMPL-X 导出

仅推理，不需要训练。保留原 GVHMR 主网络和后处理。
环境名称为 `GVHMR`，Python 3.10；依赖基于本机 E:/GVHMR 的实测组合。
不要在这条部署路线运行原来的 `pip install -r requirements.txt`，它锁定的是旧版 Linux 环境。

## 1. 环境与安装命令

本机 `D:\anaconda3\envs\GVHMR` 已安装 Python 3.10.16 和全部推理依赖，
并通过 GPU 与完整短视频测试。日常只需 `conda activate GVHMR`，无需重复安装。
以下命令用于尚未创建环境的新部署，在 Anaconda Prompt / 已初始化 Conda 的 PowerShell 中执行：

```powershell
cd D:\GitHub\GVHMR
conda create -n GVHMR python=3.10.16 pip --no-default-packages -y
conda activate GVHMR
python -m pip install setuptools==78.1.1 wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-windows.txt -c constraints-windows.txt

# 保留 E 盘实际可用的 Windows CUDA 扩展与 chumpy 的 NumPy 兼容补丁。
# 本次已生成下面三个本地 wheel；重新生成也不下载任何东西。
python tools/windows/build_local_wheels.py --source E:/GVHMR/envs
python -m pip install --no-deps wheels/pytorch3d-0.7.9-cp310-cp310-win_amd64.whl wheels/chumpy-0.70-py3-none-any.whl wheels/cython_bbox-0.1.5-cp310-cp310-win_amd64.whl
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python tools/windows/check_environment.py
```

若 PowerShell 中没有 `conda` 命令，可先执行：

```powershell
(& 'D:\anaconda3\Scripts\conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
```

若已有完整但未安装 Python 的空环境，使用 `conda install -n GVHMR python=3.10.16 pip -y`。
若安装失败后目录缺少 `conda-meta/history`、解释器却残留其他 Python 版本的包记录，
应先备份失败目录，再一次性创建包含 Python 的环境，不要反复向残缺目录叠加安装。

PyTorch 安装组合参照 [官方版本表](https://pytorch.org/get-started/previous-versions/)。
CUDA 运行库由 PyTorch wheel 提供，使用这些二进制不要求另装 CUDA Toolkit。
PyTorch3D 本地 wheel **仅适用于 CPython 3.10 / Windows x64 / torch 2.11.0+cu130**，
不是官方发布的通用 Windows wheel，不能换 torch 后继续假定其兼容。
来源记录为 facebookresearch/pytorch3d commit `b6a77ad7aaf41ed90fca80ce6a2bac3c462a7881`；
wheel 保留 E 盘安装文件，并重新计算 RECORD。chumpy wheel 包含已有 NumPy 2 兼容修改；
cython_bbox 同样复用 CPython 3.10 编译产物，避免本机重新编译它。
如无法再访问 E 盘，应保留本地 `wheels/` 备份；重新源码编译需要匹配的 CUDA/MSVC 工具链，
不属于上述安装步骤。DPVO 及旧版 torch-scatter 不纳入默认环境。

## 2. 本地模型与视频工具

用户自己的中性 SMPL-X 模型来源为：
`D:\track_dataset\models\smplx\SMPLX_NEUTRAL.npz`。

以下脚本复制它，并从 E 盘复制已下载的四组推理权重、中性 SMPL 渲染资源与 FFmpeg/FFprobe：

```powershell
python tools/windows/prepare_assets.py --smplx-dir D:/track_dataset/models/smplx --source-project E:/GVHMR
```

本次已执行，模型位于 `inputs/checkpoints/`，视频工具位于 `tools/ffmpeg/bin/`。
源模型保持不变；复制时核验 SHA-256，不覆盖内容不同的现有目标。
这些大文件及本地 wheel 不纳入 Git。`SMPL_NEUTRAL.pkl` 只用于原版预览渲染，
实际重建和动画参数使用 `SMPLX_NEUTRAL.npz`。

## 3. 不依赖 Blender 的默认输出

```powershell
python tools/demo/demo.py --video "D:/videos/input.mp4" -s --no-render
```

`-s` 适合固定相机；移动相机去掉 `-s`，使用 SimpleVO。
默认预处理批量为 4，可用 `--batch-size 2` 降低显存占用。
模型依然处理整段时序，分批预处理不代表任意长视频都不会内存溢出。
输入通过 FFmpeg 真正采样为 30 fps，防止把 60 fps 视频简单标成 30 fps 导致动作减速。

输出在 `outputs/demo/<视频名>/`：

- `hmr4d_results.pt`：原始完整预测，包含世界/相机坐标两套参数。
- `smplx_neutral.npz`：无需 Blender 的中性 SMPL-X **动画参数文件**，不是视频或自带网格的 FBX。
- 省略 `--no-render` 时，还生成原版相机视角和全局视角预览。

NPZ 使用 `allow_pickle=False` 即可读取：

- `poses [T,165]`：55 关节轴角，顺序为根、21 身体关节、下颌/双眼、左右手。
- `trans [T,3]`：原 GVHMR 世界坐标平移，Y 向上，单位米。
- `gender='neutral'`、`model_type='smplx'`、`mocap_frame_rate=30`。
- `betas [10]`：逐帧预测体型的平均值，供固定体型动画使用。
- `betas_per_frame [T,10]`：保留逐帧原始体型；身体姿态也保留在 `body_pose` 等字段。
- `incam_*`：正确的相机坐标参数，未用世界坐标代替；相机坐标不等同于世界 Y-up 坐标。

中性指人体模型的 gender，不是强制所有 betas 为零。GVHMR 不估计手指和面部动作；
完整姿态中的手指填入该中性模型的平均手势，脸和表情为零，不是新增预测。
将 `poses` 交给 SMPL-X 时应使用 `use_pca=False, flat_hand_mean=True`，
避免重复添加手部均值。精确重建用 `betas_per_frame`，固定体型展示用 `betas`。

同名视频或相机参数变更会拒绝复用旧缓存；改用 `--output_root outputs/new_run`。
旧版无清单的缓存不会自动沿用，可用下节单独转换已有 `.pt`。

## 4. 可选：自动导出中性 SMPL-X FBX

```powershell
python tools/demo/demo.py --video "D:/videos/input.mp4" -s --no-render --export fbx
```

程序先保存 NPZ，然后后台调用已安装的：

- `D:\Blender Foundation\Blender 5.1\blender.exe`
- `D:\Blender Foundation\smplx_blender_addon-1.0.3-20260511\smplx_blender_addon`

无需手工打开 Blender，不使用 E 盘男性 SMPL FBX 模板。
插件导入 neutral SMPL-X 网格、体型和身体/手部骨架，再输出带动画的 `smplx_neutral.fbx`。
启用姿态修正形状关键帧；FBX 使用固定平均体型，不保留逐帧体型变化。
插件当前提供的是 locked-head SMPL-X 网格，与 Python NPZ 模型的网格版本可能有区别；
精确数值分析请使用原始 `.pt` 或 `.npz`。FBX 的坐标表示由插件/FBX 轴转换处理，
导入机器人流程时仍应确认接收软件的轴向和单位。

也可以明确指定路径：

```powershell
python tools/demo/demo.py --video "D:/videos/input.mp4" -s --no-render --export fbx --blender "D:/Blender Foundation/Blender 5.1/blender.exe" --smplx-addon "D:/Blender Foundation/smplx_blender_addon-1.0.3-20260511/smplx_blender_addon"
```

单独转换已有结果，无须重新推理：

```powershell
python tools/export_smplx.py --input "outputs/demo/input/hmr4d_results.pt" --fbx
```

去掉 `--fbx` 仅导出 NPZ。此命令也支持 E 盘已有的 `hmr4d_results.pt`；
原版结果按 30 fps 解释，只有明确知道其真实时基时才调整 `--fps`。

## 5. 验证

```powershell
python tools/windows/check_environment.py
python tools/bench/test_smplx_export.py
```

环境检查实际执行 PyTorch3D CUDA KNN 和光栅化，并加载 neutral SMPL/SMPL-X 模型。
迁移初期使用 E 盘解释器测试；之后已使用新 `GVHMR` 环境独立验收。

本次实际验证（2026-09-17，E 盘 Python 加载 D 盘源码）：

- 从零处理 36 帧视频，默认 SimpleVO 分支完成检测、ViTPose、HMR2、GVHMR 和中性 NPZ。
- 固定相机分支也完成推理；72 帧 / 60 fps 输入正确转换为 36 帧 / 30 fps。
- 原版两路预览渲染及视频合并成功。
- NPZ 按完整姿态和逐帧体型重建的顶点，与原模型参数重建最大差值为 0（所测三帧）。
- D 盘 Blender 5.1.2 导出后重导入 FBX，确认 neutral、10,475 顶点、56 骨骼（含 root）、
  36 帧动画和非零网格运动。重导入测试明确设为 30 fps、`anim_offset=0`。
- 新 `GVHMR` 环境安装完成后，`pip check` 无依赖冲突，GPU KNN/光栅化及中性模型检查通过。
- 新环境使用全新输出目录处理 36 帧视频，完成 SimpleVO、GVHMR、中性 NPZ、D 盘 Blender FBX、
  两路预览和视频合并，进程退出码 0；在项目目录之外导入 `hmr4d` 也正确定位到 D 盘源码。
- 长视频与 DPVO 不在此次验收范围。

初期测试产物在 `outputs/migration_check/`；新环境完整结果在 `outputs/environment_check/short/`。
安装清单和运行日志分别为 `outputs/migration_check/GVHMR-installed.txt`、
`outputs/migration_check/gvhmr_environment_run.log`，均不纳入 Git。
