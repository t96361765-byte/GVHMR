# 蘑菇全旋重建的离线后处理

这条流程读取已经生成的 GVHMR 结果，不训练网络，不修改 demo、ViT 或原始预测。
首个验证样例是 `mushroom_child_8`。支持准备、全旋、下器械；完整文件保留原帧数，
另存剪去前后致意的 trimmed 文件。不同视频必须配置自己的阶段与背景区域。

## 数据与约束

- 固定视频人物原有的 SMPL-X `betas`，通过参数优化维持骨架与网格的一致性。
- 先优化根平移和场景相机，再优化根朝向、身体关节。形状不变，但部分被遮挡关节和手腕可能明显调整。
- BVH 保留原始 23 关节层级，以其全旋周期建立相位模板。关节方向是弱先验，接触时序以视频观测修正。
- 视频自己的相位、圈数、节奏保留；没有把整段 BVH 拉伸或逐帧拷贝到视频上。
- 约束包含二维 COCO 关键点、支撑高度与滑动、圈间支撑区域、弱骨盆周期稳定、连续性和碰撞。
- 视频背景估计的逐帧旋转独立于人体优化。相机旋转不作为可以任意吸收人体漂移的自由变量。
- 内部采用米制 Z-up；`.pt/.npz` 继续使用 GVHMR 的米制 Y-up。输出世界原点位于拟合蘑菇的轴线与地面交点。

## 复现样例

### 一条命令运行全部流程

```powershell
& 'D:\anaconda3\envs\GVHMR\python.exe' 'D:\GitHub\GVHMR\tools\optimize_mushroom.py' --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --start 24 --end 266 --circle-start 92 --circle-end 223
```

从任意工作目录运行。依次执行背景相机估计、优化、二维预览、完整网格检查与三维预览、Blender 场景及 FBX 导出；某一步失败就停止并显示错误。
默认输出 `D:\track_dataset\GVHMR_correct_results\输入文件夹名`，已有结果需追加 `--overwrite` 或指定新的 `--output-root`。
四个帧号均对应原始输入视频，从 0 开始，结束帧不包含：准备 `[start,circle-start)`、全旋 `[circle-start,circle-end)`、下器械 `[circle-end,end)`。

新视频首次运行会弹窗：先框选静止背景并回车，再按顺序点击蘑菇顶部轴心、帽沿左端、帽沿右端、底座前方地面点并回车。按 R 重选点，Esc 取消。
默认显示 `--start` 帧；蘑菇被遮挡时可用 `--reference-frame` 指定清晰帧。标注存入输出根目录的 `_setup/输入文件夹名/annotations.json`，后续复用。
原有 child8 的确切输入路径复用已标注配置。更改动作区间时清除旧接触区间及旧评估圈边界，避免误用；如需细化准备/下器械手脚接触，可编辑保存的 JSON，或用 `--config` 指定自己的配置（其动作区间应与命令一致）。

默认沿用本机 BVH、其稳定全旋区间 `[740,1170)`、Blender 和 SMPL-X 插件路径。可通过 `--bvh`、`--bvh-range 起始 结束`、`--blender`、`--addon` 覆盖。
`--camera static` 仅适用于确认固定的相机；默认 `jitter` 适用于轻微转动抖动，背景跟踪失败不会静默当作固定相机。
`--dry-run` 只检查输入并打印各步骤，不写文件、不优化。新视频需先有含 `camera_roi` 和 `apparatus_pixels` 的配置才能 dry-run。
统一入口简化调用，并不代表自动识别器械或已验证任意动作、视角与分辨率。仍需检查每条输出的二维对齐和接触；只给三段边界不能精确指定上下器械时的手脚接触。

在 GVHMR 环境中，从仓库根目录运行。下面的绝对 Python 路径已在本机验证。

```powershell
& 'D:\anaconda3\envs\GVHMR\python.exe' tools/estimate_mushroom_camera.py --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --output 'D:\track_dataset\GVHMR_correct_results\camera_child8' --roi 485 10 1430 320 --reference-frame 60

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/refine_mushroom.py --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --bvh 'D:\track_dataset\Flare_bvh\marker53_optimized_ground.bvh' --config tools/configs/mushroom_child_8.json --camera-motion 'D:\track_dataset\GVHMR_correct_results\camera_child8\camera_motion.npz' --output-root 'D:\track_dataset\GVHMR_correct_results' --iterations 1100

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/preview_mushroom.py --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8'

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/audit_mushroom_mesh.py --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8'

& 'D:\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'D:\GitHub\GVHMR\tools\export_mushroom_blender.py' -- --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8' --addon 'D:\Blender Foundation\smplx_blender_addon-1.0.3-20260511\smplx_blender_addon'
```

输出目录非空时，优化器默认拒绝覆盖。换一个 `--output-root` 可保留各次实验；只有明确要重做时使用 `--overwrite`。
重做后还需重新生成预览和 Blender/FBX，避免查看旧动画。`--export-fbx` 也可直接调用既有导出器，仅生成 trimmed FBX。
若确定相机固定，可以省略 `--camera-motion`。当前背景估计适用于小幅转动/抖动；明显相机平移、视差、变焦或剪辑不在验证范围内。

## 换视频时需要配置什么

复制 `tools/configs/mushroom_child_8.json`，不要直接沿用它的帧号和像素位置。

| 字段 | 含义 |
|---|---|
| `keep_range` | 保留的准备至下器械区间；零起始、左闭右开 |
| `circle_range` | 连续全旋所在区间，用于自动估计相位与完整周期 |
| `bvh_circle_range` | BVH 稳定全旋区间，使用 BVH 自己的帧率；换视频不必换此区间 |
| `extra_hand_contacts` | 准备/下器械时明确的手支撑区间；每项为起始、终止、左右手索引，0 左 1 右 |
| `ground_contacts` | 明确的脚落地区间，左右索引同上；不要给腾空脚加地面约束 |
| `apparatus_pixels` | 参考帧上的蘑菇顶部轴心、帽沿左端、帽沿右端、底座前方地面点，依次为四个 `[x,y]` |
| `evaluation_cycle_boundaries` | 可选，固定评估区间以便与历史报告比较；不参与优化，不应从样例复制到其他视频 |
| `occluded_keypoints` | 可选，`[起始,终止,[COCO关节索引]]`，降低已知遮挡区域的二维权重 |
| `surface_stride` | 默认 3，身体表面采样步长；手脚额外保留完整顶点，设为 1 可使用全部表面顶点 |

背景 ROI 必须放在稳定的墙面、梁柱等位置，避开人群、器械和黑边。脚本还会排除目标人物框的并集。
`apparatus_pixels` 对应的参考帧应与背景跟踪的 `--reference-frame` 一致。
当前损失中像素尺度按样例 1920×1080 设置，换分辨率时应检查关键点误差与接触权重，不保证直接泛化。
相反方向的全旋需要显式处理 BVH 左右镜像，脚本会拒绝静默反转。

## 输出与检查

- `hmr4d_results.pt`、`smplx_neutral.npz`：完整原时长的修正结果。
- `hmr4d_results_trimmed.pt`、`smplx_neutral_trimmed.npz/.fbx`：保留区间，采样率不变。
- `*_corrected.blend`：人体、拟合器械、地面和固定查看相机，便于直接回放。
- `diagnostics.npz`：相机、相位、支撑权重、修正量、优化前后关节与采样表面。
- `metrics.json`：优化指标，包括只改平移阶段的结果。
- `validation.json`：完整 10475 顶点检查、导出一致性、固定历史区间的漂移。
- `blender_validation.json`：Blender 骨架与优化 FK 的数值一致性。
- 将 FBX 重新导入 Blender 时，把导入选项的动画偏移设为 0。Blender 默认偏移 1 会把动作从 1–242 帧移到 2–243 帧；配套 blend 已经是正确的 1–242 帧。
- `trajectory_comparison.png`、`reprojection_comparison.mp4`、`world_comparison.mp4`：对比图和视频。
- `provenance.json`：原始文件及 BVH 的哈希、路径、裁剪区间。

世界与相机参数由同一条动作及相机变换生成，不能继续复用旧的 `smpl_params_incam`。
修正 `.pt` 特意不包含旧 `net_outputs`，因为旧网络内部输出仍描述修正前的动作。

## 解释结果时的边界

1. 圈间均值位移是拟合目标，不是现实世界三维精度。双腕中点也不是质心或器械轴的真值。
2. 骨架样式相似不意味着 BVH 是视频逐帧真值。当前只验证一个视频，不应声称已解决所有运动员和镜头。
3. 器械为依据视频粗配准的圆柱加弧形顶面；不是原 blend 中蘑菇资产的精确形状/尺寸。
   原 blend 帽宽约 0.532 m、高约 0.443 m，而拟合结果使用重建人物的模型尺度。两者不能未经配准直接混合。
4. 碰撞统计采用拟合实体的内部深度代理，不能解释成精确网格有符号距离。手指仍为未观测的 SMPL-X 平均手姿，局部残余穿透和手掌朝向要看完整网格。
5. 原始动作在网格对比中只做一次整体配准，以第一圈平均双腕位置对齐；没有逐圈消除原结果的漂移。
6. 这不是动力学仿真或机器人可执行性验证；以后做重定向仍需检查真实器械、支撑与运动学约束。
