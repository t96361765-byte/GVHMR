# 蘑菇全旋重建的离线后处理

这条流程读取已经生成的 GVHMR 结果，不训练网络，不修改 demo、ViT 或原始预测。
首个验证样例是 `mushroom_child_8`。支持准备、全旋、下器械；完整文件保留原帧数，
另存剪去前后致意的 trimmed 文件。不同视频必须配置自己的阶段与背景区域。

## 数据与约束

- 固定视频人物原有的 SMPL-X `betas`，通过参数优化维持骨架与网格的一致性。
- 先优化根平移和场景相机，再优化根朝向、身体关节。形状不变，但部分被遮挡关节和手腕可能明显调整。
- BVH 保留原始 23 关节层级，以其全旋周期建立相位模板。关节方向是弱先验，接触时序以视频观测修正。
- 准备支撑也使用 BVH：在已知全旋区间内估计低速手腕的支撑位置，再搜索此前低速、近似等高且位于该支撑区域的双手稳定段。以视频中稳定双手接触的开始、释放和起旋三个事件对应 BVH，而不是把两段完整视频线性拉伸。准备段只弱约束双臂和躯干方向。
- 从消除相机抖动后的二维手腕位置/速度识别接触，并保留抬手间隔；低位、低速且可信的脚部候选提供保守的落地约束。默认 `preparation_only: false`，准备、全旋及下器械阶段均保留手腕支撑高度、支撑范围、滑动和掌部贴合约束，按照识别出的接触权重应用，不强制双手全程贴面。`stage_support.applied_scope` 描述准备/下器械的新增接触检测范围；`active_refinement.palm_contact_scope` 描述掌部约束范围。它们是运动学推断，不是受力测量；`auto_stage_contacts: false` 可关闭自动阶段接触及准备姿态先验，再使用手工 `extra_hand_contacts` 和 `ground_contacts`。
- 接触阶段用固定的解剖掌心侧采样点贴近帽面，排除手背和手侧面；包含全旋时交替支撑的手掌。手指姿态仍未从视频重建，这不是精确手掌压力或受力模型。仍可显式设 `preparation_only: true` 限制新增约束，但这不是当前默认值。
- 掌心法线使用手腕、食指根部和小指根部定义的掌面，左右手符号分别处理。软朝向损失指向帽面的局部内法线，默认容许 30 度倾斜，且不固定掌面内的转向。接触权重低于 0.2 时该损失和新增掌心吸引严格为零；触地/离地附近按约 1–2 帧平滑过渡，悬空阶段加强对原始局部手腕姿态的保留。碰撞与时间平滑仍适用，所以不承诺悬空姿态逐帧完全不变。
- 可选参数：`palm_orientation_weight` 默认 0.6（0 关闭法线项），`palm_orientation_tolerance_deg` 默认 30，`palm_contact_min_confidence` 默认 0.2。`palm_support_weight`、`palm_normals_zup`、`palm_orientation_degrees` 和固定掌心顶点编号写入诊断；接触标签仍是运动学推断。
- 原有固定手腕离面高度是粗略代理。姿态优化阶段在可靠掌心接触时允许 `palm_wrist_height_slack_m`（默认 0.015 m）的高度容差，由实际掌心表面位置参与确定支撑高度；不调整时设为 0。它不允许掌心或手指任意穿透，原有碰撞损失继续适用。
- 已删除新增 Fourier 周期轨迹损失、曲线系数及相位调整变量，保留原始旋转相位目标；`image_weight` 默认为 1.0。原有去漂移、二维匹配、弱 BVH 肢体方向和防穿透约束仍保留。旧配置中的 `periodic_weight` 会归零，不能再启用该实验；诊断中的周期统计只用于观察，不参与优化。
- 视频自己的相位、圈数、节奏保留；没有把整段 BVH 拉伸或逐帧拷贝到视频上。
- 统一入口在基础优化后运行并腿步骤 `tools/refine_mushroom.py --stage legs`，仅调整全旋区间内部的双侧髋、膝、踝。准备、下器械、全旋首尾各两帧的参数逐值保留；根部、上身、手臂、相机、器械、体型全程固定。SMPL-X 姿态形变可能令手部皮肤发生毫米级变化，手腕骨架与掌面骨架法线不变。
- 并腿先验取 BVH 全旋段的踝/膝距离、股骨/胫骨方向差和屈膝统计，按目标腿长缩放并留容差；不做逐帧模仿。约束采用单侧软惩罚，保留自然不对称，加入二维观测、腿间胶囊采样防交叉、器械/地面碰撞及修正量的时间平滑。胶囊采样不是精确网格自碰撞检测。
- 并腿修正采用区间内五次平滑渐变，默认每端约 0.4 秒；先限制修正幅度、再乘权重，不能通过放大优化变量抵消渐变。只在全旋中段充分施加并腿约束，阶段边界附近允许保留原姿态。未新增圆或椭圆轨迹拟合。
- `closed_leg_refinement` 默认 `true`，设为 `false` 可在统一入口关闭该步骤；`closed_leg_iterations` 默认 800，`closed_leg_fade_seconds` 默认 0.4。修改 `_setup/视频名/annotations.json` 或通过 `--config` 提供。诊断新增 `closed_leg_envelope`、`closed_leg_delta`、`pre_closed_leg_*`，指标新增 `closed_legs`。本次样例的准备和结束“不变”是相对于已完成手掌优化的基础结果，不是未经优化的原始 GVHMR。
- 并腿之后的 `tools/refine_mushroom.py --stage feet` 只优化左右踝的局部旋转，髋、膝、踝位置和其余关节参数固定。由 BVH 脚趾末端方向及静止坐标系的足背方向，提取相对于小腿解剖坐标系的平均姿态；通过 SMPL-X 大脚趾、小脚趾、脚跟三点匹配脚长轴和脚面方向，不直接复制两种骨架的欧拉角。该先验随身体旋转，且保留软角度容差。
- 绷脚同样只在全旋区间内渐入/渐出，首尾各两帧和准备/结束完全保留。低位脚尖接近地面时由地面避碰与软姿态先验折中，不为追求绷直去移动膝、踝或改变下器械姿态。脚间胶囊采样减少交叉，但不是精确网格自碰撞保证。
- `foot_refinement` 默认 `true`，`foot_iterations` 默认 600，`foot_fade_seconds` 默认 0.4，`foot_orientation_tolerance_deg` 默认 6；角度容差是软惩罚的开始位置，不是硬性误差上限。诊断新增 `foot_envelope`、`foot_delta` 和少量脚部标记点，指标为 `feet`。
- 约束包含二维 COCO 关键点、支撑高度与滑动、圈间支撑区域、弱骨盆周期稳定、连续性和碰撞。
- 视频背景估计的逐帧旋转独立于人体优化。相机旋转不作为可以任意吸收人体漂移的自由变量。
- 内部采用米制 Z-up；`.pt/.npz` 继续使用 GVHMR 的米制 Y-up。输出世界原点位于拟合蘑菇的轴线与地面交点。

## 复现样例

### 一条命令运行全部流程

```powershell
& 'D:\anaconda3\envs\GVHMR\python.exe' 'D:\GitHub\GVHMR\tools\optimize_mushroom.py' --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --start 24 --end 266 --circle-start 92 --circle-end 223
```

从任意工作目录运行。依次执行背景相机估计、基础优化、全旋局部并腿优化、绷脚优化、二维预览、完整网格检查与三维预览、Blender 场景及 FBX 导出；某一步失败就停止并显示错误。
默认输出 `D:\track_dataset\GVHMR_correct_results\输入文件夹名`，已有结果需追加 `--overwrite` 或指定新的 `--output-root`。
四个帧号均对应原始输入视频，从 0 开始，结束帧不包含：准备 `[start,circle-start)`、全旋 `[circle-start,circle-end)`、下器械 `[circle-end,end)`。

新视频首次运行会弹窗：先框选静止背景并回车，再按顺序点击蘑菇顶部轴心、帽沿左端、帽沿右端、底座前方地面点并回车。按 R 重选点，Esc 取消。
默认显示 `--start` 帧；蘑菇被遮挡时可用 `--reference-frame` 指定清晰帧。标注存入输出根目录的 `_setup/输入文件夹名/annotations.json`，后续复用。
所有视频均只复用自己的缓存标定，或用 `--config` 显式提供标定，不再根据 child8 文件夹名套用样例。更改动作区间时清除旧接触区间及旧评估圈边界，避免误用；如需细化准备/下器械手脚接触，可编辑保存的 JSON（其动作区间应与命令一致）。

默认沿用本机 BVH、其稳定全旋区间 `[740,1170)`、Blender 和 SMPL-X 插件路径。可通过 `--bvh`、`--bvh-range 起始 结束`、`--blender`、`--addon` 覆盖。
统一入口默认优化 1400 次，可用 `--iterations` 修改。`metrics.json` 的 `stage_support` 记录自动识别的 BVH/视频支撑区间和事件对齐；`trajectory_*` 记录按相同绝对相位比较的圈间差异，不能当成真实运动精度。图区间共享边界采样点，避免分段绘图制造的缺口。
`--camera static` 仅适用于确认固定的相机；默认 `jitter` 适用于轻微转动抖动，背景跟踪失败不会静默当作固定相机。
`--dry-run` 只检查输入并打印各步骤，不写文件、不优化。新视频需先有含 `camera_roi` 和 `apparatus_pixels` 的配置才能 dry-run。
统一入口简化调用，并不代表自动识别器械或已验证任意动作、视角与分辨率。仍需检查每条输出的二维对齐和接触；只给三段边界不能精确指定上下器械时的手脚接触。

在 GVHMR 环境中，从仓库根目录运行。下面的绝对 Python 路径已在本机验证。

```powershell
& 'D:\anaconda3\envs\GVHMR\python.exe' tools/estimate_mushroom_camera.py --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --output 'D:\track_dataset\GVHMR_correct_results\camera_child8' --roi 485 10 1430 320 --reference-frame 60

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/refine_mushroom.py --input 'D:\track_dataset\GVHMR_results\mushroom_child_8' --bvh 'D:\track_dataset\Flare_bvh\marker53_optimized_ground.bvh' --config tools/configs/mushroom_child_8.json --camera-motion 'D:\track_dataset\GVHMR_correct_results\camera_child8\camera_motion.npz' --output-root 'D:\track_dataset\GVHMR_correct_results' --iterations 1100

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/refine_mushroom.py --stage legs --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8' --overwrite

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/refine_mushroom.py --stage feet --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8' --overwrite

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/preview_mushroom.py --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8'

& 'D:\anaconda3\envs\GVHMR\python.exe' tools/audit_mushroom_mesh.py --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8'

& 'D:\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'D:\GitHub\GVHMR\tools\export_mushroom_blender.py' -- --input 'D:\track_dataset\GVHMR_correct_results\mushroom_child_8' --addon 'D:\Blender Foundation\smplx_blender_addon-1.0.3-20260511\smplx_blender_addon'
```

输出目录非空时，优化器默认拒绝覆盖。换一个 `--output-root` 可保留各次实验；只有明确要重做时使用 `--overwrite`。
重做后还需重新生成预览和 Blender/FBX，避免查看旧动画。单独的并腿命令同样需要随后更新预览/导出，且拒绝在已并腿的结果上叠加运行；要重做可从统一入口重新生成基础结果。`--export-fbx` 也可直接调用既有导出器，仅生成 trimmed FBX。
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
背景跟踪优先直接匹配参考帧；匹配不足时在上一帧的背景区域补充特征并进行相邻帧恢复。恢复仍要求至少 15 个内点且旋转拟合中位误差不超过 2 像素，最多连续恢复 30 帧，超过则停止。恢复帧写入 `camera_motion.json` 的 `adjacent_recovery_frames`；这些帧的拟合误差是局部匹配误差，不代表累计相机精度。此恢复已在 `20260625_mushroom2` 上运行验证，不能替代对明显平移、视差和剪辑的专门处理。
`apparatus_pixels` 对应的参考帧应与背景跟踪的 `--reference-frame` 一致。
像素阈值和重投影尺度现在以 1920 宽为参考，按实际视频宽度缩放；这解决分辨率缩放问题，不能替代对不同镜头远近、裁剪和遮挡的检查。统一参数、消融预设及命令见 [MUSHROOM_CONSTRAINTS.md](MUSHROOM_CONSTRAINTS.md)。
相反方向的全旋需要显式处理 BVH 左右镜像，脚本会拒绝静默反转。

## 输出与检查

- `hmr4d_results.pt`、`smplx_neutral.npz`：完整原时长的修正结果。
- `hmr4d_results_trimmed.pt`、`smplx_neutral_trimmed.npz/.fbx`：保留区间，采样率不变。
- `*_corrected.blend`：人体、拟合器械、地面和固定查看相机，便于直接回放。
- `diagnostics.npz`：默认只保存相机、相位、支撑权重、修正量、优化前后关节和少量标记点。移除体积较大的 `original_surface_zup`、`corrected_surface_zup` 网格缓存，预览、完整网格检查及 Blender 导出均兼容精简文件；需要调试完整缓存时，统一入口追加 `--full-diagnostics`。最终动画 `.blend/.fbx` 保持完整，不属于诊断缓存。
- `metrics.json`：优化指标，包括只改平移阶段的结果。
- `validation.json`：完整 10475 顶点检查、导出一致性、固定历史区间的漂移。
- `blender_validation.json`：Blender 骨架与优化 FK 的数值一致性。
- 将 FBX 重新导入 Blender 时，把导入选项的动画偏移设为 0。Blender 默认偏移 1 会把动作从 1–242 帧移到 2–243 帧；配套 blend 已经是正确的 1–242 帧。
- `trajectory_comparison.png`、`reprojection_comparison.mp4`、`world_comparison.mp4`：对比图和视频。
- `provenance.json`：原始文件及 BVH 的哈希、路径、裁剪区间。
- `constraints.json`：小型完整有效约束配置，与指标/来源文件中的 `constraints_sha256` 对应，用于复现实验。

世界与相机参数由同一条动作及相机变换生成，不能继续复用旧的 `smpl_params_incam`。
修正 `.pt` 特意不包含旧 `net_outputs`，因为旧网络内部输出仍描述修正前的动作。

## 解释结果时的边界

1. 圈间均值位移是拟合目标，不是现实世界三维精度。双腕中点也不是质心或器械轴的真值。
2. 骨架样式相似不意味着 BVH 是视频逐帧真值。少数视频上的数值回归或完整运行不能证明所有运动员和镜头都能达到相同视觉质量。
3. 器械为依据视频粗配准的圆柱加弧形顶面；不是原 blend 中蘑菇资产的精确形状/尺寸。
   原 blend 帽宽约 0.532 m、高约 0.443 m，而拟合结果使用重建人物的模型尺度。两者不能未经配准直接混合。
4. 碰撞统计采用拟合实体的内部深度代理，不能解释成精确网格有符号距离。手指仍为未观测的 SMPL-X 平均手姿，局部残余穿透和手掌朝向要看完整网格。
5. 原始动作在网格对比中只做一次整体配准，以第一圈平均双腕位置对齐；没有逐圈消除原结果的漂移。
6. 这不是动力学仿真或机器人可执行性验证；以后做重定向仍需检查真实器械、支撑与运动学约束。
