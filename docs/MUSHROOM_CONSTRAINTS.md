# 并腿全旋约束与消融实验

统一运行入口仍是 `tools/optimize_mushroom.py`。四个阶段帧号、背景 ROI、蘑菇关键点是每条视频的正常标定输入，仍由使用者提供；它们不属于视频专用的算法规则。

## 配置与代码位置

- `hmr4d/utils/mushroom_config.py`：唯一的参数默认值、配置校验、旧配置兼容和消融预设定义。
- `tools/configure_mushroom_constraints.py`：生成/修改配置的命令行工具，无需改优化代码。
- `tools/configs/mushroom_constraints.json`：可直接编辑的完整默认配置；通过 `--constraints` 显式传入。它是默认值的导出模板，不会被所有命令自动读取。
- `mushroom_refine.py`：基础轨迹、器械和手部联合优化；`mushroom_contact.py` 提供掌面几何与接触门控；`mushroom_priors.py` 处理 BVH 与视频支撑事件。
- `mushroom_legs.py`、`mushroom_feet.py`：全旋区间内的局部腿、脚优化。两阶段的变量、先验和保护范围不同，保留分模块以便隔离消融。
- 腿脚共用的时间过渡函数并入 `mushroom_priors.py`；`mushroom_io.py` 统一所有阶段的裁剪、动画保存与精简诊断输出。

上述 `mushroom_*.py` 模块位于 `hmr4d/utils/`。基础、腿、脚共用 `tools/refine_mushroom.py` 的参数解析和配置流程；两个仅转发的旧入口已删除。总入口 `optimize_mushroom.py` 的命令不变。单独运行局部阶段改为 `refine_mushroom.py --stage legs` 或 `--stage feet`，原有 `--input`、`--overwrite` 等参数保留；基础阶段默认 `--stage base`，也可省略。

## 正常运行

```powershell
python "D:\GitHub\GVHMR\tools\optimize_mushroom.py" `
  --input "D:\track_dataset\GVHMR_results\20260625_mushroom2" `
  --start 24 --end 364 `
  --circle-start 165 --circle-end 311 `
  --constraints "D:\GitHub\GVHMR\tools\configs\mushroom_constraints.json" `
  --overwrite
```

省略 `--constraints` 时，继续使用代码默认值与已保存的该视频配置。换视频时更换输入路径与四个帧号，首次运行按原流程标定；不会按文件夹名称自动套用 child8 的标定。

参数优先级：代码默认值 → 标定文件中的旧扁平参数 → 标定文件中的 `constraints` → 显式 `--constraints` 文件 → 命令行 `--iterations`。最后一项在统一入口只覆盖基础优化次数；腿/脚次数由配置控制。做正式实验时建议每次显式传入完整配置，避免旧缓存设置影响实验。

## 参数分组

| 组 | 主要作用 | 常用参数 |
|---|---|---|
| `shared` | 随机种子、图像残差尺度与遮挡权重 | `seed`、`image_sigma_px`、`scale_pixels_with_image` |
| `base` | 二维一致性、地面/器械碰撞、相位、弱 BVH 姿态先验、平滑和漂移控制 | `iterations`、`weights.*` |
| `hands` | 支撑检测、腕部位置/滑移、掌面接触、掌心朝向、准备阶段姿态 | `enabled`、`weights.*`、`detection.*`、`orientation_tolerance_deg` |
| `legs` | 并腿距离、大腿/小腿平行、屈膝、碰撞与原动作保留 | `enabled`、`weights.*`、`fade_seconds`、各距离/角度余量 |
| `feet` | 相对小腿的脚部姿态、双脚平行、脚尖距离和地面约束 | `enabled`、`weights.*`、`orientation_tolerance_deg`、`fade_seconds` |

所有损失权重集中到各组 `weights`。将某个权重设为 `0`，可移除该损失项的梯度；关闭整个局部阶段请用 `enabled=false`，它会完全跳过优化，而不是让优化器运行零权重损失。

参数单位：`_m` 为米，`_deg` 为度，`_rad` 为弧度，`_seconds` 为秒，`_px` 为像素。像素尺度默认以 1920 宽为参考，按实际视频宽度缩放；支持检测中的器械相对距离按帽沿宽度计算。统一入口自动保存 `image_size`，直接调用底层函数而未提供该值时，会用相机内参的主点估计图像宽度。裁剪或非中心主点视频应显式提供真实 `image_size`。

示例：降低脚部朝向约束并扩大容许角度：

```powershell
python "D:\GitHub\GVHMR\tools\configure_mushroom_constraints.py" `
  --input "D:\GitHub\GVHMR\tools\configs\mushroom_constraints.json" `
  --set feet.weights.orientation=1.0 `
  --set feet.orientation_tolerance_deg=10 `
  --output "D:\GitHub\GVHMR\tools\configs\feet_soft.json"
```

未知字段、负权重、非法类型及越界参数会报错，不会静默忽略。重复写已有文件需要 `--overwrite`。

## 消融实验

| `--preset` | 准确含义 |
|---|---|
| `default` | 从默认值生成；若同时给 `--input`，保留其值，不追加消融修改 |
| `no-hands` | 关闭手部组全部直接损失、准备上肢先验及支撑时腕部姿态放松 |
| `no-palm-orientation` | 只关闭掌心朝向损失，保留接触位置、滑移和碰撞等 |
| `no-hand-release` | 关闭新增离器证据与统一支撑门控，恢复旧版位置接触权重；用于定位抬手被压制的来源 |
| `no-arm-body-collision` | 关闭前臂/手与躯干、腿部的表面防穿透，保留人体与器械碰撞 |
| `no-legs` | 跳过局部并腿阶段，仍执行基础和脚部阶段 |
| `no-feet` | 跳过局部脚部姿态阶段，仍执行基础和并腿阶段 |
| `no-local-stages` | 同时跳过并腿、脚部阶段，保留基础及手部优化 |

`no-hands` 不等于删除所有手部信息：二维重投影、相位/初始配准及基础 BVH 姿态先验仍可能使用手腕或手臂。`no-feet` 也不等于冻结脚踝：基础优化和腿部阶段仍可改变踝关节。预设用于分别消融明确的约束组/优化阶段，不宣称各组完全没有运动学耦合。

关闭脚部阶段的示例：

```powershell
python "D:\GitHub\GVHMR\tools\configure_mushroom_constraints.py" `
  --preset no-feet `
  --output "D:\GitHub\GVHMR\tools\configs\no_feet.json"

python "D:\GitHub\GVHMR\tools\optimize_mushroom.py" `
  --input "D:\track_dataset\GVHMR_results\20260625_mushroom2" `
  --start 24 --end 364 `
  --circle-start 165 --circle-end 311 `
  --config "D:\track_dataset\GVHMR_correct_results\_setup\20260625_mushroom2\annotations.json" `
  --constraints "D:\GitHub\GVHMR\tools\configs\no_feet.json" `
  --output-root "D:\track_dataset\GVHMR_ablation_results\no_feet"
```

各实验都从同一份**原始 GVHMR 结果**重新运行，复用同一组帧号、BVH、器械标定、相机模式、种子和迭代次数，使用不同输出根目录。已有优化结果上关闭参数不会撤销已发生的修正。局部命令拒绝重复优化，也不允许借局部运行修改其他组后误标为整套新配置。

每次运行保存小型 `constraints.json` 完整有效配置，并在 `metrics.json` / `provenance.json` 写入 `constraints_sha256`。原标定仍在 `config.json`。这些不是大型逐帧诊断数据；对比图、视频和动画的输出方式不变。

## 手部离器与人体自身穿透

全旋中的离器证据来自两腕三维相对高度，以及手腕超出器械支撑区上界的二维距离，分别按前臂长度与蘑菇帽沿宽度归一化，并结合关键点可信度。二维判断不直接比较两腕像素高度，避免把器械前后两只支撑手的透视高度差当作抬手。这样即使抬手最高点瞬时速度接近零，也不会仅因“慢”而判为支撑。该证据只在指定全旋区间内抑制接触；BVH 接触先验不能盖过明确的离器证据。高度、半径、防滑、掌面位置与朝向共用平滑后的支撑权重，悬空时一起退出，过渡不是二值开关。

- `hands.detection.release_enabled`：启用上述机制。`release_lift_forearm_start/end` 与 `release_lift_width_start/end` 控制两个归一化证据的平滑过渡；均为无量纲比例，不是固定帧号或厘米。
- `hands.weights.self_collision`：前臂/手对躯干、头、腿部的表面防穿透权重。按 SMPL-X 解剖区域取样，几何随原运动员体型变化；基础阶段和后续并腿阶段均使用，避免并腿时把腿移入固定的手中。
- `hands.self_collision_tolerance_m` / `self_collision_sigma_m`：允许的微小表面交叠及惩罚尺度。它是稀疏表面的软约束，不保证完全无穿透，不强制悬空手掌朝向。

上述机制不含视频名、圈数或具体帧索引，不新增人工接触标注。准备/下器械仍使用原阶段检测。离器判断假设画面基本正立、全旋时通常有另一只手支撑；两腕严重遮挡、原三维高度严重错误或特殊腾空动作需复核。仅二维置信度高不能保证三维高度正确，因此仍须结合输出视频检查。

## 通用性检查与边界

1. 约束核心没有写死当前视频的全旋帧号、圈数或文件夹名称。手部由观测与 BVH 推断支撑；腿部参考 BVH 的分布并按目标腿长缩放；脚部参考相对小腿的解剖坐标，不依赖世界朝向。
2. 腿、脚增量严格限制在手动标出的全旋区间，准备/结束帧原样保留；全旋首尾各两帧不动，中间用五次平滑过渡，默认渐变时长 0.4 秒。这里的“保留”指相对于基础优化后的动作，基础阶段仍负责原有手部和轨迹修正。
3. 保留原有跨圈漂移控制，未重新引入将每圈轨迹拟合成圆或椭圆的约束。
4. 当前适用范围是相近模式的并腿全旋，固定或轻微旋转抖动相机、无变焦。旋转方向需与 BVH 一致；镜像需另行明确处理。基础漂移估计要求至少两个完整圈。
5. 仍依赖本 BVH 的关节命名、米制和 Z-up，以及 SMPL-X 几何。换动捕格式/单位、分腿全旋、剪式动作、明显相机平移或严重遮挡不保证直接可用。器械是近似曲顶模型，极端身高/器械尺寸可能需要调整初始化和几何边界。
6. 默认权重主要在 30 fps 数据验证。接触检测和腿脚过渡已按时间/分辨率处理，但部分平滑项仍基于采样差分；帧率或动作速度差异明显时应复核平滑权重。物理距离余量可调，并非所有距离均按人体比例缩放。

自动化检查：

```powershell
python -m unittest tools.bench.test_mushroom_priors tools.bench.test_mushroom_config -v
```

测试涵盖悬空手不受掌向损失、脚部先验对世界旋转不变、跨分辨率接触一致性、阶段过渡与保护范围、消融梯度隔离、配置优先级和校验。

此前结构整理时的验证记录（新增手部约束的验证另记于下文）：

- 20 项自动化测试通过；默认配置与旧实现对照，当前视频基础阶段抽测 120 次迭代、腿部完整 800 次、脚部完整 600 次，两种坐标系下全部 SMPL-X 参数的最大差异均为 0。入口合并后也重新完成上述数值对照，检查的关节、重投影与掌面间隙诊断逐值一致；共用 CLI 的三个阶段另以少量迭代验证了参数传递、保存、裁剪及精简输出。
- `mushroom_child_8` 使用其阶段/器械标定，去除旧手工接触区间后，完成固定相机模式下 1400 + 800 + 600 次三阶段优化。保护帧逐值不变，输出有限，裁剪与完整数据一致，配置哈希一致；双脚方向夹角的全旋中段中位数从 43.80° 降至 6.14°。
- `no-hands` 实际运行 100 次迭代，确认手部组关闭、掌向权重为 0、支撑姿态放松门控为 0；`no-feet` 统一入口 dry-run 确认跳过脚部命令。已关闭的腿/脚函数也经过精确不修改数据的单元检查。
- 验证使用独立目录 `outputs/mushroom_constraints/`，未覆盖当前已接受的动画。第二条视频这次没有做视觉质量审核或 Blender/FBX 导出验证，不能据此宣称所有视频达到相同质量。

2026-09-25 手部离器与表面防穿透更新的验证：

- 23 项自动化测试通过，包含抬手最高点零速度、人体缩放/根平移、左右手互换、图像缩放、支撑手透视高差、低可信度观测，以及防穿透梯度方向和刚体变换不变性。
- `mushroom_child_8` 和 `20260625_mushroom2` 使用完全相同的默认约束配置，各自完成 1400 + 800 + 600 次三阶段优化，复用各自人工阶段/器械标定和背景相机轨迹。未加入逐视频手工接触修补。
- child_8 左腕相对右腕的最高抬起量由旧优化的 3.6 cm 恢复到 28.8 cm；四个原始左手抬起峰值帧的左手支撑权重为零、右手为一。mushroom2 同一峰值从 35.2 cm 到 38.1 cm，原有抬手保留。这不是掌面离器的绝对测量。
- 全旋中至少三个前臂/手网格面与非相邻身体区域相交的帧数，child_8 左/右由 47/41 到 30/18，mushroom2 由 38/8 到 19/11。最大相交面数分别从 197/187 到 51/29、206/23 到 51/13。计数包含微小交叠；mushroom2 右侧的交叠帧数未下降，不能宣称各部位均无穿透。
- 二维加权重投影误差分别由 11.33 到 8.16 px、15.09 到 14.00 px。阶段外腿脚保护、完整/裁剪 PT/NPZ 一致性、相机/世界坐标一致性均通过；全部对比视频完整解码。Blender/FBX 均重新导出并回导抽查，FBX 最大关节偏差分别为 1.30 mm、1.11 mm。
- 验证后的动画、预览和报告已覆盖对应 `D:\track_dataset\GVHMR_correct_results` 子目录。之前的数据快照及独立全网格对比保存在 `outputs/mushroom_hand_update/`，未向结果目录输出大型表面缓存。两条视频验证不等于所有视角/遮挡条件均已验证。
