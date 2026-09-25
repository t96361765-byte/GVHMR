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

本次实际验证：

- 20 项自动化测试通过；默认配置与旧实现对照，当前视频基础阶段抽测 120 次迭代、腿部完整 800 次、脚部完整 600 次，两种坐标系下全部 SMPL-X 参数的最大差异均为 0。入口合并后也重新完成上述数值对照，检查的关节、重投影与掌面间隙诊断逐值一致；共用 CLI 的三个阶段另以少量迭代验证了参数传递、保存、裁剪及精简输出。
- `mushroom_child_8` 使用其阶段/器械标定，去除旧手工接触区间后，完成固定相机模式下 1400 + 800 + 600 次三阶段优化。保护帧逐值不变，输出有限，裁剪与完整数据一致，配置哈希一致；双脚方向夹角的全旋中段中位数从 43.80° 降至 6.14°。
- `no-hands` 实际运行 100 次迭代，确认手部组关闭、掌向权重为 0、支撑姿态放松门控为 0；`no-feet` 统一入口 dry-run 确认跳过脚部命令。已关闭的腿/脚函数也经过精确不修改数据的单元检查。
- 验证使用独立目录 `outputs/mushroom_constraints/`，未覆盖当前已接受的动画。第二条视频这次没有做视觉质量审核或 Blender/FBX 导出验证，不能据此宣称所有视频达到相同质量。
