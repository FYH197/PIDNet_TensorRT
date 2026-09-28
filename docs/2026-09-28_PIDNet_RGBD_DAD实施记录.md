# PIDNet-S RGB-D 最小 DAD 实施记录

日期：2026-09-28。工作区：`F:\PIDnet_RT\PIDNet_TensorRT`。

## 1. 本次范围和版本

- 分支：`feature/da-pidnet`；起点：`9b53d57`，基线标签：`baseline-rgbd-v1`。
- 本次只实现最小 DAD，不实现 DG-BAG，不加入采样 B 或 C0。
- 目标是桌子、椅子等 15 类语义分割；不新增桌腿、椅腿等部件类别。
- 不改变数据划分、RGB-D 预处理、损失定义、Light-BAG、优化器、学习率、训练周期和 batch size。
- 替换的旧构造调用、导入和接口参数以注释保留；另有完整修改前快照。
- 未启动正式训练，未执行独立测试集评估，未提交或推送 Git。

## 2. 结构与插入点

输入为 `N×4×480×640`。第 4 阶段、当前 `pag4` 融合之前：

| 特征 | 尺寸 |
|---|---|
| P（`layer4_` 输出） | `N×64×60×80` |
| I（`relu(layer4(...))` 输出） | `N×256×15×20` |
| D（`diff4` 注入之后） | `N×64×60×80` |

```text
P → 1×1 Conv(64→16) → BN ──────────────────┐
I → 1×1 Conv(256→16) → BN → 双线性上采样 ──┤
                                           ↓
                                      abs(P'−I')
                                           ↓
                                3×3 Depthwise Conv → BN → ReLU
                                           ↓
                               1×1 Conv(16→64，零初始化)
                                           ↓
D（原 diff4 加法后）───────────────────────相加
                                           ↓
                               保存 temp_d → 原边界监督
                                           ↓
                                  layer5_d → 原 Light-BAG
```

归一化用于减轻特征尺度差异，并不证明差值等于语义错误。最终投影没有 ReLU，允许正负修正；使用零初始化输出投影，等价于让残差从零开始，不另加可学习缩放参数。

第一个反向步骤主要更新输出投影；它变为非零后，上游投影与深度卷积可以收到非零梯度。已用连续两步仅边界损失验证此路径。

新增模块在原模型初始化完成之后构造，并用 CPU RNG 隔离上下文保留原随机状态；因此同 seed 下原有参数初始化及后续 CPU 随机流不被新增层改变。现有训练入口在 CPU 构造模型后再移动到 GPU。

新增插值使用 `align_corners=False`，与原网络内部插值一致。原有损失端和预测端插值语义均保留。

## 3. 文件和位置

| 文件 | 位置及内容 |
|---|---|
| `models/model_utils.py` | 约第 316 行新增 `DiscrepancyAwareDerivative`；16 通道投影、绝对差、深度卷积、零初始化输出 |
| `models/pidnet.py` | 约第 129 行构造参数；第 223 行实例化；第 291 行计算差异；第 301 行注入 D；第 329、392 行训练/预测工厂参数传递 |
| `configs/default.py` | 第 38 行默认 `DAD_ENABLED=False`、`DAD_CHANNELS=16` |
| `configs/yaml/pidnet_s_sunrgbd15_rgbd_dad_4227_1057.yaml` | 新建独立配置，启用 DAD；其余配置与 baseline 相同 |
| `tools/train.py` | 第 196 行日志、第 427 行结果记录加入 DAD 开关与通道数 |
| `tools/inference_rgbd.py` | 第 35 行参数、第 74 行模型构造同步 |
| `tools/export_indoor_onnx.py` | 第 29 行参数、约第 58 行权重检查、约第 100 行模型构造同步 |
| `tools/export_indoor_onnx_mask.py` | 第 49 行参数、第 71 行构造同步 |
| `tests/verify_dad.py` | 新增可复现集成测试及短训练步计时工具 |

所有路径相对于仓库根目录。行号为本次完成时位置，后续修改可能移动。

原 baseline YAML 已与 `9b53d57` 逐文本核对，未改变。默认关闭时不实例化 DAD，不增加参数或执行额外卷积。

推理/导出需要明确使用 `--dad-enabled --dad-channels 16`。权重检查拒绝：DAD 权重配 baseline 结构、baseline 权重配 DAD 结构、DAD 通道数不匹配、缺失 DAD 参数，以及 DAD 推理权重缺失任何模型所需张量。训练时仍允许原 ImageNet 权重只初始化已有参数；这与推理必须加载完整训练权重是两个不同场景。

## 4. 已完成的验证

| 检查 | 结果 |
|---|---|
| 语法与 Git 空白检查 | 通过 |
| 关闭 DAD 与修改前模型比较 | 同 seed 的参数、键名和 CPU 输出完全一致 |
| 开启 DAD 的初始输出 | 零残差时与 baseline 三个输出逐元素完全一致 |
| 同 seed 原有参数及 RNG | 完全一致 |
| ImageNet 初始化 | 开启/关闭 DAD 的全部共享张量一致；保留原 RGB→RGB-D 首层适配逻辑 |
| 训练输出 | 语义辅助、主预测、边界预测分别为 `N×15×60×80`、`N×15×60×80`、`N×1×60×80` |
| 边界监督梯度 | 连续两步有限；第二步全部 DAD 可训练参数梯度非零 |
| 严格恢复 | 模型和 SGD momentum 状态恢复后，下一个相同更新的模型状态完全一致 |
| 错误和不完整权重 | 正确拒绝 |
| 真实数据链路 | 仅验证集首张 `sun_000006` 前向、损失有限检查通过 |
| 实际 RGB-D 推理脚本 | 生成 `640×480` 掩码，通过 |
| logits / mask ONNX | 两个实际导出入口均导出成功，并通过 `onnx.checker` |
| CUDA 完整训练步 | batch 44、FP32、原 OHEM＋边界损失、SGD，损失有限，未发生 OOM |

导出测试权重由短集成测试生成，仅证明接口可运行，不能当作正式模型或精度结果。训练模型导出时跳过的 26 个张量属于推理模型不需要的辅助头。当前没有 ONNX Runtime，因此没有验证其数值一致性，也没有在 TensorRT 或 Jetson Nano 上构建、测速。

## 5. 参数、训练步耗时与显存

参数统计包含训练辅助头：

| 模型 | 参数量 |
|---|---:|
| baseline | 7,720,191 |
| DAD | 7,726,575 |
| 增量 | 6,384（约 0.083%） |

设备为 RTX 5060 Ti。输入固定 `44×4×480×640`、FP32；使用原损失流程和 SGD。数据驻留 GPU，为合成图像和标签，不计磁盘读取、数据增强或验证时间。

先完成每组 2 步预热＋4 步计时的初测，发现预热波动，保留原记录；随后执行 A→DAD→DAD→A，每组 10 步预热＋20 步计时，并在计时边界同步 CUDA。

| 组别 | 20 步耗时中位数 | 峰值 allocated 显存 |
|---|---:|---:|
| baseline，第 1 组 | 0.62525 秒/步 | 13,198.17 MiB |
| DAD，第 1 组 | 0.62044 秒/步 | 13,255.71 MiB |
| DAD，第 2 组 | 0.62318 秒/步 | 13,266.31 MiB |
| baseline，第 2 组 | 0.61990 秒/步 | 13,199.15 MiB |

合并各模型 40 个计时步，中位数 baseline 为 0.62254 秒，DAD 为 0.62127 秒。差异很小且存在波动，不能解释为 DAD 加速。两模型各组最大 allocated 显存相差约 67.16 MiB（0.51%）。结论仅为：本次短测未见明显训练步降速，batch 44 可运行；不是正式整轮耗时、收敛、精度或目标板性能结论。

## 6. 证据和备份

本机目录：`F:\PIDnet_RT\dad_implementation_20260928`。

- `before/`：7 个被修改文件的原始副本。
- `before_sha256.json`：修改前文件校验值。
- `verification/verification_short_warmup.json`：短预热初测。
- `verification/verification.json`：延长预热后的原始逐步计时、显存和集成检查。
- `verification/additional.json`：修改前 baseline 对比、真实验证样本与语法检查记录。
- `final_checks/verification.json`：补充完整权重检查后的最终集成检查。
- `verification/dad_smoke.onnx`、`dad_smoke_mask.onnx`、`dad_smoke_mask.png`：集成测试产物，非正式训练成果。
- `after_sha256.json`：本次代码、配置、测试文件的完成态校验值。

桌面和仓库 `docs/` 各保存一份本记录并核对哈希一致。大型测试权重与证据位于仓库外，不混入代码提交。

## 7. 后续复现与训练方式

在仓库根目录执行检查（下面是 PowerShell 示例）：

```powershell
& F:\conda_env\PIDnet\python.exe -B tests\verify_dad.py --output F:\PIDnet_RT\dad_checks_new
```

需要重新测试速度时增加 `--benchmark`。工具默认每组预热 10 步、计时 20 步，使用合成数据，不产生正式训练结果。

正式训练前先审阅并提交当前代码，记录新 Git 提交编号。以下命令仅供后续使用，本次未执行：

```powershell
& F:\conda_env\PIDnet\python.exe tools\train.py --cfg configs\yaml\pidnet_s_sunrgbd15_rgbd_dad_4227_1057.yaml --seed 304 --run-name pidnet-s_sun15_rgbd_dad_s304 TRAIN.RESUME False
```

保持 batch 44、300 epoch、原验证频率和其余超参数；新运行使用新的运行目录，不把集成测试权重作为正式起点。需要恢复正式 DAD 训练时，使用其自身配置和 `last.ckpt`；不要混用 baseline 的 optimizer 状态。

目前未提交 Git，也未推送；既有 `.idea/`、`deploy/` 和独立脚本未改动。审阅后应显式选择本次修改文件和 docs/tests/新 YAML 提交，避免 `git add .` 混入已有未跟踪内容。

## 8. 证据边界与下一步

当前完成的是最小 DAD 的工程接入及验证，没有证明桌子、椅子 IoU 或整体 mIoU 改善。没有启用 DG-BAG，也没有正式单 seed 或三 seed 结果。

下一步是提交一个可追溯版本，再在固定协议下启动 DAD 单 seed 验证，观察收敛与各类别表现；只有确有收益后，才决定是否进一步实现 DG-BAG。独立测试集继续冻结。
