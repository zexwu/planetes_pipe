# PLANETES 0.5：按科学步骤组织

这次将 DARK、FLAT、WAVE、P2VM 和 REDUCE 放在同一层级。
每个文件保留自己的文件输入、参数、科学计算、内部 helper 和返回结果。
完整处理入口放在文件前部，下面是数组计算与算法细节。
Python、PyEsoRex 和 GUI 调用同一份实现。

## 文件与入口

| 步骤 | 文件入口 | 同文件的数组计算 | 主要输入 → 输出 |
| --- | --- | --- | --- |
| DARK/BAD | dark.calibrate_dark_file | compute_dark / compute_badpix | raw dark → mean、temporal RMS、坏像素图 |
| FLAT | flat.calibrate_flat_file | compute_flat / trace_spectrum | 四台望远镜 flat + DARK/BAD → 响应与提取几何 |
| WAVE | wavelength.calibrate_wavelength_file | compute_wavelength / line matching | lamp + DARK/FLAT + line list → 波长图与拟合残差 |
| P2VM | p2vm.calibrate_p2vm_file | extract_calibration_files / compute_p2vm | telescope/baseline 序列及标定 → 完整物理响应与逆矩阵 |
| REDUCE | reduction.reduce_file | reduce_spectra / compute_gdelay | science + DARK/FLAT/WAVE/P2VM → 仪器级观测量 |

每一步的 CPL 参数名称、说明和别名在本文件 `PARAMETERS` 中声明；
Python 函数签名给出默认值。字典本身不依赖 CPL。
PyCPL 在 recipes/ 中将这些定义翻译成 ParameterList。
WAVE 没有新增可调参数，保留原有检测、匹配和剔除常量。

```text
planetes/
├── dark.py                DARK/BAD 输入、参数与计算；load_dark
├── flat.py                FLAT 输入、aperture 参数、response 与 traces
├── wavelength.py          lamp/reference 输入、谱线匹配、波长拟合
├── p2vm.py                序列、目标网格、通道识别、P2VM 计算
├── reduction.py           science 输入、观测量、GD；FrameReducer/load_reducer
├── extraction.py          共用像素提取、波数重采样、16帧分块提取
├── products.py            具名数组、单位、轴、必要检查和父标定指纹
├── instrument.py          telescope/baseline/output 约定与探测器配置
├── qc.py / plotting.py    质量统计和绘图
├── io/                    raw 元数据、产品 FITS、keywords、不覆盖写入
└── recipes/               SOF、CPL 参数翻译与 DFS 产品登记
```

`calibration/`、`tasks.py` 和 `identity.py` 已删除。
P2VM 特有的序列组合、望远镜交集和调制通道识别归入 p2vm.py；
extraction.py 保留多个步骤确实共用的提取功能。产品内容指纹作为
products.py 的一个函数保留，用于防止同形状但来自不同标定的产品混用。
没有增加 dispatcher、兼容转发模块或另一套算法目录。

CPL import 全部限定在 recipes/。SOF 适配器负责角色与数量，
raw 的分类和科学关联由实际步骤检查。QC 读取产品，不再依赖步骤模块；
普通诊断中的条件数直接针对已保存的 P2VM 逆矩阵计算。
当前22个非初始化模块不存在循环 import，包括函数内部的 import。

## 单位与检查

波长、OPD、GD 的 Python 数组、函数参数和 FITS 数值统一使用 **m**。
删除 um/m 往返、原始编码复用和私有 `_encoding` 缓存。
例如 `wavelength_m`、`gd_search_min_m=-30e-6`；
粗搜步长为 `1e-8 m`，物理范围和 Newton 算法保持原定义。
这是 GRAVITY 的波长数据表使用的米制约定；相位保留 rad，差分相位保留 deg，
通量保留 ADU。GRAVITY 源码同时定义的 `... UM SC` QC 卡仅在写 header 时换算，
不进入计算过程。Python 像素索引仍为零基，FITS 输出按参考使用一基。

检查保留在会影响结果的边界：

- 输入类别、曝光和探测器几何匹配；防止错扣 dark。
- 父标定、波长覆盖和 ABCD/telescope 映射；防止同形状标定混用。
- 稀疏索引、必要数组形状及参数范围；防止像素越界和无意义计算。
- 外部 FITS 的单位、checksum、内容指纹和不覆盖写入。

删去逐字段的通用 metadata/类型遍历、仅检查 hash 字符串外观的逻辑，以及
SOF、文件入口、分块读取之间重复的 raw 分类检查。P2VM 序列不再在每次提取时
重新检查同一批 raw/DARK。实时循环不计算产品指纹。
QC 继续只报告，不新增 rejection、阈值或权重。

## 流程图

![代码组织与调用关系](architecture.png)

![数据和标定的流向](dataflow.png)

## 实际验证结果

| 验证 | 结果 |
| --- | --- |
| 维护测试 | 54项通过，包括真实 PyCPL/PyEsoRex、父标定、单位、flags 与文件保护 |
| 普通 Python | 阻止导入 CPL/PyEsoRex 后，五个步骤和实际 DARK/QC 计算仍能运行 |
| 完整台架数据 | 复用之前逐像素核对的15份标准化 raw；3份 DARK、FLAT、WAVE、P2VM、science，冷/热各完整执行一遍 |
| 结构变化的对照 | 旧算法直接使用米制输入，仅将有量纲的 GD 网格步长改为米制；每轮37个数组、合计74个数组逐字节一致 |
| 离散选择 | 相对原微米计算，BAD、visibility flags、提取像素、baseline 映射、最终保留谱线的坐标不变 |
| 文件 | 两轮共20份产品均能读回并通过内容指纹；米制长度连续读写不发生单位往返 |
| 现有 Qt6 GUI | offscreen fake-data 启动、capture 元数据、浮点像素、显示 binning 一致性、dark-off/single-frame-dark 通过 |
| 安装后调用 | 0.5.0 wheel 在项目目录外导入，5个recipe可发现，使用随包 line list 的真实 WAVE 执行通过 |
| 包与依赖 | wheel/sdist 构建、uv lock检查通过；wheel 不包含已删除模块或研究 PDF |

结构整理与单位变化分开评估。直接把旧算法的输入全部改为米，再比较新实现，
能检验本次代码移动和职责收敛是否引入额外改变。两者冷/热各37数组完全一致。
旧微米算法与新米制计算的最大绝对差异如下，不能把它们描述为与旧版逐字节一致：

| 数量 | 冷执行 | 热缓存执行 |
| --- | ---: | ---: |
| wavelength map [m] | 1.33e-17 | 1.56e-16 |
| VISAMP | 4.36e-8 | 3.68e-7 |
| science GD [m] | 7.10e-15 | 5.82e-14 |
| differential phase [deg] | 2.96e-6 | 2.52e-5 |

对照使用同一 Python/NumPy/Numba 环境；冷、热分别比较，因为原算法本身存在
已观测到的冷/热数值差异。拟合公式、物理默认值、原始像素、权重和科学切选未修改。
GUI 验证使用记录数据，未连接真实台架硬件。

## 使用与恢复

本轮源码和 GUI 修改前备份：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_000358_step_modules/`

本轮数据产品、数值对照和 GUI 结果：

`reduced/step_modules/20261028_000358/`

Python 导入按上表对应的步骤；`load_dark` 在 dark.py，
`load_reducer` 在 reduction.py。现有 GUI 与便携补丁已同步更新。

版本为 **0.5.0 / product format 4 / raw format 3**。
已有标准化 raw 可直接使用；旧 calibration/science 产品需要重新生成。
旧 `*_um` 参数和 CLI `*-um` 别名改为 `*_m` / `*-m`，参数数值相应使用米。
没有保留旧格式的隐式转换缓存。
