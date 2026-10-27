# PLANETES 0.5.1：科学步骤与 ESO 接口的边界

这轮落实“借鉴 GRAVITY 的物理阶段、产品语义和标定链，保留适合 Python
的简单步骤结构”。基于0.5.0已有的五个同级步骤，进一步明确输入、参数、
返回产品和 ESO 接口各自归属。科学算法、单位、默认值和文件格式保持一致。

## 代码从哪里看

| 步骤 | 文件入口 | 同文件的计算函数 | 返回结果的物理含义 |
| --- | --- | --- | --- |
| DARK/BAD | dark.calibrate_dark_file | compute_dark / compute_badpix | 背景均值、时间 RMS、坏像素 bits |
| FLAT | flat.calibrate_flat_file | compute_flat / trace_spectrum | 响应、光谱 trace、稀疏提取几何 |
| WAVE | wavelength.calibrate_wavelength_file | compute_wavelength / line matching | 每个输出、每个像素的波长及拟合残差 |
| P2VM | p2vm.calibrate_p2vm_file | extract_calibration_files / compute_p2vm | 全24输出物理响应、反演矩阵、实际波长网格 |
| P2VMRED | reduction.reduce_file | reduce_spectra / compute_gdelay | 逐帧 photometry、复相干通量、VISAMP、GD、相位、flags |

每个文件的入口位于前部，下面是本步骤的数组计算和 helper。共享提取/重采样
位于 extraction.py，产品定义位于 products.py，物理通道约定位于 instrument.py，
FITS 布局和 keywords 位于 io/。现有 GUI 通过 FrameReducer 复用同一计算。

新增 [examples/reduce.py](../examples/reduce.py) 是普通线性 `# %%` 脚本。
修改 calibration raw、science raw 和新输出目录后即可运行。它显式选择
三组 DARK，完成五步并保存8份标定/逐帧产品。默认未提供仪器相位参照；
需要时，在 P2VM 调用处显式传入参考 raw 和对应 DARK。

## 本轮实质改动

- 科学函数保留物理参数、单位和默认值。CPL 参数名称、CLI alias、说明移至
  对应 recipes/planetes_*.py；科学模块不再含 CPL 参数注册字典。
- FLAT/P2VM recipe 各自声明重复 SOF 输入的数量。通用 select_inputs() 接受
  显式声明，不再按隐藏的全局类别规则推断数量和返回类型。
- 参数翻译仍从科学函数签名读取默认值，不新增另一份默认配置或任务调度系统。
- 文件入口、提取器、FrameReducer 的类型标注明确表达物理输入和返回对象。
  除文档/类型标注外，46个科学函数的计算体和默认值与0.5.0相同。
- 五个 recipe 的 synopsis/description 直接说明实际物理操作。既有参数名称、
  别名、默认值和调用接口保持一致。
- 图和产品合同明确区分仪器标定、逐帧信号、后续统计平均和天文传递函数标定。

## 科学阶段与标定链

DARK/BAD、FLAT、WAVE、P2VM 描述仪器。P2VMRED 保存每一帧的仪器级测量，
其通量为 ADU，长度为 m；它的波长网格和 FLAT/WAVE 父关系来自实际 P2VM 标定。
main、lamp、science 数据组分别使用其明确选定的 DARK。

后续平均阶段需要定义 VIS/VIS2/T3 的估计量、权重、去偏差和误差。
天文标定还需要 calibrator 观测及源模型。P2VM 的仪器 phase reference
和天文 calibrator 的作用分别说明；本轮只在文档中画出后续阶段。

本轮保留现有掩码、阈值、剔除、归一化、相位约定和报告型 QC。
必要检查仍集中于物理关联、形状/索引、波长覆盖和外部文件边界。

## 流程图

![入口、科学步骤和公共部分](architecture.png)

![标定、逐帧结果和后续科学阶段](dataflow.png)

## 验证结果

| 验证 | 实际结果 |
| --- | --- |
| 维护测试 | 55项通过，包含真实 PyCPL/PyEsoRex 执行 |
| 外部参数 | 五个 recipe 的 canonical name、CLI alias、默认值保持一致；非默认波长网格、GD范围和phase开关实际传入科学函数 |
| SOF规则 | 重排输入可正常处理；不完整 FLAT/P2VM 数量被对应 recipe 拒绝 |
| 完整真实数据 | 原有15份标准化 raw，三份DARK、FLAT、WAVE、P2VM、science，旧/新冷/热各完整执行 |
| 数值和离散结果 | 每轮37个数组，合计74个逐字节一致，包含 flags、提取索引、accepted谱线和全部科学数组 |
| 产品语义 | 两轮共20产品的指纹、实际参数、父标定和所有ESO QC数值完全相同 |
| 普通Python示例 | 阻止CPL/PyEsoRex导入后，完整处理512帧并保存8份产品；保存标定可用于同一实时核心 |
| 现有Qt6 GUI | offscreen fake-data capture、元数据、显示binning一致性、dark-off/single-frame-dark通过 |
| 安装后运行 | 0.5.1 wheel项目外导入、5个recipe发现及随包line list的实际WAVE执行通过 |
| 静态检查 | Ruff、格式检查、无循环import；CPL import全部限定在recipes/ |

本轮对照基线为 `e4e2b69`（0.5.0），使用同一数值环境并分别匹配冷/热缓存状态。
GUI 使用记录数据与假相机。这些验证证明本轮接口整理保持已有结果；
独立物理准确度、误差传播和真实硬件验收有各自的科学验证范围。

## 兼容与恢复

版本 **0.5.1 / product format 4 / raw format 3**。
0.5.0的Python科学入口、recipe参数和format-4产品可继续使用，本轮无需重生成标定。
CPL参数注册字典属于recipe内部，不再从科学模块导入。

修改前源码备份：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_202453_step_interfaces/`

本轮产品、数值比较、GUI与示例结果：

`reduced/step_interfaces/20261027_202453/`

0.5.0的单位迁移、原有目录合并及数值变化记录保存在
[上一版修改报告](https://github.com/zexwu/planetes_pipe/blob/e4e2b69f02403ff9c008d450cf2a2b345053b175/docs/CHANGE_REPORT.md)。
