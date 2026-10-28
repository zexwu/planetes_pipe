# PLANETES：P2VM 可读性整理，第1步

本轮基于 `2cc9e95`（0.5.1），完成计划中的第1步：整理 p2vm.py 的科学流程与
局部命名。计算、参数、返回产品和 FITS format 4 保持一致。

## 从哪里读代码

`calibrate_p2vm_file()` 仍负责文件输入与标定组合；
`compute_p2vm()` 从199行缩短到125行，按实际物理操作调用同文件的 helper：

| 函数 | 科学职责 |
| --- | --- |
| fit_ellipse_linear | 对 A,C,B,D 输出拟合相位和归一化 ellipse radius |
| _estimate_scan_opd | 用中央半波段计算 OPD/GD，并去除 OPD 零点 |
| _fit_interferometric_response | 对全部24输出拟合 cos/sin/DC，保留相位、幅度、DC和拟合RMS |
| _fit_photometric_transmission | 以各望远镜光谱的frame median构造transmission，保留原初始归一化 |
| _normalize_response | 按原顺序完成A输出相位、transmission和coherence归一化 |
| build_v2pm_matrix / _invert_v2pm | 构造完整响应矩阵，并逐波长求伪逆 |
| _correct_phase_reference | 可选仪器相位参照修正；随后按同一路径重建、求逆 |

所有计算继续位于 [p2vm.py](../planetes/p2vm.py)。没有增加算法目录、中间状态类
或单独的预处理产品。求逆 helper 由相位修正前后两处共用。

## 命名和物理含义

- baseline、output、telescope_spectra、baseline_spectra、wavelength_m 和
  dc_term_adu 等局部名称直接说明实际物理量。
- 旧局部 visamp 改成 ellipse_radius。它是归一化椭圆坐标的半径，
  不直接代表最终物理 visibility amplitude。
- 相位参照修正中的旧局部 opl 改成 telescope_phase_rad：原公式给出的是
  参考波长上的相位，而非以米表示的光程。只改名称，原公式保持一致。
- phase-reference 数据反演得到的相干通量使用 reference_coherent_flux_adu，
  避免原 vis_ft 名称暗示存在另一台 FT 探测器。

原有7个函数的参数列表保持一致。持久化字段 ellipse_amplitude、OPD/GD、
完整24输出响应及所有 provenance 字段也保持原义和原值。

## 验证

| 检查 | 实际结果 |
| --- | --- |
| 改动前基线 | 冻结0.5.1源码并重新运行，55项测试通过 |
| 改动后测试 | 56项通过，包含真实PyCPL/PyEsoRex、非默认参数及现有科学检查 |
| 新增物理例子 | 无噪声A,C,B,D条纹contrast相差8倍，拟合ellipse radius均为1，并保留原相位方向；旧实现也通过此检查 |
| 真实数据 | 15份标准化raw，三份DARK、FLAT、WAVE、P2VM、science，旧/新冷/热分别完整运行 |
| 科学数组 | 每轮37项，两轮共74个数组的dtype、shape、字节完全一致 |
| 产品内容 | 两轮共20份产品的指纹、实际参数、父标定和全部ESO QC值一致 |
| 相位参照分支 | 启用、禁用、缺失 × 15/17通道，使用非默认GD范围；6个受控例子中66个P2VM数组及指纹/参数完全一致 |
| 静态检查 | Ruff、格式、语法检查通过；现有模块依赖无环 |

对照使用相同Python/NumPy/Numba环境，并分别匹配冷/热缓存状态。
数值一致性支持本轮组织改动保留已有结果；不替代独立的台架准确度实验。

## 兼容和后续

Python调用入口、软件版本0.5.1和产品format 4保持一致，已有标定可继续使用。
本轮没有改变FITS/QC/raw的接口、处理规则或科学选择。

下一步按已讨论的顺序整理WAVE/REDUCE，再独立处理产品/provenance边界、
bench raw兼容和可选QC。

修改前源码及一次性验证脚本/日志备份：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_095032_p2vm_readability/`

本轮数据产品和比较结果：

`reduced/p2vm_readability/20261028_095032/`

[上一轮接口整理报告](https://github.com/zexwu/planetes_pipe/blob/2cc9e9597b528bc09909f49a0abeeca01fc2b459/docs/CHANGE_REPORT.md)
保留0.5.1的接口、安装和GUI验证说明。
