# PLANETES 0.10.0：计算64位，浮点存盘32位

基于 `3688449`，将精度策略集中到 `io/hdu.py`。算法内部不再选择float32/64，
不做精度转换。现有物理单位、阈值、SOF和recipe参数保持。

## 修改

- raw每16帧读入后转换为float64；浮点产品读入为float64，复数为complex128。
  GUI的raw frame和当前帧dark在输入边界转换一次。数组级调用者提供计算精度数组。
- 新写出的浮点image/table数组统一float32，复数complex64；整数raw、索引、
  BAD bits和bool flags保持原类型。FITS header标量沿用标准文本表示。
  旧raw文件不重写。产品format仍为6，继续读取format 4/5/6。
- 移除dark、提取、flat、波长、P2VM及reduction里的精度dtype和astype。
  ndarray类型标注不再夹带存盘精度，没有新增wrapper或配置层。
- 暗场统计改为合并每块mean与中心平方和M2，返回sqrt(M2/N)。仍是population
  RMS；避免E[x²]−E[x]²的大数相减，不再把计算结果转回32位。
- 按用户确认，波长也存32位。覆盖检查只容忍一次存盘舍入的相对误差上界
  `2**-24`，该常量同样在I/O定义。真实越界仍拒绝。

## 实测影响

使用同一数值环境、15份既有raw，对比0.9.0和新recipe完整链。31个数组的shape
保持，三个BAD、提取像素索引、science flags、物理映射、setup和实际输入保持。
这次精度策略会改变浮点数值，不以逐字节相同作为接受条件。

| 最终科学量 | 新旧完整链最大绝对差 | RMS差 |
| --- | --- | --- |
| VISAMP | 1.7263e-4 | 1.3319e-5 |
| 差分相位 | 0.0109904 deg | 0.0008951 deg |
| GD | 3.5623e-11 m | 1.0059e-11 m |

另跑全64位内存链，再用新代码读取已存盘的标定计算，分离三个来源：

| 对照 | VISAMP最大差 | 差分相位最大差 |
| --- | --- | --- |
| 旧版 → 新全64位内存计算 | 6.7904e-7 | 4.4897e-5 deg |
| 新全64位内存标定 → 标定产品32位存盘后再读入 | 1.7261e-4 | 0.0109956 deg |
| 新science计算结果 → 最终32位写出 | 5.9605e-8 | 3.7957e-6 deg |

因此这批数据的主要变化来自标定文件量化经过后续标定链传播，不是最终science
文件写出的舍入。保留用户确认的32位存盘策略，并记录这个数值代价；这些对照
不能证明天文准确度提高。

## 验证

- 12项测试通过，数量未增加：稳定方差、存盘精度往返、64位读入、complex
  signed zero/NaN、已知物理映射和真实覆盖越界、PyEsoRex science链。
- 新生成文件中58个实数字段均为32位、1个复数字段为complex64；36个内存浮点
  数组均为float64/complex128，整数和flags保持。
- 10份format-4和10份format-5产品可读；新format-6产品可读。
- 真实Python 3.12 Qt窗口可处理记录帧、模拟采集并拒绝中途相机设置变化。
  本机GUI与四份可迁移补丁同步。没有操作硬件。

源码及GUI备份、一次性数值对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_131544_precision_boundary/`

产品、测试、GUI快照和comparison/precision_control结果：

`reduced/precision_boundary/20261028_131544/`

[架构图](architecture.png) · [数据流](dataflow.png) · [产品接口](PRODUCT_CONTRACT.md)

端到端分块science reduction仍是后续工作；本轮只保留16帧raw读取和稳定分块统计。
[上一版记录](https://github.com/zexwu/planetes_pipe/blob/3688449ffd555d90c7a271d6705d47433323c9bc/docs/CHANGE_REPORT.md)。
