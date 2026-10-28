# QC：当前保留的描述性统计

`qc.measure_quality(product)` 返回 `{metric: (value, comment)}`，
`io/keywords.py` 负责 FITS keyword 映射。只报告已测、有限的值，不设阈值，
不修改拟合、mask 或权重。未知或无法计算的统计直接省略，不填零，
不再维护 `missing`、`QC_STATUS_KEYS` 或 `qc_status`。

| 产品 | 保留的统计 / FITS keyword（省略 ESO QC） | 定义与单位 |
| --- | --- | --- |
| Dark | MEDIANDARK SC / DARKRMS SC | temporal mean / population RMS 图像的空间 median，ADU；未测 RMS 时省略 |
| Bad | BADPIX SC / BADPIX_DARK SC / BADPIX_RMS SC | 非零 flags、hot或dead (`flags & 3`)、noisy (`flags & 4`) 的像素数 |
| Flat | FLATFLUX SC1..4 | 四个减暗 mean-frame 的总 ADU/DIT；顺序 T4,T3,T2,T1 |
| Wave | MINWAVE SC / MAXWAVE SC | 所有输出共同覆盖的波长上下界，m；保留原 UM SC 文件单位版本 |
| Wave | RMSWAVE SC | 实际接受的 lamp fit residual 的 sample stdev（N−1），m；N<2 时省略 |
| P2VM | P2VM_COHERENCE_SC43 等六条基线 | coherence/(2 sqrt(T1) sqrt(T2))，对四输出和全波段平均；关联 transmission 非正时省略 |

这些定义沿用当前实现，参考本机 GRAVITY 1.9.6：`gravi_calib.c:183,1470–1479,2158`、
`gravi_wave.c:2027–2045,2266`、`gravi_p2vm.c:1219–1243,1284–1303`。
名称与定义参照 GRAVITY，不表示 PLANETES keyword dictionary 已获 ESO 审核。

移除 profile center/width QC、P2VM 汇总/分位数统计，以及 RON、CHI2、SNR、
GRAVITY GD rejection、最终平均 VIS 等不适用指标的占位说明。原 trace/sigma
数组仍保留；可选 P2VM diagnostics 保存拟合中间量，均不变成自动剔除规则。

`qc.diagnostics()` 仍可单独返回逆矩阵条件数、visibility flag fraction 和均值，
不写成自行命名的 ESO QC。Python `write_product` 和正式 recipe 默认附加上述
少量 QC；`include_qc=False` 可关闭。`encode_product(..., quality=...)` 只编码传入结果。
