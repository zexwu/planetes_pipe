# PLANETES QC 参数定义

这组描述性参数写入产品主头，辅助检查数据与标定状态。它们不修改数组、mask、拟合权重或样本选择，也没有新增合格判定阈值。关键词和定义仍需 ESO 审核，正式 QC dictionary 将在该审核后定稿。

| 关键词 | 单位 | 定义和适用产品 |
| --- | --- | --- |
| `ESO QC BAD PIXEL FRACTION` | 无量纲 | flat 中 `bad_pixel_flags != 0` 的像素数除以探测器像素总数；已有 hot/dead/noisy 标志的并集 |
| `ESO QC WAVE MIN` | um | wave map 的最小波长，用于描述输出覆盖范围，不代表拟合精度 |
| `ESO QC WAVE MAX` | um | wave map 的最大波长，用于描述输出覆盖范围，不代表拟合精度 |
| `ESO QC P2VM CONDITION MAX` | 无量纲 | 由已有 transmission、coherence amplitude、coherence phase 重建每个波长的 Cartesian V2PM，计算二范数条件数并取最大值；非有限时不写该值 |
| `ESO QC VIS FLAG FRACTION` | 无量纲 | reduced 中 `flags == True` 的 frame/baseline/wavelength 样本数除以样本总数 |

`ESO DRS PHASE REF APPLIED` 另外记录相位参考校正是否实际执行。它与用户请求的 `phase_correction` 参数分别保存，以区分“开关开启但未提供相位参考”和“已执行校正”。

计算定义见 `planetes/algorithms/quality.py`，关键词写入见 `planetes/eso/common.py`。波长残差、探测器噪声模型及最终科学误差仍需要各自的测量和验证，不能由上述覆盖范围或标志比例代替。
