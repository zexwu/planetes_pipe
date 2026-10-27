# QC：GRAVITY 1.9.6 定义与 PLANETES 适用范围

`qc.measure_quality()` 返回物理量名称、数值、单位说明以及不可用原因。
例如 `dark_median_adu` 在 `io/keywords.py` 映射为 `ESO QC MEDIANDARK SC`；
QC 计算本身不拼装 FITS keyword，Python 与 CPL 写出共用同一映射。
这些指标不修改拟合、mask、剔除或阈值。参考源码固定为本机
`/Users/zexwu/software/gravi_pipeline/gravity-1.9.6/gravi/`；关键词见 gravi_pfits.h。
SC 仅指本仪器唯一光谱相机。指标名称相同不代表已通过 ESO 的 PLANETES 字典审核。

| 关键词（省略 ESO QC 前缀） | 定义、单位与统计对象 | 参考实现 / 本仪器差异 |
| --- | --- | --- |
| MEDIANDARK SC | temporal mean 图像的空间 median，ADU；样本数为全部探测器像素 | gravi_calib.c:183；原始 dark 全帧均值不变 |
| DARKRMS SC | temporal population RMS 图像的空间 median，ADU；同一像素域 | gravi_calib.c:183；不是独立测得的纯读出噪声 |
| BADPIX SC | 非零 BAD 像素数 | gravi_calib.c:2158；使用现有 bits 并集 |
| BADPIX_DARK SC / BADPIX_RMS SC | 分别数 hot或dead (`flags & 3`) / noisy (`flags & 4`) 像素 | 本仪器 bits 是 hot=1/dead=2/noisy=4，不能把 GRAVITY 的坏像素类别当位编号 |
| FLATFLUX SC1..4 | 每个扣暗 flat 的 mean-frame 总 ADU，ADU/DIT；全部像素求和 | gravi_calib.c:1470–1479；序号稳定对应 T4,T3,T2,T1 |
| PROFILE_CENTER SC1/SC13 MED, LEFT, RIGHT | 对 region 0/12 的一基 trace，取全列 median、索引 floor(nx/6)、floor(5nx/6)，pixel；按参考 floor(value×1e6)/1e6 | gravi_calib.c:1485–1519；保留本仪器平滑后的 trace |
| PROFILE_WIDTH SC1/SC13 MED | 同 region 已接受 Gaussian 拟合 sigma 的 median，pixel，按同样方式截断；有效 N 写入注释 | gravi_calib.c:1495–1500；无法拟合的列 NaN 不参与，aperture width 不充当 sigma |
| MINWAVE SC / MAXWAVE SC | 全 24 输出共同覆盖：max(各输出最小值) / min(各输出最大值)，m | gravi_wave.c:2027–2045；不是全图极小/极大值 |
| MINWAVE UM SC / MAXWAVE UM SC | 同上，um | 同一物理量的参考单位版本 |
| RMSWAVE SC / RMSWAVE UM SC | 最终实际接受的 lamp surface-fit residual 的样本标准差，m / um；N 为保留谱线匹配数，N<2 不可用 | gravi_wave.c:1429,2266–2267 使用 sample stdev，并按偏振平均；本仪器单偏振、lamp 残差，原 clipping/拟合不变 |
| P2VM_COHERENCE_AVG_SC | 24 个关联输出的光谱 coherence mean 再平均 / (2×关联 transmission mean 的平均)，无量纲 | gravi_p2vm.c:1180–1278；使用每输出对应基线/两台 telescope，全波段 |
| P2VM_COHERENCERMS_AVG_SC | 上式分子换成各输出光谱 coherence 的 sample stdev 后平均 | 同上；ddof=1，与 CPL cpl_column_get_stdev 相同 |
| P2VM_PHASERMS_AVG_SC | 各关联输出的光谱 PHASE 样本标准差再平均，rad | 同上；保留参考直接角度统计，不另做 unwrap |
| P2VM_TRANS_AVG_SC / MIN_SC / MAX_SC | 各输出两台关联 telescope 的光谱 mean transmission 的平均，再对24输出取 mean/min/max，最后 /2 | 同上；不丢弃保存的其他串扰项 |
| P2VM_COHERENCE_SC43 等6基线 | 每输出每波长 coherence/(2 sqrt(T1) sqrt(T2))，对四输出和全波段平均 | gravi_p2vm.c:1219–1243,1284–1303；任一关联 transmission 非正时不报告该基线值 |
| P2VM_MINCOHERENCE_SC43 / MAXCOHERENCE_SC43 等 | 每输出先按光谱排序，取 floor(0.05 N) / floor(0.95 N) 元素，再跨四输出取 min/max | 同上及 gravi_cpl.c:2510；不是 NumPy 插值分位数，N=nwave |

Dark temporal RMS 使用 population 定义；上述谱域/残差 sample stdev 使用 N−1。
两者不同是参考定义和物理统计对象不同。波长残差来自本次拟合实际保留点，
其定义为 lamp 标定残差；不能将它解释为独立实测的天文波长准确度。

没有充分物理依据的数值不填 0，而在 PLANETES_META 的 qc_status 中说明：

- BADPIX_FLAT：保留的 mask 没有 flat-based 判坏规则。
- RON、电子/光子 flux rate、SNR、CHI2：缺独立噪声分解、gain 或传播误差。
- PROFILE_WIDTH：没有 accepted Gaussian fits 时不可用；缺 fitted residual 时 RMSWAVE 不可用。
- REFWAVE、FT/metrology/ACQ：参考 K 波段/硬件不适用。
- GRAVITY GD rejection 指标：本仪器只有 coarse initialization 区间，Newton 不被截断。
- 最终 VISAMP/VISPHI 平均、REJECTED RATIO：产品仍为逐帧仪器级结果；channel invalid flags 不是 GRAVITY 的 frame rejection 百分比。

保存的 P2VM 逆矩阵条件数和 channel-level flag fraction 保留在普通 `qc.diagnostics()` 中，
不再写成自行命名的 ESO QC。相位校正的请求/实际执行状态在具名 provenance
参数中保存，不继续发明 ESO DRS 关键词。普通拟合 RMS 保存在
PLANETES_P2VM_FIT_RMS [ADU]，不冒称有权重的 chi-square。
