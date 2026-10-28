# PLANETES 0.9.0：精简 QC、raw 读取和产品状态

基于 `0563593`，落实确认的三项修改。科学算法、物理单位、默认阈值、mask、
SOF标签和五个步骤结构保持。

## 具体修改

- **QC**：`qc.py` 从234行减到101行。保留暗场median/RMS、BAD像素数、
  四个flat flux、共同波长覆盖/拟合残差和六条基线coherence。
  删除profile/P2VM扩展统计，以及不适用指标的missing字典、关键词状态映射和
  `qc_status`。可用统计的定义不变；未知值省略，不填零，QC仍可关闭。
- **Raw**：`io/raw.py` 只读取物理字段，`RawMetadata`从8个字段减到3个：
  category、shutters、setup。采集record及header/writer移到`io/acquisition.py`，
  使用`CaptureMetadata`；host时间、配置来源仍保存在raw文件中。
  身份识别返回已读metadata，flat/P2VM复用它做兼容检查，不再重复解析同一文件。
  已知ROI、曝光、detector、单位及快门矛盾仍报错。
- **Products**：固定配对在`instrument.BASELINE_PAIRS`集中定义；从P2VMSpectra、
  P2VM和P2VMRed删除`baseline_telescopes`。实际detector output映射保留。
  P2VM扫描OPD/GD、ellipse phase/radius、DC、fit RMS六组诊断默认不保留；
  Python `keep_diagnostics=True`或PyEsoRex `--keep-diagnostics=true`可保留。
  这些选项不改变P2VM拟合、归一化或反演。

Qt6采集helper与可迁移补丁同步更新，没有兼容别名或新的workflow框架。
新产品写**format 6**，因为扫描诊断不再必需；仍可读format 4/5。
Raw writer继续写format 3，原始像素和已有文件不覆盖。

## 验证

| 检查 | 结果 |
| --- | --- |
| 开发测试 | 12项通过，数量不增加；覆盖默认/含诊断P2VM的FITS往返 |
| 真实数据 | 15份既有raw，两版各运行一轮全部五个recipe |
| 常规产品 | 31个保留数组的dtype、shape、字节一致；默认省略6个诊断数组 |
| 可选诊断 | 开启诊断的P2VM全部11个数组与旧版逐字节一致 |
| 产品含义 | 10份产品的物理映射、setup、实际输入一致；原recipe参数值保持，新增keep-diagnostics参数 |
| QC | 31个保留keyword值一致，少写26个扩展QC keyword |
| 旧文件 | 10份format-4产品转存format 6后数组、映射、setup一致；另10份format-5产品读入的37个数组字节一致 |
| 物理冲突 | 有效checksum的错误曝光、ROI、快门及部分已知快门冲突仍被拒绝 |
| GUI | 真实Python 3.12 Qt窗口处理记录帧、读取无诊断P2VM、模拟写出raw和中途设置变化拒绝通过 |
| Qt6补丁 | 四份补丁可应用到原始文件并复现本机消费者 |

源码与GUI备份、一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_185200_simplify_boundaries/`

测试日志、产品、对照结果及GUI快照：

`reduced/simplify_boundaries/20261028_185200/`

[架构流程图](architecture.png) · [标定与科学数据流](dataflow.png) · [保留的QC](QC_PARAMETERS.md)

本轮未改SOF身份表达、file-level workflow归属和全量science内存处理。
目前提取仍会拼接整段光谱，GD搜索也处理全部帧；端到端分块应另作一次明确的数值对照。

[0.8.1修改记录](https://github.com/zexwu/planetes_pipe/blob/056359362d4f5ed531ace26756933b69ae91689d/docs/CHANGE_REPORT.md)
保留Dark合并及Bad重命名的证据。
