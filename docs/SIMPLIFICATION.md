# PLANETES 0.15.0：明确输入，直接计算

调用处选择文件和标定，读取函数负责读取，科学步骤负责计算。
不自动猜输入类型，不维护JSON metadata或采集历史。

## 科学步骤

| 文件流程 | 数组计算 |
| --- | --- |
| make_dark_products | measure_dark / detect_bad_pixels |
| make_flat | build_flat |
| make_wave | fit_wavelength_solution |
| make_p2vm | fit_p2vm |
| reduce_science | reduce_spectra |

FLAT/P2VM使用固定的`{identity: Path}`：望远镜`4,3,2,1`，
基线`43,42,41,32,31,21`。SOF身份读取在recipe入口完成。
路径映射始终只存路径，不在中途变成path/metadata tuple。

`make_wave`明确传入线表；默认文件由recipe或脚本选择。
P2VM相位参考是已减暗、提取并对齐到扫描网格的光谱，缺失用None表示。
科学函数不再接收相位开关或参考暗场，不自行选择默认文件或暗场。

## 产品与文件

产品是简单的`Dark`、`Bad`、`Flat`、`Wave`、`P2VM`、`P2VMRed`数据类。
Dark必需mean，RMS和帧数在未测量时为None；BAD检测要求实际测得的RMS。
P2VMRed保存逐帧信号，平均后的VIS/OIFITS和天文传递函数标定属于后续阶段。

物理metadata只放在FITS header。raw reader返回`RawMetadata`，包含类别、四个
快门状态及`DetectorSetup`；后者只含曝光、ROI、binning和detector。
采集writer移出本package，测试只构造固定的最小FITS fixture。
删除CaptureMetadata、JSON读写、PLANETES_META扩展、采集来源、legacy状态、
采集时间历史及header/JSON调和检查。没有补缺兼容层。

不再写入或强制检查私有格式号和BAD位定义字符串，按实际产品字段读取。
raw类别由调用者明确给出，HDU默认IMAGING_DATA_SC；非标准数据显式传hdu=0等。
删除自动HDU搜索、类别猜测、重复维度检查和仅包装getheader的函数。
随包LINE_LIST去掉旧metadata和指纹，保留原表精度、全部谱线值与标签。

## 保留的计算边界

- 计算float64/complex128，writer统一缩为float32/complex64；整数和flags保留。
- 稀疏像素范围及波长覆盖在进入Numba提取前检查。
- 数组入口检查关键物理轴；保留telescope/baseline身份、标定曝光/ROI匹配及V2PM秩检查。
- QC仅报告少量可用统计，脚本/recipe显式计算并传给writer；不改变mask、拟合或权重。
- 每个输出独立暂存、生成checksum并拒绝覆盖；删除多文件回滚、inode核对及同目录约束。
  checksum只检查一次；完整round-trip由现有测试检查。
- CPL参数与DFS实际输入由recipe记录；CPL import仅存在于recipes目录。

Qt6项目独立维护。本仓库不含GUI补丁、manifest或部署文档。
`FrameReducer(flat, wave, p2vm)`保留为数组级科学接口。

[修改报告](CHANGE_REPORT.md) · [架构图](architecture.png) ·
[产品接口](PRODUCT_CONTRACT.md) · [科学约定](SCIENTIFIC_METHODS.md)
