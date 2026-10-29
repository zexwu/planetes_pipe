# PLANETES 0.13.0：简单科学产品与明确的步骤 API

0.7曾将`products.py`从401行缩减到119行；0.8统一步骤、字段和参数名称，
0.8.1合并Dark类型后为109行；0.9.0移除固定配对副本后为106行，继续使用普通数据类。
每个物理步骤仍在自己的文件中定义输入、参数、计算与局部 helper。

| 原名称 | 当前名称 | 内容 |
| --- | --- | --- |
| DarkCalibration / DarkImage | Dark | 必需 mean，可选 temporal RMS、frame count 和 detector setup |
| BadPixelMap | Bad | hot=1、dead=2、noisy=4 的坏像素 flags |
| FlatCalibration | Flat | detector flat、trace 和稀疏提取几何 |
| WavelengthCalibration | Wave | 逐输出波长图及可选拟合诊断 |
| P2VMCalibration | P2VM | 完整24输出响应、逆矩阵；扫描诊断可选 |
| VisibilityObservables / ReducedObservables | P2VMRed | 逐帧仪器信号；实时预览可以省略 GD/差分相位 |

旧名称直接替换，没有兼容别名。0.8.1统一使用Dark，删除DarkImage和DarkInput，
并将坏像素产品简称Bad。Dark的mean_adu必需，temporal_rms_adu和n_frames
未知时为None；单帧记录n_frames=1，RMS仍为None。减暗只用均值，BAD检测
要求RMS；完整DARK文件写出仍要求已测RMS和至少两帧。

## 删除的机制

通用 `validate()`、`validate_gd_range()`、`validate_wavelength()`、`positive()`，
产品 SHA-256 指纹、父标定依赖图、`require_parent()`、`Provenance` 以及
数值函数中的文件路径/参数来源记录全部删除。输入文件由 Python 调用者明确选择，
Python 脚本保留所用路径和参数；PyEsoRex 的输入与 recipe 参数由 CPL DFS 记录。

## 保留的必要检查

- 固定的 telescope/baseline 配对和24输出的 ABCD 映射，在实际计算和 FITS 边界检查。
- 关键输入维度在 P2VM/reduction 入口检查，防止 NumPy 广播掩盖物理轴不匹配。
- 稀疏像素范围和波长覆盖/顺序，在进入 Numba 提取与插值前检查一次。
- V2PM 必须具有16个独立数据通道，数值秩使用 NumPy 默认定义；原逐波长伪逆保留。
- 采集/文件边界保留已知曝光、ROI、单位和快门矛盾检查，以及 FITS checksum 和不覆盖写入。

这些检查直接针对所处物理步骤，没有通用的产品验证框架。

## 数据流

```mermaid
flowchart LR
    Raw["Raw FITS + 明确物理身份"] --> Steps["dark / flat / wavelength / p2vm / reduction"]
    Steps --> Data["Dark / Flat / Wave / P2VM / P2VMRed"]
    Data --> Python["Python：显式保存"]
    Data --> Recipe["recipe：DFS 来源与参数"]
    Python --> FITS["io.fits：format 6"]
    Recipe --> FITS
    Data --> QC["可选 qc.measure_quality"]
    QC --> FITS
```

物理标定顺序是 DARK/BAD → FLAT → WAVE → P2VM。科学处理使用自己选择的 DARK，
以及 FLAT、WAVE 和 P2VM，返回逐帧 `P2VMRed`。平均后的 VIS/OIFITS 和天文
传递函数标定属于后续科学阶段。

## 文件与调用兼容性

新产品写 format 6，仍可读 format 4/5；旧私有 hash/provenance 被忽略。
原数组扩展、物理单位和坐标约定保持，完整 raw 仍是 format 3。
原始文件不会被读取或保存过程改写。

保留当前 main 的台架输入支持：Python 显式望远镜/基线映射、不完整 header，
非 ADU 拒绝、多 cube 歧义拒绝或显式低层 hdu 选择。SOF 类别保持不变。
采集程序可调用`io.acquisition`的`CaptureMetadata`、`build_raw_header()`和`write_raw()`。
Qt6项目独立维护；本仓库不再携带GUI补丁、manifest和部署文档。

QC 仍由发布入口的 `include_qc` 选择，保留当前 Python/recipe 默认 True；
传 False 可跳过测量和附加。encoder 只编码传入结果，产品数据类不管理 QC。
只保留暗场统计、BAD计数、flat flux、波长覆盖/残差和各基线coherence；
未测或非有限值省略，删除missing字典、状态映射和qc_status。

## 两个可直接阅读的层次

| Raw files → products | Arrays → scientific result |
| --- | --- |
| make_dark_products | measure_dark / detect_bad_pixels |
| make_flat | build_flat |
| make_wave | fit_wavelength_solution |
| make_p2vm | fit_p2vm |
| reduce_science | reduce_spectra |

`P2VM.matrix`保存逆矩阵，`baseline_outputs`保存输出映射；
内部光谱类叫`P2VMSpectra`。`aperture_width_pixels`表示提取全宽，
`include_phase`控制GD/差分相位。`trace_spectrum`固定返回三个数组。

删除`wavelength_grid`、`FrameReducer.calibration_dir`；`build_flat`直接接收
已减暗图像与BAD，不再依赖Dark。`_invert_v2pm`名称已经准确，继续保留。
发布函数叫`publish_outputs`，不把多文件发布描述成整体原子事务。

## 0.9.0 的两个边界

`RawMetadata`仅含category、四个可未知的shutter状态、setup；reader不再重建采集记录。
`CaptureMetadata`在`io/acquisition.py`中保存采集时间和配置来源，并由writer使用。
0.9曾通过嵌套path/metadata tuple复用解析结果；0.12改为入口显式映射，
步骤按各自文件循环读取metadata，避免让这一优化决定科学接口。
ROI、曝光来源、detector、单位和已知shutter矛盾检查保留。

固定配对只在`instrument.BASELINE_PAIRS`定义；从三个产品类删除
`baseline_telescopes`。实际从数据推断的`baseline_outputs`仍保留。
`keep_diagnostics=False`默认不保存P2VM扫描/ellipse/DC/residual数组；设为True
可保留原来的六组诊断，计算方法不变。ESO参数对应`--keep-diagnostics=true`。

## 0.10.0 的精度边界

`io/hdu.py`集中定义浮点存盘32位、计算64位；复数对应complex64/128。
算法不声明float32/64，不做精度转换。索引和flags继续用整数/bool。
raw每16帧读入后转换一次，产品读入和数组输入边界同理；写出时统一舍入。
数组级调用者在入口准备计算精度数组，不增加逐算法wrapper或dtype参数。

暗场统计合并每块的mean/M2，避免E[x²]−E[x]²。保留population RMS和原判坏规则。
按用户确认，所有浮点数组均按32位写出；波长覆盖检查仅容忍一次存盘舍入的
相对误差上界2**-24，实际越界仍拒绝。

## 0.11.0 的 P2VM 流程

`fit_p2vm`直接展示扫描响应拟合 → transmission → 归一化 → V2PM求逆 →
可选phase reference。输入数组契约、扫描诊断、文件读取和物理output识别分别有
清楚的位置，仍在同一个步骤文件内。没有引入新的框架或产品类。

删除椭圆拟合的全1 `envs`参数，仍用原`pinv`及全部原阈值。
phase reference明确用T4及43/42/41，不再通过`telescope-1`隐含配对。
原有相位操作顺序、双重transmission归一化、全24输出cross-talk响应保持。

保留12项开发测试。真实数据对照及椭圆兜底统计见[当前修改报告](CHANGE_REPORT.md)；
0.10精度改动见[上一版报告](https://github.com/zexwu/planetes_pipe/blob/08904a20ce7a3d4942b4af8c418765172aeff14e/docs/CHANGE_REPORT.md)。

[P2VM流程图](p2vm_flow.png) · [科学约定](SCIENTIFIC_METHODS.md) ·
[产品接口](PRODUCT_CONTRACT.md)

## 0.12.0 的直接文件流程

FLAT/P2VM只接受`{identity: Path}`。移除支持Mapping/Iterable两种模式的
`identify_raw_exposures`，路径在科学步骤中保持同一含义。SOF侧用
`read_shutter_id`确定重复raw项的身份；Python直接提供映射。

`read_sof`只分组路径、检查数量和分类DFS frames；每组始终是路径列表。
各recipe明确读取DARK/FLAT/WAVE等产品，不再通过共享加载器返回objects字典。
保存统一调用`save_products`；离线reduction直接准备extractor。

raw reader分别解释header和保存的setup，一次选择最终物理字段。header已知值
优先、JSON补缺、来源标志按保存记录保留；已知矛盾继续报错。

[修改报告](CHANGE_REPORT.md)列出本轮接口变化、备份和数值对照。

## 0.13.0 的明确输入

读取函数负责读取，科学步骤负责计算。文件入口检查关键文件语义；数组入口
保留必要的轴检查；内部流程不再重新猜输入。默认文件与标定选择留在调用处。

- 删除`load_dark`：分别读取DARK产品、从raw生成暗场，或用内存帧构造Dark。
- `make_wave`必传reference；默认线表仅由recipe或脚本选择。
- `fit_p2vm`/`make_p2vm`接收参考光谱或None，不接受相位开关或参考暗场。
  reference文件及其DARK在recipe/脚本中明确选定和提取，不再用零帧数组表示缺失。
- `P2VMSpectra`只含telescope/scan光谱、输出身份和波长。
- DFS直接填写刚创建的stub，不再用额外helper重新判断stub是否符合内部约定。
- 写出保留checksum、DFS签名、临时发布和不覆盖；删除自动读回解码。
  现有测试承担round-trip对照，读取外部产品仍检查单位、checksum和身份。
- 删除Qt6集成补丁、manifest、专用文档和`load_reducer`目录包装；
  `FrameReducer(flat, wave, p2vm)`作为数组科学接口保留。

raw HDU搜索、format-3 header/JSON调和规则不在本轮改动。
