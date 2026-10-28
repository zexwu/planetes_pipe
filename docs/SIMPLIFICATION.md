# PLANETES 0.8.0：简单科学产品与明确的步骤 API

0.7已将`products.py`从401行缩减到119行；0.8保持这些普通数据类，
统一步骤、字段和参数名称。
每个物理步骤仍在自己的文件中定义输入、参数、计算与局部 helper。

| 原名称 | 当前名称 | 内容 |
| --- | --- | --- |
| DarkCalibration | Dark | mean、temporal RMS、frame count 和可选 detector setup |
| FlatCalibration | Flat | detector flat、trace 和稀疏提取几何 |
| WavelengthCalibration | Wave | 逐输出波长图及可选拟合诊断 |
| P2VMCalibration | P2VM | 完整24输出响应、逆矩阵及扫描诊断 |
| VisibilityObservables / ReducedObservables | P2VMRed | 逐帧仪器信号；实时预览可以省略 GD/差分相位 |

旧名称直接替换，没有兼容别名。`DarkImage` 只包含减暗用的平均图像，
与完整 `Dark` 分开。`DarkInput = Dark | DarkImage` 只是减暗函数的类型标注。

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
    Python --> FITS["io.fits：format 5"]
    Recipe --> FITS
    Data --> QC["可选 qc.measure_quality"]
    QC --> FITS
```

物理标定顺序是 DARK/BAD → FLAT → WAVE → P2VM。科学处理使用自己选择的 DARK，
以及 FLAT、WAVE 和 P2VM，返回逐帧 `P2VMRed`。平均后的 VIS/OIFITS 和天文
传递函数标定属于后续科学阶段。

## 文件与调用兼容性

新产品写 format 5，仍可读 format 4；旧私有 hash/provenance 被忽略。
原数组扩展、物理单位和坐标约定保持，完整 raw 仍是 format 3。
原始文件不会被读取或保存过程改写。

保留当前 main 的台架输入支持：Python 显式望远镜/基线映射、不完整 header，
非 ADU 拒绝、多 cube 歧义拒绝或显式低层 hdu 选择。SOF 类别保持不变。
Qt6采集helper使用`build_raw_header()`。GUI自己持有目录路径，不再从
FrameReducer读取目录；overlay使用`baseline_outputs`。portable patches同步更新。

QC 仍由发布入口的 `include_qc` 选择，保留当前 Python/recipe 默认 True；
传 False 可跳过测量和附加。encoder 只编码传入结果，产品数据类不管理 QC。
启用 QC 时，不可用原因放在 FITS 的最小 metadata 中。

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

## 验证

保留12项开发测试，不恢复补丁中的整批测试或 golden NPZ。完整真实数据的新旧
冷/热对照保留74个科学数组的字节；20份产品的物理映射、DFS 参数、used inputs
和 QC 值相同。六种相位参照/实时预览情形共138个数组逐字节一致。
10份旧 format-4 产品可读并转存为 format 5，数组、几何与 detector setup 相同。

[当前修改报告](CHANGE_REPORT.md)记录证据位置；
[产品接口](PRODUCT_CONTRACT.md)列出字段、文件布局和参数。
