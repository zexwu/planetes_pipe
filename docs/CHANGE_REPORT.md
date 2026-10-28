# PLANETES 0.12.0：固定输入形式，直接展示文件流程

基于 `742d754`。这轮按“减少模式、间接调用和中间状态”的要求整理文件入口。
科学计算函数、阈值、归一化、mask、物理单位与 FITS 格式保持。

## 具体改变

### 文件映射始终只是文件映射

删除 `identify_raw_exposures`。`make_flat`、`make_p2vm`、
`extract_p2vm_spectra`只接收明确的`Mapping[str, Path]`：

- telescope keys：`4,3,2,1`；
- baseline keys：`43,42,41,32,31,21`。

移除“字典还是列表”“用户指定还是自动推断”的内部模式切换。
步骤直接按固定物理身份循环：取路径 → 读metadata → 检查所选标定 → 读图像或提取光谱。
路径不会中途变成`(Path, RawMetadata)`，调用方不再用`[0]`、`[1].setup`拆这种tuple。

Python直接提供文件身份。PyEsoRex在recipe入口用`read_shutter_id`读取SOF raw
文件的快门组合，再构造同样的路径映射。身份缺失、文件重复、已知快门矛盾和
ROI/曝光矛盾检查保留；未知shutter的bench文件仍可通过Python显式映射处理。
这里允许入口与步骤分别读取小量metadata，以保持内部数据结构直接。

### Recipe明确加载所选标定

删除`load_recipe_inputs`。新的`read_sof`只分组路径、检查数量并保留分类后的DFS
输入frames。每个tag的值始终是路径列表，不再混用单个Path和列表，也不返回一个
由共享函数自动加载的objects字典。

每个recipe直接展示自己的输入，例如FLAT：

```python
paths, used = read_sof(frameset, {"DARK": 1, "BAD": 1, "FLAT_RAW": 4})
dark = read_product(paths["DARK"][0], Dark)
bad = read_product(paths["BAD"][0], Bad)
flats = {read_shutter_id(path, "FLAT_RAW"): path for path in paths["FLAT_RAW"]}
product = make_flat(flats, dark, bad, **options)
return save_products([("planetes_flat.fits", product)], used, self)
```

SOF tags、数量、CPL参数及DFS实际输入顺序保持。LINE_LIST的默认文件仍由recipe
显式加入DFS输入。关闭phase correction时，reference/dark仍从实际使用输入中排除。

### Raw metadata一次选择最终值

header与保存的JSON分别解析为`header_setup`、`saved_setup`，先检查已知矛盾，
再一次选择最终物理字段：header/cube已知值优先，JSON补缺。
保存的`exposure_source`和`legacy`标志明确处理。快门值在一个循环中核对并补全。
移除多次`asdict`、字典叠加覆盖及对象/字典往返转换。

raw writer继续写format 3，包含现有JSON记录；产品format仍为6。
本轮保持已有字段来源规则，未重新定义采集格式。

### 合并薄包装与无收益的间接调用

- 删除只把一个产品包装成列表的`save_product`，所有recipe统一调用`save_products`。
- `reduce_science`直接调用`prepare_extraction`，随后提取并计算；不再构造
  `FrameReducer`只为了取它的extractor。逐帧复用的`FrameReducer`接口保持。
- 删除未使用的`flat_setup`旁路参数，校准检查直接接收所选Flat。

没有新增模块、产品类、输入配置对象或通用校验框架。
[代码风格](CODE_STYLE.md)现将输入形式、直接流程和明确的数据来源放在命名/排版之前。

## 验证与接口变化

- 现有12项测试通过，数量未增加；调用更新为显式文件映射。
- 同一批15份既有raw完整运行五种recipe：10份产品的31个数组逐字节相同，
  QC、setup、物理映射、参数与实际输入保持。
- 相同提取光谱下六种P2VM配置及后续计算的63个存盘前对照数组逐字节相同。
- 33个header/JSON样例，完整metadata结果或异常与旧版相同。
- DARK、FLAT、WAVE、P2VM、extraction与reduction的科学计算函数体保持。
  Ruff与diff检查通过。

这是Python文件入口及recipe内部helper的接口简化，版本升为0.12.0。
Python旧调用中的raw文件列表需改成具名路径映射；SOF文件、产品字段与文件布局保持。

## 备份与结果

完整修改前源码和一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_181919_direct_workflows/`

本轮独立产品、metadata对照、存盘前数组和`comparison.json`：

`reduced/direct_workflows/20261028_181919/`

[架构图](architecture.png) · [完整数据流](dataflow.png) · [P2VM计算](p2vm_flow.png)

[上一版报告](https://github.com/zexwu/planetes_pipe/blob/742d75499e0b5a58b4fcb734a62dde8a27e27e05/docs/CHANGE_REPORT.md)
保留此前P2VM、精度和排版改动的验证记录。
