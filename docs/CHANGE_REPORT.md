# PLANETES 0.14.0：删除JSON与采集来源机制

基于 `3121978`。此前自行加入的JSON metadata、采集来源和legacy状态已从读写两端删除。
物理字段只在FITS header中保存一份。

## 删除

- `PLANETES_META`扩展、JSON序列化/解析函数及全部JSON import。
- header/JSON之间的类别、setup、曝光来源、快门核对，以及补缺/优先级逻辑。
- `CaptureMetadata`及配置来源、host/exposure时间记录、original header历史记录。
- `DetectorSetup.exposure_source`、`legacy`和仅存于私有metadata的`orientation`。
- `readback`、`legacy-property`、legacy-role-map等状态及其分支。
- 私有raw格式版本和raw版本兼容判断。
- 随包LINE_LIST中旧的metadata扩展和残留指纹卡。

## 当前接口

`RawMetadata`只包含类别、四个快门状态和物理`DetectorSetup`；raw reader和writer
共用这个类型，不再分别维护一套采集记录。

```python
physical = RawMetadata(category, shutters, setup)
write_raw(path, cube, physical)
```

`DetectorSetup`只含曝光、ROI、binning和detector。raw reader从header和cube尺寸
直接取值；DARK/FLAT的setup同样从header读取。writer将已提供的detector写入现有
`ESO DET ID`，曝光写入DIT；不再根据来源状态选择keyword。缺失值保持未知。

新产品格式为7，旧生成标定需要重新生成。旧raw文件仍可按物理header/cube读取，
额外扩展不参与解释，没有JSON补缺或兼容分支。原始文件没有被改写。
随包LINE_LIST保留原表精度、全部30个波长及origin标签；仅更新文件封装。
ESO DFS的recipe参数、实际输入记录和签名保持。

## 验证

- 现有12项测试通过，数量不变；补充现有测试中的无JSON扩展、物理字段往返、
  raw不覆盖及checksum断言。
- 同一批15份真实raw完整运行五种recipe，生成10份产品：31个数组与0.13.0逐字节一致，
  QC一致，所有输出HDU checksum有效，没有metadata扩展。
- 六个科学模块的32个计算函数AST保持相同；阈值、mask、单位、精度和相位顺序未改。
- 整个`planetes/`和`examples/`中不存在JSON机制、来源/legacy字段或CaptureMetadata。
- Ruff及diff检查通过，接口文档与流程图同步更新。

## 备份与结果

修改前源码：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261029_103701_fits_headers_only/`

真实数据输出和文本对照：

`reduced/fits_headers_only/20261029_103701/chain/comparison.txt`

本轮对照记录使用文本与NumPy数据，不另存JSON。

[架构图](architecture.png) · [产品接口](PRODUCT_CONTRACT.md) ·
[0.13.0报告](https://github.com/zexwu/planetes_pipe/blob/3121978/docs/CHANGE_REPORT.md)
