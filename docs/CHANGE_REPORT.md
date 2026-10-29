# PLANETES 0.15.0：让I/O直接表达读了什么、写了什么

## 后续维护：保留 raw 文件名中的时间信息

在 mba 的 `/Users/zexwu/PLANETES_new/planetes_pipe` 开发。默认命名沿用
GRAVITY 的 reference-raw 规则：完整 raw stem 加小写、去下划线的 PRO CATG。
五种 recipe 显式选择命名参考：DARK/BAD、WAVE 和 science 分别使用本步 raw；
FLAT 使用 T4，P2VM 使用 43 基线，使 SOF 排序不影响文件名。DFS header
也从同一份 reference raw 继承，保留已有 DATE-OBS 等观测时间卡。
`--static-name=true` 保留此前固定名称，不改变禁止覆盖的写入行为。

Python 增加 `product_filename(reference, product)`，示例和 SOF 模板同步更新。
`write_product` 仍严格使用调用者给定的路径，不自动改名。时间戳、毫秒和
`TUNKNOWN.000x` 按原样保留，不从当前时间或 mtime 补造 DATE-OBS/时区。

在 mba 建立独立 `.venv` 与 CPL 7.4 安装，使用 Python 3.13.11、PyCPL
1.0.4.post1、PyEsoRex 1.0.4。17 项测试全部通过，无跳过；涵盖动态/固定命名、
反序 SOF 的 P2VM 参考选择、科学数组和 DFS 输入记录。Ruff 与格式检查通过。
另只读核对 `2026_02_05/raw` 的物理 header 和 8 个已有产品，确认命名对应关系。
科学计算未改，已有 raw/product 未重命名或覆盖，bench 环境未修改。

## 后续维护：尺寸检查、ESO日志与函数说明

基于 `276c1ff`，补齐两处实际运行问题：FLAT 在减暗前核对 DARK 数组尺寸，
即使 setup 未知也拒绝广播；五种 recipe 在调用期间将科学日志转发给 CPL，
遵守 PyEsoRex 的终端/文件日志级别，并在成功或异常退出时恢复 Python logger。

函数说明统一使用 NumPy 风格，逐项解释参数、数组轴、单位、返回值、原地修改
和关键科学约定。更新了 WAVE/FLAT 的固定选区说明、波数插值与 flat 归一化、
ABCD 相位符号、GD 初始化范围和 flags 含义；不改变原有科学判据。

新增回归覆盖 DARK 广播拒绝、合成 raw DARK→FLAT→WAVE 的已知 trace/色散、
热像素排除和独立 lamp DARK，以及 recipe 异常后的日志恢复。现有 ESO 链测试
同时核对终端 INFO 抑制与 logfile DEBUG 保留。共 15 项测试通过（含真实执行器），
Ruff 和格式检查通过。移除 docstring 后的 AST 对照确认：生产代码的可执行变化
仅限上述尺寸检查和日志接入，数值算法、阈值、mask 与运算顺序保持。

下文为此前 0.15.0 I/O 整理及其真实数据对照记录。

基于 `256efee`。本轮清理的是额外的功能、隐含选择和重复处理；保留科学数组与文件中的物理含义。

## 当前io目录

| 文件 | 职责 |
| --- | --- |
| raw.py | 从明确的类别和HDU读取物理header、分块读取cube |
| fits.py | 各产品的具体FITS字段编解码，以及显式保存 |
| hdu.py | 精度/大小端、checksum、单文件暂存发布 |
| keywords.py | 明确的FITS名称、类别和QC映射 |

删除acquisition.py和publication.py。原来的6个功能文件收敛为4个；未新增manager、
serializer、注册器、输入配置对象或通用validation层。

## 具体修改

### 明确输入

raw默认读取`IMAGING_DATA_SC`。非标准bench文件必须由调用者传`hdu=0`或实际扩展名，
不再自动遍历HDU寻找图像。五个文件步骤和提取入口都接受这个明确的hdu参数。
`read_raw_metadata`必须传category，只核对文件中已声明的DPR type，不再自行推断类别。

删除`parse_raw_header`与仅包装Astropy的`read_raw_header`。维度检查集中在选定cube的
入口；单位、快门、ROI和曝光检查仍在文件边界。标定配对规则回到各步骤，直接调用
`check_setup(raw.setup, selected.setup, ...)`，不再由I/O层决定使用什么标定。

### 明确输出

Python调用直接写为：

```python
quality = measure_quality(product)
write_product(path, product, quality=quality)
```

省略quality就不写QC。writer和encoder都不再计算统计；recipe在调用处显式测量QC。
DFS writer接收稳定的`(filename, product, quality)`条目，不改变中途数据结构。

每个文件使用同目录临时路径，写完后通过不替换已有目标的硬链接发布。删除多文件
回滚、inode核对和输出必须同目录的约束。后面的文件失败时，前面完整发布的结果保留。
recipe仍在发布前完成CPL DFS签名。

### 删除自定义格式门槛和无用模式

删除`PLANETES FORMAT`与`PLANETES BAD BITS`的写入及强制检查。文件是否可读取决于
类别、实际必需HDU/列、单位和身份；缺失字段直接报错，不增加schema迁移或JSON兼容机制。
这也撤销了0.14.0因格式号而一律拒绝旧标定的行为。

checksum保留一次显式检查，删除Astropy自动检查与手动检查的重复。
`_column`删除无人传入的dim参数和已被统一存储精度排除的D/M分支。
QC已有m/um keyword逐项声明，移除通过replace派生keyword的逻辑，实际名称和值保持。
删除raw定义表中仅供已移除采集writer使用的字段，编码器统一使用已有keyword常量。

随包LINE_LIST去掉私有格式号卡；其谱线值、origin标签和原表精度保持不变。
生产package不再提供raw采集写入接口，测试只用一个简短函数生成固定synthetic fixture。
外部采集/GUI项目未改动。

## 验证

- 现有12项测试通过，数量未增加；测试明确选择primary/extension HDU，验证版本号
  不再阻止读取、QC显式传入、输出不覆盖和临时文件清理。
- 同一批15份真实raw完整运行五种recipe：10份产品、31个数组与0.14.0逐字节一致。
  QC、物理setup、输出身份、recipe参数和实际输入记录保持，全部HDU checksum有效。
- 31个数组科学计算函数的AST保持相同；未改变阈值、mask、单位、精度或相位顺序。
- 单独验证DFS第二个输出遇到已有文件时：第一个完整输出保留，已有文件不变，临时文件清理。
- Ruff和diff检查通过；接口文档、example和三张流程图已同步。

## 备份与结果

修改前源码及一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261029_105420_direct_io/`

独立结果：`reduced/direct_io/20261029_105420/`。
`chain/comparison.txt`是全链对照，`checks.txt`是计算函数与发布行为核对，`tests.log`是现有测试。
没有新增JSON或验证框架。

[架构图](architecture.png) · [产品接口](PRODUCT_CONTRACT.md) ·
[0.14.0报告](https://github.com/zexwu/planetes_pipe/blob/256efee/docs/CHANGE_REPORT.md)
