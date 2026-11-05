# PLANETES 0.25.0：简洁的科学图版式

- 报告改为简洁的科学图风格：页边距加大（左右约 2.3 cm），细边框、四边朝内的刻度和次刻度，
  图例无边框。
- 多面板图共用坐标轴：x 轴（以及可比较时的 y 轴）只在最外侧面板标刻度，轴标签每组只写一次；
  基线名写在面板角上，不再每个面板写标题；每节只有一个图例，放在面板上方。
- 表格改为三线表：只有横线（顶线、表头线、底线），浅色底纹标出失败、通过和该基线的望远镜。
- 新的公用函数 `pages.panels`、`pages.baseline_panels`、`pages.legend`；去掉
  `pages.baseline_grid`。数值结果不变。

# PLANETES 0.24.0：竖版 A4 报告，版式按 GRAVITY

- 诊断 PDF 改为竖版 A4，便于打印，版式按 GRAVITY `gravi_visual`：首页大标题；之后是
  依次排下来的小节，每节有粗体标题、一行说明和内容（图或表），一页放得下就接着放；每页
  页脚为 "Page n / N"、产品文件名与 pipeline 版本。报告第一节是产品 header 中的身份信息
  （类别、recipe、pipeline、写出时间、DIT、raw 输入）。
- 各模块的 `pages()` 改为 `sections()`，返回 `pages.Section`（标题、说明、高度、绘图函数）；
  `pages.write_pdf` 负责分页。表格的行标签作为第一列，留在页边距之内。
- 数值结果不变。

# PLANETES 0.23.0：读取压缩的 raw

- `read_raw` 读取 tile 压缩的 raw（`CompImageHDU`）：返回 `raw.CompressedCube`，按需逐块
  解压，与 memmap 一样不会一次读入整个 cube；普通图像仍然 memmap。旧的 int32 文件照常读取。
- bench 写 raw 改为 int16（12 位数据可原样保存）+ 无损 Rice 压缩，每帧一个 tile：比
  int32 小约 5 倍（768 帧的扫描约 4 GB → 0.8 GB）。test_data 处理结果与 reference 逐位相同；
  处理时间约为未压缩时的 1.8 倍（每帧解压约 7 ms）。
- 数值结果不变。

# PLANETES 0.22.0：每种产品一个诊断 PDF

- `plots.py` 拆成 `plots/` 包，与 `steps/` 一一对应：`dark`、`flat`、`wave`、`p2vm`、
  `science`，以及共用的 `pages`。每个模块的 `pages(product, ...)` 生成 matplotlib
  `Figure`（不经过 pyplot，因此不需要 GUI 后端）。0.16 时曾把 `plots/` 合并成一个文件；
  这次按产品拆分，规则与 `steps/` 相同。
- `plots.report(product_file)` 按 PRO.CATG 选模块，在产品旁写出 `<product>.pdf`；命令行
  `python -m planetes_pipe.plots FILE...`。不覆盖已有 PDF。`save_calibration_plots` 与
  `save_reduction_plots` 保留原签名，由各模块的页面组成。
- P2VM 报告按 GRAVITY `gravi_visual_p2vm.py` 补齐：质量汇总（输出相对透过率 0.7–1.3；
  中心 70 % 波段内归一化 coherence 0.80–1.03，超出不得多于 5 % 的通道；另加 raw 饱和），
  GOOD/WARNING/BAD 判定；每路输出的相对透过率；coherent flux 传递矩阵；visibility 传递
  函数；输出相位表；c-matrix。另加两页扫描诊断：ABCD 椭圆的单位圆，以及条纹相位减去
  2π·OPD/λ 的残差（不 unwrap，按 wrap 后的差值）。
- DARK/BAD、FLAT、WAVE 各有自己的页面：探测器图与分布、按类型的坏像素、光迹与宽度、
  望远镜通量、波长解与谱线残差。
- 数值结果不变。

# PLANETES 0.21.0：记录 raw 饱和

- `io/raw.py` 新增 `SATURATION_ADU = 4095`（探测器 12 位读出的截断值）、
  `Raw.saturated()`（所有帧中达到该值的采样数）与 `record_saturation(header, raws)`。
- FLAT、WAVE、P2VM 与科学产品的 header 记录每个 raw 输入的饱和采样数：
  `QC.NSAT.T4…T1`（flat）、`QC.NSAT.SC43…SC21`（扫描）、`QC.NSAT.LAMP`、`QC.NSAT.SCI`；
  有 FLAT 时只计提取用到的像素（`Flat.extracted_pixels()`），FLAT 本身计整个探测器；
  不为零时记录 warning。QC 只做描述，不改变拟合与产品数据。
- 原因：2026-11-04 的扫描 43 在 B/D 输出（约 1240–1265 nm）有 39,725 个采样截断在
  4095 ADU，P2VM coherence 因而在该波段出现 dip；2026-02-05 测试数据的扫描 43 同样有
  7,912 个截断采样。
- 新增 keyword `DATE-OBS`、`MJD-OBS`、`TPL.ID`、`TPL.START`，供 bench 记录拍摄时间与
  拍摄脚本的运行（产品从 reference raw 继承这些卡片）。
- 数值结果不变。

# PLANETES 0.20.0：探测器设置直接使用 ESO DET 卡片

- 删除 `DetectorSetup`、`check_setup`、`read_setup`/`write_setup` 与 `UIT1` 回退。探测器
  设置就是 raw header 的 DET 卡片（DICD §4.4.2.7：`DET.ID`、`DET.WIN1.STRX/STRY/NX/NY/
  BINX/BINY`、`DET.WIN1.DIT1`），bench 的 `camera_cards()` 直接写这些卡片。
- `Raw` 保存 primary header（`Raw.header`、`Raw.detector`、`Raw.shutters`）；DARK/FLAT
  产品的 `setup` 字段改为 `detector`（原样的 DET 卡片）。
- 兼容性检查只剩一条规则 `raw.check_match`：两边都记录且值不同即报错，报错写出 ESO
  keyword 名；DARK 比较窗口与 DIT，FLAT 比较窗口（`keywords.WINDOW`）。DIT 精确比较。
- `instrument.py` 只保留通道顺序；`shutter_identity` 移入 `io/raw.py`。
- bench 采集脚本的 `SETUP = DetectorSetup()` 改为 `DETECTOR = {}`（ESO keyword → 值）。
- 数值结果不变。

# PLANETES 0.19.0：目录分为 steps/ 与 io/

- 五个步骤及共用的 `extraction` 移入 `steps/`，`keywords`、`raw`、`products` 移入 `io/`；
  顶层只保留 `instrument.py`、`plots.py` 和 `eso/`。导入改为
  `from planetes_pipe.steps import dark, flat, ...` 与 `from planetes_pipe.io.raw import read_raw`。
- 代码内容不变。

# PLANETES 0.18.0：keyword 按 ESO 字典方式定义

- 新增 `keywords.py`：每个 keyword 是一个 `Keyword`（ESO 简写名、注释、单位，对应
  DICD §7.3 的字典条目），读写都通过它：`DIT.get(header)`、`DIT.set(header, value)`、
  `SHUTTER.at(4)`。代码中不再拼写 `"HIERARCH ESO ..."` 字符串；同一个 keyword 的读写
  不再分两处维护。QC keyword 定义在各步骤的 `qc()` 旁，`qc()` 返回 `{Keyword: value}`。
- 删除 `set_card`、`write_qc`；`products.save_product` 写全部 DICD §4.7.1 PRO keyword，
  `provenance` 只写继承卡片与 PRO.REC1 记录。
- 按 DICD 调整：`DATE` 带毫秒（§4.5）；卡片按类别排序（§3.1.1.3）；扩展头写
  `INHERIT = T`；DET 卡片都带单位注释；QC 单位改为 `adu`、`m`。
- 数据与 QC 数值与 0.17.0 完全相同；只有 header 注释、顺序与上述卡片变化。

# PLANETES 0.17.0：产品按 float64 原样存储

- 产品不再降为 float32 存储：删除 `_storage_array` 与 `STORAGE_RTOL`，波长覆盖按精确值检查；
  读取只做字节序转换（FITS 大端 → 本机，numba 需要）；读取时不再重新校验 checksum
  （写出时仍写）。0.17.0 之前的产品是 float32，建议重新生成。
- `files.py` 按方向拆为 `raw.py`（`Raw`、raw 读取、DET/快门 header）和 `products.py`
  （产品读写、扩展、QC 卡片、溯源）；像素坐标转换移入 `flat.py`。`read_raws` 改为一条
  明确规则：每种快门配置恰好一个文件。
- 2026-02-05 数据相对 float32 版本：标定读回不再舍入，lamp 拟合多接受一条谱线
  （544 → 545），WAVE 中位偏移 0.9 pm（最大 26 pm，拟合残差约 127 pm），P2VM 波长网格
  最大偏移 2.4 pm；science 的 VISAMP 中位变化 1e-4，差分相位 0.008°，GD 0.3 nm，flags 不变。
  文件约为原来的两倍（DARK 21 MB，P2VM 26 MB，science 91 MB）。

# PLANETES 0.16.0：每个产品一个模块

- 删除 `model/`、`io/`、`processing/` 三层结构。每个步骤（dark、flat、wave、p2vm、
  reduce）一个模块，包含数据类、算法、`qc()`、FITS 读写和 `run()`。
- `run()` 是 Python 与 PyEsoRex 共用的入口：文件进、产品文件出，返回
  `{类别: 路径}`。所有产品写同一套 DICD PRO 溯源 header；EsoRex 只负责签名
  （DATAMD5）和登记。P2VM 相位参考提取移入 `p2vm.extract_phase_reference()`。
- `keywords.py` 集中声明全部 keyword、raw tag、QC 和 recipe 参数；QC 由 dataclass
  改为 `{keyword: value}`。新增 `ESO QC NWAVE SC`，取代 RMSWAVE 注释中的 `N=`。
  `tests/test_keywords.py` 检查未声明的 keyword 和超长卡片。
- raw 由一次性迭代器和 context manager 改为 memmap：`files.read_raw()` 返回
  `Raw(cube, …)`，`Raw.chunks()` 可重复迭代；不再需要 `with`/`read_many`。
- raw 快门写为 `ESO INS SHUTn ST`；2026-11 前的 `PLANETES INS SHUTn ST` 仍可读取。
- 全部 numba kernel 移入 `kernels.py`；PyEsoRex recipe 合并为 `eso/recipes.py`。
- 暂时移除价值不大的功能：recipe 的 `static-name` 与 `phase-correction` 选项
  （SOF 中有 PHASE_REFERENCE 即使用）、`*_UM` 重复 QC、未使用的 `diagnostics()`、
  暂存再硬链接的写出（改为 `writeto`，已存在文件直接报错）。
- 删除只重复校验本程序刚写出数据的检查：读取时的单位/逻辑编码检查、INSTRUME、
  `_check_baselines`、P2VMRED 行顺序与站点、P2VM REGNAME 交叉检查。保留针对
  用户错误的检查（标定 setup 不匹配、快门缺失/重复、标定选择不唯一等）。
- 仓库整理：测试按模块命名（`test_<module>.py`，共用 `fixtures.py`）；文档合并为
  `methods.md`、`products.md`（含 QC 与 recipe 参数）、`code_style.md`、`changes.md`；
  删除过时的 0.15 设计说明与流程图。
- 自查后的修正：`fits.py` 改名 `files.py`（避免与 `astropy.io.fits` 混淆）；`p2vm.py`
  拆为 `p2vm/__init__.py`（步骤）与 `p2vm/fit.py`（矩阵拟合）；帧平均改为
  `Raw.mean()`，去掉 extraction 的循环 import 绕行；更新过时的 docstring。
- 精简结构：`p2vm` 恢复为单个文件；numba kernel 回到调用它的模块，删除
  `kernels.py`；删除 `keywords.py`（QC 注释写在 `qc()`，raw tag 在 `files.py`，
  recipe 帮助文本在 `eso/recipes.py`）；`plots/` 合并为 `plots.py`；`reduce` 改名
  `science`。recipe 选项名改为 Python 参数名加连字符（`bad-pixel-threshold`、
  `aperture-width-pixels`）。
- 选择输入文件（原 `association.py`）移出 pipeline，放入 bench 的
  `scripts/reduce_calibrations.py`；示例脚本改为明确写出输入。
- 入口与 helper 重写：`run()` 参数统一为 `*_file(s)`，按"读入 → 计算 → 写出"顺序书写；
  删除只转发参数的 `make_p2vm` 与 `extract_phase_reference`（相位参考与扫描共用一个
  extractor）；`files.read_raws` 读入时按快门确定一次物理身份，`Raw` 带 `path`；
  `check_calibrations` 取代成对的 `check_setup`；删除 `as_compute_array`、`table()`、
  `output_paths`，`product_filename` 改为 `product_path`，像素坐标转换合为
  `to_fits_pixels`/`from_fits_pixels`；读写函数和 `plots.py`、`wave.py` 不再使用嵌套函数。
  修复 `files.set_card` 上残留的 `@contextmanager`。
- 数值算法、阈值和产品 FITS 布局不变。

# PLANETES 0.15.0：让I/O直接表达读了什么、写了什么

## 后续维护：与 bench 的包名及产品接口同步

科学软件使用独立 distribution `planetes-pipe` 和 Python namespace `planetes_pipe`，
恢复 `PLANETES_new/bench` 已采用的接口约定。硬件项目 `catkit2-planetes` 继续拥有
`planetes` namespace，两者可在同一进程导入。ESO recipe 名和产品类别保持不变。
更新包目录、资源查找、构建配置、测试与文档；同步 pipeline/bench 的 uv 环境，
未升级第三方科学依赖，科学函数 AST 保持一致。

bench 新增独立 Git 仓库与两项离线联调测试，检查 raw 写入/只读回放、动态命名标定
按 PRO CATG 加载、明确 DARK 的逐帧/离线一致性，以及重复标定类别的拒绝行为。
同步 acquisition 示例和数据目录的 prepare_reduce.py，产品名统一调用
`product_filename`，不再从 archive 重算命名。非 Git 文件修改前已保存在 bench/archive。

mba 验证：pipeline 17 项测试通过；bench 2 项联调通过。实际 2026-02-05 science
回放读取 1024×1280 raw 帧，输出 6×151 个有限 VISAMP，明确使用 science DARK；
与已存 float32 首帧的最大差值为 5.96e-8。验证仅使用离线回放，未操作硬件。

## 后续维护：保留 raw 文件名中的时间信息

在 mba 的 `/Users/zexwu/PLANETES_new/planetes_pipe` 开发。默认命名沿用
GRAVITY 的 reference-raw 规则：完整 raw stem 加小写、去下划线的 PRO CATG。
五种 recipe 显式选择命名参考：DARK/BAD、WAVE 和 science 分别使用本步 raw；
FLAT 使用 T4，P2VM 使用 43 基线，使 SOF 排序不影响文件名。DFS header
也从同一份 reference raw 继承，保留已有 DATE-OBS 等观测时间卡。
`--static-name=true` 保留此前固定名称，不改变禁止覆盖的写入行为。

Python 增加 `product_filename(reference, product)`，示例和 SOF 模板同步更新。
各产品模块的 `write()` 仍严格使用调用者给定的路径，不自动改名。时间戳、毫秒和
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
| files.py | 各产品的具体FITS字段编解码，以及显式保存 |
| hdu.py | 精度/大小端、checksum、单文件暂存发布 |
| 各产品的 encode/decode | 直接声明 FITS 名称；QC 字段在 model/quality.py |

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
product_io.write(path, product, quality=quality)
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

波长参考谱线直接定义在 `processing/wave.py`；保留原表数值精度和来源注释。
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

[架构图](architecture.svg) · [产品接口](products.md) ·
[0.14.0报告](https://github.com/zexwu/planetes_pipe/blob/256efee/docs/CHANGE_REPORT.md)
