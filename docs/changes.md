# PLANETES 0.40.0：望远镜统一记为 S1–S4

- 报告、日志和关键字注释里的望远镜从 T1–T4 改为 S1–S4，与基线 S4–S3 … S2–S1 一致：平场 QC 表头、
  P2VM 透过率表、科学报告的光通量图例，`Shutter S4 open`、`Total flux per DIT, S4` 等关键字注释，
  以及 P2VM 对各路平场不一致的报错。闭合三角形记为 S4–S3–S2 等。
- 平场的饱和计数关键字改为 `QC.NSAT.S4…S1`（原 `QC.NSAT.T4…T1`）；P2VM 报告读新名字，
  0.40.0 以前的产品重新出报告时不显示这几项平场的饱和数，重新 reduce 即可。
- `OI_T3`、`T3AMP`、`T3PHI` 是 GRAVITY/OIFITS 的表名和列名，不变。数值结果不变。

# PLANETES 0.39.0：不再支持 2026-11-05 以前的扫描，删除 `dl_gain`

- P2VM 扫描必须在 DL_SCAN 中记录 `OPDGAIN`（2026-11-05 起 `take_p2vm_calibration` 都会写入）。
  `find_reversed_baselines` 对缺少它的扫描报错，不再警告后当作未反向。
- 删除 `p2vm.run`、`find_reversed_baselines`、`delay_line_opd` 的 `dl_gain` 和 PyEsoRex 参数
  `planetes.calib.dl-gain`。较早的数据（如 2026-02-05 的 test_data）不再支持。
- 数值结果不变（11-06 的 P2VM 与 0.38.0 逐位相同，见下）。

# PLANETES 0.38.0：报告页边距变窄，密集的点画成图像

- 所有报告（`plots.pages`）的页边距从 2.3 cm（顶 2.4 cm、底 2.2 cm）改为两侧约 1.1 cm、顶 1.5 cm、底 1.4 cm；
  首页标题和页脚随之靠近页边。
- 小节标题下的说明可以有多行：标题区的高度随说明的行数（每行 150 字符）增加，不再固定为一行。
- P2VM、科学和 WAVE 报告去掉开头的 Definitions 表（删除 `pages.definitions_section`）：每个量的公式写在用到它的
  小节说明里（如 $T_{ot}$、$V_{ob}$、$c_{ob}$ 的拟合模型、VISAMP、GD、DIFFPHASE、T3PHI、SNR 与误差、WAVE 的拟合与残差）。
- `write_pdf` 保存前把密集的图画成图像：超过 100 个点的散点（`RASTER_MARKERS`，如每帧一个点）和超过 600 个点的
  曲线（`RASTER_POINTS`；约 520 点的光谱仍是矢量）。坐标轴、文字和其余曲线仍是矢量。每个点一条矢量路径时，
  相位诊断报告约 0.7 MB。图像按 matplotlib 默认的 100 dpi 输出（waterfall 一直按此重采样）。
- 数值结果不变。

# PLANETES 0.37.0：估计科学数据的群延迟前先去掉色散

- 色散使相位在波数上弯曲，延迟峰的顶部变平，噪声使每帧的 GD 沿平顶滑动。T4 的三条基线弯曲 4-11 rad（从波段中心到
  边缘），延迟峰半高宽约 30 µm（无色散时约 11 µm）。2026-11-05T173504 的 41 基线 GD 每帧散布 1.1 µm，
  11-06T175306 的 41 和 43 分别为 0.71 和 0.56 µm；去掉色散后为 0.04、0.05 和 0.01 µm。
- `extraction.tracked_group_delay` 在第一次估计后，用有条纹（峰值 ≥ 20）的帧求出数据自身的色散相位
  （`extraction.dispersion_phase`：各帧按 GD 和平均相位对齐后求和，相位用波数的 4 次多项式拟合，去掉常数和线性项），
  去掉后重新估计每帧的 GD，重复两次（第一次的色散受模糊 GD 影响；模拟 11 rad 弯曲时 GD 偏 0.8 µm，第二次后 0.03 µm）。
  少于 64 帧有条纹时（如实时单帧）不做。
- 色散相位没有线性项，GD 的含义不变（平均波数处的群延迟）。VISDATA、VISAMP 和闭合相位不用 GD，不受影响；
  DIFFPHASE 随 GD 变化。

# PLANETES 0.36.0：科学数据的群延迟在 ±1000 µm 内跟踪条纹

- 科学数据的 GD 以前只在 ±100 µm 内搜索。条纹在范围外时 GD 是随机值，VISAMP 也错：2026-11-06T175510 的 T4
  在 −300 µm（GD 散布在 −100..0 µm，T4 基线 VISAMP 0.8）；2026-11-05T173504 的六条基线在 +217..+668 µm。
- 现在用 `extraction.tracked_group_delay`：每 64 帧用 `stacked_group_delay` 在 ±1000 µm 内找到条纹，每帧再在
  其 ±50 µm 内搜索。`gd_search_min_m` / `gd_search_max_m`（科学）默认改为 ±1000e-6，含义是找条纹的范围。
  某块帧在范围内没有条纹（峰值 < 20）时给出警告。
- bench 的实时显示仍传 ±100 µm，速度不变；`center_fringes` 已在 ±1000 µm 内搜索。

# PLANETES 0.35.0：扫描 OPD 只由条纹决定

- P2VM 扫描每帧的 OPD 现在只由条纹决定，不再使用延迟线记录的位置。每帧的整数条纹数由群延迟确定
  （`p2vm.scan_group_delay`：先用 `stacked_group_delay` 在 ±1000 µm 内由所有帧求出扫描中心，再在每帧中心
  ±150 µm 内搜索）。群延迟取决于相位随波长的变化，与整数条纹无关。一个条纹以内的位置仍由相位决定：
  群延迟每帧精度为 0.04-0.12 µm，相位为 nm 量级。
- 以前整数条纹数由延迟线位置（`DL_SCAN` × `OPDGAIN`）决定。扫描每步跨好几个条纹，延迟线的回差超过半个条纹，
  2026-11-06 的扫描有 63-89% 的帧差了 1-9 个条纹。OPD 在波段中心是对的，在两端不对，P2VM 的 contrast 在两端
  降到 0.14-0.4，科学数据的 VISAMP 在两端达到 2-5（中心约 1.05）。现在 11-06 17:18 P2VM 的 GD − OPD rms 从
  1.4-4.5 µm 降到 0.04-0.12 µm，contrast 在整个波段是平的，17:40 科学数据的 VISAMP 为 1.00-1.02。
- `OPDGAIN` / `dl_gain` 现在只决定基线方向（`find_reversed_baselines`，两者须一致，都没有时给出警告）。
  删除 `read_scan_opd` 和 `fit_p2vm` 的 `scan_opd_m`；`gd_search_min_m`、`gd_search_max_m` 只用于相位参考。
- 延迟线记录的 OPD 仍保存在 P2VM（`PLANETES_DL_OPD`，`keep_diagnostics` 时），只用于对比。P2VM 报告的
  "Recovered scan delays" 把它叠加在条纹 OPD 上；新的 "Delay-line OPD error" 页画出两者之差，标出 gain、rms、
  最大误差和 ± 半个条纹。
- WAVE 报告增加内容（参考最早 `recipes/wave.py` 的诊断图）：汇总表（覆盖波段、色散、参考谱线数、拟合/剔除的谱线、
  残差的 nm 和像素值、每个 output 的谱线数、灯的饱和）；灯的图像，标出接受和剔除的谱线；所有 output 的灯光谱画在
  各自的波长解上，标出参考谱线是否被使用；波长解、色散 dλ/dx、与各 output 中位数之差；每个 output 的谱线数；
  残差随 x、随 y 的分布（含剔除的谱线）；每条参考谱线的残差。为此 WAVE 产品保存了所有拟合的谱线（含剔除的，
  `PLANETES_WAVE_FIT` 增加 `WAVELENGTH`、`OUTPUT`、`REJECTED` 列）、`PLANETES_LAMP_SPECTRA` 和
  `PLANETES_LAMP_IMAGE`。`fit_residual_m` 等字段现在包含剔除的谱线，QC 只用接受的（`wave.accepted_residual_m`）。
- **0.35.0 以前的 P2VM 都需要重做**，用它们处理的科学数据也要重做。

# PLANETES 0.34.0：多帧群延迟

- `extraction.stacked_group_delay`：把多帧的相干功率相加，得到一个群延迟和峰值高度（峰值 / 搜索范围内的中位数）。
  11-05 的数据里，有条纹时峰值至少 25，没有条纹时最多 8。bench 的 `center_fringes.py` 使用它。
- 群延迟和所用波段有关：去掉 1150 nm 以下的弱光通道后，17:18 文件的群延迟闭合误差从约 7 µm 降到约 2 µm。
  科学产品里的 `GDELAY` 仍使用整个 P2VM 波段。

# PLANETES 0.33.0：可限制 P2VM 波段

- `p2vm.run` 增加 `wave_min_m`、`wave_max_m`，限制 P2VM 的波段。默认不限制；bench 的 `reduce_night.py` 和
  `examples/reduce_calibrations.py` 设为 1100-1270 nm，避开 SuperK 泵浦（约 1064 nm）和两端的弱光部分。

# PLANETES 0.32.0：S3-S2 的方向；迹线跟踪窗口

- P2VM 现在统一基线方向（`C_ij = E_i E_j*`）。扫描的 `OPDGAIN`（或 `dl_gain`）为正的基线（目前只有 S3-S2）的
  响应相位反号（`p2vm.find_reversed_baselines`），于是 `VISDATA`、群延迟、差分相位和闭合相位的约定一致。
  以前含 3-2 的两个闭合相位（T4-3-2、T3-2-1）完全不闭合。这条基线的 B−A 现在显示为 270°。
  **旧的 P2VM 仍是反的，需要重做。**
- FLAT：判断某列是否有光时，窗口跟随追踪到的迹线中心，不再固定在 `ref_y` 附近。迹线是斜的，以前走出窗口后范围
  就被截断（如 11-05 的 output 23 从 x=661 才开始）。
- P2VM 报告的输出相位表改用圆平均；以前相位在 ±180° 附近时，C−A 会算错（如显示 259°）。

# PLANETES 0.31.0：P2VMRED 增加误差和闭合相位；报告改用瀑布图；迹线范围

- SINGLE_SCI_P2VMRED 增加 GRAVITY 的列：`FLUXERR`、`VISERR`、`CHI2`、`VISPHI`、`F1F2`、`SNR`，以及新表
  `OI_T3`（`T3AMP`、`T3PHI`、`T3PHIERR`）。误差：输出方差 = 信号 / 增益 + DARK RMS 给出的读出噪声，经
  P2VM² 传播，如 `gravi_p2vm_reduce`。增益 36 e-/ADU（`science.GAIN_E_PER_ADU`），由 2026-11-05 平场的
  帧差 photon transfer 测得（34-40）。11-05 科学数据的 CHI2 中位数 0.83，帧差噪声 / 误差约 0.9。
- 科学报告按 `gravi_visual_p2vmreduced` 重排：平均光谱、通量、相干通量和 SNR 画在一张图上（颜色区分望远镜或
  基线）；通量、VISAMP、差分相位用瀑布图；相位和群延迟用共享时间轴的六行图；增加闭合相位和 CHI2。
  FLAT 报告的 Trace centres 也合成一张图。
- 迹线范围：在沿 x 平均 21 列的平场上跟踪，只用噪声阈值（不再用各迹线峰值的 5%，否则范围取决于 SuperK
  泵浦峰），取最长的连续好列。11-05 17:02 平场的公共波段从约 272 个通道增加到 447 个（1150-1296 nm）。

# PLANETES 0.30.0：报告和 P2VMRED 采用 GRAVITY 的名称

- SINGLE_SCI_P2VMRED 的表改名为 GRAVITY 的 `OI_WAVELENGTH`（`EFF_WAVE`）、`OI_FLUX`（`FLUX`）、
  `OI_VIS`，带 `INSNAME`。新增 `TIME` [us]、`MJD`、`INT_TIME`，由原始文件的 MJD-OBS 和 DIT 估算
  （假设每个 DIT 一帧；相机没有逐帧时间戳）。没有误差、目标和阵列数据，所以不是合法的 OIFITS。
  `read_reduced` 仍可读取旧的 `PLANETES_*` 产品。
- P2VM、DARK、BAD、WAVE 和科学报告的标题与说明改用 GRAVITY `gravi_visual` 的用语（只在内容相同时）；
  BAD 报告增加 Statistics 表（超过 10000 个坏像素时标色）。

# PLANETES 0.29.2：报告说明每个量的定义

- P2VM 和科学报告开头增加 Definitions 表（`pages.definitions_section`），给出透射率 T、相干度 C、
  归一化相干度 V、VISAMP、群延迟等的公式；所有报告的小节说明也写明公式或定义。
- 说明文字换行时不再拆开 `$公式$`。

# PLANETES 0.29.1：望远镜通量悬殊时也能找到迹线

- `flat.build_flat` 取行轮廓中最突出（prominence）的 24 或 25 个峰作为迹线，不再要求高于轮廓的
  第 75 百分位。2026-11-05 17:02 的平场 T4 比 T1 亮约 19 倍，只由暗望远镜照亮的迹线低于该
  阈值，只找到 21 条。最弱迹线与下一个峰的突出度之比低于 3 时给出警告
  （`flat.TRACE_PROMINENCE_MARGIN`）。

# PLANETES 0.29.0：暗淡平场的迹线与输出映射；条纹对比度

- 修复 2026-11-05 的 P2VM 失败。那天的平场比 11-04 暗约 7 倍，轮廓不对称（陡升、长尾），
  峰值与 Gaussian 中心相差 2-3 像素。迹线跟踪从峰值出发，第一次拟合就被
  "每列移动 < 2 像素" 拒绝，此后一直卡住：输出 0 只拟合了 39 列，在其余位置偏离光谱，
  提取不到光。`flat.trace_spectrum` 现在先在起始列用重新居中的质心确定起点。
- P2VM 的输出映射不再取每个望远镜最亮的 12 个输出，而是取超过其典型亮输出
  （最亮 12 个的中位数）10% 的输出（`p2vm.ILLUMINATED_FRACTION`）。弱但被照亮的输出仍算，
  噪声水平的不算。不一致时错误信息列出相关输出的平场通量。
- `p2vm.normalized_coherence`（从 plots 移来）和新的 `p2vm.instrumental_visibility(p2vm)`：
  标定扫描时每条基线的条纹对比度 (baseline, wavelength)。VISAMP 乘以它就是一帧的条纹
  对比度；bench 实时显示用它来优化偏振。

# PLANETES 0.28.0：扫描自带 OPD 符号

- P2VM 扫描的每一步约 1.9 µm OPD（约 1.5 个条纹），超过半个条纹，单靠条纹相位无法解缠绕，
  必须用延迟线读数指导。以前这需要手动传 `dl_gain`，忘了就得到错误的 P2VM 而没有提示。
- bench 现在把每条基线的 OPD 符号写进扫描文件：DL_SCAN 头里的 `OPDGAIN`
  （`take_p2vm(..., opd_gain=...)`，取值见 `take_p2vm_calibration.OPD_GAIN`）。
- `p2vm.read_scan_opd` 优先用记录的 `OPDGAIN`；`dl_gain` 只给没有记录的旧扫描用，两者都有时
  必须一致，否则报错。两者都没有时给出明确警告（并报告延迟线步长），而不是静默地继续。
- 2026-11-05 之前的数据仍需 `dl_gain`；例子和 bench 的 `reduce_night.py` 保留它。

# PLANETES 0.27.0：自动找 dark

- `io.raw.find_dark(raw_file, directory)`：为一个曝光找对应的 DARK_RAW，规则与 bench 的
  night 相同：探测器窗口和 DIT 必须一致，然后优先同一次采集（TPL.START），再取 MJD-OBS
  最近的；找不到或无法判断时抛出 LookupError。通用版本 `io.raw.closest_raw(...)`。
- `examples/reduce_calibrations.py` 不再需要手动填 dark：flat/scan 和灯的 dark 都自动找，
  选中的文件和时间差写入日志。

# PLANETES 0.26.0：清楚的日志

- 每条日志都写明时间、级别、步骤和函数：
  `11:31:15 INFO    [WAVE] wave.fit_wavelength_solution: lines matched per output: 14-18`。
  新模块 `planetes_pipe.logs`；每个模块用自己的 logger（`logging.getLogger(__name__)`）。
- 每个步骤的 `run()` 由 `@logs.step(...)` 包装：开始时列出输入文件，结束时列出写出的产品和
  用时；出错时记录哪一步、多久之后、什么错误，再抛出。报告（`plots.report`）同样标为 [REPORT]。
- INFO 只留摘要（每步几行）：DARK 的电平和坏像素数，FLAT 的 trace 范围，WAVE 的谱线匹配和
  拟合残差，P2VM 每条基线的 OPD/GD 范围和拟合 RMS，SCIENCE 的 group delay 中位数。逐条
  trace、逐个 output、逐次迭代的细节改为 DEBUG。去掉 "--- Step ... [DONE] ---" 一类的行。
- `configure_logging(level="INFO", file=None, file_level="DEBUG")`：可同时写日志文件（带毫秒）；
  再次调用会替换原来的 handler。PyEsoRex 下 CPL 显示 module.function，消息带 [步骤]。
- 例子脚本调用 `configure_logging()`；bench 的 `reduce_night.py` 用同一格式写 `reduce_night.log`。

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
