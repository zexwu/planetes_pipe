# PLANETES 0.8.0：明确的科学步骤与 Python API

基于 `ecce961`，统一当前 package 的函数、字段与参数名称，并清理与职责不符的
依赖。原有科学公式、默认值、FITS format 5/4兼容性、raw format 3、SOF类别、
五个PyCPL recipe名和现有CPL参数/CLI别名保持。

## 文件入口与数组计算

| Raw files → products | Arrays → scientific result |
| --- | --- |
| make_dark_products | measure_dark / detect_bad_pixels |
| make_flat | build_flat |
| make_wave | fit_wavelength_solution |
| make_p2vm | fit_p2vm |
| reduce_science | reduce_spectra |

入口仍在对应的dark.py、flat.py、wavelength.py、p2vm.py、reduction.py中。
保存由调用者或recipe明确执行，没有新增Pipeline/Task对象、wrapper或兼容别名。

## 实际简化

- `build_flat(flat_images, bad)`只接收已减暗图像与BAD；不再接收未用于减暗的Dark。
  文件入口make_flat负责选择并携带raw detector setup。
- `trace_spectrum`固定返回centers、position errors和Gaussian sigma三个数组。
- `aperture_width_pixels`明确表示提取孔径全宽，默认10 pixel；与Gaussian sigma分开。
  对外CPL名称仍是planetes.calib.profile-width，CLI仍是--profile-width。
- `include_phase`控制GD和差分相位；完整处理默认True，FrameReducer预览默认False。
- `P2VM.matrix`替代同名数值字段；baseline_outputs统一表示24输出的基线映射；
  P2VM拟合用的内部类改叫P2VMSpectra。FITS扩展和列名不变。
- 删除wavelength_grid封装，直接构造np.linspace；保留拟合对中央波段采样的物理要求。
- 删除FrameReducer.calibration_dir，目录由GUI持有。审阅中“目录未使用”的判断不准确：
  当前GUI在初始化和切换时确实读取它，因此这两个调用点与main入口一起改动。

## 其他名称

P2VM内使用fit_abcd_ellipse、extract_p2vm_spectra、estimate_group_delay、
_estimate_transmission、build_v2pm、_normalize_response_inplace和_apply_phase_reference。
_invert_v2pm已准确描述逐波长伪逆，保留该名称。

提取层使用extract_sparse_spectra、extract_spectra_from_file和_readonly_copy；
波长匹配区分match_lines_ransac与match_peaks_grid。I/O和recipe helper也明确说明
返回对象或操作：_detector_hdu、_detector_table、build_raw_header、_verify_checksums、
load_recipe_inputs、populate_dfs_product。绘图入口为save_reduction_plots。

发布context manager叫publish_outputs，继续逐文件无覆盖发布及失败回滚；
不将它描述为多文件整体原子事务。QC名称、指标、数值和可选发布行为保持。

## 流程图

![当前架构](architecture.png)

![标定与科学处理](dataflow.png)

可直接运行的调用顺序见[README](../README.md)和[examples/reduce.py](../examples/reduce.py)。

## 验证

| 检查 | 结果 |
| --- | --- |
| 保留的开发测试 | 12项通过，没有新增测试模块或增加测试数量 |
| 完整真实数据 | 15份既有raw，新旧各跑冷/热两轮全部五个recipe |
| 科学数组 | 两轮共74个数组的dtype、shape、字节一致；比较时只统一matrix字段的名称 |
| 产品语义 | 20份产品的物理映射、detector setup、DFS参数、实际输入和QC值一致 |
| 相位参照与预览 | 六种情形共138个P2VM/P2VMRed数组逐字节一致 |
| 孔径参数 | 非默认3.5 pixel全宽通过原--profile-width调用，新旧FLAT数组一致 |
| GUI | Python 3.12真实Qt窗口初始化、显示路径、切换标定目录和真实单帧处理通过；可读format 4 |
| 可迁移补丁 | 四份Qt6补丁均可应用到原始文件，结果与本机文件及manifest哈希一致 |

这些是对命名与接口整理的数值对照；不改变原有科学方法或替代台架准确度实验。

修改前源码、GUI消费者备份及一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_110344_step_names/`

日志、数据输出、GUI快照及比较结果：

`reduced/step_names/20261028_110344/`

[上一版产品简化记录](https://github.com/zexwu/planetes_pipe/blob/ecce9613842a20ad1bc6ca06a500c613f9375ec1/docs/CHANGE_REPORT.md)
保留0.7的删除机制和文件兼容说明。
