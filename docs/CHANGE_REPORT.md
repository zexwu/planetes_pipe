# PLANETES 0.3 修改汇报

2026-10-27。新版按科学任务整理整个 package，Python、PyEsoRex 和实际 Qt6
GUI 共用科学实现。与旧版处理完整台架数据时，在同一数值环境和对应的 Numba
缓存状态下，27 个科学字段全部逐位一致。

## 主要修改

| 原来的组织 | 当前组织与作用 |
| --- | --- |
| processing.py 与 algorithms/ 同时组织任务 | dark、flat、wavelength、p2vm、reduction 各自包含文件入口和数组计算 |
| product.py 与 io/products.py 容易混淆 | models.py 定义科学对象、单位和轴；fits.py 负责读取与编码 |
| dark 附着在 flat | 独立暗场任务；完整 DarkCalibration 与仅相减的 DarkImage 分开 |
| 实时维护独立算法和网格 | FrameReducer 共用提取、归一化和 flags，采用 P2VM 保存的网格 |
| 同形状标定可能错配 | flat/wave/P2VM 保存父标定指纹，准备处理时验证依赖 |
| 部分参数隐藏在不同层 | 普通函数定义默认值与校验，PyCPL 映射实际参数 |
| 写出分步失败可能留下空产品 | 临时目录完成、签名和读回后原子发布，已有文件不覆盖 |

schema 2 明确保存科学单位、Python 数组轴、零基索引表、真实暗场帧数、实际
参数和来源；复数与布尔 flags 无损保存，各 HDU 有校验和。30 条静态谱线的
数值和 origin 标签保留。详见 [产品契约](PRODUCT_CONTRACT.md) 和
[科学方法](SCIENTIFIC_METHODS.md)。

## 流程图

![模块职责](architecture.png)

![数据流程](dataflow.png)

main、lamp 和 science dark 按用途独立选择。flat 的 BPM 由建立它的完整暗场
确定；science dark 的变化不重新定义该 mask。相位参考可选独立 dark，缺省
记录主 dark 复用。所有输出必须覆盖 P2VM 目标网格，不静默外推。

## 验证结果

迁移回归使用 15 个完整原始 FITS，总计 10,401,966,720 字节。旧版从冻结的
38341e0 源码运行，新版执行真实 PyEsoRex recipes，另跑完整 Python 链。

| 检查 | 结果 |
| --- | --- |
| 冷启动对应冷启动 | 27/27 个旧科学字段逐位一致 |
| 缓存复跑对应缓存复跑 | 27/27 个旧科学字段逐位一致 |
| Python 对应 ESO | 7 个产品、44 个字段/科学指纹比较一致 |
| 迁移后的安装包 | 原 48 项测试通过；五个 recipe 可发现，默认谱线 wave 实际执行通过 |
| Qt6 实际窗口 | 录制数据下扣暗、binning 不变性、暗场启停、二维 dark 和错误输入通过 |
| 实时核心 | 1024×1280 raw 帧预热后 128 次，均值 1.104 ms，中位数 1.100 ms |

实时计时不含采集/绘图，不计算 GD/差分相位。真实台架部署仍需实际运行验证。
旧版自身有冷编译/热缓存的微小差异；上述逐位结果要求匹配缓存状态，没有
放宽容差或修改 fastmath/cache。保留的科学算法、默认 mask、相位和归一化
定义见科学方法文档。

## 清理后的维护结构

测试整理为四个文件：科学计算、标定关系、FITS 读写、可选 ESO 集成。
删除了重复的坏像素/谱线检查、迁移期导入检查和绘图 mock 测试；几何、网格
与逆矩阵快照检查合并。保留 43 项针对科学正确性及文件安全的测试。
一次性的 `validation/`、`bench/`、旧报告、构建日志和计划审查记录移出活动
仓库，不再随源码包发布。科学实现未修改。

完整迁移记录仍可从 Git 提交 `ad0daf5` 获取。本机另有逐文件校验的清理前快照：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_212301_cleanup/source/`

原版源码和外部 GUI 备份保留在：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_132125_restructure/`

完整真实数据回归的输出保留在：

`reduced/restructure_validation/20261027_132125/`

## 使用和后续

[README](../README.md) 提供五个 recipe 和 Python 用法；[SOF 示例](../examples/)
指定输入角色；[GUI 集成](GUI_INTEGRATION.md) 保留实际 viewer 补丁与恢复说明。
旧 Python API、schema-1 FITS 和 NPZ 不自动兼容，应从 raw 生成 schema-2 标定。

当前输出是仪器级观测量。ESO raw 元数据、分类字典、EDPS 关联、科学误差、
传递函数、闭合相位/OIFITS 和 Phase 3 仍是后续工作。
