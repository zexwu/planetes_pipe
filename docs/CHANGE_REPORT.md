# PLANETES 0.4 修改汇报

2026-10-27，基于 61f519f，按确认的 GRAVITY 对齐方案实施。五个 recipe 和
Python/GUI 共用科学实现；本轮改变数据合同、命名和 QC，不改变现有科学算法。
原始文件与原环境保留，新产品使用 format 3，旧 schema-2 产品需要重新生成。

## 改了什么

| 部分 | 当前行为与原因 |
| --- | --- |
| DARK / BAD | 独立科学对象与两个产品；dark recipe 同时产生；保留 hot=1/dead=2/noisy=4 和原判据；先校验全部，再不覆盖发布 |
| SOF | 使用重复 FLAT_RAW/P2VM_RAW，通过原始头的物理快门识别望远镜/基线；输入顺序和文件名不决定身份 |
| 文件任务 | 完整 DARK 为正式任务必需；检查已知 DIT、ROI、物理 binning、探测器和 orientation；历史未知项明确记录 |
| 原始数据 | IMAGING_DATA_SC、真实分类、物理快门、曝光/ROI/来源；保留像素与 dtype；旧 raw 写成独立副本 |
| FITS 产品 | 参考 GRAVITY 1.9.6 的同义结构/字段和单位；文件用 m、一基像素坐标，内部仍用 um、零基坐标 |
| 科学内容 | 完整24输出响应、inverse、OPD/GD、ellipse、flags 保留；C_MATRIX 保存拟合 DC 系数，inverse 独立命名 |
| QC | 按源码公式补齐适用 dark/BAD/flat/wave/P2VM 指标；没有测量依据的指标写缺失原因，自定义诊断退出 ESO QC 命名空间 |
| 参数与来源 | 显式映射 planetes.calib.* / planetes.reduce.*；保留四个 flat、六个 baseline 和实际参考的完整来源 |
| 内容关联 | 保留少量私有指纹；载入后保留原文件单位表示，避免重复转换漂移；修改数组/参数会使旧关联失效 |
| 自动采集 + Qt6 | 同一 raw writer；设置在采集开始记录；配置来源明确；不再把所有图像强制转 int16；没有操作真实台架 |

[数据合同](PRODUCT_CONTRACT.md) 给出字段/shape/单位及参考符号对应关系；
[QC 文档](QC_PARAMETERS.md) 列出公式、统计域、样本数和缺失条件。
参考固定为本机 GRAVITY 1.9.6 源码，未声称这是 GRAVITY 最新版或已获 ESO 审核。

## 流程图

![模块职责](architecture.png)

![数据流程](dataflow.png)

main、lamp、science DARK 分别选择；FLAT mask 由其 DARK/BAD 父关系固定。
PyCPL 负责 recipe 参数、FrameSet 与 DFS，科学计算和 FITS 合同不依赖它。
当前输出是逐帧仪器级结果，保留必要 PLANETES 扩展。

## 实际验证

真实回归使用 15 个 raw FITS，共 10,401,966,720 字节；标准化副本全部逐像素、
shape/dtype 验证。原文件只读。旧源码冻结于 61f519f，使用同一 Python 3.13
数值环境并分别检查冷/热缓存。

| 检查 | 结果 |
| --- | --- |
| 维护测试 | 53 项通过，包括实际 PyCPL/PyEsoRex 执行、独立 QC 期望值、单位/指纹、元数据冲突、发布失败/并发保护 |
| 相同数组输入的旧/新核心 | 冷、热各23个科学数组逐字节一致；BAD bits 另行逐位一致，基线映射保持一致 |
| 完整新 recipes 对旧版 | 冷、热合计60个数组全部在事先固定的格式转换控制界限内 |
| 普通 Python 对 recipes | 10个产品、37个数组逐字节一致，物理映射一致 |
| 两个实际采集入口 | 自动脚本 fake 完整13个 raw；Qt6 实际窗口 offscreen 测试通过，像素/来源/配置变化、dark及显示 binning 科学不变性通过 |
| 干净包安装 | 项目外 wheel 安装、5个 recipe 发现、默认打包谱线的真实 wave 执行通过 |
| 实时科学核心 | 1024×1280 raw 帧预热后128次，中位数1.000 ms，均值1.039 ms；不含采集/绘图和GD/差分相位 |

格式边界的 um↔m 和一基坐标转换会产生少量浮点舍入。因此先运行“冻结旧核心
+ 仅格式转换”的控制链，在运行候选版前锁定每个数组的差异界限；没有事后放宽
容差。60项中45项与转换控制直接数值一致，其余也均未超过原先界限。
同数组核心比较使用逐字节判定，未以格式转换为理由放宽。旧核心自身已有
冷/热差异，fastmath/cache 保持原设置。

## 备份、证据与维护

本轮开始前的完整源码、四个外部采集/GUI文件及其哈希保存在：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_223235_gravity_conventions/`

完整标准化数据、新旧产品、冻结转换界限和比较结果在：

`reduced/gravity_conventions/20261027_223235/`

一次性运行脚本、日志及批准的计划/审查记录归档到上述备份目录的 evidence/；
不纳入源码包。维护测试仍集中在原来的四个 test 模块。外部代码的精确补丁与
before/after 哈希在 integration/qt6/，[GUI 文档](GUI_INTEGRATION.md) 说明部署与恢复。

## 下一步

在台架环境确认真实相机、物理 binning、orientation 和快门配置，并做实际采集
验收。当前本地 GUI 环境没有 catkit2；fake/offscreen 通过不能替代硬件验收。
之后再推进 ESO 字典确认、EDPS 关联、误差传播、传递函数和最终 OIFITS/Phase 3。
这些工作不通过伪造时间、有效带宽、gain、SNR 或误差字段来提前宣称完成。
