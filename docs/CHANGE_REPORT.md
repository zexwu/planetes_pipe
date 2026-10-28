# PLANETES 0.7.0：产品层与科学核心简化

基于 `675d715`，移植用户提供的 v0.7.0 简化方案，同时保留当前 main 已验证的
P2VM 物理拆分、台架 raw 支持和12项开发测试。

## 修改

| 位置 | 当前职责与修改 |
| --- | --- |
| products.py | 401 → 119行；Dark / Flat / Wave / P2VM / P2VMRed 是普通数据类；DarkImage 独立 |
| 科学步骤 | 删除通用校验调用、指纹、父关系和Provenance；P2VM已有物理helper保留，WAVE局部名称更明确 |
| instrument.py / 数值入口 | 私有的固定物理映射检查；关键维度检查；伪逆前用NumPy数值秩拒绝缺少独立通道的V2PM |
| io/raw.py | 移除公共raw validator；已知采集检查放入raw header组装/解析，保留ADU、cube选择、快门映射修复 |
| io/fits.py | format 5写出、format 4兼容读取；保留units/checksum/身份/不覆盖写入；移除私有产品hash与来源图 |
| recipes/ | 只选择输入、调用科学步骤、请求可选QC并登记/签名DFS；来源与参数由CPL记录 |
| Qt6 acquisition | 本机scripts/utils.py及portable patch同步使用raw_header；更新manifest哈希 |

产品类没有旧名称别名。`P2VMRed`表示未平均的逐帧中间信号；实时预览的
GD和差分相位为None，正式保存需要完整诊断。输出仍是仪器级ADU、可见度、
GD与差分相位，未新增平均VIS/OIFITS阶段。

QC保留当前发布默认True，可通过include_qc=False跳过；测量与FITS编码分开。
启用时的QC定义和数值保持，缺失原因保存在FITS metadata中。

## 流程图

![架构](architecture.png)

![标定和科学数据流](dataflow.png)

更紧凑的流程与迁移说明见[简化说明](SIMPLIFICATION.md)。

## 验证

| 检查 | 结果 |
| --- | --- |
| 保留的开发测试 | 12项通过，包括FITS读写、物理映射/维度和真实PyEsoRex调用 |
| 完整真实数据 | 15份既有标准化raw，新旧版本各跑冷/热两轮DARK/BAD→FLAT→WAVE→P2VM→science |
| 科学数组 | 每轮37个，共74个dtype、shape和字节一致 |
| 文件语义 | 20份产品的物理映射、detector setup、DFS参数、实际使用文件和QC值一致 |
| 相位参照与预览 | 启用/禁用/缺失 × 15/17通道；含有/无诊断的P2VMRed，138个数组逐字节一致 |
| 旧文件迁移 | 10份format-4产品读取并转存format 5，全部数组、几何和setup一致；不再写旧hash |
| 物理边界 | 非ADU、多cube歧义、无DPR TYPE的FLAT映射、秩亏V2PM均按预期处理 |
| Qt6消费者 | 实际GUI Python 3.12读取format 4并处理真实单帧；fake camera元数据及raw发布成功；portable patch与manifest一致，未连接硬件 |

对照使用同一数值环境和分别匹配的冷/热缓存。源码整理保持既有数值结果；
实际台架测量准确度和最终ESO数据字典仍由后续实验/接口确认。

修改前源码、Qt6 helper备份与一次性脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_104015_minimal_products/`

日志、真实数据输出及比较结果：

`reduced/minimal_products/20261028_104015/`

[上一版接口整理与测试精简记录](https://github.com/zexwu/planetes_pipe/blob/675d7150bd9eb0f31861fedb2486bf8441eeda95/docs/CHANGE_REPORT.md)
保留历史说明。
