# PLANETES 0.11.0：按物理流程整理 P2VM

基于 `08904a2`。这轮集中修改 `p2vm.py` 的可读性：保持科学算法、阈值、
相位操作顺序和计算精度，将入口检查、扫描诊断、文件读取与物理映射分开。

## 现在怎么读代码

`fit_p2vm`直接展示五个阶段：

1. 扫描响应拟合：每条baseline的ABCD ellipse → OPD/GD与零点 → 全24输出响应。
2. Telescope flats估计transmission。
3. 归一化phase、transmission和coherence amplitude。
4. 构造V2PM，检查rank，逐波长用原`pinv`求逆。
5. 可选仪器phase reference：校正响应相位，再构造并求逆。

扫描诊断只在扫描函数内收集，通过普通字典写入已有产品字段。
数组入口检查集中在`_check_input_axes`；两个文件入口置于模块后部。
`_identify_output_mapping`单独表达flat照明和baseline调制必须给出同一组outputs。
没有新增文件框架、产品类或通用validator。

`_apply_phase_reference`明确分为重建reference、恢复逐帧telescope phase、
去时间变化并求和、去残余GD、更新响应相位。用T4与43/42/41的物理配对
取代`telescope-1`索引。各数组轴、单位与相位符号在代码及科学说明中明确。

## 接口与科学约定

- `fit_abcd_ellipse(spec)`删除全1的`envs`参数。它原来是方程右端，
  并不是residual weights。内部使用一维全1右端，继续`pinv`，不改成`lstsq`。
- 保留原两次transmission归一化及运算顺序，守住浮点基线。
- A输出相位归零仍作用于该quartet的全部六个响应分量，包括cross-talk。
  phase reference则仍只修正各output所属的物理baseline。
- 非正/非有限photometric normalization与V2PM rank检查保留。
  未新增condition-number拒绝阈值。
- FITS format仍为6；计算64位、浮点存盘32位的策略保持。
  版本升至0.11.0以标明ellipse公开函数签名变化。

## 数值验证

同一数值环境、同一批15份既有raw，在独立目录运行旧版和新版完整五种recipe。
两版均显式选择main/lamp/science DARK；reference使用独立science DARK。

- 10份产品的31个科学数组逐字节相同，包括BAD、提取索引、P2VM与science flags。
  mapping、setup、QC、recipe参数及输入选择相同（运行根目录归一后比较）。
- 另用完全相同的提取光谱比较**存盘前**结果：reference启用、提供但禁用、
  未提供而启用，分别开/关诊断，共六种配置。连同V2PM、science reduction和
  退化ellipse样例，共63个数组逐字节相同；浮点科学数组为float64/complex128。
- 新recipe开启诊断后，五个共有P2VM数组与默认产品逐字节相同。
- 现有12项测试通过，数量未增加；Ruff与diff检查通过。

## 椭圆兜底的实测情况

临时观测基线版本的`pinv`调用，返回原inverse不作修改。六条baseline × 151
channels，共906次拟合：`std_x/std_y <= 1e-9`均为0次，两个非正二次项触发的
`1e-6`兜底也均为0次。此统计仅描述这批数据。

全部原阈值与兜底保留。全零扫描对照仍会经兜底返回零phase/radius，不能因此
视为有效标定。其他数据下如何判定、拒绝失败channel，仍需另行确定科学标准。

## 备份与结果

修改前完整源码、一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_172501_p2vm_readability/`

本轮产品、存盘前数组、`comparison.json`、`ellipse_fallbacks.json`：

`reduced/p2vm_readability/20261028_172501/`

[P2VM流程图](p2vm_flow.png) · [科学约定](SCIENTIFIC_METHODS.md) ·
[总体架构](architecture.png) · [完整数据流](dataflow.png)。

[0.10.0精度修改报告](https://github.com/zexwu/planetes_pipe/blob/08904a20ce7a3d4942b4af8c418765172aeff14e/docs/CHANGE_REPORT.md)
保留计算精度、标定存盘量化和最终写出舍入的原始证据，不与本轮重构混淆。
