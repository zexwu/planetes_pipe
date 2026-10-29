# PLANETES 0.13.0：明确选择输入，删除自动识别与重复回读

基于 `d0ecbe0`。调用处明确输入含义；读取函数负责读取，科学步骤负责计算。
文件入口检查关键文件语义，数组入口保留必要的物理轴检查；内部流程不逐层兜底。

## 本轮修改

- 删除 `dark.load_dark`。读取 DARK 产品用 `read_product(path, Dark)`；
  原始暗场计算用 `make_dark_products(path)`；临时内存暗场直接构造 `Dark(frame)`。
  不再自动寻找第一个有数据的 HDU、猜输入类型或在读取函数里决定是否计算。
- `make_wave(..., reference=lines)` 必须明确传入线表；删除
  `load_wavelength_reference`。默认线表由 recipe 或脚本选择。WAVE recipe
  仍将所用 LINE_LIST 写入 DFS 实际输入。
- `fit_p2vm` 与 `make_p2vm` 的 `phase_reference` 只接收已减暗、已提取、
  与扫描使用相同网格的参考光谱，或 None。删除科学接口的 `phase_correction`、
  `phase_dark`，不再自动复用主暗场。`P2VMSpectra` 只含 telescope/scan
  光谱、物理输出身份和波长，不再用零帧数组暗示缺少参考。
- P2VM recipe 在调用处选择参考文件与 DARK，并明确提取。SOF 和
  `--phase-correction` 的名字、默认值保持；未选 PHASE_DARK_CALIB 时，recipe
  明确采用其主 DARK。关闭或没有 reference 时，不加载可选 reference/DARK，
  并从 DFS 实际输入排除；单独出现的未使用 phase DARK 不再触发错误。
- DFS 在 `save_products` 内直接填写自己刚创建的临时文件；删除
  `populate_dfs_product` 及其空 stub 猜测检查，合并为一次打开。
- 写出只做编码、checksum 生成、临时发布与禁止覆盖；删除通用 `verify`
  回调和 `verify_raw`，不再自动读回并完整解码刚写出的文件。
  FITS round-trip 由现有测试检查。外部产品读取仍核对单位、checksum 和身份，
  recipe 仍执行 CPL DFS 签名，多产品发布失败时仍回收本次已建立的链接。

## Qt6 项目分离

删除 `integration/qt6/` 中的五份补丁/manifest，以及专用的
`docs/GUI_INTEGRATION.md`；从 README 和架构图移除对应维护与部署说明。
删除 GUI 专用的 `load_reducer(directory)` 包装。

数组科学接口仍直接使用：

```python
flat = read_product(flat_path, Flat)
wave = read_product(wave_path, Wave)
p2vm = read_product(p2vm_path, P2VM)
reducer = FrameReducer(flat, wave, p2vm)
```

没有修改外部 GUI 项目。没有增加产品类、通用 validator、新依赖或测试文件。

## 保留的科学约定

计算仍为 float64/complex128，I/O 仍为 float32/complex64；raw format 3、
product format 6、物理单位、波长覆盖舍入容差、mask、阈值和相位操作顺序保持。
Numba 索引范围、关键数组轴、标定 ROI/曝光、物理输出映射和 V2PM 秩检查保留。
raw HDU 搜索及 header/JSON 双份 metadata 的调和规则留待下一步处理。

## 验证

- 现有 **12 项测试通过**，包括 PyEsoRex recipe；测试数量未增加。
- 同一批 15 份既有 raw，旧版和新版分别在独立目录运行完整五种 recipe：
  **10 份产品、31 个数组逐字节相同**；QC、setup、物理映射和实际输入一致。
- 使用同一批提取光谱，六种相位参考/诊断配置及后续 reduction 的
  **63 个存盘前数组逐字节相同**。
- CPL 参数名字、默认值和实际值一致；phase-correction 现在由 recipe
  单独声明并列在参数表末尾，所以 FITS 的 PARAMn 编号顺序有所变化。
  参数对照按名字比较，而非将编号当作参数身份。
- 五种 recipe 选择情况通过：独立参考暗场、明确复用主暗场、关闭参考、
  缺少参考、只有未使用的参考暗场。未使用的可选 FITS 即使内容无效也不会
  被 pipeline 加载或记入 DFS 实际输入。
- 校验了这些选择输出和 15 份输入 fixture 的所有 HDU checksum/DATASUM。
  六个模块内 **32 个科学计算函数的 AST 保持相同**。
- Ruff 与 `git diff --check` 通过；架构图、数据流图和 P2VM 图同步更新。

## 备份与对照结果

修改前源码、原 Qt6 补丁及一次性对照脚本：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261029_101722_explicit_boundaries/`

隔离输出与对照记录：

`reduced/explicit_boundaries/20261029_101722/`

`comparison.json` 保存数组及 metadata 对照，`selection_v2/summary.json`
保存选择分支检查；`tests.log` 保存现有测试结果。

[架构图](architecture.png) · [数据流](dataflow.png) · [P2VM](p2vm_flow.png)

[0.12.0 修改报告](https://github.com/zexwu/planetes_pipe/blob/d0ecbe07d9b46a18c925d8ebc28254bc63af59e5/docs/CHANGE_REPORT.md)
