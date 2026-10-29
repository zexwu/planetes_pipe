# PLANETES 代码风格

目标：从名字和相邻几行代码读懂物理过程。函数内部沿用简洁的科学叫法；
跨模块的接口、产品字段和 FITS 定义提供完整含义。

## 先让流程清楚

- 调用处明确输入含义；读取函数负责读取，科学步骤负责计算。默认文件与
  所用标定由脚本/recipe选择，内部函数不自动识别、加载或兜底。
- 缺少可选输入用None表示，不用空数组暗示，也不同时用开关决定是否使用它。
- 文件入口检查文件语义；数组入口检查必要的物理轴、像素范围和维度。
  内部已经确定的数据不逐层重新验证。数值mask、秩检查和拟合约束按科学定义保留。

- 每个入口接受一种明确的数据形式。FLAT/P2VM 文件入口使用具名路径映射；
  SOF 的分组和快门身份推断在 recipe 入口完成。
- 一个名字在流程中保持同一含义。路径映射始终是路径映射，不能中途变成
  `{identity: (path, metadata)}` 再要求后面的代码反复拆包。
- 主函数直接展示本步骤的正常顺序。共用函数应承担具体工作；仅改变参数形式、
  或把一项包装成列表的函数可以合并。
- 物理字段只在FITS header中保存一份。禁止JSON metadata、采集来源/legacy
  状态和额外的历史记录框架；缺失字段保持未知，不增加补缺兼容层。
- 允许少量直白的重复。只在确实重复同一项工作时提取共用函数，避免为了少写
  几行引入模式开关、通用配置或临时对象。
- 结构审查先看输入模式、间接调用和数据形状变化。数值正确性问题单独处理，
  结构整理保持现有运算、阈值和物理身份检查。

- 文件入口明确给出raw类别；标准HDU作为默认值，非标准数据显式指定hdu。
  不自动搜索图像或从缺失信息推测身份。
- QC在脚本/recipe中明确计算，writer只编码传入的quality。每个输出独立完成写出，
  不为成组文件增加事务、回滚或状态管理。

## 命名

- 名字表达当前作用，不重复函数名、产品名和已知数组轴。例如响应拟合内部用
  `amplitude`、`phase_rad`、`dc_adu`，不再每次重复 `coherence_...`。
- 波长、OPD、GD 保留 `_m`；相位保留 `_rad` / `_deg`。局部光度与复相干通量
  分别用 `flux_adu`、`coherent_adu`；visibility amplitude 用已有领域叫法 `visamp`。
- 多组同类数据才加限定词：`scan_phase_rad` 与响应 `phase_rad`，
  `detector_flat` 与 `grid_flat`，`line_x` 与 `fit_x`。不用含义模糊的 `data1` / `tmp2`。
- `x`、`y`、`a`、`b`、`c` 等保留给局部坐标和数学系数；物理身份使用
  `telescope`、`baseline`、`output`。`xs` / `ys` 表示像素索引，`nx` / `ny` 表示尺寸。
  `spec`、`coeffs`、`rms`、`opd`、`gd` 是允许的常见科学缩写。
- 名字跟随实际物理量：复数相位信号用 `phasor`，取角度后才叫 `phase_deg`。
- 字段到局部名的映射放在入口和返回处；产品字段、公开参数、FITS keyword、
  SOF tag 和 QC 名称按其科学契约维护，避免随着局部排版一起改名。

例如，reduction 内部计算使用：

```python
channels = np.einsum("box,ofx->fbx", p2vm.matrix, spectra)
flux_adu = channels[:, :N_TELESCOPES, :]
coherent_adu = channels[:, REAL_SLICE, :] + 1j * channels[:, IMAG_SLICE, :]
visamp, flags = normalize_coherent_flux(coherent_adu, flux_adu)
```

保存结果时仍明确对应产品的 `photometric_flux_adu`、`coherent_flux_adu`、
`visibility_amplitude` 字段。数组轴与符号约定写在函数 docstring 和产品字段旁。

## 排版

- 4 空格缩进、88 列、双引号；普通 Python 交给 Ruff formatter。
- 能放下的短 import 和调用放一行；末尾逗号不强制拆行。长参数、复杂表达式
  按数学步骤换行。用空行分隔物理阶段。
- 常量、FITS 名称、field/extension/unit 映射和 recipe 元信息等声明表，
  在**同一小组内**对齐赋值或值列。用局部 `# fmt: off` / `# fmt: on` 保留排版。
- 数学计算、条件分支、函数调用按普通格式书写；不为整段算法补空格凑列。
- 注释说明轴、单位、符号或处理原因；避免复述变量赋值。

```python
# fmt: off
DETECTION_SIGMA   = 3.0
SURFACE_DEGREE    = 3
RANSAC_ITERATIONS = 1_000_000
# fmt: on
```

## 改动边界与检查

可读性修改保持运算顺序、阈值、mask、归一化和文件语义。
计算精度继续在 I/O 边界定义，算法内部使用收到的计算数组。
需要改变科学定义时，单独说明依据并确认。

Ruff 配置集中在 `pyproject.toml`：

```bash
ruff format planetes tests examples
ruff check planetes tests examples
```

现有科学测试与必要的数值对照用于检查计算行为；纯排版不新增测试。
