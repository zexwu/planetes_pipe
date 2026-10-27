# PLANETES 0.4.1：职责、命名与依赖整理

本轮基于 0.4 的 format-3 实现整理整个 package。用户指出的主要问题是：
从文件名看不出入口、科学计算和文件格式之间的关系，且内部存在循环依赖。
本轮拆开这些职责，保留原科学公式、默认值、FITS 字段和产品指纹。

## 现在从哪里读代码

```text
planetes/
├── calibration/           数组标定：dark、flat、wavelength、p2vm
├── extraction.py          数组扣暗、提取、波数重采样
├── reduction.py           数组观测量、GD、实时 FrameReducer
├── tasks.py               文件任务：读取 → 调用计算 → 返回科学对象
├── products.py            具名科学产品、单位、轴、有效性和父关系
├── instrument.py          物理通道约定、DetectorSetup 和配置匹配
├── identity.py            内容指纹与私有的规范数量表示
├── qc.py                  物理量的质量统计
├── plotting.py            科学结果绘图
├── io/
│   ├── raw.py             原始 FITS、采集元数据及匹配检查
│   ├── fits.py            科学产品 FITS 编解码
│   ├── keywords.py        类别、扩展名、QC keyword 映射
│   ├── hdu.py             公用 FITS 数组、JSON、checksum 操作
│   └── publication.py     校验、不覆盖发布和失败回滚
└── recipes/
    ├── planetes_*.py      五个明确命名的 PyEsoRex 接入口
    ├── parameters.py      Python 参数 ↔ CPL 名称和 CLI 别名
    └── dfs.py             CPL FrameSet、输入选择和 DFS 产品登记
```

从完整处理流程开始看 tasks.py；看算法时进入 calibration/、extraction.py 或
reduction.py；查 FITS 字段时进入 io/。CPL import 全部位于 recipes/。
原 eso/、models.py、根目录 fits.py 以及根目录四个标定模块被实际迁移，
没有保留另一套同名实现或兼容转发文件。

## 实质修改

- 原 `fits → qc → p2vm → reduction → extraction → fits` 循环已消除；
  包括函数内部 import 在内，24 个非初始化模块构成无环依赖。
- 数组计算不读取文件。文件提取、五类文件任务、GUI 初始加载集中在 tasks.py；
  FrameReducer 只处理传入的标定和 raw 数组。
- raw 与科学产品的 FITS 读写分开。DFS stub 填充迁至 recipes/dfs.py，
  文件安全发布接受显式校验函数，独立于产品类型和 CPL。
- QC 计算返回 `dark_median_adu` 等物理量名称，文件层才映射成现有
  `ESO QC MEDIANDARK SC`。数值公式、单位、统计对象和外部名称保持一致。
- `compute_wave` → `compute_wavelength`，`twopx_interp` → `resample_spectra`，
  `compute_reduce` → `reduce_spectra`；Python dark 任务参数改为
  `bad_pixel_threshold`。CPL 名称、CLI 别名及默认值不改，映射只声明一次。
- DetectorSetup/配置匹配归到 instrument.py，RawMetadata 归到 io/raw.py，
  规范数量表示/指纹实现归到 identity.py；保留必要的私有 `_encoding`，
  避免单位往返导致指纹漂移。format-3 BAD 的私有 provenance 参数键保留，
  从而保持原产品身份。
- 当前 Qt6 GUI、自动采集共用工具和精确部署补丁同步更新导入及加载接口。

## 流程图

![职责与调用关系](architecture.png)

![科学数据流程](dataflow.png)

## 验证

| 检查 | 结果 |
| --- | --- |
| 维护测试 | 54 项通过，含真实 PyCPL/PyEsoRex 执行 |
| 科学层独立性 | 禁止导入 Astropy、CPL、文件任务层后，科学模块与 dark/QC 计算仍正常 |
| 完整真实数据 | 复用已逐像素验证的15个标准化 raw；五类 recipes 冷/热各跑一遍 |
| 对0.4科学数组 | 每轮37项，合计74项逐字节一致 |
| 产品身份及 QC | 两轮共20产品，科学指纹与所有 ESO QC 数值完全一致 |
| GUI与自动采集 | 真实 Qt6 窗口 offscreen、完整13个 fake 自动采集再次通过 |
| 安装包 | 干净构建、项目外安装、5个recipe发现和默认参考的真实wave执行通过 |

本轮没有新增文件单位转换；format 3 和科学产品可直接继续使用。
0.3/schema-2 产品仍需按0.4迁移规则重新生成。真实台架硬件验收仍单独进行。

本轮源码恢复点与一次性脚本/日志归档于：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261027_233421_package_boundaries/`

真实回归输出与数值比较保存在：

`reduced/package_boundaries/20261027_233421/`

[README](../README.md) 的 Python 导入和 recipe-dir 已同步更新；
[产品合同](PRODUCT_CONTRACT.md) 与 [QC 定义](QC_PARAMETERS.md) 保留对应源码依据。
