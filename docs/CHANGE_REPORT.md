# PLANETES 0.8.1：统一 Dark，简化 Bad

基于 `8e62fad`，删除 `DarkImage` 和 `DarkInput`，统一使用 `Dark`；
`BadPixelMap` 改为 `Bad`，不保留兼容别名。`products.py` 现为109行。

## Dark 的内容与用途

```python
@dataclass(slots=True)
class Dark:
    mean_adu: Array
    temporal_rms_adu: Array | None = None
    n_frames: int | None = None
    setup: DetectorSetup | None = None
```

- 多帧暗场仍测量均值、temporal population RMS 和帧数，计算方法不变。
- 单帧记录 `n_frames=1`、`temporal_rms_adu=None`；独立二维图像的帧数未知。
- 减暗只需要均值，Python 标定与科学处理入口不再要求完整的暗场统计。
- BAD 检测需要已测 RMS；QC 报告可用均值，并说明缺失 RMS，不能填零。
- 完整 DARK FITS 写出仍要求已测 RMS 和至少两帧，不新增文件类别。

`Bad` 继续保存 `bad_pixel_flags`，FITS 类别仍为 `BAD`，
hot=1、dead=2、noisy=4。算法、阈值、单位、望远镜/基线映射与文件布局保持。
产品写 format 5，仍可读 format 4。已知曝光、ROI 和 detector 矛盾检查保留。

## 调用点与 GUI

更新 dark、flat、提取、P2VM、reduction、FITS、QC 和 recipe 中的类型引用。
Qt6 viewer 的当前帧暗场也使用 `Dark`；状态根据 RMS 是否存在显示
“RMS not measured”。本机 viewer 和可迁移补丁已同步，manifest 更新为0.8.1。

## 本次验证

| 检查 | 结果 |
| --- | --- |
| 保留的开发测试 | 12项通过，未增加测试数量或测试模块 |
| 完整真实数据 | 15份既有 raw，旧版和新版各运行一轮全部五个 recipe |
| 科学数组 | 37个数组的 dtype、shape、字节完全一致 |
| 产品语义 | 10份产品的物理映射、detector setup、DFS 参数、实际输入和 QC 值一致 |
| GUI | 实际 Python 3.12 Qt 窗口中，完整暗场与相同均值的二维暗场产生完全相同的 visibility 数组；单帧暗场按钮和后续处理通过 |
| 文件兼容 | 旧 BAD 产品读取为 Bad；完整 DARK 文件合同保持 |
| 可迁移补丁 | 四份补丁均可应用到原始文件，并复现本机消费者文件 |

这次对照确认正常标定链的数值结果保持，并验证缺少 RMS 的暗场用于减暗。

修改前源码和 GUI 备份：

`/Users/zexwu/zexwu_lib/planetes_pipe_backups/20261028_113453_dark_bad/`

测试日志、两版输出、比较结果和 GUI 快照：

`reduced/dark_bad/20261028_113453/`

[流程图：架构](architecture.png) · [流程图：数据流](dataflow.png)

[0.8.0 修改报告](https://github.com/zexwu/planetes_pipe/blob/8e62fad201c469d68594cfa4488627dc5a86f166/docs/CHANGE_REPORT.md)
保留上一次步骤命名与接口整理的记录。
