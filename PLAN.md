# Plan: PLANETES 科学处理与 ESO 接口整理
_Locked via grill and independent review; user approved implementation on 2026-10-27._

## Goal

将 PLANETES 整理成一套可以从 Python、PyEsoRex 和台架 GUI 调用的科学实现。按暗场、flat、波长、光谱提取、P2VM 和科学 reduction 组织代码，明确数组、标定对象、文件和框架之间的边界。允许重整 Python 接口和重新设计 FITS，不要求兼容 v0.2 文件；保留仪器级科学输出与既有科学定义，明确修复网格错配、实时标志不一致和失败写出问题。原始数据、已有产品及旧环境不覆盖。

审查基准：当前仓库提交 38341e0；已完成的真实回归使用 15 个原始 FITS、10,401,966,720 字节。旧版存在 Numba 冷编译与热缓存的微小数值差异，精确比较必须固定环境并匹配缓存状态。

## Approach

1. 冻结基准和恢复点。
   - 从当前版本建立 codex/ 下的新工作分支，保留既有原版备份。
   - 对即将修改的外部 GUI 和其必要接入文件逐文件备份并记录 SHA256。/Users/zexwu/zexwu_lib/devel 不是 Git 仓库，不假定它能依靠 Git 恢复。
   - 在同一个已验证的科学环境中建立基准；不混用旧 NPZ 字段、旧缓存结果和不同依赖版本。
   - 将回归执行和逐字段比较脚本纳入仓库，参数只需显式的数据根目录和全新的输出目录，不重新引入运行 YAML。

2. 按科学职责整理 package。
   - 采用以下职责划分，模块名在实现前固定到文档和测试；避免通用 processing 层与 algorithms 层并行承担同一任务。

         planetes/
             models.py       科学对象、单位、轴和科学有效性约束
             instrument.py   4T、6B、24 输出及 A,C,B,D 约定
             fits.py         原始 cube 读取与产品 FITS 编解码
             dark.py         暗场统计和既有坏像素判定
             flat.py         flat 图、trace 和稀疏提取几何
             wavelength.py   波长解和谱线匹配
             extraction.py   扣暗、平场响应、提取、重采样的共享实现
             p2vm.py         物理标定项、矩阵构造、求逆与相位校正
             reduction.py    仪器观测量及共享单帧计算
             qc.py           描述性 QC
             plotting.py     显式调用的诊断图
             eso/            五个 PyCPL recipe 和公共适配工具

   - 文件级入口放在对应任务模块内，以清楚的 *_file 或任务名称区分文件任务与数组计算。它们负责读取、明确的任务顺序和返回内存对象，不默认保存结果。
   - 每项任务的数组计算只有一个实现；PyCPL adapter 不再拥有暗场相减、波长网格或科学拟合逻辑。
   - 与领域相关的 GD 算法放回 reduction；P2VM 复用这一公共计算函数。通用分块统计与原始读取分别由 dark 和 fits 按实际职责承接。
   - 去掉 import 时的绘图全局设置；日志由明确的调用入口配置。数值库导入不依赖 PyCPL 或 Qt。
   - 当前仓库 bench 只保留使用公共接口的示例和计时，不保留第二份科学算法。build/lib 是生成物，不作为第二个源码树维护。

3. 定义科学对象和新的 FITS 契约。
   - 使用简单 dataclass 表达 DarkImage、DarkCalibration、FlatCalibration、WavelengthCalibration、P2VMCalibration、VisibilityObservables、ReducedObservables 和 WavelengthReference。CalibrationSpectra、基础实时观测量与预备提取状态只用于内存，不增加文件缓存阶段。
   - 每个对象明确自身的数组轴、单位、标志意义和仪器顺序，避免一个跨产品的字段名表决定不同对象的轴。科学约束由独立于 FITS/CPL 的模型边界维护，FITS 模块只补充文件表达的校验。
   - 完整 DarkCalibration 保存 mean_adu、temporal_rms_adu、既有 bit flags 和实际帧数，由至少两帧的原始三维暗场计算。temporal RMS 不称为已测定的纯读出噪声；不引入增益、泊松误差或新的噪声模型。
   - DarkImage 只保存一张用于相减的二维 mean_adu 和来源，明确没有测得 temporal RMS 或坏像素标定；不填零、不借用其他文件的统计量，也不将未知叠加帧数写成 1。二维输入以及只有一帧的 cube 可由公共扣暗输入加载器读成 DarkImage，但不能用于建立 flat 的坏像素 mask。
   - flat 保存 flat_map_adu、trace_y_pixels、profile_x_pixels 和 profile_y_pixels。提取 mask 由建立 flat 时选用的暗场坏像素图确定。
   - wave 保存 detector-output/pixel 的 wavelength_map_um 与源谱线记录。
   - 定义可执行的依赖契约：wave 保存建立它时的 flat 指纹；P2VM 保存建立它时的 flat 和 wave 指纹。准备 P2VM、science 或实时状态时，验证 wave→flat、P2VM→flat/wave 链，拒绝形状相同但依赖不同的组合。science 自己选用的 dark 不要求与 flat 的 dark 相同。
   - 指纹按明确的科学类型、单位、轴、仪器顺序以及实际消费的科学数组计算 SHA256，包括 flat 响应和带哨兵的提取几何、wave 坐标及适用参数；数组使用固定字节序、形状和 dtype 的规范表示。排除文件路径、时间戳、CPL 签名等易变文件头。读回和任务准备时重算校验，不依赖可被数组原地修改绕过的缓存 ID。预备提取状态持有所需数组的只读快照，在载入/切换时计算指纹，不在逐帧循环中反复哈希。
   - P2VM 保留逆矩阵、明确的目标波长坐标、基线/望远镜/输出映射、全部 24 输出的 transmission 和 coherence amplitude/phase，以及现有 OPD、GD、椭圆拟合诊断。保留交叉项；Cartesian V2PM 只用于计算和验证。
   - VisibilityObservables 提供 photometric/coherent flux、visibility amplitude、flags、波长和基线映射。ReducedObservables 扩展这些基础观测量，增加实际计算的 GD 和差分相位。两者均保留 frame 轴，单帧时长度为 1。实时端可以只返回前者；规范的 reduced FITS 仅接受完整 ReducedObservables，不以零值或缺失字段伪装已经算过的 GD/相位。
   - WavelengthReference 明确纳入 schema 2：保留静态谱线的 wavelength_um 和逐行 origin 标签，使用带单位的表扩展及 PLANETES_LINE_LIST 类别。从原 ECSV 重新生成同名包内 FITS，保留 ECSV 原文件、来源文件名和 SHA256；30 个值与原 wavelength_nm / 1000 逐位一致。默认参考和显式参考共用该读取契约。
   - FITS 使用新的 schema 版本 2 和明确类别，空 primary 加命名数据扩展。保存实际单位、轴、顺序、索引约定、版本、输入来源和实际参数；每个 HDU 有 CHECKSUM/DATASUM。复数分实部/虚部，布尔标志无损保存。
   - Python 与 ESO 共享一份 payload 编解码。运行来源信息属于该次产品的轻量元数据，不成为全局配置或调度对象。记录实际生产入口；Python 产物不伪称由 PyEsoRex 执行。缺少的观测元数据不猜测、不伪造。
   - v0.2 FITS 和旧 NPZ 不自动转换、不覆盖；需要时从原始数据生成新的标定。FITS schema、各扩展及 metadata 的具体字段形成单份可检查的契约文档。
   - 产品先校验，写到最终目录所在文件系统的临时目录，并保持最终 basename。完成适用的签名、全部 HDU 校验和及读回验证后，使用原子且不覆盖的发布操作，例如 os.link(temp, final)，再清理临时目录。目标已存在（包括符号链接）时必须失败；不支持该操作的文件系统清楚报错，不退回会覆盖目标的 rename/replace。失败不能留下看似完成的空 DFS 产品。

4. 统一科学处理和参数。
   - 新增独立 dark 文件级任务和 planetes_dark recipe，接收三维、至少两帧的原始 dark cube，生成完整 DarkCalibration。flat 必须显式传入完整 DarkCalibration；wave、p2vm、reduce 的相减输入可为 DarkCalibration 或 DarkImage，不从 flat 产品取得暗场。
   - 公共 load_dark 接口用于 Python/GUI 的扣暗输入：schema-2 暗场产品按类别读取；原始多帧 cube 调用同一个完整 dark 任务；二维图或单帧 cube 仅生成 DarkImage。对已声明为其他产品/科学角色的 FITS 必须拒绝，不能因其为二维而当作暗场。原始元数据缺失时由调用者明确指定 dark 角色并记录未知项，不伪造测量值。
   - P2VM 的 main_dark 显式用于四个 telescope flat 和六个 baseline cube。phase_reference 可选，phase_dark 单独可选；有 reference 而未提供 phase_dark 时，按公开的默认规则复用 main_dark，并记录实际暗场指纹及复用事实。提供 phase_dark 却没有 reference 时直接报错。相位校正关闭时标记未执行，不能把未用于计算的相位参考登记成已使用的标定。
   - 科学 target 的暗场与建立 flat 所用暗场可以不同；后者的坏像素 mask 不因 target dark 的不同而自动改变。这保留现有科学 mask。
   - 开放提取宽度后，仍使用原来的严格空间选点不等式及原有像素累加顺序。不同输出的稀疏列表以成对 (-1,-1) 补齐，不能裁短长行、改变 mask 或通过重新排序获得矩形数组；校验有效索引范围、哨兵配对及每个输出至少存在有效样本。
   - 五项标准文件任务读取全部帧，保留 16 帧分块；不会新增截取、binning 或采样规则。
   - 标定 P2VM 时明确指定默认 1.15–1.30 um、151 通道的目标网格。science 和实时端直接采用 P2VM 保存的坐标。
   - 在提取前检查坐标有限、正、严格递增，flat/wave/P2VM 的几何及科学角色一致，且每个输出覆盖整个目标网格。覆盖不足立即报出输出编号和缺失波段，不延伸边界值或静默缩短网格。
   - 必要的一致性检查位于加载和任务边界，不在每个数值循环中反复增加 guard。参数或标定变化时重新准备缓存，不能继续使用旧网格的预计算状态。
   - 本轮开放参数见下表。Python 与 PyEsoRex 使用相同名称含义、默认值、校验和实际值记录。固定仪器布局不成为可调参数；不顺带开放所有内部常数。
   - 参数使用共享的普通 Python 定义与校验，PyCPL 参数只作映射：阈值和提取宽度为有限正实数；波长上下限为有限正数且下限小于上限；通道数为整数（不接受 bool），生成后严格递增且 central slice [n//4:(3*n)//4] 至少包含两个波长样本。GD 上下限有限、下限小于上限，按原 0.01 um 步长得到非空有限粗搜索网格；相位开关必须为 bool。非法参数在进入科学计算前失败。

     | 阶段 | 参数 | 保留的默认值 |
     | --- | --- | --- |
     | dark | 坏像素判定阈值倍率 | 5.0，沿用现有暗场均值/RMS 判定 |
     | flat | 提取宽度 pixel | 10 |
     | p2vm | 目标波长下限、上限、通道数 | 1.15 um、1.30 um、151 |
     | p2vm | 相位参考校正 | true；无参考则明确记录未执行 |
     | p2vm | 标定 GD 搜索上下限 | -30 um、30 um |
     | reduce / realtime | 科学 GD 搜索上下限 | -100 um、100 um |

   - 标定 GD 设置贯穿 baseline 拟合以及相位参考中的全部 GD 搜索；science GD 设置用于完整 science reduction。公开上下限只限定粗网格初始化，后续 Newton 更新仍可越界，不改成最终结果截断或剔除条件。
   - 现有 GD 网格步长与 Newton 次数、波长拟合阶数/RANSAC/剔除规则、相位符号与零点、归一化和 fastmath/cache 保持原定义。允许参数取值变化不等于默认算法发生变化。
   - 单帧和分块共用同一提取及可见度归一化函数。非正/非有限光度积、非有限相干通量的处理沿用正式离线算法：NaN 加无效标志。实时端可以跳过 GD/差分相位等额外输出，但不能换一种可见度定义。

5. 收敛 ESO 和实时接入。
   - ESO 层只负责 FrameSet 标签、ParameterList、调用普通 Python 任务和 DFS 产品登记。使用显式的角色→允许产品类别/原始类型→CPL group/type 表，不能根据 tag 是否等于产品类别来猜测 RAW/CALIB。每个角色最多一个文件；同一文件可以明确用于两个不同角色。

     | Recipe | 必需 SOF 角色 | 可选角色及规则 |
     | --- | --- | --- |
     | planetes_dark | DARK_RAW：三维原始暗场，至少两帧 | 无；生成 PLANETES_DARK |
     | planetes_flat | DARK_CALIB：完整 PLANETES_DARK；TEL_FLAT_4/3/2/1：raw | 无；生成 PLANETES_FLAT |
     | planetes_wave | DARK_CALIB、PLANETES_FLAT、WAVE_RAW | PLANETES_LINE_LIST，缺省加载包内 schema-2 静态参考 |
     | planetes_p2vm | DARK_CALIB、PLANETES_FLAT、PLANETES_WAVE、四个 TEL_FLAT_*、六个 P2VM_RAW_* | PHASE_REFERENCE raw；PHASE_DARK_CALIB，缺省按已公开规则复用 main dark |
     | planetes_reduce | DARK_CALIB、SCIENCE_RAW、PLANETES_FLAT、PLANETES_WAVE、PLANETES_P2VM | 无 |

   - wave/p2vm/reduce 的 DARK_CALIB 及 PHASE_DARK_CALIB 允许 PLANETES_DARK 或仅含扣暗图的 PLANETES_DARK_IMAGE，均属 CALIB；它们在 SOF 中应为显式保存的 schema-2 产品。raw 角色属 RAW。静态 PLANETES_LINE_LIST 属 CALIB/TABLE。Python/GUI 的原始暗场便捷加载不混入其他 recipe 的科学处理逻辑。
   - 新暗场步骤后，一个原始 dark 可生成一个对应标定，并按实际用途分别关联。GUI 已有目录选择可继续解析最新文件为候选，但必须展示实际选用文件并经过公共 loader 和兼容性检查；最新、同名或同形状都不等于兼容。其他任务不自动选文件，不新增 EDPS 关联引擎。
   - 仅迁移 /Users/zexwu/zexwu_lib/devel/camview_p2vm.py 及该入口确有必要的接入代码。Qt5 历史 GUI 不纳入迁移；不能删除仍被历史 GUI 引用的独立旧实现。
   - 主 GUI 直接依赖已安装 planetes 公共接口，不再使用 devel/planetes_pipe 中的独立科学实现；新 Python API 不增加旧路径兼容层。
   - 保留 Camera/Visibility/History 布局、采集、硬件设置和原始文件保存流程。原始未 bin 的帧用于科学提取，binning 仅影响显示。
   - 保留当前 GUI 的运行规则：暗场关闭、不可用或不匹配时暂停 P2VM 更新并显示原因。严格区分原始帧、已扣暗帧和用于显示的帧，确定一个明确的扣暗位置，离线与实时各只扣一次。fake 源默认返回 raw，不预扣暗后再次传入扣暗流程。
   - 按第 15 项决定，GUI 保留二维暗场用于科学扣暗，通过公共 load_dark 返回 DarkImage；状态明确显示其没有噪声/坏像素统计。多帧暗场调用共享 dark 任务，不在 GUI 内另写 mean/RMS 算法。取当前帧作暗场及已有二维暗场的操作保留，但不能由这些输入建立 flat 的坏像素标定。
   - GUI 的 VISAMP 曲线和均值遵循公共 flags；无效波段不进入显示均值。绘图与 ABCD 标签使用 instrument 的真实 A,C,B,D 约定。
   - 更换暗场/标定时使相关预备状态和显示缓存失效，避免残留上次成功结果被标为本次有效结果。单次数据无效与整个标定不兼容要分别显示。

6. 统一可复现验证和运行环境。
   - 独立的新验证环境采用已有 CPL/PyCPL 依赖，保留原 .venv。原生 CPL 已链接的安装不直接搬目录；以显式 CPLDIR 复用或重新构建到经确认的位置。
   - 先核对实际 GUI 解释器及 Qt 依赖，复用现有可用环境。不要对采集环境执行无差别依赖同步；任何尚未授权的新依赖安装另行说明。数值 package 不强制依赖 Qt。
   - 为匹配/不匹配网格、覆盖不足、参数一致性、raw 保护、异常写出、复数与 flags 往返和校准关联添加有实际科学意义的检查。已有验证按新公开接口迁移，不以补丁路径或文件名本身作为测试目标。
   - 补齐有针对性的反例：同形状而 flat/wave 依赖不同；P2VM 主暗场与相位暗场不同和显式复用；二维暗场用于扣暗但被 flat 拒绝；同形状错误类别产品被 loader 拒绝；新宽度产生 ragged profile 和 detector-edge profile；通道数 1/2 或非法 GD 网格被拒绝。
   - 验证不同的合法 GD 设置真正传到 baseline 及相位参考搜索，且 Newton 仍保留原越界语义；不同合法宽度不改变严格选点规则，补齐哨兵不参与累加。
   - 对发布失败做注入测试：科学校验失败、DFS 临时头创建后失败、签名/读回失败，以及发布时目标被另一写入者抢先建立。均不得覆盖目标或留下最终名半成品；清理后可重试，既有目标的原始字节保持不变。
   - 自动化串行执行五个真实 PyEsoRex recipe，包含 default 和显式参数；同时从 Python 调用文件级任务，对比相同输入与设置下的科学结果。
   - 在新的输出目录完整重跑已有约 10.4 GB 数据，与保存的正式离线基准逐字段比较。由于拆出 dark 和重设计 FITS，按科学含义映射字段；不要求 FITS 文件字节相同。数组默认要求逐位保持，映射保持一致，Numba 缓存状态匹配。任何差异先定位，不放宽容差或更换基准掩盖它。
   - 目标波段严格覆盖已检查：当前基准 wave 的 24 输出均严格递增，共同覆盖约 0.948954–1.370011 um，足以覆盖现有 1.15–1.30 um 网格。
   - 逐帧和分块两条路径在同一帧、标定、暗场及设置下比较通量、相干通量、可见度与 flags，包含坏像素哨兵、非正光度积、不同 wave/P2VM 网格、显示 binning 和更换暗场的情形。
   - 主 GUI 以 fake 输入完成离屏检查：启动、标定切换、暗场启停、错误状态和曲线数据来源；只模拟硬件接口，不自动连接或操作台架设备。真实台架部署验证由实际运行环境完成，不能以 fake 检查宣称完成。
   - 对实际单帧工作负载测量预热后的计算耗时并报告，与现有 GUI 使用节奏比较；不要通过减少帧、改变 mask 或归一化来达到速度。
   - 构建 wheel，项目外安装后验证科学 API、五个 recipe 和静态参考文件；实际执行 wave recipe 的默认参考路径，确认不会回落到源码目录的旧 schema-1 资产。检查 30 个谱线值及 origin 标签与原 ECSV 一致。README 和 examples 使用明确的新环境与新入口，移除旧命令作为有效入口的提示。

7. 分阶段交付。
   - 每个可逆阶段保留清楚的提交和验证记录；外部 GUI 的改动另有备份、精确补丁和使用说明，因为它不在当前 Git 仓库内。
   - 更新修改汇报、科学对象/FITS 契约、参数表、运行示例及模块/数据流程图；标出语义修复和正常数据不变的证据。
   - 提供新源码归档与 wheel，明确格式版本和旧文件再生成方式。GitHub 推送、PR 和合并在实施结束时按用户明确授权执行，不在计划阶段进行。

## Key decisions & tradeoffs

以下 15 项由本轮访谈确认：

| 项 | 决定 |
| --- | --- |
| 1 | Python 与 PyEsoRex 共用正式处理函数 |
| 2 | 实时与离线共用科学定义和无效数据规则 |
| 3 | Python 默认只返回内存结果，保存显式调用 |
| 4 | science 与实时提取采用 P2VM 保存的网格 |
| 5 | 覆盖不足立即报错 |
| 6 | 提供完整文件级任务，同时保留分步调试 |
| 7 | 将实际相机 GUI 接入统一核心 |
| 8 | 允许重整 Python API，不维护旧接口兼容层 |
| 9 | 允许重设计 FITS 并重新生成产品 |
| 10 | 本轮止于仪器级科学结果 |
| 11 | 主 GUI 为 devel/camview_p2vm.py |
| 12 | 以现有正式离线算法为科学基准 |
| 13 | 一并开放列出的科学参数，两个入口一致 |
| 14 | 暗场独立为任务和 FITS 产品 |
| 15 | 保留二维暗场用于扣暗；不虚构 RMS/坏像素统计，建立 flat 仍需完整暗场标定 |

文件级任务和数值函数具有不同输入职责，但按科学任务放在一起，避免增加一个泛称 workflow/processing 框架。模型表达科学含义，FITS 表达磁盘格式，ESO 表达执行协议；它们共享定义但不相互复制科学逻辑。

## Risks / open questions

- 新 FITS 与 Python API 是一次明确迁移，需要使用新示例并重新生成标定。旧产品保留可追溯，不靠猜测旧字段含义做转换。
- 当前离线精确复现受 Numba 冷/热缓存影响；固定环境与对应缓存状态是验收条件，不能将跨状态差异误归因于重构。
- 当前实验原始 FITS 的曝光单位、逐帧时间和 ESO 观测元数据尚不完整。本轮记录实际已知内容，不虚构匹配条件；ESO 正式字典和归档关联仍需仪器方确认。
- GUI 的现场 Python/Qt/采集环境可能与本机不同；本轮可交付源代码接入及 fake 验证，不把实际硬件部署未经验证地标为完成。
- 新参数带来的科学输出变化必须可由实际参数记录解释；不在重构中混入未讨论的算法升级。

## Out of scope

- 天文传递函数标定、最终 OIFITS、闭合相位、完整误差传播、Phase 3 验收和 EDPS 规则。
- 修改原始 FITS、覆盖既有 reduced 产品、自动迁移采用中的旧 NPZ，或删除旧环境和原生库。
- 改变采集、硬件控制、GUI 布局、Qt5 历史入口及其仍在使用的旧实现。
- 新工作流引擎、全局 PipelineConfig、运行 YAML、自定义命令行框架，以及与本轮实际任务无关的通用架构。

