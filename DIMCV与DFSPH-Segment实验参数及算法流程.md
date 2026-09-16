# DIMCV 与 DFSPH+Segment 实验参数及算法流程

> 整理对象：`D:\Simulation\SourceCode\DIMCV-SPH`
>
> 本文档依据实验场景 JSON、配置读取代码和求解器源码整理。参数值以场景文件中的实际配置为准；“未设置”表示 JSON 中没有显式字段，运行时可能使用代码默认值。

---

## 1. 实验范围与场景文件

本次整理对应当前曲线对比中使用的实验组：

| 实验组 | DIMCV 场景 | DFSPH+Segment 场景 | 当前状态 |
|---|---|---|---|
| sphere | `data/scenes/DIM_von_karman_vortex_3d_dimcv_sphere.json` | `data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_sphere.json` | 已找到 |
| sphere_coarse | `data/scenes/DIM_von_karman_vortex_3d_dimcv_sphere_coarse.json` | `data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_sphere_coarse.json` | 已找到 |
| propeller | `data/scenes/DIM_von_karman_vortex_3d_dimcv_propeller_tank_fluid_block.json` | `data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_propeller_tank_fluid_block.json` | 已找到 |
| propeller_tall250 | `data/scenes/DIM_von_karman_vortex_3d_dimcv_propeller_tank_fluid_block_tall250.json` | `data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_propeller_tank_fluid_block_tall250.json` | 已找到 |
| propeller_tall500 | 未找到 | 未找到 | 当前工程中没有对应 JSON |
| baffle | `data/scenes/DIM_von_karman_vortex_3d_dimcv_baffle_tank_fluid_block.json` | `data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_baffle_tank_fluid_block.json` | 已找到 |

`propeller_tall500` 目前只有曲线/结果目录线索，没有对应的场景 JSON，不能把 `tall250` 参数直接当作 `tall500` 参数使用。

### 1.1 方法编号

| `Configuration.simulationMethod` | 方法 | 运行入口 |
|---:|---|---|
| `0` | DIMCV + DFSPH | `run_simulation.py` |
| `1` | 纯 DFSPH | `run_simulation_dfsph.py` |
| `2` | DFSPH + Segment | `run_simulation_dfsph_segment.py` |

方法选择由 `particle_system.py::build_solver()` 决定，不能只依据场景文件名判断。

---

# 2. DIMCV 方法参数

## 2.1 DIMCV 场景参数

DIMCV 参数位于各场景 JSON 的 `Configuration` 节点，主要由 `config_builder.py::SimConfig.get_cfg()` 读取，并由 `dimcv_sph.py` 使用。

| 参数 | 作用 | sphere | sphere_coarse | propeller | tall250 | baffle |
|---|---|---:|---:|---:|---:|---:|
| `simulationMethod` | 选择 DIMCV solver | 0 | 0 | 0 | 0 | 0 |
| `particleRadius` | SPH 粒子半径，影响粒子体积、支持半径和采样分辨率 | 0.0045 | 0.01 | 0.01 | 0.01 | 0.01 |
| `vortexEnforcingDomainStart` | DIMCV 涡量增强/采样区域起点 | `[0.2,0,0]` | `[0.2,0,0]` | `[0,0,0]` | `[0,0,0]` | `[0,0,0]` |
| `vortexEnforcingDomainEnd` | DIMCV 涡量增强/采样区域终点 | `[4,1,1]` | `[4,1,1]` | `[3,1,1]` | `[3,1,1]` | `[3,1,1]` |
| `volumeScale` | 涡量采样粒子有效体积缩放 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| `GenProbPara` | 动态采样生成概率参数，代码中赋给 `a` | 5 | 5 | 5 | 5 | 5 |
| `DelProbPara` | 动态采样删除概率参数，代码中赋给 `b` | 3.5 | 3.5 | 3.5 | 3.5 | 3.5 |
| `divergenceTransitArea` | 散度处理的过渡区域 | `[0,0.02,0.02]` | `[0,0.02,0.02]` | `[0,0.02,0.02]` | `[0,0.02,0.02]` | `[0,0.02,0.02]` |

### 2.1.1 DIMCV 参数结论

- 五个已找到的 DIMCV 场景使用完全相同的 DIMCV 动态采样参数：`volumeScale=0.2`、`GenProbPara=5`、`DelProbPara=3.5`。
- 主要实验差异来自粒子分辨率和计算域：`sphere` 使用更细粒子 `particleRadius=0.0045`，其余四组为 `0.01`。
- sphere 的涡量增强区域从 `x=0.2` 开始，计算域终点为 `x=4.0`；水槽类场景从 `x=0` 到 `x=3.0`。

## 2.2 DIMCV 动态采样逻辑

`dimcv_sph.py` 中读取：

```python
self.a = self.ps.cfg.get_cfg("GenProbPara")
self.b = self.ps.cfg.get_cfg("DelProbPara")
```

运行中维护 `is_sample`、`sample_idx` 等状态：

1. 根据 SPH 粒子的涡量/运动学量计算 KVN 等采样判据。
2. 对尚未采样的粒子，根据 `GenProbPara` 计算生成概率。
3. 对已采样且 KVN 较低的粒子，根据 `DelProbPara` 计算删除概率。
4. 采样粒子参与 DIMCV 的涡量样本表示和相关计算。
5. SPH 的 DFSPH 压力约束仍然负责流体不可压缩性。

因此，`GenProbPara` 和 `DelProbPara` 不是 Segment 环量参数，而是 DIMCV 的动态采样/删样控制参数。

## 2.3 DIMCV 求解器中的固定常量

以下参数当前直接写在 `dimcv_sph.py::DIMCVSPHSolver.__init__()` 中，不由场景 JSON 配置：

| 源码变量 | 值 | 作用 |
|---|---:|---|
| `eps` | `1e-6` | 数值稳定用小量 |
| `inv_eps` | `1/eps` | `eps` 倒数 |
| `m_max_iterations_v` | `100` | 速度/涡量相关迭代上限 |
| `m_max_iterations` | `100` | DIMCV 迭代上限 |
| `m_eps` | `1e-5` | 迭代收敛阈值 |
| `max_error_V` | `0.1` | 体积/速度约束误差上限 |
| `max_error` | `0.05` | 常规误差上限 |

---

# 3. DFSPH+Segment 方法参数

DFSPH+Segment 的公共参数位于 `Configuration`，Segment 参数位于 `SegmentConfiguration`。

## 3.1 公共 SPH/DFSPH 参数

| 参数 | sphere | sphere_coarse | propeller | tall250 | baffle |
|---|---:|---:|---:|---:|---:|
| `simulationMethod` | 2 | 2 | 2 | 2 | 2 |
| `particleRadius` | 0.0045 | 0.01 | 0.01 | 0.01 | 0.01 |
| `timeStepSize` | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |
| `simulationTime` | 12.0 | 12.0 | 6.0 | 6.0 | 14.0 |
| `density0` | 1000 | 1000 | 1000 | 1000 | 1000 |
| `stiffness` | 50000 | 50000 | 50000 | 50000 | 50000 |
| `exponent` | 7 | 7 | 7 | 7 | 7 |
| `gravitation` | `[0,0,-9.8]` | `[0,0,-9.8]` | `[0,0,-9.8]` | `[0,0,-9.8]` | `[0,0,-9.8]` |
| `domainStart` | `[0,0,0]` | `[0,0,0]` | `[0,0,0]` | `[0,0,0]` | `[0,0,0]` |
| `domainEnd` | `[4,1,1]` | `[4,1,1]` | `[3,1,1]` | `[3,1,1]` | `[3,1,1]` |
| `deleteFluidParticlesOutsideDomain` | true | true | true | true | true |
| `domainCullMargin` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

这些参数决定基础 SPH/DFSPH 的粒子分辨率、时间推进、物理域和状态方程；Segment 只是叠加在这个基础求解器上的涡量离散与反馈系统。

## 3.2 Segment 表示与生命周期参数

| `SegmentConfiguration` 参数 | 作用 | sphere | sphere_coarse | propeller | tall250 | baffle |
|---|---|---:|---:|---:|---:|---:|
| `segmentMaxNum` | Segment 最大容量 | 220000 | 220000 | 180000 | 180000 | 180000 |
| `biotSavartModel` | 诱导速度模型 | `finite_segment` | `finite_segment` | `finite_segment` | `finite_segment` | `finite_segment` |
| `pointVortexLifetimeMode` | 点涡寿命模式 | `persistent` | `persistent` | `persistent` | `persistent` | `persistent` |
| `regularizationRadiusR` | Biot-Savart 正则化半径 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| `gammaDecay` | 每步环量衰减因子 | 0.93 | 0.98 | 0.98 | 0.98 | 0.98 |
| `splitLengthThreshold` | 超过此长度触发分裂 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| `enableSplitSegments` | 是否分裂 | true | true | true | true | true |
| `enableMergeSegments` | 是否合并 | false | false | false | false | false |
| `enableDeleteWeakSegments` | 是否删除弱段 | true | true | true | true | true |
| `enableGpuDeleteCompact` | 是否启用 GPU 删除/压缩 | true | true | true | true | true |
| `deleteGammaThreshold` | 弱段删除阈值 | `1e-6` | `1e-6` | `1e-6` | `1e-6` | `1e-6` |
| `deleteOutsideDomain` | 是否删除域外段 | true | true | true | true | true |
| `outflowDeleteCenterBeyondX` | 中心超过该 X 坐标后删除 | 4.0 | 4.0 | 3.0 | 3.0 | 3.0 |
| `initType` | 初始 Segment 生成方式 | `none` | `none` | `none` | `none` | `none` |

当前实验没有预置 Segment 涡环或随机涡段，`initType=none`；Segment 主要依赖后续边界涡量注入生成。

### 3.2.1 关键差异

- 精细 sphere：`gammaDecay=0.93`、`sphSegmentBsCoupling=0.3`，Segment 涡量衰减和反馈均较保守。
- sphere_coarse：`gammaDecay=0.98`、`sphSegmentBsCoupling=0.6`，用于增强粗粒子场景中的涡量保持和 Segment 反馈。
- propeller、tall250、baffle：使用 `gammaDecay=0.98`、`sphSegmentBsCoupling=0.6`。

## 3.3 Segment 平流参数

| 参数 | 作用 | 五组值 |
|---|---|---|
| `segmentAdvectionMode` | Segment 平流模式 | `sph_velocity` |
| `segmentAdvectionUseBiotSavart` | 平流是否使用 Segment 自身 BS 速度 | false |
| `segmentAdvectionUseSphVelocity` | 平流是否使用 SPH 速度 | true |
| `segmentAdvectionUseBackground` | 平流是否使用背景速度 | false |
| `segmentAdvectionBackgroundVelocity` | 背景平流速度 | `[0,0,0]` |
| `sphVelocityToSegmentAdvectionEnabled` | 是否采样 SPH 速度用于 Segment 平流 | true |
| `sphVelocityToSegmentAdvectionBlend` | SPH 速度混合比例 | 1.0 |
| `sphVelocityToSegmentAdvectionScale` | SPH 速度缩放 | 1.0 |
| `sphVelocityToSegmentAdvectionSkipBoundary` | 是否跳过边界段 | true |

当前设置的含义是：Segment 的位置推进主要跟随 SPH 速度，而不是使用 Segment 自身的 Biot-Savart 速度进行平流；Biot-Savart 仍用于 Segment 对 SPH 的速度反馈。

## 3.4 SPH 边界涡量到 Segment 的注入参数

| 参数 | 作用 | sphere | sphere_coarse | propeller | tall250 | baffle |
|---|---|---:|---:|---:|---:|---:|
| `sphBoundaryVorticityInjectionEnabled` | 开启 SPH 边界涡量注入 | true | true | true | true | true |
| `sphBoundaryVorticityInjectionIntervalSteps` | 注入间隔 | 4 | 4 | 4 | 4 | 4 |
| `sphBoundaryVorticityInjectionStartStep` | 开始注入步号 | 20 | 20 | 500 | 500 | 500 |
| `sphBoundaryVorticityInjectionThreshold` | 注入涡量阈值 | 35.0 | 35.0 | 10.0 | 10.0 | 10.0 |
| `sphBoundaryVorticityInjectionSource` | 注入源字段 | `vorticity` | `vorticity` | `vorticity` | `vorticity` | `vorticity` |
| `sphBoundaryVorticityInjectionThresholdSource` | 阈值源字段 | `vorticity` | `vorticity` | `vorticity` | `vorticity` | `vorticity` |
| `sphBoundaryVorticityInjectionUseResidualVorticity` | 使用残差涡量 | true | true | true | true | true |
| `sphBoundaryVortexResidualSupportRadiusScale` | 残差支持半径缩放 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `sphBoundaryVorticityInjectionRegion` | 注入区域类型 | `box` | `box` | `box` | `box` | `box` |
| `sphBoundaryVorticityInjectionDistance` | 边界附近距离 | 0.035 | 0.035 | 0.035 | 0.035 | 0.035 |
| `sphBoundaryVorticityInjectionMaxPerStep` | 每步最大注入数量 | 140 | 400 | 400 | 400 | 400 |
| `sphBoundaryVorticityInjectionSelection` | 候选选择策略 | `top_strength` | `top_strength` | `top_strength` | `top_strength` | `top_strength` |
| `sphBoundaryVorticityInjectionSegmentLength` | 新 Segment 长度 | 0.018 | 0.018 | 0.018 | 0.018 | 0.018 |
| `sphBoundaryVorticityInjectionGammaScale` | 新 Segment 环量缩放 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `sphBoundaryVorticityInjectionSegmentTypeId` | 新 Segment 类型 | 0 | 0 | 0 | 0 | 0 |
| `sphBoundaryVorticityInjectionOrientation` | 方向构造方式 | `omega` | `omega` | `omega` | `omega` | `omega` |
| `sphBoundaryVorticityInjectionGammaMode` | 环量计算方式 | `omega_dot_t` | `omega_dot_t` | `omega_dot_t` | `omega_dot_t` | `omega_dot_t` |

### 3.4.1 注入策略差异

- sphere 和 sphere_coarse：第 20 步开始注入，阈值为 `35.0`。
- propeller、tall250、baffle：第 500 步开始注入，阈值为 `10.0`。
- sphere 每步最多注入 `140` 条，其他场景每步最多 `400` 条。
- 这些 Segment 的方向由局部 SPH 涡量 `omega` 构造，环量使用 `omega_dot_t` 方式计算。

## 3.5 Segment → SPH 反馈与 SPH → Segment 沉积

| 参数 | 作用 | sphere | sphere_coarse | propeller | tall250 | baffle |
|---|---|---:|---:|---:|---:|---:|
| `sphSegmentFeedbackMode` | Segment 对 SPH 反馈模式 | `bs_direct` | `bs_direct` | `bs_direct` | `bs_direct` | `bs_direct` |
| `sphSegmentBsCoupling` | BS 速度反馈耦合系数 | 0.3 | 0.6 | 0.6 | 0.6 | 0.6 |
| `sphSegmentBsFeedbackSkipBoundary` | 反馈时跳过边界段 | true | true | true | true | true |
| `sphToSegmentDepositEnabled` | SPH 涡量沉积到已有 Segment | false | false | false | false | false |

当前六组实验的实际耦合主线为：

```text
SPH 边界涡量 -> 新建 Segment
SPH 速度 -> Segment 平流
Segment Biot-Savart 速度 -> SPH 速度反馈
```

而不是每一步都用 SPH 涡量覆盖已有 Segment 的 `gamma`，因为 `sphToSegmentDepositEnabled=false`。

---

# 4. DFSPH+Segment 算法逻辑链

## 4.1 系统组成

DFSPH+Segment 是一个双向耦合系统：

```text
SPH/DFSPH 流体求解器
        ↕
Segment 涡量离散系统
        ↕
Biot-Savart 诱导速度场
```

SPH 粒子表示连续流体，Segment 用线段表示离散涡量。每条 Segment 至少包含：

```text
x_minus       负端点
x_plus        正端点
gamma         环量 Γ
active        活跃标记
age           年龄
seg_type      类型
center        中心
 tangent      切向方向
length        长度
```

Segment 的几何关系为：

\[
\mathbf{c}_i=\frac{\mathbf{x}_i^-+\mathbf{x}_i^+}{2},\qquad
\mathbf{d}_i=\mathbf{x}_i^+-\mathbf{x}_i^-,\qquad
L_i=\|\mathbf{d}_i\|.
\]

切向量为：

\[
\mathbf{t}_i=\frac{\mathbf{d}_i}{\|\mathbf{d}_i\|}.
\]

## 4.2 初始化链

```text
1. run_simulation_dfsph_segment.py 启动
2. 初始化 Taichi/CUDA
3. 读取 scene JSON
4. 创建 SimConfig
5. 创建 ParticleSystem
6. 分配 SPH 粒子场和邻居网格
7. 根据 simulationMethod=2 创建 DFSPHSegmentHybridKarmanSolver
8. 创建 SegmentConfig、SegmentSystem、SegmentSolver
9. 父类 DFSPH solver 初始化
10. SegmentSystem 清空并按 initType 初始化
11. 当前实验 initType=none，不预置 Segment 涡环
12. 初始化边界状态并更新 Segment 几何缓存
13. 进入主时间循环
```

关键源码职责：

| 文件 | 职责 |
|---|---|
| `run_simulation_dfsph_segment.py` | 程序入口、时间循环、导出与统计 |
| `particle_system.py` | SPH 粒子、邻居网格、solver 选择 |
| `dfsph_segment_hybrid_solver.py` | SPH 与 Segment 双向耦合 |
| `segment_config.py` | 读取 `SegmentConfiguration` |
| `segment_system.py` | Segment 数组和几何缓存 |
| `segment_solver.py` | Segment 速度、平流、注入、拓扑和清理 |
| `flow_statistics.py` | 统一统计 CSV |

## 4.3 单个物理子步的真实逻辑

源码中 hybrid solver 的 `substep()` 逻辑可以概括为：

```text
A. SPH 预测
   1. compute_densities
   2. compute_DFSPH_factor
   3. divergence_solve
   4. compute_non_pressure_forces
   5. predict_velocity

B. 得到预测阶段的 SPH 涡量
   6. compute_vorticity
   7. compute_vorticity_vis

C. 推进 Segment 子系统
   8. 同步移动刚体/圆柱姿态
   9. 按调度执行边界虚拟 Segment 注入
  10. 执行周期性 Segment 发射（若启用）
  11. 将 SPH 边界残差涡量注入为 Segment
  12. 更新 Segment 几何缓存
  13. 若启用，执行 SPH 涡量 -> Segment 环量沉积
  14. 若启用，执行 SPH 速度 -> Segment 平流速度采样
  15. 计算 Segment 端点速度
  16. RK4 推进 Segment 两端点
  17. 更新几何缓存
  18. 删除障碍物内部/域外/老化/弱 Segment
  19. split 过长 Segment
  20. merge 满足条件的 Segment
  21. 删除弱 Segment并压缩数组

D. Segment 反馈到 SPH
  22. 当前配置为 bs_direct
  23. 在 SPH 流体粒子位置累加 Segment Biot-Savart 速度
  24. v_sph += sphSegmentBsCoupling * u_segment

E. 反馈后的压力约束和粒子推进
  25. 再次 pressure_solve
  26. 重新计算 vorticity/vorticity_vis
  27. 保存临时位置
  28. advect 推进 SPH 粒子位置
  29. 若使用 ghost displacement，执行位置反馈
```

重要顺序：Segment 反馈发生后还会再进行一次 `pressure_solve`，所以 Segment 产生的速度修正会重新接受 DFSPH 的压力/不可压缩约束。

## 4.4 SPH/DFSPH 阶段

### 4.4.1 密度与不可压缩约束

密度通过 SPH 邻域核估计：

\[
\rho_i=\sum_j m_j W(\mathbf{x}_i-\mathbf{x}_j,h).
\]

随后计算 DFSPH 系数并进行散度求解，使：

\[
\nabla\cdot\mathbf{v}\approx 0.
\]

### 4.4.2 非压力力与预测速度

非压力力通常包含重力、黏性和场景相关外力。预测速度抽象为：

\[
\mathbf{v}_i^*=\mathbf{v}_i^n+\Delta t\,\mathbf{a}_i^{non-pressure}.
\]

Segment 子系统在这个预测阶段之后读取 SPH 速度和涡量。

## 4.5 SPH 边界涡量注入

当前实验最主要的 Segment 生成来源是 `sphBoundaryVorticityInjectionEnabled=true`。

执行链：

```text
1. 更新移动边界姿态
2. 选择边界附近 SPH 粒子/候选区域
3. 读取原始 SPH 涡量 vorticity
4. 根据 threshold 筛选高强度候选
5. 使用 top_strength 选择候选
6. 使用 omega 构造 Segment 方向
7. 通过 omega_dot_t 计算环量
8. 按 segmentLength 创建新 Segment
9. 提交到 SegmentSystem
10. 更新几何缓存
```

边界投影路线在启用时还会经过：

```text
update_boundary_pose
-> generate_boundary_segments
-> compute_k_matrix
-> compute_rhs
-> solve_linear_system
-> commit_boundary_segments
```

其目标是通过边界虚拟段满足边界速度约束，抽象形式为：

\[
K\Gamma\approx U.
\]

当前五组已找到的 Segment 场景均将 `initType` 设为 `none`，因此初始时没有预置涡段；注入流程负责逐步建立涡段云。

## 4.6 Segment 速度计算与平流

### 4.6.1 Biot-Savart 诱导速度

默认模型为有限长 Segment：

```text
biotSavartModel = finite_segment
regularizationRadiusR = 0.006
```

每个 Segment 的环量在其两端点和 SPH 粒子位置产生诱导速度。正则化半径用于避免查询点接近涡段时的数值奇异。

### 4.6.2 RK4 端点平流

Segment 两个端点分别满足：

\[
\frac{d\mathbf{x}_i^-}{dt}=\mathbf{u}(\mathbf{x}_i^-),\qquad
\frac{d\mathbf{x}_i^+}{dt}=\mathbf{u}(\mathbf{x}_i^+).
\]

采用 RK4：

\[
\mathbf{x}^{n+1}=\mathbf{x}^{n}+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\]

本实验使用 `segmentAdvectionMode=sph_velocity`，因此 Segment 位置主要由 SPH 速度推进；Segment 自身的 Biot-Savart 速度主要用于反馈 SPH，而不是作为 Segment 平流来源。

## 4.7 Segment 拓扑和生命周期

### split

当：

\[
L_i > splitLengthThreshold
\]

即 `L_i > 0.12` 时，在中点处分裂为两条 Segment。新段继承原环量和类型，但年龄重新开始。

### merge

当前实验 `enableMergeSegments=false`，因此默认不执行合并。若开启，通常要求 Segment 中心足够接近且方向满足角度条件，并尽量保持离散涡量向量：

\[
\mathbf{w}=\Gamma_i\mathbf{d}_i+\Gamma_j\mathbf{d}_j.
\]

### delete/compact

删除条件包括：

- `abs(gamma) < deleteGammaThreshold`；
- Segment 超出计算域；
- Segment 中心超过 `outflowDeleteCenterBeyondX`；
- Segment 进入障碍物内部；
- Segment 超过寿命限制；
- GPU 删除压缩条件满足。

当前启用 GPU 删除/压缩：`enableGpuDeleteCompact=true`。

## 4.8 Segment 到 SPH 的 Biot-Savart 反馈

当前配置为：

```text
sphSegmentFeedbackMode = bs_direct
sphSegmentBsFeedbackSkipBoundary = true
```

执行链：

```text
1. 对每个流体 SPH 粒子建立查询点
2. 累加所有符合条件的活跃 Segment 的 Biot-Savart 速度
3. 跳过配置要求忽略的边界 Segment
4. 乘以 sphSegmentBsCoupling
5. 修正 SPH 粒子速度
6. 进入后续 pressure_solve
```

抽象表达为：

\[
\mathbf{v}_p\leftarrow\mathbf{v}_p+\beta\mathbf{u}^{seg}_p,
\]

其中：

```text
beta = sphSegmentBsCoupling
```

各组 `beta`：

```text
sphere          0.3
sphere_coarse   0.6
propeller       0.6
tall250         0.6
baffle          0.6
```

该速度反馈可能暂时破坏速度场散度，因此必须在反馈之后重新进行压力求解。

## 4.9 反馈后的压力求解与最终推进

Segment 反馈后：

```text
Segment BS feedback
        ↓
pressure_solve
        ↓
重新计算 SPH 涡量
        ↓
advect SPH 粒子
```

这样 DFSPH 重新校正 Segment 反馈带来的速度/密度误差，之后 SPH 粒子才完成当前子步的位置更新。

## 4.10 输出与统计

达到输出周期后，入口程序依次执行：

```text
FlowStatisticsRecorder.record()
solver.export_png()
solver.export_ply()（若启用）
```

统计流程会重新计算 `vorticity` 和 `vorticity_vis`，再对活跃流体粒子统计：

```text
vortex_particle_fraction
vorticity_mean_signed_z
vorticity_mean_abs_z
vorticity_mean_magnitude
vorticity_rms_magnitude
kinetic_energy_total
kinetic_energy_mean_per_particle
vortex_kinetic_energy_total
vortex_kinetic_energy_mean_per_particle
```

涡粒子筛选条件为：

\[
\|\boldsymbol{\omega}_p\|\ge statisticsVorticityThreshold.
\]

---

# 5. 方法对照总结

| 对照项 | DIMCV | DFSPH+Segment |
|---|---|---|
| 方法编号 | `simulationMethod=0` | `simulationMethod=2` |
| 涡量表示 | DIMCV 动态采样粒子/样本 | 线段 Segment 云，携带环量 |
| 主要控制参数 | `volumeScale`、`GenProbPara`、`DelProbPara` | `segmentMaxNum`、`gammaDecay`、注入阈值、`sphSegmentBsCoupling` |
| 涡量生成/维护 | 动态生成与删除采样粒子 | 边界注入、Segment 平流、split/delete |
| 速度耦合 | DIMCV 与 DFSPH 内部耦合 | Segment BS 速度反馈到 SPH |
| 当前场景的主要差异 | 五组已找到的 DIMCV 参数基本相同 | sphere 与其他场景在衰减、反馈和注入策略上有明显差异 |
| 是否预置涡段 | 不适用 | `initType=none`，不预置 |
| SPH 涡量沉积到已有涡段 | 不适用 | `sphToSegmentDepositEnabled=false` |

## 最重要的当前配置结论

1. DIMCV 的动态采样参数在已找到的五组实验中保持统一：
   `volumeScale=0.2`、`GenProbPara=5`、`DelProbPara=3.5`。
2. DFSPH+Segment 当前不是“预先生成一堆 Segment 再独立推进”，而是主要通过 SPH 边界涡量注入动态生成 Segment。
3. 当前 Segment 平流主要跟随 SPH 速度，Segment 的 Biot-Savart 速度主要通过 `sphSegmentBsCoupling` 反馈回 SPH。
4. 当前 `sphToSegmentDepositEnabled=false`，所以不会每步把 SPH 涡量重新沉积到已有 Segment 的环量中。
5. sphere 和 sphere_coarse 虽然几何场景相近，但 Segment 参数不同：
   - sphere：`gammaDecay=0.93`、`sphSegmentBsCoupling=0.3`、最大注入 `140/step`；
   - sphere_coarse：`gammaDecay=0.98`、`sphSegmentBsCoupling=0.6`、最大注入 `400/step`。
6. `propeller_tall500` 当前没有对应场景 JSON，参数需要补充真实配置后才能纳入严格的实验参数对照。

---

## 6. 主要源码来源

- `run_simulation.py`
- `run_simulation_dfsph_segment.py`
- `particle_system.py`
- `config_builder.py`
- `dimcv_sph.py`
- `dfsph_segment_hybrid_solver.py`
- `segment_config.py`
- `segment_system.py`
- `segment_solver.py`
- `flow_statistics.py`
- `SEGMENT_SOLVER_FLOW.md`
- `实验场景与运行命令.md`
