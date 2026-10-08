# 3D SPH-Segment 统一粗粒子残差环量持久守恒方法

## 1. 实验与代码入口

场景文件：

data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_rotating_mixer_unified_coarse_residual_circulation_persistent_conservative_decay098.json

运行命令：

    python run_simulation_dfsph_segment.py --scene_file ./data/scenes/DIM_von_karman_vortex_3d_dfsph_segment_rotating_mixer_unified_coarse_residual_circulation_persistent_conservative_decay098.json

入口为 run_simulation_dfsph_segment.py，混合求解器为 dfsph_segment_hybrid_solver.py 中的 DFSPHSegmentHybridKarmanSolver。SPH 粒子负责密度、压力和速度求解；Segment 集合负责离散涡量的持久存储及 Biot-Savart 反馈。

## 2. 核心思想

每个 Segment i 由端点 x⁻ᵢ、x⁺ᵢ、有符号环量 Γᵢ、中心 cᵢ、切向量 tᵢ 和长度 Lᵢ 表示：

    dᵢ = x⁺ᵢ − x⁻ᵢ
    cᵢ = (x⁻ᵢ + x⁺ᵢ) / 2
    Lᵢ = |dᵢ|
    tᵢ = dᵢ / (Lᵢ + ε)

本实验采用 residual_circulation 和 persistent 模式。SPH 只负责检测当前未被已有 Segment 表示的残差并生成新段；已有段不再被 SPH 涡量重写，而是在平流后按 gammaDecay=0.98 衰减。这使得离散环量具有时间持久性，同时保留可控耗散。

## 3. 场景参数

### 3.1 SPH/DFSPH 参数

| 参数 | 值 |
|---|---:|
| 计算域 | [0,0,0] 到 [4,1,1] |
| 粒子半径 | 0.01 |
| 时间步长 Δt | 0.002 |
| 仿真时间 | 12.0 |
| 参考密度 ρ₀ | 1000 |
| 重力 | [0,0,-9.8] |
| 刚度 | 50000 |
| 状态方程指数 | 7 |
| simulationMethod | 2 |

搅拌器刚体 objectId=2，中心约为 [0.65,0.5,0.42]，从 t=2 s 开始以 350°/s 旋转，运行周期为 2 s 开、2 s 关。

### 3.2 Segment 参数

| 参数 | 值 | 作用 |
|---|---:|---|
| segmentMaxNum | 220000 | Segment 容量 |
| segmentStrengthConvention | residual_circulation | Γ 为残差线环量 |
| biotSavartModel | finite_segment | 有限长度线段模型 |
| regularizationRadiusR | 0.006 | Biot-Savart 正则化半径 |
| pointVortexLifetimeMode | persistent | Segment 持久存在 |
| residualCirculationPersistent | true | 开启持久残差模式 |
| residualSegmentGammaBlend | 0.0 | 禁止 SPH 重写已有 Γ |
| gammaDecay | 0.98 | 每步环量衰减 |
| splitLengthThreshold | 0.12 | 分裂长度阈值 |
| enableSplitSegments | true | 开启分裂 |
| enableMergeSegments | true | 开启合并 |
| mergeDistanceLambda | 0.05 | 合并中心距离 |
| mergeAngleThreshold | 60° | 反向环量角容差 |
| mergeIntervalSteps | 1 | 每步合并 |
| deleteGammaThreshold | 5×10⁻⁷ | 弱段删除阈值 |
| deleteOutsideDomain | true | 删除域外段 |
| outflowDeleteCenterBeyondX | 4.0 | 出口删除位置 |

Segment 位置采用 SPH 速度平流：

    segmentAdvectionMode = sph_velocity
    sphVelocityToSegmentAdvectionBlend = 1.0
    sphVelocityToSegmentAdvectionScale = 1.0

Segment 自身 Biot-Savart 速度不用于位置平流；它主要用于反馈 SPH。

## 4. SPH 涡量与残差

SPH 密度由标准核估计：

    ρᵢ = Σⱼ mⱼ W(xᵢ − xⱼ, h)

在边界候选点 x 处，已有 Segment 的涡量估计写为：

    ωᵛ(x) = [Σⱼ w(x−cⱼ) Γⱼ qⱼ] / [Σⱼ w(x−cⱼ)]

其中 w 是 cubic kernel，qⱼ 为 Segment 的方向向量。SPH 当前涡量记为 ωˢᵖʰ，残差涡量定义为：

    ωʳ(x) = ωˢᵖʰ(x) − ωᵛ(x)

本实验设置 sphBoundaryVorticityInjectionUseResidualVorticity=true，因此新段方向和环量均来自 ωʳ，而不是完整 SPH 涡量。

候选区域为 [0,0.02,0.05] 到 [4,0.98,0.433333...]，边界距离带为 0.035，阈值为 0，每步最多注入 400 条，选择策略为 proportional_sign。

## 5. 新 Segment 的生成

三维情况下，新段切向量为：

    t_new = ωʳ / (|ωʳ| + ε)

新段长度固定为 ℓ_new=0.018，中心为候选粒子位置 x：

    x⁻_new = x − 0.5 ℓ_new t_new
    x⁺_new = x + 0.5 ℓ_new t_new

在 residual_circulation 约定下，代码实际使用的三维新段环量为：

    Γ_new = gammaScale · |ωʳ| · Vₚ / ℓ_new

其中 gammaScale=1，Vₚ 是 SPH 粒子体积。除以 ℓ_new 是因为 SPH 的 ωV 是涡量向量强度，而 Segment 存储的是线环量 Γ。对应离散涡量向量为：

    β_new = Γ_new d_new

注意：配置中仍保留 sphBoundaryVorticityInjectionGammaMode=omega_dot_t，但三维残差持久分支优先执行上述“残差模长 × 体积 ÷ 段长”公式。

## 6. 持久环量与衰减

### 6.1 已有段的保守持久性

已有残差段的可选 SPH 松弛形式是：

    Γᵢ ← (1 − b) Γᵢ + b Γᵢ,target

其中：

    Γᵢ,target = [ωˢᵖʰ(cᵢ) · tᵢ] · Vᵢ / Lᵢ

本实验 b=residualSegmentGammaBlend=0，因此：

    Γᵢ ← Γᵢ

所以已有 Segment 不会被当前 SPH 涡量重新覆盖；SPH 残差只生成新段。

### 6.2 对流衰减

Segment 端点平流后执行：

    Γᵢⁿ⁺¹ = 0.98 Γᵢⁿ
    ageᵢⁿ⁺¹ = ageᵢⁿ + Δt

离散因子对应的连续等效衰减率为：

    λ_decay = −ln(0.98) / 0.002 ≈ 10.1017 s⁻¹
    Γ(t) ≈ Γ(0) exp(−λ_decay t)

衰减发生在本步端点速度计算和 RK4 更新之后，因此本步使用旧 Γ 计算速度，衰减后的 Γ 用于下一步。

## 7. 单步完整算法

### 7.1 SPH 预测

1. 计算密度和 DFSPH 系数。
2. 求解速度散度约束。
3. 计算重力、黏性等非压力力。
4. 计算预测速度、涡量和涡量损失场。
5. 计算预测阶段的可视化涡量。

### 7.2 Segment 更新

1. 同步旋转搅拌器姿态。
2. 执行边界流程和周期发射流程。
3. 对已有持久残差段执行可选环量松弛；本实验因 blend=0 不修改 Γ。
4. 根据 SPH 边界残差生成新 Segment。
5. 更新几何缓存。
6. 关闭 SPH 到已有 Segment 的环量沉积：sphToSegmentDepositEnabled=false。
7. 采样 SPH 速度作为 Segment 端点平流速度。
8. 使用 RK4 推进两个端点：

    k₁ = u(xⁿ)
    k₂ = u(xⁿ + 0.5 Δt k₁)
    k₃ = u(xⁿ + 0.5 Δt k₂)
    k₄ = u(xⁿ + Δt k₃)
    xⁿ⁺¹ = xⁿ + Δt (k₁ + 2k₂ + 2k₃ + k₄) / 6

9. 执行 Γ←0.98Γ 和年龄更新。
10. 删除障碍物内部、域外和低速无效段。
11. 对长段执行 split。
12. 对近邻反向段执行 merge。
13. 删除弱段并压缩数组。

拓扑顺序严格为：

    平流/衰减 → 删除无效段 → split → merge → 删除弱段

因此每个时间步先分裂再合并。

### 7.3 Segment 到 SPH 的反馈

对 SPH 粒子 xₚ，有限线段 Biot-Savart 速度为：

    u_seg(xₚ) = Σⱼ u_BS(xₚ; x⁻ⱼ, x⁺ⱼ, Γⱼ, R)

反馈速度修正为：

    vₚ ← vₚ + β u_seg(xₚ)

本场景中 sphSegmentFeedbackMode=bs_direct，sphSegmentBsCoupling=1.0，并跳过边界类型 Segment。反馈后再次执行 DFSPH pressure_solve，重新满足不可压缩约束。

### 7.4 最终粒子推进

    Segment Biot-Savart 反馈
    → pressure_solve
    → 重新计算涡量
    → 保存临时位置
    → advect 粒子

## 8. 分裂与合并

### 8.1 分裂

当 Lᵢ>0.12 时，在中点 mᵢ=cᵢ 处分裂：

    [x⁻ᵢ, x⁺ᵢ] → [x⁻ᵢ, mᵢ] + [mᵢ, x⁺ᵢ]

子段继承 Γ 和段类型，点涡体积各取原体积的一半，并满足：

    Γ₁d₁ + Γ₂d₂ = Γᵢdᵢ

### 8.2 合并

两个段 i、j 必须满足：

    |cᵢ−cⱼ| ≤ 0.05

定义有符号环量向量：

    wᵢ = Γᵢ dᵢ
    wⱼ = Γⱼ dⱼ

反向条件为：

    (wᵢ·wⱼ) / (|wᵢ||wⱼ|) ≤ −cos(60°) = −0.5

合并后的量为：

    w_new = wᵢ + wⱼ
    c_new = (cᵢ + cⱼ) / 2
    L_new = (Lᵢ + Lⱼ) / 2
    t_new = w_new / |w_new|
    Γ_new = |w_new| / L_new
    x⁻_new = c_new − 0.5 L_new t_new
    x⁺_new = c_new + 0.5 L_new t_new

因此：

    Γ_new d_new = Γᵢ dᵢ + Γⱼ dⱼ

合并保持离散涡量向量，同时减少粗粒子场景中的 Segment 数量。

## 9. 论文式算法摘要

    给定 SPH 粒子状态和持久 Segment 集合：

    1. 用 DFSPH 计算预测速度和 SPH 涡量 ωˢᵖʰ。
    2. 估计已有 Segment 涡量 ωᵛ。
    3. 构造残差 ωʳ=ωˢᵖʰ−ωᵛ，并筛选边界候选点。
    4. 以 ωʳ 的方向生成长度 0.018 的新 Segment，
       设置 Γ_new=|ωʳ|V/0.018。
    5. 保持已有 Segment 的 Γ 不变，因为 blend=0。
    6. 使用 SPH 速度对端点进行 RK4 平流，并执行 Γ←0.98Γ。
    7. 删除无效段，先对 L>0.12 的段分裂，
       再对距离≤0.05 且环量向量反向的段合并。
    8. 用有限段 Biot-Savart 速度反馈 SPH，
       随后重新执行 DFSPH 压力求解。
    9. 推进 SPH 粒子并进入下一时间步。

## 10. 源码对应

| 文件 | 作用 |
|---|---|
| run_simulation_dfsph_segment.py | 场景读取、主循环、统计和导出 |
| dfsph_segment_hybrid_solver.py | SPH-Segment 耦合、残差注入、反馈和子步顺序 |
| segment_system.py | Segment 状态数组和几何缓存 |
| segment_solver.py | Biot-Savart、RK4、衰减、分裂、合并、删除 |
| segment_config.py | SegmentConfiguration 读取 |
| segment_boundary.py | 可选边界虚拟段处理 |

关键函数包括：

    DFSPHSegmentHybridKarmanSolver.substep
    _inject_segments_from_sph_boundary_vorticity
    _update_persistent_residual_segments
    SegmentSolver.advect_segments_rk4
    SegmentSolver.split_segments
    SegmentSolver.merge_segments
    SegmentSolver.delete_weak_segments

