# DIM 2D DFSPH + Persistent Point Vortex Ghost 实验 Step 伪代码

命令：

```bash
python run_simulation_dfsph_segment.py --scene_file ./data/scenes/DIM_von_karman_vortex_2d_dfsph_segment_velocity_deposit.json
```

核心配置：

```json
{
  "biotSavartModel": "point_vortex_2d",
  "pointVortexLifetimeMode": "persistent",
  "sphSegmentFeedbackMode": "vortex_ghost_displacement",
  "vortexGhostFeedbackVelocitySource": "direct_bs",
  "segmentAdvectionMode": "sph_velocity",
  "segmentAdvectionUseBiotSavart": false,
  "segmentAdvectionUseSphVelocity": true,
  "segmentSphAdvectButGhostUseBsVelocity": false,
  "enableBoundaryInjection": false,
  "sphBoundaryVorticityInjectionEnabled": true,
  "sphBoundaryVorticityInjectionUseResidualVorticity": true
}
```

SPH 解速度/压力/涡量；从 SPH residual 涡量注入 persistent 点涡；点涡由 SPH 速度推进；ghost/点涡 gamma 在 SPH 粒子位置直接计算 2D Biot-Savart 速度；该速度以 displacement 方式反馈到 SPH 位置。

---

## 1. 主循环

```text
load scene config
init ParticleSystem
build DFSPHSegmentHybridKarmanSolver
solver.initialize()

cnt = 0
cnt_output = 0
t = 0

while t <= simulationTime and no stop file:
    repeat numberOfStepsPerRenderUpdate times:
        solver.step()

    cnt += 1
    t += dt

    if cnt % output_interval == 0:
        export SPH PNG
        export segment / point-vortex PNG
        export ghost PNG
        if exportPLY:
            export PLY
        cnt_output += 1

    if cnt % 50 == 0:
        print simulation status
```

当前：

```text
dt = 0.002
output_interval = int(0.016 / dt) = 8
```

所以每 8 个 simulation step 导出一次 PNG / PLY。

---

## 2. solver.step()

```text
function step():
    ps.initialize_particle_system()
        update_activity()
        particle_partition()
        rebuild_neighbor_grid()

    if cnt % emit_interval == 0:
        emit SPH inlet particles
        ps.rebuild_neighbor_grid()

    cnt += 1

    compute_moving_boundary_volume()

    if fluid_particle_num <= 0:
        return

    substep()

    if bounceBackBoundary:
        enforce_boundary_2D / enforce_boundary_3D
```

---

## 3. substep()

```text
function substep():
    compute_densities()
    compute_DFSPH_factor()
    divergence_solve()
    compute_non_pressure_forces()
    predict_velocity()

    compute_vorticity_pre()
    compute_vorticity_vis_pre()

    advance_segments_coupled()
    apply_segment_feedback_to_sph()

    pressure_solve()

    compute_vorticity_post()
    compute_vorticity_vis_post()

    copy_x_temp()
    advect()

    apply_vortex_ghost_displacement_feedback()
```

对应 timing：

```text
compute_densities
compute_DFSPH_factor
divergence_solve
compute_non_pressure_forces
predict_velocity
compute_vorticity_pre
compute_vorticity_vis_pre
advance_segments_coupled
apply_segment_feedback_to_sph
pressure_solve
compute_vorticity_post
compute_vorticity_vis_post
copy_x_temp
advect
apply_vortex_ghost_displacement
```

---

## 4. SPH 部分

```text
compute_densities():
    for each SPH particle i:
        density[i] = sum_j m_j * W(x_i - x_j)

compute_DFSPH_factor():
    compute pressure / divergence solve factor

divergence_solve():
    iteratively correct velocity divergence

compute_non_pressure_forces():
    compute gravity / viscosity / external non-pressure forces

predict_velocity():
    v += dt * acceleration

compute_vorticity():
    compute raw SPH curl / omega

compute_vorticity_vis():
    compute smoothed vorticity for visualization and injection
```

---

## 5. advance_segments_coupled()

当前 boundary 关闭，因此 `boundary_pipeline` 基本为空。

```text
function advance_segments_coupled():
    boundary_pipeline()              # disabled currently
    periodic_emit()                  # parallel-X repeat disabled currently
    residual_vorticity_injection()
    update_segment_geometry_pre()
    vorticity_deposit_to_segments()  # disabled because sphToSegmentDepositEnabled=false
    sph_velocity_deposit_to_segments()

    compute_endpoint_velocity()
    advect_segments_rk4()

    if segment_initial_sph_then_bs:
        decay_initial_sph_advect_velocity()

    update_segment_geometry_post()
    delete_inside_obstacles()
    delete_slow_point_vortices()

    if not point_vortex_2d:
        split_segments()
        merge_segments()
        restore_frozen_segment_geometry()

    delete_weak_segments()

    if segmentSphAdvectButGhostUseBsVelocity:
        recompute_bs_velocity_for_ghost()   # false currently

    sim_step_index += 1
```

当前主要执行：

```text
residual_vorticity_injection
sph_velocity_deposit_to_segments
compute_endpoint_velocity
advect_segments_rk4
delete_inside_obstacles
delete_slow_point_vortices
delete_weak_segments
```

---

## 6. residual_vorticity_injection()

当前配置：

```json
"sphBoundaryVorticityInjectionEnabled": true,
"sphBoundaryVorticityInjectionUseResidualVorticity": true,
"sphBoundaryVorticityInjectionThreshold": 70.0,
"sphBoundaryVorticityInjectionMaxPerStep": 400,
"sphBoundaryVorticityInjectionIntervalSteps": 1
```

伪代码：

```text
function residual_vorticity_injection():
    if injection disabled:
        return
    if step < start_step:
        return
    if step % injection_interval != 0:
        return

    compute SPH source vorticity omega_sph

    if use_residual_vorticity:
        estimate omega represented by existing point vortices
        omega_residual = omega_sph - omega_vortex
    else:
        omega_residual = omega_sph

    candidates = SPH particles where abs(omega_residual) > threshold
    select up to max_per_step candidates

    for each selected candidate:
        create persistent point vortex
        set position near SPH particle
        set gamma from residual/source vorticity
        set segment type = 0
        initialize point vortex volume and geometry
```

作用：把 SPH 中强涡量区域转成 persistent point vortices。

---

## 7. SPH velocity deposit 到点涡

点涡由 SPH 速度推进。

```text
function sph_velocity_deposit_to_segments():
    for each active point vortex / segment i:
        if skipped segment type:
            sph_advect_velocity_minus[i] = 0
            sph_advect_velocity_plus[i] = 0
            continue

        xm = x_minus[i]
        xp = x_plus[i]

        v_m = kernel_average_sph_velocity_at(xm)
        v_p = kernel_average_sph_velocity_at(xp)

        sph_advect_velocity_minus[i] = blend(old_minus, scale * v_m)
        sph_advect_velocity_plus[i]  = blend(old_plus,  scale * v_p)
```

`kernel_average_sph_velocity_at(x)` 当前使用 SPH neighbor grid：

```text
find grid cell of x
visit neighboring 3x3 cells
for each SPH particle p within support radius:
    weight = m_V[p] * W(|x - x_p|)
    accumulate weight * v[p]
return weighted average velocity
```

---

## 8. 点涡推进

当前推进不用点涡之间 BS，不用背景速度，只用 SPH 沉积速度。

```text
compute_endpoint_velocity():
    for each point vortex i:
        v_minus[i] = sph_advect_velocity_minus[i]
        v_plus[i]  = sph_advect_velocity_plus[i]

advect_segments_rk4():
    for each point vortex i:
        x_minus[i] += dt * v_minus[i]
        x_plus[i]  += dt * v_plus[i]

update_segment_geometry():
    center[i]  = 0.5 * (x_minus[i] + x_plus[i])
    tangent[i] = normalize(x_plus[i] - x_minus[i])
    length[i]  = |x_plus[i] - x_minus[i]|
```

---

## 9. 点涡删除

```text
delete_inside_obstacles():
    for each internal point vortex:
        if center inside obstacle / cylinder:
            delete it
    compact remaining point vortices

delete_slow_point_vortices():
    for each internal point vortex:
        speed = |0.5 * (v_minus + v_plus)|
        if speed < deleteSlowPointVortexSpeedThreshold:
            delete it
    compact remaining point vortices

delete_weak_segments():
    gamma *= gammaDecay
    if abs(gamma) < deleteGammaThreshold:
        delete it
    compact remaining point vortices
```

---

## 10. apply_segment_feedback_to_sph()

当前 feedback 来源是 `direct_bs`。

```text
function apply_segment_feedback_to_sph():
    skip boundary type if needed

    direct_bs_source = true
    copy_ghost_velocity = false

    sync_vortex_ghost_particles_from_segments():
        for each active internal point vortex:
            ghost_x     = segment center
            ghost_gamma = segment gamma
            ghost_mV    = point vortex volume
            ghost_v     = 0

    build_vortex_ghost_grid()

    compute_vortex_ghost_bs_velocity_for_fluid():
        for each SPH fluid particle i:
            u_i = 0
            query neighboring ghost grid cells
            for each nearby ghost j:
                r = x_i - ghost_x[j]
                if |r| < support:
                    u_i += Gamma_j / (2*pi) * [-r_y, r_x] / (|r|^2 + R^2)
            u_vortex_ghost_sph[i] = normal_sign * u_i

    if feedback mode is velocity:
        ps.v[i] += beta * u_vortex_ghost_sph[i]
    else if feedback mode is displacement:
        keep u_vortex_ghost_sph for later displacement
```

当前是 displacement 模式，因此这里只缓存 `u_vortex_ghost_sph`，不立刻改速度或位置。

---

## 11. Pressure solve 与 ghost displacement 时序

当前顺序：

```text
apply_segment_feedback_to_sph()
pressure_solve()
copy_x_temp()
advect()
apply_vortex_ghost_displacement_feedback()
```

即：

```text
pressure_solve():
    project / correct SPH velocity to reduce density error

advect():
    x_i += dt * ps.v[i]

apply_vortex_ghost_displacement_feedback():
    x_i += dt * beta * u_vortex_ghost_sph[i]
```

最终近似：

```text
x_i^{n+1} = x_i^n + dt * v_sph_i + dt * beta * u_ghost_bs_i
```

---

## 12. Direct-BS 公式

对 SPH 粒子 `i` 和 ghost/point vortex `j`：

```text
r = x_i - x_j
R = regularizationRadiusR
Gamma = ghost_gamma[j]
```

2D 点涡速度：

```text
u_i += normal_sign * Gamma / (2*pi) * [-r_y, r_x] / (|r|^2 + R^2)
```

当前：

```json
"pointVortex2DNormalDirection": "out"
```

因此 `normal_sign = +1`。

---

## 13. 完整 step 高级伪代码

```text
function solver.step():
    update / compact SPH particles
    emit inlet SPH particles
    rebuild SPH neighbor grid

    compute SPH density and DFSPH factors
    divergence_solve()
    compute non-pressure forces
    predict SPH velocity
    compute SPH vorticity

    inject persistent point vortices from SPH residual vorticity
    deposit SPH velocity to point vortices through SPH neighbor grid
    advect point vortices by deposited SPH velocity
    delete obstacle-inside / slow / weak point vortices

    sync point vortices to ghost particles
    build ghost grid
    compute direct BS velocity at SPH particles from ghost gamma

    pressure_solve()
    recompute SPH vorticity
    copy_x_temp()
    advect SPH particles by pressure-projected SPH velocity
    apply ghost displacement:
        x += dt * beta * direct_bs_velocity
```

---

## 14. 导出阶段

每 8 个 simulation step：

```text
export_png():
    export SPH vorticity image
    export segment / point vortex gamma image
    export ghost panel:
        draw SPH particles
        draw ghost particles
        draw green arrows for u_vortex_ghost_sph

if exportPLY:
    export SPH particle PLY
    export segment PLY
```

绿色箭头表示：

```text
SPH 粒子位置上的 direct-BS velocity u_vortex_ghost_sph
```

不是 ghost 自身速度。

---

# 中文版流程摘要

这一节保持前面的详细伪代码不变，只用中文重新概括当前实验每一步做了什么。

---

## A. 整体思路

当前实验不是纯 SPH，也不是纯点涡方法，而是一个混合流程：

```text
SPH 负责求解流体的速度、压力和涡量；
从 SPH 的涡量场中提取强涡量区域，生成 persistent 点涡；
这些点涡由 SPH 速度场推进；
点涡自身的 gamma 被当作涡量源；
在 SPH 粒子位置直接由这些点涡 gamma 计算 Biot-Savart 速度；
最后把这个 BS 速度转换成一个额外位移，加到 SPH 粒子位置上。
```

也就是说，当前点涡主要有两个作用：

```text
1. 记录/携带 SPH 中抽取出来的涡量结构；
2. 通过 direct Biot-Savart velocity 反过来影响 SPH 粒子的运动。
```

---

## B. 每个 simulation step 的中文流程

一个完整的 `solver.step()` 可以理解为：

```text
1. 更新 SPH 粒子系统
   - 删除出域粒子
   - 压缩粒子数组
   - 重建 SPH neighbor grid

2. 从入口发射新的 SPH 粒子
   - 当前是左侧入口持续发射
   - 发射速度为 [2.0, 0.0]

3. 执行 SPH / DFSPH 求解
   - 计算密度
   - 计算 DFSPH factor
   - 做 divergence solve
   - 计算非压力力
   - 预测速度

4. 计算 SPH 涡量
   - 计算 raw vorticity
   - 计算平滑后的 vorticity_vis

5. 从 SPH 涡量注入点涡
   - 找到涡量超过阈值的 SPH 区域
   - 如果启用 residual vorticity，则扣除已有点涡已经表示的涡量
   - 从剩余涡量中选择候选粒子
   - 最多注入 sphBoundaryVorticityInjectionMaxPerStep 个点涡

6. 用 SPH 速度推进点涡
   - 对每个点涡，在点涡位置查询 SPH neighbor grid
   - 用核函数加权平均附近 SPH 粒子的速度
   - 得到点涡的推进速度
   - 用这个速度推进点涡位置

7. 删除不需要的点涡
   - 删除进入障碍物内部的点涡
   - 删除速度接近 0 的滞留点涡
   - 删除 gamma 太弱的点涡

8. 把点涡同步成 ghost 粒子
   - ghost_x = 点涡位置
   - ghost_gamma = 点涡 gamma
   - ghost_mV = 点涡体积权重
   - ghost_v = 0，因为当前 direct_bs 模式不使用 ghost 自身速度

9. 在 SPH 粒子位置计算 ghost direct-BS 速度
   - 对每个 SPH 粒子，查询附近 ghost grid
   - 使用 ghost gamma 直接计算 2D 点涡 Biot-Savart 速度
   - 结果保存到 u_vortex_ghost_sph

10. 做 pressure solve
    - DFSPH 压力投影修正 SPH 速度
    - 使 SPH 速度满足近似不可压条件

11. 推进 SPH 粒子
    - 先用 pressure solve 后的 SPH 速度推进：x += dt * v_sph
    - 再加 ghost BS 产生的额外位移：x += dt * beta * u_vortex_ghost_sph
```

---

## C. 当前最关键的反馈公式

当前 ghost 对 SPH 的反馈不是直接修改速度，而是修改位置。

首先，在 SPH 粒子位置计算 direct-BS 速度：

```text
u_ghost(x_i) = sum_j Gamma_j / (2*pi) * [-r_y, r_x] / (|r|^2 + R^2)
```

其中：

```text
r = x_i - x_ghost_j
Gamma_j = 第 j 个点涡的 gamma
R = regularizationRadiusR
```

然后把这个速度变成位移：

```text
dx_ghost = dt * vortexGhostVelocityCouplingBeta * u_ghost
```

最终 SPH 粒子的位置更新可以理解为：

```text
x_new = x_old + dt * v_sph + dx_ghost
```

也就是：

```text
x_new = x_old + dt * v_sph + dt * beta * u_ghost
```

---

## D. 当前哪些功能是关闭的

当前为了性能和实验目标，关闭了 segment boundary 相关功能：

```text
enableBoundaryInjection = false
boundaryInjectionSchedule = disabled
enableGpuBoundarySolve = false
rhsIncludeInternalSegments = false
```

因此当前不会每步做 boundary virtual segment 求解。

当前也关闭了点涡自身 BS 推进：

```text
segmentAdvectionUseBiotSavart = false
segmentSphAdvectButGhostUseBsVelocity = false
```

所以点涡之间不会互相用 BS 速度推进，点涡主要跟随 SPH 速度场移动。

---

## E. 当前图像中各元素含义

### SPH 图

```text
显示 SPH 粒子，颜色通常代表 vorticity_vis.z。
```

### Segment / Point Vortex 图

```text
显示 persistent 点涡，颜色代表点涡 gamma。
```

### Ghost 图

```text
显示 ghost/点涡和 SPH 粒子。
绿色箭头表示 SPH 粒子位置上由 ghost gamma 直接计算出的 BS 速度 u_vortex_ghost_sph。
黄色箭头原本表示 ghost 自身速度，但当前 direct_bs 模式下 ghost_v 被置零，因此通常不会显示。
```

---

## F. 当前性能主要看哪里

当前最重要的 timing 是：

```text
[hybrid-substep-timing]
[segment-advance-timing]
```

其中 `advance_segments_coupled` 里面最可能耗时的是：

```text
residual_vorticity_injection
sph_velocity_deposit_to_segments
delete_inside_obstacles
delete_slow_point_vortices
delete_weak_segments
```

目前 boundary pipeline 已经关闭，所以它不应该再是主要瓶颈。

---

## G. 当前实验一句话总结

```text
这是一个 SPH 驱动 persistent 点涡、persistent 点涡再通过 direct Biot-Savart displacement 反馈 SPH 的双向耦合实验。
```

更具体地说：

```text
SPH 产生涡量；
涡量生成点涡；
点涡跟随 SPH 速度移动；
点涡 gamma 产生 BS 速度；
BS 速度以位移形式影响 SPH 粒子。
```

