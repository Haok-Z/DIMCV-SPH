# 2D DFSPH + Persistent Point Vortex Ghost 实验配置说明

对应场景：`DIM_von_karman_vortex_2d_dfsph_segment_velocity_deposit.json`

当前实验核心：SPH 解速度/压力/涡量，按 SPH 涡量注入 persistent 点涡；点涡由 SPH 速度推进；ghost/点涡的 `gamma` 在 SPH 粒子位置直接计算 2D Biot-Savart 速度，并以 displacement 方式反馈到 SPH。

---

## 1. SPH / DFSPH 基础配置 `Configuration`


| 配置项                                         | 当前值                   | 说明                                                            |
| ------------------------------------------- | --------------------- | ------------------------------------------------------------- |
| `domainStart` / `domainEnd`                 | `[0,0]` / `[4,1]`     | SPH 仿真域。                                                      |
| `particleRadius`                            | `0.0045`              | SPH 粒子半径；邻域核半径通常为 `4 * particleRadius`。                       |
| `timeStepSize`                              | `0.002`               | SPH 时间步长。                                                     |
| `simulationTime`                            | `12.0`                | 总仿真时间。                                                        |
| `numberOfStepsPerRenderUpdate`              | `1`                   | 每次主循环执行的 step 数；PNG 实际由脚本 `int(0.016/dt)` 控制，当前每 8 step 导出一次。 |
| `simulationMethod`                          | `2`                   | 使用 DFSPH + vortex segment/point vortex hybrid solver。         |
| `density0`                                  | `1000`                | 参考密度。                                                         |
| `gravitation`                               | `[0,0]`               | 无重力。                                                          |
| `deleteFluidParticlesOutsideDomain`         | `true`                | 删除流出仿真域的 SPH 粒子。                                              |
| `reservedCapacity`                          | `800000`              | 粒子系统预留容量。                                                     |
| `exportPLY`                                 | `true`                | 是否导出 SPH PLY。                                                 |
| `cylinderCenter` / `cylinderRadius`         | `[0.65,0.5]` / `0.08` | 圆柱障碍物参考位置和半径。                                                 |
| `imageVorticityVmin` / `imageVorticityVmax` | `-40` / `40`          | SPH 涡量图色标范围。                                                  |
| `imageFluidPointSize` / `imageFluidAlpha`   | `1.0` / `1`           | SPH 粒子绘制大小与透明度。                                               |


---

## 2. 点涡 / Segment 基础配置 `SegmentConfiguration`


| 配置项                                  | 当前值               | 说明                             |
| ------------------------------------ | ----------------- | ------------------------------ |
| `segmentMaxNum`                      | `80000`           | 最大点涡/segment 数量。               |
| `biotSavartModel`                    | `point_vortex_2d` | 使用 2D 点涡 Biot-Savart 模型。       |
| `pointVortexLifetimeMode`            | `persistent`      | 点涡持久存在，直到被删除规则移除。              |
| `pointVortex2DNormalDirection`       | `out`             | 2D 点涡法向方向；会影响 BS 速度方向。         |
| `regularizationRadiusR`              | `0.01`            | BS 正则化半径，避免近场奇异。               |
| `gammaDecay`                         | `0.99`            | 每步 gamma 衰减。                   |
| `backgroundVelocity`                 | `[0,0,0]`         | segment solver 背景速度，当前基本不用于推进。 |
| `segmentAdvectionBackgroundVelocity` | `[0,0,0]`         | segment advect 背景速度，当前关闭。      |


---

## 3. SPH ↔ 点涡 / Ghost 耦合


| 配置项                                 | 当前值                         | 说明                                                                        |
| ----------------------------------- | --------------------------- | ------------------------------------------------------------------------- |
| `sphSegmentFeedbackMode`            | `vortex_ghost_displacement` | ghost 对 SPH 的反馈方式：不直接加速度，而是在 advect 后对 SPH 位置加位移。                         |
| `vortexGhostVelocityCouplingBeta`   | `0.8`                       | ghost feedback 强度，位移为 `dt * beta * u_direct_bs`。                          |
| `vortexGhostFeedbackVelocitySource` | `direct_bs`                 | 在 SPH 粒子位置直接由 ghost/点涡 `gamma` 计算 BS 速度。                                  |
| `sphSegmentBsCoupling`              | `1`                         | 旧 direct BS velocity feedback 参数，当前主要用 `vortexGhostVelocityCouplingBeta`。 |
| `sphSegmentBsFeedbackSkipBoundary`  | `true`                      | 旧 BS feedback 跳过 boundary 段。当前 boundary 已关闭。                              |


当前真正作用在 SPH 上的是：

```text
u_i = sum_j normal_sign * Gamma_j / (2*pi) * [-r_y, r_x] / (|r|^2 + R^2)
x_i += dt * beta * u_i
```

其中 `normal_sign` 由 `pointVortex2DNormalDirection` 控制。

---

## 4. 点涡推进：SPH velocity deposit


| 配置项                                         | 当前值            | 说明                                                |
| ------------------------------------------- | -------------- | ------------------------------------------------- |
| `segmentAdvectionMode`                      | `sph_velocity` | 点涡由 SPH 速度推进。                                     |
| `segmentAdvectionUseBiotSavart`             | `false`        | 点涡推进不使用点涡之间的 BS 速度。                               |
| `segmentAdvectionUseSphVelocity`            | `true`         | 点涡推进使用 SPH velocity deposit。                      |
| `segmentAdvectionUseBackground`             | `false`        | 不叠加背景速度。                                          |
| `segmentSphAdvectButGhostUseBsVelocity`     | `false`        | 不再为 ghost 自身额外计算 BS 速度；SPH feedback 已用 direct BS。 |
| `sphVelocityToSegmentAdvectionEnabled`      | `true`         | 启用 SPH 速度沉积到点涡。                                   |
| `sphVelocityToSegmentAdvectionBlend`        | `1.0`          | 完全使用本步沉积速度。                                       |
| `sphVelocityToSegmentAdvectionScale`        | `1.0`          | 沉积速度缩放。                                           |
| `sphVelocityToSegmentAdvectionSkipBoundary` | `true`         | 沉积时跳过 boundary type。当前 boundary 已关闭。              |


---

## 5. SPH 涡量注入 persistent 点涡


| 配置项                                                 | 当前值                 | 说明                                         |
| --------------------------------------------------- | ------------------- | ------------------------------------------ |
| `sphBoundaryVorticityInjectionEnabled`              | `true`              | 启用从 SPH 涡量注入点涡。                            |
| `sphBoundaryVorticityInjectionIntervalSteps`        | `1`                 | 每步执行注入。增大可提速。                              |
| `sphBoundaryVorticityInjectionStartStep`            | `20`                | 第 20 步后开始注入。                               |
| `sphBoundaryVorticityInjectionThreshold`            | `70.0`              | 注入阈值，涡量低于阈值不注入。                            |
| `sphBoundaryVorticityInjectionMaxPerStep`           | `400`               | 每步最多注入 400 个点涡。                            |
| `sphBoundaryVorticityInjectionUseResidualVorticity` | `true`              | 使用 residual vorticity，扣除已有点涡贡献后再注入。更准但更耗时。 |
| `sphBoundaryVortexResidualSupportRadiusScale`       | `1.0`               | residual 估计支持半径缩放。                         |
| `sphBoundaryVorticityInjectionSource`               | `vorticity`         | 注入 gamma 的 SPH 涡量来源。                       |
| `sphBoundaryVorticityInjectionThresholdSource`      | `vorticity`         | 阈值判断来源。                                    |
| `sphBoundaryVorticityInjectionRegion`               | `all`               | 全域可注入。                                     |
| `sphBoundaryVorticityInjectionOrientation`          | `omega`             | 注入方向按涡量方向。                                 |
| `sphBoundaryVorticityInjectionGammaMode`            | `source_component`  | gamma 使用源涡量分量。                             |
| `sphBoundaryVorticityInjectionDistance`             | `0.035`             | 注入位置偏移距离。                                  |
| `sphBoundaryVorticityInjectionSelection`            | `proportional_sign` | 按符号与强度比例选择候选。                              |
| `sphBoundaryVorticityInjectionSegmentLength`        | `0.02`              | 注入点涡对应的可视化 segment 长度。                     |
| `sphBoundaryVorticityInjectionGammaScale`           | `1.0`               | 注入 gamma 缩放。                               |
| `sphBoundaryVorticityInjectionSegmentTypeId`        | `0`                 | 内部注入点涡 type id。                            |


---

## 6. Segment boundary 功能（当前关闭）


| 配置项                                | 当前值        | 说明                            |
| ---------------------------------- | ---------- | ----------------------------- |
| `enableBoundaryInjection`          | `false`    | 关闭虚拟边界段生成/求解。                 |
| `enableGpuBoundarySolve`           | `false`    | 关闭 GPU boundary solve。        |
| `boundaryProjectionCache`          | `false`    | 关闭 boundary projection cache。 |
| `boundaryInjectionSchedule`        | `disabled` | 不执行 boundary pipeline。        |
| `boundaryReplaceCommittedEachStep` | `false`    | 不每步替换 boundary segments。      |
| `rhsIncludeInternalSegments`       | `false`    | boundary RHS 不包含内部点涡。         |


保留但当前基本不生效的 boundary 参数包括：`boundarySampleSource`、`boundaryCircleCenter`、`boundaryCircleRadius`、`boundarySegmentTypeId`、`numBoundarySamples`、`numGeneratedBoundarySegments`、`boundaryLeastSquaresEps` 等，用于将来重新启用 boundary 时参考。

---

## 7. 点涡删除 / 清理


| 配置项                                              | 当前值       | 说明                     |
| ------------------------------------------------ | --------- | ---------------------- |
| `enableDeleteWeakSegments`                       | `true`    | 启用弱 gamma 删除。          |
| `enableGpuDeleteCompact`                         | `true`    | 使用 GPU compact 删除弱段。   |
| `deleteGammaThreshold`                           | `0.00025` | `abs(gamma)` 低于该阈值则删除。 |
| `deleteInteriorSegmentsInsideObstacles`          | `true`    | 删除障碍物内部点涡。             |
| `deleteInsideObstacleMargin`                     | `0.001`   | 障碍物内部判定 margin。        |
| `deleteInsideObstacleSegmentTypeIds`             | `[0]`     | 只删除内部点涡 type 0。        |
| `deleteInsideObstacleIncludeRigidBlocks`         | `true`    | 检查 rigid block 内点涡。    |
| `deleteInsideObstacleRigidBlockExcludeObjectIds` | `[2]`     | 删除检测排除 object id 2。    |
| `deleteInsideObstacleIncludeCylinders`           | `true`    | 检查圆柱内部点涡。              |
| `deleteSlowPointVorticesEnabled`                 | `true`    | 删除速度过小的滞留点涡。           |
| `deleteSlowPointVortexSpeedThreshold`            | `1e-5`    | 点涡推进速度低于该阈值则删除。        |
| `outflowDeleteCenterBeyondX`                     | `4.00`    | 删除中心超过出流边界的点涡。         |


---

## 8. 拓扑 split / merge 配置

当前是 `point_vortex_2d`，split/merge 对点涡主流程影响较小，多数情况下会跳过。


| 配置项                           | 当前值     | 说明                            |
| ----------------------------- | ------- | ----------------------------- |
| `enableSplitSegments`         | `true`  | 是否允许 split。                   |
| `enableMergeSegments`         | `true`  | 是否允许 merge。                   |
| `splitLengthThreshold`        | `0.12`  | split 长度阈值。                   |
| `mergeDistanceLambda`         | `0.005` | merge 距离阈值系数。                 |
| `mergeAngleThreshold`         | `2.0`   | merge 角度阈值。                   |
| `mergeSpatialHashEnabled`     | `true`  | merge 使用空间哈希。                 |
| `mergeSpatialHashCellSize`    | `0.01`  | merge hash cell 大小。           |
| `mergeIntervalSteps`          | `1`     | merge 间隔。                     |
| `mergeRequireSameSegmentType` | `true`  | 只 merge 相同 type。              |
| `boundarySkipTopology`        | `true`  | boundary 跳过拓扑；当前 boundary 关闭。 |


---

## 9. 初始化 / 周期发射配置（当前不生效）


| 配置项                            | 当前值     | 说明                         |
| ------------------------------ | ------- | -------------------------- |
| `initType`                     | `none`  | 不使用预设 segment 初始化。         |
| `parallelXRepeatEnabled`       | `false` | 不周期性发射 parallel-X segment。 |
| `parallelXRepeatIntervalSteps` | `13`    | 周期发射间隔，当前不生效。              |
| `parallelXRepeatStartStep`     | `13`    | 周期发射起始步，当前不生效。             |
| `parallelXRepeatMaxBatches`    | `0`     | 最大发射批次，当前不生效。              |


其余 `parallelXLayers`、`parallelXFilamentsPerLayer`、`parallelXSegmentsPerLine`、`parallelXDomainStart/End` 等仅在 `initType` 或 repeat 开启时使用。

---

## 10. 可视化 / 导出配置

### 10.1 输出目录


| 配置项                            | 当前值                         | 说明                            |
| ------------------------------ | --------------------------- | ----------------------------- |
| `imageExportSubfolderSph`      | `sph_velocity_deposit`      | SPH PNG 子目录。                  |
| `imageExportSubfolderSegments` | `segments_velocity_deposit` | segment/point vortex PNG 子目录。 |
| `imageExportSubfolderGhost`    | `ghost_velocity_deposit`    | ghost panel PNG 子目录。          |
| `exportSegmentPLY`             | `true`                      | 导出 segment PLY。               |
| `exportPLYAxisConvention`      | `xy`                        | segment PLY 坐标约定。             |


### 10.2 Ghost 图


| 配置项                                  | 当前值           | 说明                                                        |
| ------------------------------------ | ------------- | --------------------------------------------------------- |
| `imageExportVortexGhostPanel`        | `true`        | 导出 ghost panel。                                           |
| `imageVortexGhostPanelIncludeSph`    | `true`        | ghost 图叠加 SPH 粒子。                                         |
| `imageShowVortexGhostVelocityArrows` | `true`        | 显示 ghost 自身速度箭头；当前 direct-BS 模式下 ghost 自身速度置 0，通常不显示黄色箭头。 |
| `imageVortexGhostArrowColor`         | `[255,230,0]` | ghost 自身速度箭头颜色，黄色。                                        |
| `imageShowSphGhostBsVelocityArrows`  | `true`        | 显示 SPH 粒子位置上的 direct-BS feedback 速度。                      |
| `imageSphGhostBsArrowColor`          | `[0,255,80]`  | direct-BS 绿色箭头颜色。                                         |
| `imageSphGhostBsArrowStride`         | `5`           | 每隔几个 SPH 粒子画一个绿色箭头；越小越密，PNG 导出越慢。                         |
| `imageSphGhostBsArrowMaxCount`       | `2200`        | 绿色箭头最大数量。                                                 |


### 10.3 Segment / 点涡图


| 配置项                                    | 当前值                | 说明                                                 |
| -------------------------------------- | ------------------ | -------------------------------------------------- |
| `imageSegmentColormap`                 | `coolwarm`         | 点涡 gamma 色图。                                       |
| `imageSegmentGammaAutoScale`           | `false`            | gamma 色标不自动缩放。                                     |
| `imageSegmentGammaVmin/Vmax`           | `-0.0002 / 0.0002` | gamma 色标范围。                                        |
| `imageSegmentDrawWhiteUnderlay`        | `true`             | 点涡/segment 下方绘制白色底层增强可见性。                          |
| `imageSegmentPanelIncludeSolid`        | `true`             | segment 图显示固体。                                     |
| `imageSegmentPanelDrawCircleObstacle`  | `true`             | segment 图画圆柱。                                      |
| `imageSegmentPanelCircleRadius`        | `0.1`              | 绘图圆柱半径，注意与物理圆柱 `0.08` 不同。                          |
| `imageSegmentPanelOpaqueBackground`    | `true`             | 使用不透明背景。                                           |
| `imageSegmentPanelUseSimulationDomain` | `true`             | 使用仿真域作为显示范围。                                       |
| `imageShowBoundarySegments`            | `true`             | 若存在 boundary type 段则显示；当前 boundary 关闭，一般不会出现黄色边界段。 |


---

## 11. Debug / Timing


| 配置项                                 | 当前值     | 说明                                  |
| ----------------------------------- | ------- | ----------------------------------- |
| `debugHybridSubstepTiming`          | `true`  | 打印 hybrid substep 大阶段耗时。            |
| `debugHybridSubstepTimingInterval`  | `10`    | 每 10 步打印一次。                         |
| `debugHybridSubstepTimingSync`      | `true`  | 每个计时段前后 `ti.sync()`，更准但更慢。正式跑可关。    |
| `debugSegmentAdvanceTiming`         | `true`  | 打印 `advance_segments_coupled` 内部耗时。 |
| `debugSegmentAdvanceTimingInterval` | `10`    | 每 10 步打印一次 segment 内部 timing。       |
| `debugInternalSegmentGamma`         | `false` | 不打印内部 gamma 统计。                     |
| `sphVorticityDebugPrint`            | `false` | 不打印 SPH 涡量 debug。                   |
| `vortexGhostVelocityDebug`          | `false` | 不打印 ghost velocity debug。           |


---

## 12. 几何对象

### FluidEmitters


| 配置项            | 当前值         | 说明              |
| -------------- | ----------- | --------------- |
| `squareCenter` | `[0.0,0.5]` | 左侧入口中心。         |
| `squareSize`   | `[0.0,0.9]` | 入口线段高度。         |
| `velocity`     | `[2.0,0.0]` | 新发射 SPH 粒子入口速度。 |
| `density`      | `1000.0`    | 新粒子密度。          |


### RigidBlocks / Cylinders

- `RigidBlocks` 中 object id `1` 主要是上下壁和右侧边界块。
- object id `2` 是一个被多处 exclude 的辅助 rigid block。
- `Cylinders[0]` 是静态圆柱，中心 `[0.65,0.5,0]`，半径 `0.08`。

---

## 13. 当前性能敏感配置


| 配置项                                                 | 当前值        | 优化提示                             |
| --------------------------------------------------- | ---------- | -------------------------------- |
| `boundaryInjectionSchedule`                         | `disabled` | boundary pipeline 已关闭，节省主要开销。    |
| `sphBoundaryVorticityInjectionIntervalSteps`        | `1`        | 每步注入较贵；增大到 2/3/5 可提速。            |
| `sphBoundaryVorticityInjectionMaxPerStep`           | `400`      | 越大点涡越多，后续推进/删除/ghost 越贵。         |
| `sphBoundaryVorticityInjectionUseResidualVorticity` | `true`     | residual 更准但更耗时。                 |
| `deleteInteriorSegmentsInsideObstacles`             | `true`     | 每步障碍物内删除有成本。                     |
| `deleteSlowPointVorticesEnabled`                    | `true`     | 每步低速删除有成本。                       |
| `imageSphGhostBsArrowStride`                        | `5`        | 绿色箭头较密，只影响 PNG 导出速度。             |
| `debugHybridSubstepTimingSync`                      | `true`     | 精确 timing 增加同步开销，正式跑可设为 `false`。 |


