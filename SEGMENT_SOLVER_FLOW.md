# Segment 求解主流程说明（结合 Vortex Segment 论文）

本文档给出独立 `Segment` 求解器的完整主流程说明，并将流程与论文  
**Incompressible Flow Simulation on Vortex Segment Clouds (TOG 2021)** 的关键思想对应起来。

---

## 1. 方法目标与定位

`Segment` 方法将涡量离散为一组线段（segment cloud），每条线段携带：

- 两端点位置：`x_minus`, `x_plus`
- 涡强标量：`gamma`
- 活跃状态与寿命：`active`, `age`

该表示相比点涡更适合表达细长、各向异性的涡结构（如涡管、尾迹、重连）。

> 对应论文第 4 节：Discrete Vortex Segments。

---

## 2. 当前代码结构（独立 Segment 版本）

当前工程中的独立 Segment 架构：

- `segment_config.py`：读取 `SegmentConfiguration`
- `segment_system.py`：段云数据结构与几何缓存
- `segment_solver.py`：时间推进与拓扑操作主流程
- `segment_boundary.py`：边界虚拟段与线性系统（骨架）
- `segment_export.py`：导出段端点 PLY
- `run_segment_simulation.py`：独立入口与时间循环

---

## 3. 一次时间步 `step()` 的完整流程

以下是推荐并已在骨架中组织好的主流程（按执行顺序）：

1. 更新边界几何（若有）
2. 生成并求解边界虚拟段
3. 计算段端点速度场
4. RK4 对流推进段端点
5. 更新段几何缓存
6. 拓扑操作：split / merge / delete
7. 可选清理：域外段、过老段、远场低贡献段

对应 `segment_solver.py` 的 `step()`。

---

## 4. 与论文关键公式/思想的对应关系

## 4.1 速度计算（论文 4.2）

速度场由背景流与所有段的 Biot-Savart 诱导速度叠加：

\[
u(x)=\sum_j u^{BS}_j(x)+u_{\infty}
\]

实现上对应：

- `compute_endpoint_velocity()`
- 对每条活跃段端点 `x_minus/x_plus` 调用 `u_at_point(x)`

第一版可用 \(O(N^2)\) 直接求和，后续再做网格/FMM 类加速。

---

## 4.2 对流与拉伸（论文 4.3）

3D 情况下通过更新段两端点进行对流（同时体现拉伸）：

\[
\frac{d x_j^-}{dt}=u(x_j^-),\quad \frac{d x_j^+}{dt}=u(x_j^+)
\]

实现上对应：

- `advect_segments_rk4()`（建议 RK4）
- 当前骨架中暂以 Euler 占位，后续替换为 RK4 子步

---

## 4.3 拓扑更新（论文 4.4）

局部重播种操作：

- **split**：段过长则分裂
- **merge**：相近且方向满足条件则合并
- **delete**：涡强过弱则删除

实现上对应：

- `split_segments()`
- `merge_segments()`
- `delete_weak_segments()`

这些操作让段云保持可控规模，并支持重连等拓扑变化。

---

## 4.4 边界处理（论文第 5 节）

边界通过“虚拟段 + 最小二乘”求解满足速度边界条件：

\[
K\Gamma = U,\quad
\Gamma = (K^T K + \epsilon I)^{-1}K^T U
\]

实现上对应 `segment_boundary.py`：

1. `update_boundary_pose()`
2. `generate_boundary_segments()`
3. `compute_k_matrix()`
4. `compute_rhs()`
5. `solve_linear_system()`
6. `commit_boundary_segments()`

---

## 5. 伪代码（建议版本）

```python
def step():
    if has_boundary:
        boundary.update_boundary_pose()
        boundary.generate_boundary_segments()
        boundary.compute_k_matrix()
        boundary.compute_rhs()
        boundary.solve_linear_system()
        boundary.commit_boundary_segments()

    compute_endpoint_velocity()   # 内部段 + 边界段
    advect_segments_rk4()         # 端点 RK4 对流

    update_segment_geometry()
    split_segments()
    merge_segments()
    delete_weak_segments()
    cull_segments()
```

---

## 6. `SegmentConfiguration` 参数建议

关键参数及作用：

- `timeStepSize`：时间步长
- `simulationTime`：总时长
- `segmentMaxNum`：段容量上限
- `regularizationRadiusR`：BS 正则半径
- `backgroundVelocity`：背景流
- `splitLengthThreshold`：分裂阈值
- `mergeDistanceLambda`：合并距离阈值
- `mergeAngleThreshold`：合并方向阈值
- `deleteGammaThreshold`：删除阈值
- `gammaDecay`：每步衰减
- `enableBoundaryInjection`：是否开启边界注涡
- `exportInterval`：导出间隔
- `exportPLY`：是否导出

---

## 7. 推荐实现顺序（从可跑到可用）

1. **先打通主循环**：背景流 + Euler 占位（已完成）
2. **补初始段生成**：`seed_initial_segments()`
3. **实现 BS 速度**：`compute_endpoint_velocity()`
4. **替换 RK4**：`advect_segments_rk4()`
5. **实现 split/delete**（先不 merge）
6. **实现 merge**
7. **实现边界最小二乘注涡**
8. **最后做性能加速**（网格/多层近似）

---

## 8. 当前状态总结

当前代码已完成：

- 独立架构拆分
- 主流程调用顺序骨架
- 配置与导出通路
- 边界流程函数拆分

当前未完成（待填）：

- 段初始化
- BS 诱导速度
- RK4 正式实现
- split/merge 细节
- 边界线性系统求解与写回

这意味着：**工程框架已就位，下一步进入“物理细节逐项实现”阶段。**

---

## 9. 引入 DIMCV 式蒙特卡洛采样以加速 Segment-BS（建议）

对于 `Segment` 方法，Biot-Savart 求和通常是主要瓶颈。  
可借鉴 DIMCV 的重要性采样思想，将 BS 计算改为：

- **近场精确求和**
- **远场蒙特卡洛估计**

从而在维持主要精度的同时降低计算成本。

### 9.1 核心分解

对查询点（段端点）`x`：

\[
u(x)=\sum_{j \in \mathcal{N}(x)} u_j^{BS}(x)\;+\;\sum_{j \in \mathcal{F}(x)} u_j^{BS}(x)
\]

其中：

- \(\mathcal{N}(x)\)：近场段集合（精确求和）
- \(\mathcal{F}(x)\)：远场段集合（采样估计）

远场无偏估计器：

\[
\hat u_{\text{far}}(x)=\frac{1}{M}\sum_{m=1}^{M}\frac{u_{j_m}^{BS}(x)}{p(j_m|x)}, \quad j_m \sim p(\cdot|x)
\]

最终：

\[
\hat u(x)=u_{\text{near}}(x)+\hat u_{\text{far}}(x)+u_\infty
\]

---

### 9.2 重要性分布设计

建议按“贡献强度”构造采样权重：

\[
w_j = |\gamma_j| \cdot L_j \cdot \phi(r_{xj}), \quad
p_j = \frac{w_j}{\sum_k w_k}
\]

可选衰减函数：

- \(\phi(r)=1/(r^2+R^2)\)
- 或 \(\phi(r)=1/(r^2+R^2)^{3/2}\)（更强距离衰减）

其中：

- \(\gamma_j\)：段强度
- \(L_j\)：段长度
- \(r_{xj}\)：查询点到段中心距离
- \(R\)：正则半径

---

### 9.3 推荐实现：两级采样（更高效）

直接全局按段采样开销仍高，推荐：

1. 先按网格/桶把段分组（cells）
2. 先采样远场 cell
3. 再在 cell 内采样 segment

这样可显著减少构建与更新采样分布的成本，且更适合 GPU 并行。

---

### 9.4 方差与稳定性控制

建议同时使用以下策略：

- 固定每端点采样数 `M`（例如 16/32）
- 近场始终精确（不要采样）
- 对估计速度做幅值上限裁剪（防偶发大值）
- 保持 BS 正则半径 `R > 0`
- 可加入“最强若干段确定性 + 其余MC”混合策略

---

### 9.5 在当前代码中的落地位置

主要改造 `segment_solver.py` 的 `compute_endpoint_velocity()`：

1. 保留全求和基线版本（便于误差对比）
2. 增加开关 `useMCSegmentBS`
3. 计算流程改为：
   - `u_near_exact(x)`
   - `u_far_mc(x, M)`
   - `u = u_inf + u_near_exact + u_far_mc`
4. 对 `x_minus/x_plus` 两端点统一使用该流程

---

### 9.6 建议新增配置项（SegmentConfiguration）

可在 `SegmentConfiguration` 中新增：

- `useMCSegmentBS`：是否启用 MC 远场估计
- `nearFieldRadius`：近场半径
- `mcSamplesPerEndpoint`：每端点采样数
- `mcDistancePower`：距离权重幂次（例如 1 或 1.5）
- `mcVelocityClamp`：速度裁剪阈值
- `mcUseTwoLevelSampling`：是否启用两级采样

---

### 9.7 推荐推进顺序

1. 先完成全求和 BS，建立性能/误差基线
2. 仅将远场替换为 MC（近场仍精确）
3. 引入重要性分布（\(|\gamma|L\phi(r)\)）
4. 引入两级采样与自适应采样预算
5. 最后再调参数做误差-速度平衡

该路径风险最低，且便于逐步验证。

