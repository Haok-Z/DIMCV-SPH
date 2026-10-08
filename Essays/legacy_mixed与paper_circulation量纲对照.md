# `legacy_mixed` 与 `paper_circulation` 的量纲与强度对照

## 结论

`legacy_mixed` 看起来不过强，不是因为它采用了更温和的速度混合。它和当前方法一样使用：

```json
"sphSegmentFeedbackMode": "bs_direct",
"sphSegmentBsCoupling": 0.6
```

`legacy_mixed` 中的 `mixed` 指的是 `gamma` 的物理意义前后混用，不是 velocity mixing。

真正原因是：旧版 `gamma` 先被当作涡量体积投影量保存，随后有限段 Biot-Savart 又把它乘以段长作为环量使用。这样在离散涡量向量中额外引入了两次段长因子，使每条新生涡段的 Biot-Savart 强度被压低约 `3086` 倍。

## 符号定义

设：

- $\boldsymbol{\omega}_{res}$：用于 Segment 注入的 SPH 残差涡量，单位为 $1/s$；
- $\mathbf{t}$：涡段单位切向量，无量纲；
- $V$：SPH 粒子体积，单位为 $m^3$；
- $L$：涡段长度，单位为 $m$；
- $\mathbf{d}=L\mathbf{t}$：涡段有向长度向量，单位为 $m$；
- $\Gamma$：有限涡段 Biot-Savart 公式使用的物理环量，单位为 $m^2/s$；
- $\gamma$：代码中存储的 Segment 强度字段，具体物理意义取决于约定。

将 SPH 残差涡量投影到 Segment 方向并乘以粒子体积，定义：

$$
B=(\boldsymbol{\omega}_{res}\cdot\mathbf{t})V
$$

其单位为：

$$
[B]=\frac{1}{s}m^3=\frac{m^3}{s}
$$

这里的 $B$ 是涡量体积投影量，不是有限涡段公式所需的环量 $\Gamma$。

## `legacy_mixed`

### 注入时保存的 `gamma`

旧版注入直接保存：

$$
\gamma_{legacy}=B
$$

因此：

$$
[\gamma_{legacy}]=\frac{m^3}{s}
$$

它实际上仍然是“涡量乘体积”的量，而不是环量。

### 有限段 Biot-Savart 中使用的环量

legacy 分支在有限段公式中执行：

$$
\Gamma_{legacy}=\gamma_{legacy}L=BL
$$

对应代码：

[segment_solver.py](D:/Simulation/SourceCode/DIMCV-SPH/segment_solver.py:1251)

即：

```python
circulation = gamma * length
```

这样得到的量纲为：

$$
[\Gamma_{legacy}]
=
\frac{m^3}{s}m
=
\frac{m^4}{s}
$$

这与 Biot-Savart 所需的环量单位 $m^2/s$ 不一致。

### 对离散涡量向量的影响

一个有限涡段表示的离散涡量体积向量应满足：

$$
\Gamma\mathbf{d}
\approx
\boldsymbol{\omega}_{res}V
$$

但 legacy 实际得到：

$$
\Gamma_{legacy}\mathbf{d}
=
(BL)(L\mathbf{t})
=
BL^2\mathbf{t}
$$

因此，相对于目标投影量 $B\mathbf{t}$，legacy 只保留：

$$
L^2
$$

的强度。

## `paper_circulation`

### 注入时保存的 `gamma`

论文环量约定先将涡量体积投影量转换为环量：

$$
\gamma_{paper}
=
\frac{B}{L}
$$

对应代码：[dfsph_segment_hybrid_solver.py](D:/Simulation/SourceCode/DIMCV-SPH/dfsph_segment_hybrid_solver.py:819)

因此：

$$
[\gamma_{paper}]
=
\frac{m^3/s}{m}
=
\frac{m^2}{s}
$$

这与物理环量 $\Gamma$ 的单位一致。

### 有限段 Biot-Savart 中使用的环量

在 `paper_circulation` 下，有限段公式直接使用：

$$
\Gamma_{paper}=\gamma_{paper}=\frac{B}{L}
$$

因此：

$$
\Gamma_{paper}\mathbf{d}
=
\frac{B}{L}(L\mathbf{t})
=
B\mathbf{t}
=
(\boldsymbol{\omega}_{res}\cdot\mathbf{t})V\mathbf{t}
$$

这与 SPH 残差涡量的方向投影和体积积分相匹配。

## 两种约定的强度比

对于相同的 $\boldsymbol{\omega}_{res}$、$V$、$\mathbf{t}$ 和 $L$：

$$
\frac{\Gamma_{paper}}{\Gamma_{legacy}}
=
\frac{B/L}{BL}
=
\frac{1}{L^2}
$$

当前注入段长度为：

$$
L=0.018
$$

所以：

$$
L^2=0.018^2=0.000324
$$

legacy 只保留目标强度的：

$$
0.000324=0.0324\%
$$

反过来，paper 相对于 legacy 的强度倍率为：

$$
\frac{1}{0.018^2}
\approx
3086.42
$$

有限段 Biot-Savart 对环量 $\Gamma$ 是线性的。因此在几何位置、正则化半径和 Segment 数量都相同的情况下：

$$
\frac{\|\mathbf{u}_{paper}^{BS}\|}
{\|\mathbf{u}_{legacy}^{BS}\|}
\approx
3086.42
$$

这解释了为什么 legacy 实验看起来很弱，而切换到 paper 环量后 Segment 对 SPH 的影响会突然变强。

## 代码路径对照

### legacy

```text
SPH 残差涡量
  -> B = (omega_res dot t) V
  -> gamma = B
  -> Gamma_BS = gamma * L
  -> u_BS
  -> v_SPH += 0.6 u_BS
```

### paper

```text
SPH 残差涡量
  -> B = (omega_res dot t) V
  -> gamma = B / L
  -> Gamma_BS = gamma
  -> u_BS
  -> v_SPH += 0.6 u_BS
```

两条路径的最后一步相同，都是直接 BS 速度反馈；差异主要发生在 `gamma` 到 `Gamma` 的转换。

## 分裂与合并时的补充说明

在 `paper_circulation` 和 `residual_circulation` 下，`gamma` 直接表示 $\Gamma$。

分裂时，两个子段继承父段的 `gamma`，但各自长度为原来的一半：

$$
\Gamma\mathbf{d}_1+\Gamma\mathbf{d}_2
=
\Gamma\mathbf{d}
$$

因此物理环量向量守恒。

合并时，代码先计算：

$$
\boldsymbol{\beta}_i=\Gamma_i\mathbf{d}_i,
\qquad
\boldsymbol{\beta}_j=\Gamma_j\mathbf{d}_j
$$

再令：

$$
\boldsymbol{\beta}_{new}
=
\boldsymbol{\beta}_i+\boldsymbol{\beta}_j
$$

并根据新段长度重新构造：

$$
\Gamma_{new}
=
\frac{\|\boldsymbol{\beta}_{new}\|}{L_{new}}
$$

所以合并可能改变单段 `gamma`，但目标是保持总的物理环量向量。

## 最终判断

`legacy_mixed` 的“不过强”主要来自单位混用造成的数值削弱，而不是更合理的物理耦合。对于物理量自洽的实现，应使用：

$$
\Gamma
=
\frac{(\boldsymbol{\omega}_{res}\cdot\mathbf{t})V}{L}
$$

并在有限段 Biot-Savart 中直接使用这个 $\Gamma$。如果希望降低 paper 版本的反馈强度，应显式调节反馈系数、残差注入量、Segment 数量或速度限幅，而不应依赖 legacy 的长度单位错误。
