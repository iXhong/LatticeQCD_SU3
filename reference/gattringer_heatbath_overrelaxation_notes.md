# Heatbath 与 Overrelaxation 算法笔记

来源：Christof Gattringer and Christian B. Lang, *Quantum Chromodynamics on the Lattice*, Chapter 4, especially Sec. 4.1.4, 4.3.1, 4.3.2, and 4.4.2.

用途：供后续 Codex 实现 pure gauge Wilson action 下的 SU(2)/SU(3) 组态更新时参考。

## 1. 背景符号

Wilson gauge action 的单链更新只影响包含该 link 的局域 plaquette。对一条待更新链

```text
U = U_mu(n)
```

将周围所有 staples 求和，记为

```text
A = sum_i P_i
```

其中每个 `P_i` 是与 `U_mu(n)` 共同组成一个 plaquette 的另外三条 link 的有序乘积。四维中一条 link 属于 6 个 plaquette，因此有 6 个 staples。

局域 action 可写成

```text
S_loc[U] = (beta / N) Re Tr[6 I - U A]
```

若只改变这条链 `U -> U'`，action 变化为

```text
Delta S = S_loc[U'] - S_loc[U]
        = -(beta / N) Re Tr[(U' - U) A]
```

这里 `A` 在更新 `U` 时保持不变。heatbath 和 overrelaxation 都从这个局域形式出发。

局域 Boltzmann 权重为

```text
dP(U) = dU exp[(beta / N) Re Tr(U A)]
```

其中 `dU` 是规范群上的 Haar measure。

## 2. Heatbath 算法的定义

Heatbath 的思想是：不先提议候选链再 Metropolis 接受/拒绝，而是直接从固定邻居背景 `A` 所定义的局域概率分布中抽取新的 link：

```text
U' ~ dU exp[(beta / N) Re Tr(U A)]
```

因此新的 link 总是被接受。Gattringer 也指出：无限多次 multi-hit Metropolis 的极限等价于 heatbath。

## 3. SU(2) Heatbath 实现

SU(2) 情况特殊，因为 staples 之和 `A` 与某个 SU(2) 矩阵成比例。写作

```text
A = a V
a = sqrt(det A)
V = A / a
```

若 `det A = 0`，则直接随机取一个 SU(2) 矩阵。

定义

```text
X = U V
```

利用 Haar measure 的不变性，可将抽样问题转化为抽取

```text
dP(X) = dX exp[(a beta / 2) Re Tr X]
```

然后得到新链

```text
U' = X V^dagger = X A^dagger / a
```

### 3.1 SU(2) 矩阵参数化

用四维实向量表示 SU(2)：

```text
X = x0 I + i x . sigma
x0^2 + |x|^2 = 1
```

抽样分布可化为

```text
dP(X) = (1 / 2 pi^2) dcos(theta) dphi dx0
        sqrt(1 - x0^2) exp(a beta x0)
```

因此：

- `x0` 按 `sqrt(1 - x0^2) exp(a beta x0)` 分布抽样；
- `x` 的方向在三维球面上均匀抽样；
- `|x| = sqrt(1 - x0^2)`。

### 3.2 Gattringer 给出的 x0 抽样方法

引入变量

```text
x0 = 1 - 2 lambda^2,  lambda in [0, 1]
```

先从

```text
p1(lambda) = lambda^2 exp(-2 a beta lambda^2)
```

抽样，再用

```text
p2(lambda) = sqrt(1 - lambda^2)
```

做接受/拒绝修正。

实际步骤：

1. 取 `r1, r2, r3` 为 `(0, 1]` 上均匀随机数。
2. 计算

   ```text
   lambda^2 = -[ln(r1) + cos^2(2 pi r2) ln(r3)] / (2 a beta)
   ```

3. 再取一个均匀随机数 `r in [0, 1)`，若

   ```text
   r^2 <= 1 - lambda^2
   ```

   则接受该 `lambda`，否则回到第 1 步。
4. 令

   ```text
   x0 = 1 - 2 lambda^2
   |x| = sqrt(1 - x0^2)
   ```

5. 在三维方向上均匀抽取单位向量 `n_hat`，令

   ```text
   x = |x| n_hat
   ```

6. 构造

   ```text
   X = x0 I + i x . sigma
   U' = X V^dagger
   ```

## 4. SU(3) Pseudo Heatbath

Gattringer 说明没有一个直接生成 SU(3) link 的简单 heatbath 算法。实际常用 Cabibbo-Marinari 方法：把 SU(3) 更新分解为若干嵌入的 SU(2) 子群更新。

常用三个 SU(2) 子群嵌入：

```text
R acts on color indices (1, 2)
S acts on color indices (1, 3)
T acts on color indices (2, 3)
```

设当前 link 为 `U`，staple 和为 `A`。若先用 `R` 左乘更新，则局域权重指数为

```text
(beta / 3) Re Tr[R U A]
```

记

```text
W = U A
```

则 `R` 的 heatbath 只依赖 `W` 中与 `(1, 2)` 子块对应的四个元素。把这个子块当作 SU(2) heatbath 中的有效 staple，按 SU(2) 方法生成 `R`。然后依次更新：

```text
W <- R U A
生成 S
W <- S R U A
生成 T
U' = T S R U
```

实际实现时，一个 SU(3) link update 通常可以写成：

```text
for pair in [(0, 1), (0, 2), (1, 2)]:
    W = U @ A
    extract 2x2 active block from W for this pair
    generate SU(2) heatbath matrix h from the active block
    embed h into a 3x3 identity matrix H
    U = H @ U
```

注意：书中公式写为 `U -> U' = T S R U`，因此这里采用左乘更新。

## 5. Overrelaxation 算法的定义

Overrelaxation 的目标是尽可能大地移动 configuration，同时保持局域 action 不变。因此新链 `U'` 的 Boltzmann 权重与旧链 `U` 相同，接受率为 1。

它是 microcanonical 更新：只在等 action 曲面上移动，因此单独使用不遍历 canonical ensemble。必须与 heatbath 或 Metropolis 这类 ergodic 更新结合。

## 6. U(1) Overrelaxation 直观例子

令

```text
U = exp(i phi)
A = a exp(i alpha)
```

局域权重指数为

```text
beta Re(U A) = beta a cos(phi + alpha)
```

反射

```text
phi + alpha -> -(phi + alpha)
```

等价于

```text
phi -> 2 pi - 2 alpha - phi
```

该变换保持局域 action 不变，因此自动接受。

## 7. SU(2) Overrelaxation 实现

对非阿贝尔群，Gattringer 给出的 ansatz 是

```text
U' = V^dagger U^dagger V^dagger
```

其中 `V` 要选得使 action 不变。对 SU(2)，仍利用

```text
A = a V
a = sqrt(det A)
V = A / a
```

于是

```text
U' = V^dagger U^dagger V^dagger
```

并且有

```text
Tr[U' A] = Tr[U A]
```

因此局域 action 不变，更新自动接受。若 `det A = 0`，可接受任意随机 link。

实现清单：

```text
for each link U_mu(n):
    A = staple_sum(U, n, mu)
    a = sqrt(det(A))
    if a is close to 0:
        U_mu(n) = random_su2()
    else:
        V = A / a
        U_mu(n) = V^dagger @ U_mu(n)^dagger @ V^dagger
```

## 8. SU(3) Overrelaxation

Gattringer 明确提醒：SU(3) overrelaxation 更复杂，效率未必好；书中没有像 SU(2) 那样给出一个简单的完整实现。实践中若要在 SU(3) 里用 overrelaxation，通常也沿用 SU(2) 子群嵌入思路，对 `(1,2)`, `(1,3)`, `(2,3)` 子群逐个做 SU(2) overrelaxation。

但要注意：

- overrelaxation 不能单独作为 SU(3) 组态生成算法；
- 它必须和 heatbath 或 Metropolis sweep 混合；
- 如果实现复杂度较高，课程项目中可以先实现 pseudo heatbath，之后再加 overrelaxation 加速 decorrelation。

## 9. 推荐 sweep 组织方式

Gattringer 在 Sec. 4.4.2 中提到，为提高 Markov chain 的步长，常组合不同算法。例如：

```text
one combined sweep =
    1 heatbath sweep
    + 2 or 3 overrelaxation sweeps
```

一个典型 pure gauge Monte Carlo 程序结构：

```text
initialize RNG
initialize neighbor tables
initialize gauge field: cold start or hot start

for i in range(n_equil):
    update_configuration()

for i in range(n_measure):
    for j in range(n_discarded):
        update_configuration()
    measure_observables()
    save measurements
```

其中 `update_configuration()` 可以是：

```text
heatbath_sweep()
overrelaxation_sweep()
overrelaxation_sweep()
```

## 10. 实现时的检查点

### 群性质检查

每次更新后检查：

```text
U^dagger U ≈ I
det U ≈ 1
```

数值误差积累后需要 reunitarization。SU(3) 可用 Gram-Schmidt 正交化重建三行。

### Action/plaquette 检查

对 heatbath：

- 不需要 Metropolis 接受/拒绝；
- plaquette 的 Markov history 应在 thermalization 后达到平台；
- cold start 和 hot start 的 plaquette 曲线应逐渐靠近。

对 overrelaxation：

- 单次局域更新应满足 `S_loc[U'] - S_loc[U] ≈ 0`；
- 只做 overrelaxation 时 action 不应按 Boltzmann 分布采样；
- 必须与 heatbath/Metropolis 混合使用。

### 随机数检查

SU(2) heatbath 的 `lambda` 抽样中：

- `r1, r2, r3` 不能取 0，因为需要 `ln(r)`；
- 若随机数生成器给 `[0,1)`，可用 `1 - r` 转成 `(0,1]`；
- 若 `lambda^2 > 1`，自动不满足接受条件，应重新抽样。


## 11. 与 Metropolis 的关系

Metropolis：

```text
提议 U'，然后按 min(1, exp(-Delta S)) 接受
```

Heatbath：

```text
直接从局域条件分布 dP(U) 抽取 U'，总是接受
```

Overrelaxation：

```text
构造保持局域 action 不变的 U'，总是接受，但单独不 ergodic
```

因此，对 SU(3) Wilson gauge action 的课程项目，一个稳妥路线是：

```text
先实现 Metropolis 或 pseudo heatbath
用 plaquette history 验证 thermalization
再加入 overrelaxation sweep 降低自相关
```

