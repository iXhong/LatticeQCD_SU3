# Polyakov 环与 Polyakov 环关联函数

## Polyakov 环定义

Polyakov 环（也叫热 Wilson 线）是沿时间方向缠绕一圈的 Wilson 线，利用周期性边界条件闭合。

### 教科书定义（Gattringer & Lang, Eq. 3.60）

$$ P(m) = \operatorname{tr} \left[ \prod_{j=0}^{N_T-1} U_4(m, j) \right] $$

- $m$: 空间坐标
- $N_T$: 时间方向格点数
- $U_4(m, j)$: 格点 $(m, j)$ 上的时间方向连接矩阵
- 迹保证规范不变性

### 代码实现（observables.py:59-88）

```python
wilson_line = np.eye(3, dtype=np.complex128)    # 从单位矩阵开始
for time_coord in range(temporal_extent):         # 沿时间方向走
    site = geometry.index_from_coord(full_coords)
    wilson_line = wilson_line @ links[site, time_direction]  # 连乘
return complex(np.trace(wilson_line) / 3.0)      # 归一化后的迹
```

**注意**：教科书定义是纯迹 $\operatorname{tr}[W]$，代码返回 $\operatorname{tr}[W]/3$（归一化版本，对 SU(3) 最大值 1）。两种约定都常见，保持一致即可。

### 物理意义

Polyakov 环对应一个静态夸克的自由能：
$$ \langle P \rangle \sim e^{-\beta F_q} $$

作为**退禁闭相变序参量**：
- **禁闭相**：$\langle P \rangle = 0$（中心对称性 Z$_3$ 未破缺）
- **退禁闭相**：$\langle P \rangle \neq 0$（中心对称性自发破缺）

### 单格点 vs 全格

- `polyakov_loop()`: 输入一个空间坐标，返回一个复数
- `polyakov_loops()`: 遍历所有空间格点，返回一个复数数组

"一个格点的 Polyakov 环" = 固定空间位置，沿时间方向走一圈。

---

## Wilson 线（直观理解）

Wilson 线是把一串连接矩阵 $U_\mu(x)$ 依次乘起来。每个 $U$ 描述夸克从一个格点沿某方向跳到相邻格点时的颜色荷旋转。

```
t=0       t=1       t=2     ...   t=Nt-1
  ●────────●────────●── ... ──●
  U(x,0)   U(x,1)   U(x,2)   U(x,Nt-1)
```

对比：
- **Wilson line**: 开的路径，不闭合
- **Wilson loop**: 任意闭合回路
- **Polyakov 环**: 沿时间方向直穿整个格子，靠周期性边界条件闭合的特殊 Wilson loop

---

## Polyakov 环关联函数

### 定义

两个不同空间位置的 Polyakov 环的关联：
$$ \langle P(m) P(n)^\dagger \rangle $$

物理意义：静态夸克（在 $m$）和反夸克（在 $n$）对的自由能。

### 提取静态势能

$$ \langle P(m) P(n)^\dagger \rangle \propto e^{-N_T a V(r)} \quad (r = a|m-n|) $$

$$ a V(r) = -\frac{1}{N_T} \ln \langle P(m) P(n)^\dagger \rangle $$

### 势能参数化（Eq. 3.62）

$$ V(r) = A + \frac{B}{r} + \sigma r $$

- **$A$**: 无关的能量偏移
- **$B/r$**: 库仑项（单胶子交换，短距主导）
- **$\sigma r$**: 线性禁闭项，$\sigma$ 为弦张力（$\sigma \approx 900\ \text{MeV/fm}$）

### 大类距离行为

$$ \lim_{r \to \infty} \langle P(m) P(n)^\dagger \rangle = |\langle P \rangle|^2 $$

- 禁闭相：$\langle P \rangle = 0 \implies$ 关联 $\to 0$
- 退禁闭相：$\langle P \rangle \neq 0 \implies$ 关联趋于常数

### 实际测量步骤（Gattringer & Lang, Sec. 4.5.4）

1. 对每个构型，用 `polyakov_loops()` 算所有空间格点的 Polyakov 环
2. 按距离 $|m-n|$ 分组，计算 $\langle P(m) P(n)^\dagger \rangle$
3. 拟合 $-\ln\langle P(m) P(n)^\dagger \rangle / N_t$ 得到 $a V(r)$
4. 拟合 $V(r) = A + B/r + \sigma r$ 提取弦张力 $\sigma$ 和 Sommer 标度 $r_0$

---

## 参考

- Gattringer & Lang, *Quantum Chromodynamics on the Lattice*, Sec. 3.3.5, 3.4, 4.5.4, 12.1.1
- 代码: `src/lattice_su3/observables.py`
- 参考笔记: `reference/polyakov_loop_static_potential.md`
