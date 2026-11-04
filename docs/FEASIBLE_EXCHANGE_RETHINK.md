# 重新审视 Feasible-Exchange Diffusion：从支持集问题走向可核查的策略改进

审计日期：2026-11-04。本文是研究设计与数学检查，不是已完成算法或 RL 性能结果。没有启动训练，没有修改现有训练源码或远端任务。便宜的枚举检查见 `analysis/feasible_exchange_sanity.py`。

## 当前判断

“在可行解上做 swap，再用奖励训练”不足以形成强创新。更值得研究的命题是：

> 在完整可行动作图上，用满足时间调和关系的标量势，表示相对旧策略的终点概率改进；显式区分动作支持集、探索基准分布，以及生成路径上的额外正则化。

这个方向有明确动机，但其中的 Doob 变换、CTMC、指数加权和 KL 分解都是已有数学工具。真正有可能形成贡献的是：**证明并量化 solver 策略的支持集缺口，构造能补全这些动作且每一步可行的生成策略，并把可测的生成误差与动作层面的策略改进连接起来。** 需要在相同支持集、相同 critic 和相同算力下超越普通可行采样器，才能说明 diffusion 本身有价值。

当前 intervention 的线性 decoder 使用 `W.T @ c`。重复列导致同组 primitive actions 得分完全相同；一般位置的系数无法任意指定多个组的部分激活计数。这与“某个可行动作概率很小”不同：部分有效行为可能根本没有正测度的 latent 原像。完整计数与可达性证明由本次独立的支持集审计负责，本文不把直觉当作已验证的数量结论。

如果该缺口确实覆盖高价值动作，调 latent density、校正 normal-cone 体积或减少生成误差都不能修复零支持。可行交换图提供的是新的动作表达能力。但 masked autoregressive policy、全支持的直接离散 policy 也可能做到，因此它们是必须的对照。

**这个支持缺口只针对当前 `W` 特征化，不能推广成所有 linear binary solver policies 的缺陷。** 若直接生成 20 维 primitive costs，任意可行动作 a 都可由 `c=2a-1` 唯一选中，且有非零大小的对应区域。必须增加“native DPMD 更新规则 + primitive 20 维 decoder + sphere/diffusion”的强控制；早先同时改变 critic 和更新规则的 primitive harness 不能替代它。

全路径可行也不是单独足够强的优势：环境只执行最终动作，而旧 solver 本来就保证最终可行。要证明的是相同完整支持集和总成本下的混合、探索、分布建模或长期回报收益。

## 先把生成过程定义正确

本任务动作集为

\[
\mathcal A=\{a\in\{0,1\}^{20}:|a|\leq10\},\qquad |\mathcal A|=616666.
\]

用完整可行动作作为图节点，只允许以下边：

- birth：未达到预算时加入一个未选元素；
- death：删除一个已选元素；
- swap：交换一个已选和一个未选元素。

Birth/death 本身已经保证连通：从任意动作先删到空集，再加到另一动作。Swap 可缩短距离、改善局部移动，但单独使用 swap 会把图分成不同 cardinality 的分量，不能探索低于预算的动作。

在含这三种边的图中，cardinality 为 k 的节点度数是

\[
d(k)=\begin{cases}20+k(20-k),&k<10,\\10+10(20-10),&k=10.\end{cases}
\]

最大度数为 119，不是固定常数。若每步均匀选邻居，转移 `P(a,b)=1/d(a)` 的平稳分布正比于节点度数，而不是均匀动作分布。若使用对称的连续时间率

\[
q^0(a,b)=\kappa\,\mathbf1\{a\sim b\},\quad
q^0(a,a)=-\kappa d(a),
\]

则均匀分布满足详细平衡。二者差异来自 CTMC 的停留时间：记录每次 jump 后的状态，仍然会看到 degree-biased embedded jump chain；必须按固定物理时间观察 CTMC，才能使用其平稳分布结论。

离散实现可以选择统一化：取 `Lambda >= max_a sum_{b!=a} q(a,b)`，使用 `P=I+Q/Lambda`，允许 self-loop，并在 Poisson 时间点更新。或者对均匀邻居 proposal 使用接受概率 `min(1,d(a)/d(b))` 的 MH 校正，再加 lazy self-loop。它们都是已有采样技术，不是创新点。离散 diffusion 统一化已有专门收敛分析：[Chen and Ying, 2024](https://arxiv.org/abs/2402.08095)。

“均匀可行动作”也不等于“均匀选 cardinality”：前者的 `Pr(|a|=k)` 正比于 `binom(20,k)`。此任务中 `Pr(|a|=10)≈0.2996`。若实验希望 cardinality 均匀，需要明确指定另一基准测度，不能无意中混用。

## 两种时间方向，不能混写

固定环境状态 s，在单次动作生成过程中 s 不变；生成时间 t 与环境时间步不是同一概念。

**正向 noising 与反向生成。** 从目标动作分布 `rho_0` 出发，以 `q_t` 加噪，得到 `rho_t`。令反向时钟 `u=T-t`，从 `rho_T` 生成回 `rho_0` 的精确反向率为

\[
q^{\rm rev}_u(a,b)=q_{T-u}(b,a)
\frac{\rho_{T-u}(b\mid s)}{\rho_{T-u}(a\mid s)},\qquad b\ne a.
\]

对称 base rates 下，可以用一个标量网络 `v_theta(s,t,a)` 表示 log-density potential，设置边比率为 `exp(v(b)-v(a))`。这自动保证任意闭环上的 log-ratio 之和为零。但有限 noising 时间下 `rho_T` 通常不等于平稳分布；直接把初始反向样本换成均匀分布会产生近似误差。图连通不等于已经混合。

**相对旧生成过程的 Doob 改进。** 现在令 `R_s` 表示旧策略从 0 到 T 的生成路径分布，旧生成率为 `q_old,t`，终点分布为 `pi_old`。设

\[
f_s(a)=\exp(Q_\phi(s,a)/\tau),\qquad
h_t(s,a)=\mathbb E_{R_s}[f_s(X_T)\mid X_t=a].
\]

这里 t 沿生成方向增加，`h_T=f`，并满足 backward Kolmogorov 方程

\[
\partial_t h_t(a)+\sum_{b\ne a}q_{{\rm old},t}(a,b)
[h_t(b)-h_t(a)]=0.
\]

Doob 改进率是

\[
q^+_t(a,b)=q_{{\rm old},t}(a,b)\frac{h_t(b)}{h_t(a)}.
\]

这里的 `h` 是终点倾斜的条件期望，不是上段的 `rho`。把它们统称为 score 而不交代参考过程，会掩盖不同的训练问题。

## 一个重要的初始分布陷阱，以及更简洁的设计

若旧路径起点为 `mu_0`，直接把整个路径 law 倾斜为

\[
dP^*/dR=f(X_T)/Z
\]

会同时改变起点分布：

\[
P^*_0(a)=\mu_0(a)h_0(a)/Z.
\]

因此，“只用 Doob rates，但仍从原来的随机 `mu_0` 开始”一般不能精确实现 `pi_old(a) exp(Q(a)/tau)/Z`。保持初始分布的多边缘 bridge 问题更复杂；不能用一个 terminal potential 自动解决。

一个适合当前约束的干净选择是：**每次生成从确定的可行动作 `a_ref(s)` 出发，本任务取空集。** 此时 `h_0` 在唯一初始节点是常数，改变 rates 即可实现精确的 terminal tilt。整个过程始终处于完整可行动作集合，可包含 birth/death/swap 循环；它不是只增不减的顺序构造。

这是一种 anchored feasible CTMC generative policy。它不必伪装成从均匀噪声反向去噪的标准 diffusion。代价是有限时间内的距离偏置：从空集达到 cardinality k 至少需要 k 次 birth。正的边率保证理论支持，不保证实际预算下会见到这些动作。必须测量 jump 数、覆盖率、稀有动作命中率以及起点敏感性。

理论 CTMC 在有限时间内仍可能发生任意有限次跳跃；若实现另加硬上限 J，就不能直接沿用该全支持结论。从空集最多做 J 次 birth/death/swap 时，cardinality 大于 J 的动作不可达。并行独立执行多个 flip 也不自动保持预算可行，必须串行处理合法 moves，或另行证明批量转移核可行。

## 可能值得写成论文核心的结构：调和势与终点 KL

在冻结的旧生成器上学习一个标量势 `u_theta(s,t,a)`，定义

\[
q_\theta(a,b)=q_{\rm old}(a,b)
\exp(u_\theta(s,t,b)-u_\theta(s,t,a)).
\]

所有非法边的 rate 永远为零。网络学习一个动作势，而不是为每条边自由预测不相容的 ratio。它仍可能是昂贵的：一次 jump 需要全部合法邻居的 rate 总和，当前任务最多 119 个邻居。

要求 terminal boundary `u(s,T,a)=Q_phi(s,a)/tau`，并定义调和残差

\[
r_u(s,t,a)=\partial_tu(s,t,a)+
\sum_{b\ne a}q_{\rm old,t}(a,b)
\left[\exp(u(s,t,b)-u(s,t,a))-1\right].
\]

在有限图、有限率及可微时间势的条件下，路径 likelihood ratio 满足

\[
\log\frac{dP_u}{dR}
=u(T,X_T)-u(0,X_0)-\int_0^T r_u(t,X_t)dt.
\]

这给出一个明确的调试和理论接口：**残差为零时得到真正的 terminal tilt；任意标量 potential 只保证边比率的环一致性，不能自动保证正确的终点分布。**

对于确定起点，若 terminal boundary 精确，且全图全时间有 `|r_u|<=delta`，则归一化后终点密度比满足

\[
e^{-2T\delta}\leq\pi_u(a\mid s)/\pi^+(a\mid s)
\leq e^{2T\delta}.
\]

若 terminal boundary 还有一致误差 epsilon，相应指数可放宽为 `2(T delta+epsilon)`。这是由上面的 likelihood-ratio 恒等式直接得到的候选引理，并非已确认的新定理。只在采样路径上测得较小平均 residual，**不能**宣称满足全局上界；需要额外覆盖、函数类或可枚举空间假设。

路径 KL 也需要准确解释：

\[
\mathrm{KL}(P\|R)=\mathrm{KL}(\pi_P\|\pi_{\rm old})
+\mathbb E_{a\sim\pi_P}
\mathrm{KL}(P(\cdot\mid X_T=a)\|R(\cdot\mid X_T=a)).
\]

一般的 path-KL penalty 会额外限制“怎样到达同一动作”。精确 terminal Doob tilt 保留条件路径分布，使第二项为零；任意近似控制并不具有这个性质。若能在完整可行图上高效训练调和势，并通过残差校准控制这部分额外惩罚，才比“用 swap 做局部搜索”多出一个集中的研究问题。

可研究的贡献候选是：支持集缺口定理；可行 exchange graph 上的改进/误差关系；实用的局部残差估计与有限 jump 预算下的稳定改进。每项都需要进一步查重。仅把上述经典公式组合起来，不足以保证顶会创新性。

## RL 对接必须基于真实动作

训练 `Q_phi(s,a)` 或经过精确行为等价合并后的 `Q_phi(s,effect(a))`。不要复用当前 `Q(s,c)`，否则不同路径、不同编码仍可能被评价成不同决策。TD target 要对应声明的真实执行策略，例如 sampled-policy expectation；如果执行 best-of-K，就明确其改变了实际行为分布。

严格的动作层 PMD 为

\[
\pi^+=\arg\max_\pi\{\mathbb E_\pi Q-\tau\mathrm{KL}(\pi\|\pi_{\rm old})\}
\propto\pi_{\rm old}\exp(Q/\tau).
\]

如果始终相对固定的均匀 base `mu` 做终点加权，得到的是 `mu exp(Q/tau)`，对应另一种 entropy/reference-regularized objective，不是相对旧策略的 mirror-descent 更新。两种方法都可以研究，但不能混称。

PMD 无法从真正的零支持中恢复动作：`pi_old(a)=0` 则 `pi^+(a)=0`。因此必须从一开始给 connected graph 正的合法边率，或者明确使用探索参考 `pi_ref=(1-eta)pi_old+eta mu`。后者改变了参考策略；不得继续声称其目标是原来的精确 PMD。严格正率也可能产生小到无法采样的尾部概率，不能代替实际 coverage 证据。

每次改进应先冻结旧 rates 和 critic，再拟合/蒸馏新过程；不断更新旧过程同时只做一步回归，是 moving-target 近似。指数终点权重存在 ESS 崩溃，harmonic Monte Carlo 回归存在稀有终点方差，PDE residual 训练存在覆盖不足。这些问题不因“结构正确”而消失。

## 与 IDEA3 的连接：统一分布必须说明在哪个空间

可行交换去掉了 normal-fan 体积先验，却可能换来节点度数先验、cardinality 先验或动作标签 multiplicity 先验。不存在自动中性的几何。

本 benchmark 中，同组 primitive actions 的 `W` 列相同。若组大小是 `m_g`，用 `k_g` 表示该组激活数量，则完全相同的环境效果对应

\[
M(k)=\prod_g {m_g\choose k_g}
\]

个 binary action labels。均匀 binary-action prior 在效果计数上的分布正比于 `M(k)`。均匀效果计数 prior 是另一个选择，不等于同一件事。

原始对称 flip/swap CTMC 聚合到计数空间后，birth/death rates 分别带有 `m_g-k_g` 和 `k_g` 的 multiplicity；跨组 swap 带有 `k_g(m_h-k_h)`。它对 multiplicity measure 可逆。若希望探索均匀效果计数，可以直接在可行 count graph 上设置对称边率，再按声明的规则 lift 到 primitive actions。

更一般地，应先声明效果空间的参考测度 `nu`，再构造满足 `nu(a) q(a,b)=nu(b) q(b,a)` 的可逆 rates。详细平衡本身是已知条件；“均匀 binary labels”与“均匀 effects”对应不同规范性目标，不能把其中一个宣布为天然正确。当前环境的 reward/transition 只依赖 `Wa`，才使精确效果合并合理；一般未知环境不能仅凭近似特征相同就删除动作。

这给出 IDEA2 与 IDEA3 的更聚焦组合：**对精确行为等价类生成可行离散 diffusion，显式选择基准测度，并验证对无意义动作标签复制的稳健性。** 但“动作冗余会扭曲 entropy exploration”并非新发现：[Action redundancy in reinforcement learning, UAI 2021](https://proceedings.mlr.press/v161/baram21a.html) 已直接讨论并提出方法。我们需要证明的是结构化 diffusion 和策略改进在这一点上如何获得此前没有的可计算保证或明显收益。

## 已核实的最近邻工作与边界

下列原始论文在本次联网中均可访问，日期和内容均实际核查，不是把看似未来的 arXiv 号当作事实。

| 工作 | 已覆盖的关键内容 | 本方向还需证明的区别 |
|---|---|---|
| [RL-D², arXiv:2509.22963v3](https://arxiv.org/html/2509.22963v3)，2025-09 首发，2026-05 修订 | Masked discrete diffusion、在线 RL、PMD target、forward/reverse KL。 | 可行完整动作上的 birth/death/swap 过程，以及支持集缺口与终点误差/路径 KL 的结构分析。仅“离散 diffusion policy + PMD”没有新意。 |
| [Combinatorial Adjoint Matching, arXiv:2605.30920v2](https://arxiv.org/html/2605.30920v2)，2026-05 首发 | CTMC stochastic control、trajectory KL、离散 adjoint 与组合优化。正文以单 bit flip 为基本邻居，并允许 surrogate penalty 或 post-processing 处理不可行解。 | 我们要求全路径可行、处理 sequential RL 的实际动作价值、显式旧策略参考与初始 law；这些区别仍须证明价值，不能把“CTMC+组合优化”包装成新点。 |
| [DoobPO, arXiv:2610.28990v1](https://arxiv.org/html/2610.28990v1)，2026-10-20 | Doob ratio tilt、生成策略优化、改进与生成学习解耦、最小控制修改。 | 不应把 Doob transform 或 terminal tilt 本身作为贡献。候选区别是 constrained jump graph、可核查的支持/基准测度、局部 harmonic residual 与动作层误差。 |

最关键的不确定性：目前未证明这些差异足够构成强新方法。下一轮文献审查应专门查 constrained state-space diffusion、discrete Schrödinger bridges、graph ratio/score matching、action abstraction 与 CTMC residual bounds。

## 最小验证顺序与停止条件

1. **结构 oracle。** 在已有 benchmark 上确认 latent decoder 漏掉的行为计数中确实存在高价值决策；若缺口与价值无关，它不足以解释该 benchmark 上的算法差异。
2. **冻结目标分布。** 在可枚举的小图上给出已知多峰目标，比较独立 feasible sampler、degree-biased walk、对称 CTMC、potential-controlled diffusion。先看 TV/KL、mode coverage、rare-support recall、feasibility 与总 jump/NFE 成本，不先上 RL。
3. **冻结 Q 的策略改进。** 与精确 `pi_old exp(Q/tau)` 比较；分离 terminal-boundary error、harmonic residual、起始分布误差与混合不足。
4. **真正的 RL。** 第一层必须比较 native DPMD + primitive 20 维 decoder 的 sphere/diffusion 强基线，区分改掉 `W` 支持限制与改变优化规则。第二层以匹配的 action critic、交互预算和 wall-clock，比较 masked autoregressive、feasible local search、uncontrolled exchange CTMC，以及 proposed exchange diffusion，隔离生成机制。原 40 维 spherical decoder 保留为外部参考。保留全部退化臂，报告完整曲线与多 seed。

若 masked autoregressive 在相同支持集与成本下达到相同收益，只能证明原 decoder 支持受限，不能证明 diffusion 有优势。若 learned rates 退化为即时 Q-guided swap，且移除时间势/调和训练不影响结果，应把方法重新定位为局部搜索。若 endpoint-KL 近似不优于简单路径 policy gradient，或邻居枚举占据全部算力，应停止扩大训练规模。

## 本次便宜检查的实际结果

运行命令：`.venv/bin/python analysis/feasible_exchange_sanity.py`。脚本只枚举 `n=6,B=3` 的 42 个动作，使用矩阵指数和数值积分，不训练模型。

- birth/death/swap 图有 1 个连通分量；swap-only 有 4 个分量。
- 对称 CTMC 的 uniform stationarity 残差约 `2.78e-17`；均匀邻居 walk 的残差约 `1.08e-2`。
- 在同一个 terminal tilt 上，错误保持随机初始 law 的终点 TV 误差为 `0.107746`；正确初始重加权后约 `1.61e-16`。
- 确定空集起点的 Doob 终点分布匹配到机器精度。
- 该 anchored 例子的 terminal KL 为 `0.3753173492540136`；直接积分 CTMC rate-level path KL 得到 `0.3753173492540134`。

这些数字只验证本文的几条数学与实现约定。它们没有验证学习效率、在线稳定性、实际混合速度或新方法优于现有策略。
