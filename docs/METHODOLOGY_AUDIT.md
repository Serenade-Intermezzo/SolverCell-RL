# 方法论审计：从短 transport 转向 decision KL、support 与执行核

审计日期：2026-11-04。本文只做推导、源码核查与既有结果解释，不启动 RL、不修改科学源码。IDEA 2 的具体 exchange-process 设计另见 [FEASIBLE_EXCHANGE_RETHINK.md](FEASIBLE_EXCHANGE_RETHINK.md)。

**核心判断：cell endpoint 更容易回归，不等于它更忠实地实现 policy mirror descent。** 在最有利的确定性 decoder、cell 内 Q 恒定条件下，动作 PMD 的最小 KL lift 恰好保留旧策略的 cell 内条件分布；重加权旧 point samples 天然符合这一点。当前 native 实现的执行噪声和 critic bootstrap 又破坏了这些理想前提。更值得追求的结构是：明确动作或 effect 的目标分布、保证它可达、再用尽量不额外扭曲目标的生成过程实现它。

## 1. 确定性 decoder：KL 链式分解给出了正确的 lift

固定环境状态 s，省略条件 s。令 c 为成本，D(c)=a 为确定性 decoder，C_a=D^{-1}(a) 为 cell，p 为成本分布，π=D_#p 为动作分布。先假设 p_new 对 p_old 绝对连续；连续空间的表达相对于共同参考测度，条件分布按 regular conditional probability 理解。

在 C_a 上，

\[
\frac{dp_{\rm new}}{dp_{\rm old}}(c)
=\frac{\pi_{\rm new}(a)}{\pi_{\rm old}(a)}
 \frac{dp_{\rm new}(\cdot\mid a)}{dp_{\rm old}(\cdot\mid a)}(c).
\]

取对数并对 p_new 积分，得到

\[
\boxed{
\mathrm{KL}(p_{\rm new}\Vert p_{\rm old})
=\mathrm{KL}(\pi_{\rm new}\Vert\pi_{\rm old})
+\mathbb E_{a\sim\pi_{\rm new}}
 \mathrm{KL}(p_{\rm new}(\cdot\mid a)\Vert p_{\rm old}(\cdot\mid a)).
}
\tag{1}
\]

因此，在所有实现同一个 π_new 的成本分布中，KL 最小者满足

\[
p^*_{\rm new}(dc)=\sum_a \pi_{\rm new}(a)\,p_{\rm old}(dc\mid a),
\qquad
\mathrm{KL}(p^*_{\rm new}\Vert p_{\rm old})
=\mathrm{KL}(\pi_{\rm new}\Vert\pi_{\rm old}).
\tag{2}
\]

它在 π_new 的支持上唯一，且要求 π_new(a)>0 的动作原本有 π_old(a)>0。一个旧策略概率为零的 cell 不能靠有限 KL 更新凭空获得概率。

若 Q(c)=Q(D(c)) 在每个 cell 内恒定，理想的动作 PMD 为

\[
\pi^+
=\arg\max_\pi\left\{
\mathbb E_\pi Q-\eta^{-1}\mathrm{KL}(\pi\Vert\pi_{\rm old})
\right\},
\quad
\pi^+(a)=\frac{\pi_{\rm old}(a)e^{\eta Q(a)}}{Z}.
\tag{3}
\]

其 KL 最小成本 lift 正是

\[
p^+(dc)=\frac{e^{\eta Q(D(c))}}{Z}p_{\rm old}(dc).
\tag{4}
\]

故“从旧成本策略采样，按动作 Q 重加权，保留旧 point 作为 endpoint”在分布层面具有明确依据。式 (4) 不自动保证有限网络、有限数据、近似 diffusion sampler 能精确拟合；但它至少定义了正确的目标分布。

相反，把同一个动作 a 的旧 endpoint 改成新噪声的近 cell 投影，通常改变 p(c|a)。即使动作边际在训练目标层面保持不变，式 (1) 的第二项也可能增加。最小化 \(\|c-z_0\|^2\) 与最小化相对于 p_old 的 KL 是不同的优化问题。

硬投影还可能把正概率区域压到低维面。如果最终归一化后该面相对旧策略的支持仍低维，则理想 endpoint law 含 singular mass，成本 KL 可为无穷。这里必须谨慎：正 margin 的一个 affine face 经径向归一化后未必仍低维；不能仅凭“发生了投影”就断言当前球面 endpoint 分布奇异。需要检查 active-set 维数和归一化后的秩；多个独立 active constraints 是直接的风险。有限训练噪声与近似采样还可能把这种目标质量重新平滑。因此本文不声称已训练 cell policy 的 KL 已测得为无穷。

**当前 point 也不是式 (4) 的严格实现。** 它先对旧 point 做相同 margin 的 repair；native 权重还包含 batch/EMA Q 标准化、clipping、target lag，actor 是有限步回归。保留旧 endpoint 的理论解释，不能被写成当前 point 已实现精确动作 PMD 的证明。

## 2. 当前 native 的 clean cell 不是执行等价类

[NativeLSFlow.act](../solvercell/upstream.py) 先从 actor 采样 K=12 个成本，以 critic 选最大者，得到 clean center c。训练行为随后加入固定 κ=28 的切空间 Gaussian 扰动并归一化，再由 D 解码。Replay 保存 clean c，但 reward 和 next state 来自扰动后的执行动作。

定义实际执行核

\[
K_\kappa(a\mid c)
=\Pr_\epsilon[D(T_\kappa(c,\epsilon))=a].
\tag{5}
\]

一般 \(D(c)=D(c')\) 不蕴含 \(K_\kappa(\cdot|c)=K_\kappa(\cdot|c')\)。靠近 cell 边界和位于内部的两个 center 会有不同的越界概率。若先讨论一个理想的、后续策略固定的真实 action value，则

\[
Q_\kappa(s,c)=\sum_a K_\kappa(a|c)Q_{\rm action}(s,a)
\tag{6}
\]

也通常不在 clean cell 内恒定。因此用旧 c 的 critic 权重训练另一个 clean-equivalent endpoint，甚至没有保证保留同一个执行动作分布或同一个 Q 值。Cell transport 的语义问题先于回归误差问题。

源码还有三个必须保留的限定：

- **训练与评估动作核不同。** [evaluate](../solvercell/trainer.py) 调用 stochastic=False：仍从随机生成的 12 个候选中选最大 Q，但不再加入执行扰动。已有 execution-flip probe 是训练数据路径的诊断，不能称为 validation rollout 的实际 flip rate。
- **当前 critic 不是式 (6) 的已证精确 Q。** [DPMD._smoothed_value](../vendor/lsflow/algos/dpmd_rf.py) 对 target actor 的 M=16、J=1 扰动候选取 twin-min Q，排序后去掉最低 3 个，再平均剩余 13 个。这既不是 raw proposal 的期望，也不是 K=12 best-of-K 行为的期望。准确称呼是“以 noisy clean center 为输入、采用 trimmed bootstrap 的 critic”。将其解释为部署策略的标准 \(Q^\pi\)，需要额外论证。
- **native α 没有测量策略熵。** 上游每 180 步计算
  \[
  H_{\rm proxy}(x)=\tfrac12D\log[2\pi e(0.1e^x)^2],
  \quad x=\log\alpha,\quad
  L_\alpha=x\,\mathrm{stopgrad}(H_{\rm proxy}(x)-0).
  \]
  该更新只依赖 x、维数、固定目标 0、optimizer 状态和更新次数；没有 sphere、diffusion 或 decoded-action 样本熵反馈。当前 native.act_dim 始终为 40，包括内部 actor 仅 8 维的 quotient variants。它是一个确定的、由维数参数化的代理控制过程，不应宣传成实际动作熵控制。

这些是对上游适配语义的审计，不构成修改当前冻结实验的理由；修正目标定义必须作为后续独立方法设计。

## 3. 加上执行噪声后，链式分解仍成立，但任意 action lift 不再可实现

对固定 K，令联合分布 \(j_p(c,a)=p(c)K(a|c)\)，执行动作边际为 \(\pi_p(a)=\int p(dc)K(a|c)\)。因为新旧的条件核相同，

\[
\mathrm{KL}(p_{\rm new}\Vert p_{\rm old})
=\mathrm{KL}(j_{\rm new}\Vert j_{\rm old})
=\mathrm{KL}(\pi_{\rm new}\Vert\pi_{\rm old})
+\mathbb E_{\pi_{\rm new}}\mathrm{KL}
 (j_{\rm new}(c|a)\Vert j_{\rm old}(c|a)).
\tag{7}
\]

但给定任意目标 π* 的无约束最小联合 lift

\[
j^*(c,a)=j_{\rm old}(c,a)\frac{\pi^*(a)}{\pi_{\rm old}(a)}
\tag{8}
\]

一般具有

\[
K^*(a|c)
=\frac{K(a|c)\,\pi^*(a)/\pi_{\rm old}(a)}
 {\sum_bK(b|c)\,\pi^*(b)/\pi_{\rm old}(b)}
\ne K(a|c).
\tag{9}
\]

也就是说，保留旧 posterior \(j(c|a)\) 的最优联合 lift 往往要求改变执行噪声核。若只学习 p(c) 并固定 K，可达到的动作分布被限制在 \(\{pK\}\)，即可用 kernel rows 的混合，而非任意动作 simplex。简单反例：两个可用 center 的执行核为 (0.8,0.2) 和 (0.2,0.8)，无论如何混合，第一动作概率都只能在 [0.2,0.8]，不能实现一个概率为 0.9 的 action-PMD target。

把噪声显式并入变量同样不能绕过限制。基准联合律 \(p_{\rm old}(c)\rho(\epsilon)\) 经 deterministic \(a=D(T(c,\epsilon))\) 倾斜后，通常引入 c 与 ε 的相关性；固定独立噪声 ρ 的 actor 家族未必包含这个 lift。

两种不同的更新因此需要分开：

\[
\begin{aligned}
\text{固定核、成本 KL prox:}\quad
p^+(c)&\propto p_{\rm old}(c)
 \exp\{\eta\,\mathbb E_{a\sim K(\cdot|c)}Q(a)\},\\
\text{联合执行轨迹的 action tilt:}\quad
j^+(c,a)&\propto p_{\rm old}(c)K(a|c)e^{\eta Q(a)}.
\end{aligned}
\tag{10}
\]

后者的成本边际包含 \(\mathbb E_K e^{\eta Q}\)，前者包含 \(e^{\eta\mathbb E_K Q}\)；一般不相同。当前 best-of-12 又使 raw actor proposal 与被选中的 center 分布不同。任何“精确执行动作 PMD”主张都必须说明 KL 放在哪个分布上、哪些核可学习，以及 candidate selection 是否包含在该分布中。

## 4. IDEA 3：prior calibration 必须先定义目标参考测度

令 p_0 为固定成本参考分布，\(\mu_0=D_\#p_0\) 为它诱导的动作 prior。固定参考的成本正则化

\[
\max_p\{\mathbb E_p Q(D(c))-\tau\mathrm{KL}(p\Vert p_0)\}
\]

在 cell 内 Q 恒定时诱导

\[
\pi^*(a)\propto\mu_0(a)e^{Q(a)/\tau}.
\tag{11}
\]

如果希望参考分布是明确指定的 ν，且 \(\nu\ll\mu_0\)，可以构造

\[
\rho_\nu(dc)=p_0(dc)\frac{\nu(D(c))}{\mu_0(D(c))}.
\tag{12}
\]

使用固定正则项 KL(p||ρ_ν) 就得到 \(\pi^*\propto\nu e^{Q/\tau}\)。等价地，在仍用 p_0 的固定目标中加入
\(\tau\log[\nu(D(c))/\mu_0(D(c))]\)。
当 ν 在给定支持上均匀时，才可忽略常数写成 \(-\tau\log\mu_0(a)\)。这是一种**固定参考目标的重新定义**，不是每次 PMD 都应重复扣一次旧 geometric prior。

如果目标是“旧策略的一步 KL prox + 对固定 ν 的熵/相对熵正则”，正确问题是

\[
\pi^+=\arg\max_\pi\left\{
\mathbb E_\pi Q-\eta^{-1}\mathrm{KL}(\pi\Vert\pi_{\rm old})
-\tau\mathrm{KL}(\pi\Vert\nu)\right\}.
\tag{13}
\]

对 π(a) 求导并加入归一化乘子：
\((\eta^{-1}+\tau)\log\pi(a)
=Q(a)+\eta^{-1}\log\pi_{\rm old}(a)+\tau\log\nu(a)+\text{constant}\)。
因此

\[
\boxed{
\pi^+(a)\propto
\pi_{\rm old}(a)^{1/(1+\eta\tau)}
\nu(a)^{\eta\tau/(1+\eta\tau)}
\exp\{\eta Q(a)/(1+\eta\tau)\}.
}
\tag{14}
\]

从旧动作样本拟合这个目标时，其权重比为
\[
\frac{\pi^+(a)}{\pi_{\rm old}(a)}
\propto
\exp\left\{\frac{\eta}{1+\eta\tau}
\left[Q(a)-\tau\log\frac{\pi_{\rm old}(a)}{\nu(a)}\right]\right\}.
\tag{15}
\]

这里涉及当前 π_old，而不是永远用初始 μ_0 代替。τ=0 恢复未正则化 PMD；η→∞ 恢复固定参考的 soft target。区分固定 regularizer 与 consecutive-policy prox 是既有 regularized-MDP 理论中的基本问题；式 (13)–(15) 是本文对单状态目标的直接求解，不是新定理。[Geist, Scherrer & Pietquin, 2019](https://proceedings.mlr.press/v97/geist19a.html)

反例说明为什么不能每步盲目减 \(\tau\log\mu_0\)：令 Q=0、μ_0=(0.8,0.2)、ητ=1，若从 π_0=μ_0 开始重复
\(\pi_{k+1}\propto\pi_k/\mu_0\)，则依次得到 (0.5,0.5)、(0.2,0.8)、约 (0.0588,0.9412)。第一步恰好均匀，后续却持续过度补偿。式 (14) 在 ν=(0.5,0.5) 时则把 log odds 逐步减半并趋向 ν。

“均匀”也必须注明对象。均匀 primitive binary actions 与均匀 group-count effects 是不同 ν；前者给 effect r 的质量与 \(\prod_g{m_g\choose r_g}\) 成正比。消除表示重复并不等于最大化状态或 transition entropy。动作冗余及其与 transition entropy 的区别已有专门研究。[Baram, Tennenholtz & Mannor, 2021](https://proceedings.mlr.press/v161/baram21a.html)

## 5. 当前 prior 的首要问题是零支持，不只是概率不均

[solver support audit](../reports/solver_support_audit/summary.json) 对当前 instance 的 W 做了精确枚举。W 的 rank 为 8，20 个 primitive columns 合成 8 组，组大小为 [3,3,1,3,3,3,1,3]。统计单位是 group-count effect，包含空动作，预算为“≤10”：

| 对象 | 数量 |
|---|---:|
| 可行 primitive binary actions | 616,666 |
| 可行 group-count effects | 9,318 |
| canonical decoder 的正测度 effect cells | 268 |
| 仅在精确跨组 ties 上 canonical 可达 | 246 |
| 对任何成本都 canonical 不可达 | 8,804 |

所以 \(268+246+8804=9318\)。对绝对连续的 isotropic sphere prior，后两类共 9,050 个 effects 概率为零；finite precision ties 不能冒充正测度 support。已枚举的 μ_0 在 268 个正支持 effects 上最大/最小概率比为 7.010753。原 full sphere S39 与 orthonormal quotient sphere S7 诱导**完全相同的 clean μ_0**，因此降到 rank-8 quotient 本身不是 prior calibration。

但 support 的计数缺口不等于当前已训练策略的主要价值瓶颈。该审计新增的 completed quotient-point seed0 late-replay 样本中，512 个状态仅 20 个存在 positive-support 与全 effect 集之间的一步最优回报差；平均差 0.001041933，median/p95 为 0，最大 0.067417334。相比之下，synthetic states 的平均差为 0.965921，random-policy visited states 为 0.032228。不能拿前两者较大的差来解释已训练策略的长期性能；这些都是已知模型的一步 oracle 差，并非长期 optimality gap。Support 修复应作为表述与对照的必要条件，目前不足以单独担当 diffusion 超过 FM 的主要实验动机。

式 (12) 只能在已有正支持上重新分配概率。若 ν 想覆盖全部 9,318 effects，ν/μ_0 在零支持处无定义；加一个 denominator epsilon 不能创造 actor/decoder 本来无法产生的动作。必须改变可达表示、decoder、生成过程或明确缩小目标支持。

限制来自当前 W 特征映射，并非所有线性 solver policies。直接用 20 维 primitive costs 时，对任意可行动作 a，取 \(c=2a-\mathbf1\)，则
\[
c^\top a-c^\top b=\|a-b\|_1>0\quad(b\ne a).
\]
因而 a 是所有可行 binary b 中的唯一最优解，且存在正体积邻域保持该结果。归一化 c 不改变排序。**native-style PMD + primitive-cost decoder 是 exchange diffusion 必须面对的后续控制**：否则所谓结构优势可能只是修复了 W 的 support 限制。这个控制尚未由当前 native-structure campaign 实现；最初 primitive-cost matched harness 同时改了 critic、scale 和 update rule，不能替代它。

## 6. IDEA 2 的目标应是可达的 decision/path KL，不是再造一个近端点

在可行 action/effect 图上定义生成过程，可以直接控制支持、约束和参考测度，但“使用 CTMC / diffusion / Doob transform”本身不是新贡献。离散 diffusion + PMD 已出现在 [RL-D²](https://arxiv.org/abs/2509.22963)；组合优化中的 CTMC control 已出现在 [CAM](https://arxiv.org/abs/2605.30920)；ratio-tilted generative policy 与 Doob matching 已出现在 [DoobPO](https://arxiv.org/abs/2610.28990)。这些文献确立邻近空间，不自动否定一个更具体的受约束构造，也不能被绕过后声称范式首创。

更清楚的研究目标是：给定一个显式 terminal action target，设计始终可行、支持足够、terminal KL 误差可检验且计算可控的 exchange process。需要先满足：

1. **图的连通性。** 当前是 ≤B 约束，纯 swap 守恒 cardinality，不能连接空动作和非空动作；需可行 add/drop 或其他跨 cardinality moves。对 group-count 状态也要遵守每组容量。不是任意组合约束都天然具有简单 exchange 连通性。
2. **明确 reference law。** 每步均匀选邻居通常不是均匀状态 prior；degree、边率和有限生成时域都会改变分布。可用对称 CTMC edge rates 或针对 ν 的 detailed-balance 设计，但有限时间 terminal prior 不会仅因“ν 是平稳分布”就自动等于 ν。
3. **明确 path KL 与 terminal KL 的差额。** path-space 的 KL 同样分解成 terminal-action KL 加给定终点后的 path conditional KL。理想 terminal tilt 保留旧 path conditionals；更短的路径或更低回归 loss 不能代替这个约束。若 Doob 构造从随机初始律出发，还要处理 initial-law tilt，不能只改转移率后宣称任意终点目标已实现。
4. **将执行模型纳入声明。** 若末端直接输出可行动作，应说明取消了旧 cost perturbation；若保留额外执行核，应按式 (7)–(10) 限定目标可达性。二者不是同一个控制问题。

值得继续检验的创新假设应落在“约束图 + 足够支持 + 目标 KL 的实现误差/复杂度控制”的结合上。具体定理、比已有方法更强的性质和计算优势仍待建立；本文不把 Doob 或 prior 校正本身登记为新颖性。

## 7. 当前证据、最小可证伪检查与不成立的主张

本节经验依据限于已完成、源码为 030e1c799… 的 native-revision stage A，[完整报告](../reports/native_revision_20261104/summary.json)中的 seed0 last-window 为 point 335.431249、cell 289.115001、independent 288.647501。它说明当前 cell 机制没有赢过 point，不能证明理论 KL gap 就是全部因果来源。正在独立审计的 margin/seed follow-up 不在本文中被预判。

固定 cell 最终 checkpoint 的 [offline margin probe](../reports/native_revision_20261104/native_mechanism_audit/quotient_cell_offline_margin_probe.json) 在 δ=0.1/0.3/0.6 下得到 execution action-flip fraction 0.896606/0.788452/0.603149。它支持边界鲁棒性作为一个待检验因素；不能证明提高 margin 必然提高训练回报，也不等于测得策略 KL。

在任何新增长程 RL 前，以下检查可分别推翻具体假设；本文没有运行它们：

| 待检验命题 | 最小检查 | 推翻或收窄的结果 |
|---|---|---|
| Cell 仅改善拟合，不改变 PMD 目标 | 小型枚举 decoder，给定旧 p 和 cell-constant Q；精确计算 action tilt、旧 conditional 和投影 conditional | 相同动作边际却出现额外 conditional KL；不能再称它是相同 PMD 的无偏重参数化 |
| Clean-equivalent endpoints 执行等价 | 固定同一 clean cell 的两个 center，估计完整 Kκ 行与期望 action Q | 显著不同的 kernel rows；应优化执行分布或明确更换执行规则 |
| Prior calibration 实现指定 ν | Q=0，使用已知 μ_0 的枚举分布，检查一次及多次更新 | 漂离 ν、反复过度补偿，或试图给零支持 effect 分配可实现质量 |
| Exchange 的收益超出修复 W support | 同一 native-style 更新下比较 W-cost、primitive-cost、exchange；并报告可达集、采样/critic预算 | primitive-cost 已解释收益，则贡献应收窄为表示修复，而不是 exchange 独有能力 |
| Exchange 实现正确 terminal PMD | 小图上用精确 CTMC matrix exponential/枚举作 oracle，检验 terminal law、KL、feasibility 与初始律 | 路径看似合理而 terminal target 不符；先修生成过程，不能用 RL 回报掩盖目标偏差 |

目前不支持的主张包括：cell 距离更短因此 policy update 更小；clean action 相同因此 native critic 相同；native α 在控制实际策略熵；rank quotient 已校正 action prior；inverse-prior weighting 能恢复零支持动作；当前 point 是精确 PMD；任意 fixed-noise cost actor 都能实现任意 action-PMD lift；以及 exchange/Doob/离散 diffusion+PMD 作为一般概念具有首创新颖性。

保留下来的研究方向更具体：先定义需要保持的 decision distribution 与 reference measure，检查表示与执行核允许哪些更新，再寻找既保持可行性又减少生成实现误差的过程。这样，结构优美对应的是一个可检验的目标与分解，而不是一个仅有更小 MSE 的 endpoint construction。

## 8. 聚焦 diffusion 相对 FM：具体代码瓶颈与可利用的机制

最新设计重点应是把内部生成过程训练成**可学习的随机改进过程**，而非只替换 endpoint transport。下表严格区分已在代码中观察到的机制与尚待验证的性能瓶颈。

| 源码事实与精确位置 | Diffusion 可以利用的机制 | 不能据此宣称的内容 |
|---|---|---|
| 上游 DPMD 在 [dpmd_rf.py:347](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/algos/dpmd_rf.py:347) 每个 replay state 只采一个 target endpoint；[243](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/algos/dpmd_rf.py:243) 对 endpoint 评分；[359](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/algos/dpmd_rf.py:359) 只传入 state、endpoint、scalar weight | 用 stochastic transition 的条件 likelihood 学习每一步的改进与 trust region，并以 terminal action value 回传信用 | Endpoint weighting 是这种 actor update 的设计，并非所有 FM 无法采用 trajectory RL |
| [RFMPolicy._rfm_loss_core:162](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/policy.py:162) 随机 source，[179](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/policy.py:179) 建 geodesic interpolation，[191](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/policy.py:191) 用同一个 endpoint weight 加权 velocity error；没有内部 V(s,z_t,t)、advantage 或采样轨迹的 log-prob 更新 | 训练 denoising-state value/advantage，区分哪个局部 stochastic move 实际提升最终可行动作质量；不需要对 solver 求导 | FM 已有中间时间的监督，不能说它“完全没训练中间步骤”；缺的是中间步骤的 RL credit |
| [NativeLSFlow.act:122](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/upstream.py:122) 完整生成 K 个候选并以 Q argmax 选一个；[RFMPolicy.sample:122](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/policy.py:122) 给 K 个样本独立初始化，随后逐步积分并只返回终点 | 在相同预算下比较 shared-prefix branching、对未决定部分的 re-noise/denoise，或少量样本的多轮局部 repair，以 Q/feasibility 信息调配后续计算 | FM 也可以 branching、restart、加外部噪声；这些本身不是 diffusion 独占功能。当前重复度也不足以支持严重 mode collapse |
| 训练执行噪声 [upstream.py:127](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/upstream.py:127) 来自固定 κ；[rfm/service.py:25](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/service.py:25) 为固定强度切空间 Gaussian。当前 bridge [diffusion_service.py:108](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/diffusion_service.py:108) 的 g 只取决于固定 sigma/sin(time)，不依赖 state、solver slack 或 critic uncertainty | 学习条件 reverse transition 或 constraint-aware 局部随机动作，使不同 state/阶段有不同探索行为；固定噪声协方差时也可通过学习 drift 改变条件搜索方向 | 当前 FM 的 vector field 已依赖 state 和 time，[policy.py:110](/home/serenade/AI/src/RL/SolverCell-RL/vendor/lsflow/models/rfm/policy.py:110)；不能把它写成 state-unconditioned。自适应探索也不是 diffusion 专属 |
| 原 RFM 仅有 weighted velocity loss。当前 diffusion bridge 已在 [diffusion_service.py:72](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/diffusion_service.py:72) 加 interior Gaussian，[79](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/diffusion_service.py:79) 训练 noise-prediction MSE，[82](/home/serenade/AI/src/RL/SolverCell-RL/solvercell/diffusion_service.py:82) 与 velocity loss 相加；但没有实现 reverse-transition log-likelihood/path-KL RL objective | 显式保留非退化 stochastic reverse kernel，利用可计算的条件 log probability、KL 与逐步 credit，而不是仅把第二个 regression head 视为创新 | 不能说当前 bridge“没有 denoising 训练”；它有 noise-prediction regression。也不能声称任意 noise MSE 都等于已验证的 DDPM ELBO |

**一个真实的参数化差异：局部 stochastic likelihood。** 若有限步 diffusion 使用共享、非退化 covariance，
\[
P_{\theta,k}(z_{k+1}|z_k,s)
=\mathcal N(m_{\theta,k}(z_k,s),\Sigma_k(z_k,s)),
\]
则每步 Gaussian log probability 可直接计算，且同一条件下
\[
\mathrm{KL}(P_{\theta,k}\Vert P_{{\rm old},k})
=\tfrac12\|m_{\theta,k}-m_{{\rm old},k}\|_{\Sigma_k^{-1}}^2.
\]
初始律相同时，path KL 是对新路径分布求期望的这些 conditional KL 之和，并上界相同 terminal decoder 后的 action KL。Decoder 不可微不影响 score-function policy gradient；但用 noisy-center critic 代替真正 terminal action value 仍会引入目标偏差。

纯确定性 ODE 的每步条件转移是 Dirac measure，不直接提供上述有限 Gaussian likelihood ratio；改变 deterministic transition 可使对应条件测度互相奇异。这是 stochastic reverse kernels 相对**当前纯 ODE sampler**的具体优化接口优势，不是 FM 所有训练/采样方法的能力下界。FM 可以引入 stochastic sampling、直接训练 density/ratio，或采用其他控制目标，必须保留相应强对照。将 denoising 作为多步决策过程本身已有 [DDPO](https://arxiv.org/abs/2305.13301) 和 [DPPO](https://arxiv.org/abs/2409.00588)，不能作为新的核心 claim。

若学习 covariance，应使用完整有限步 Gaussian KL；不能一边改变 continuous-time diffusion coefficient，一边直接套用“只改 drift、共享 diffusion coefficient”下的 Girsanov path-KL 公式。球面 normalize 后也不能把最终 transition density 偷换成未归一化 Gaussian density；可在 Euclidean latent path 上定义 KL，再通过确定性 normalize/solver 用 data processing，或明确采用有已知密度的球面核。

**候选重复的实际程度。** 在 frozen stage-A 的三个 final checkpoints 上进行了只读 probe：各自 final replay 最后 1,000 条 transition 中无放回抽 512 states，replay seed 882310，candidate seed 882311，生成 12 个 online clean candidates并以当前 canonical decoder 解码：

| Checkpoint | 平均不同 clean actions / 12 | 平均重复比例 | 存在重复的 state 比例 |
|---|---:|---:|---:|
| quotient_sphere | 10.189453 | 0.150879 | 0.865234 |
| quotient_point | 10.750000 | 0.104167 | 0.748047 |
| quotient_cell | 11.619141 | 0.031738 | 0.320313 |

三者均没有一个 state 的 12 个候选全部相同。它们使用各自 replay states，因此不是严格配对的因果比较。Checkpoint SHA256 分别为 008c2e9f997ffcad9078fa574c1167e876c98f46618f215fd13586c299bea0bd、447367fe5a4d7c9bf1cee993415e626e9c6d458ec28138687878fbe432f94446、035f8bf363c205a75cc08d626be75dbffc037e95b6f980a4d891c15747e72e5f。所有 probe 均未更新模型或修改 checkpoint。

这些数字不支持“LSFlow 已经严重候选塌缩”的说法；cell 更不重复却回报更差，也说明 diversity 不是充分解释。在带执行噪声的训练目标下，同 clean action 的两个 center 仍可能有不同 Kκ，因此 clean duplicates 也不能全部当作毫无价值的重复计算。更合理的待检验指标是同预算下最终 action value、不同高价值 effects 的覆盖率，以及内部计算分配的收益。

**下一算法应先回答的可证伪问题：** 在相同可达表示、actor/critic 预算和 terminal target 下，把 scalar endpoint distillation 改为带 conditional likelihood/KL 的内部 stochastic improvement，是否比 endpoint-only diffusion、原 FM、以及带相同 branching/noise 的 FM 有更高的 action-value 改善效率？若收益只来自更大支持、更多网络计算、额外 critic calls 或终点重排，应归因于对应因素。若 conditional credit、可学习 reverse transitions 或对约束的逐步利用没有独立收益，就没有证据支持“利用了 diffusion 相对 FM 的结构优势”。
