# 从 diffusion 的可利用结构出发，而不是更换 actor 标签

状态：2026-11-04 的研究设计与文献核查；没有启动新训练，没有修改实验代码。**下述候选估计器已完成首轮便宜的机制检查，等 Q 成本结果为负，当前不推荐进入 RL 工程实现。** 保留设计与负结果，避免把推导正确当成实用收益。

**本轮集中检查的候选方向：在实际采样的有限 Gaussian 去噪链上做策略优化，并把最后一步通过组合解码边界的随机性解析积分掉。** 暂称“解码边界条件积分”，不先给它一个新算法缩写。目标是让离散动作的价值差直接给随机去噪转移提供低方差信用。与之配套的 continuation critic 是已有工具，不作为创新点。这个方向尚未建立论文级创新性，也未通过首轮成本门槛。

首轮证据来自 [summary.json](/home/serenade/AI/src/RL/SolverCell-RL/reports/continuation_branch_audit/summary.json) 和 [variance_comparison.csv](/home/serenade/AI/src/RL/SolverCell-RL/reports/continuation_branch_audit/variance_comparison.csv)：3 个已有训练后期状态 × 4 个坐标，每个估计器 20,000 samples，使用独立 pilot 得到的共同 baseline。完整 20 维梯度的两 Q 条件积分，相对两个独立 score samples 的 covariance trace 比值在全部 12 个情形均大于 1，范围 **1.473–1.862**。这意味着这些情形中它没有赚回额外一次 Q 的成本。

边界：测试是 \(m=0,\sigma=0.5\) 的声明 diagnostic kernel 和精确非线性 myopic action-value oracle，**不是 learned actor 的真实终层 kernel，不是长期 learned Q，更不是 RL 结果**。条件积分 quadrature 误差约 \(6.2\times10^{-15}\)，有限差分误差约 \(1.7\times10^{-10}\)，说明公式正确与成本失败并不矛盾。完整方法和限制另见 [NOISE_CONTINUATION_CRITIC.md](/home/serenade/AI/src/RL/SolverCell-RL/docs/NOISE_CONTINUATION_CRITIC.md)。

这比“可行图上的 Doob transform”更接近当前值得检验的问题：给定同样的动作支持、网络、噪声和 critic 调用预算，能否更有效地把 Q 信息转成策略更新。原有 40 维参数化的支持缺口是一个数学事实，但根任务对实际访问状态的检查没有发现足够大的 myopic gap，不能继续拿它作主要收益动机。也不能把“每个中间状态可行”当作主要收益：环境只执行最终动作，旧 solver 已保证最终可行。

## 1. 哪些优势成立，哪些只是实现选择

| 候选理由 | 对确定性 FM 的区别 | 科学边界 |
| --- | --- | --- |
| 每一步随机转移有显式条件密度 | 有限 Gaussian 转移的 log probability 精确可算；确定性 ODE 步是 Dirac kernel，不能直接使用同样的转移似然比 | 优势属于“带显式随机 kernel 的生成器”。ReinFlow 已给 FM 注入噪声并取得这一性质 |
| 无需对离散 solver 求导即可进行策略梯度 | 可以对去噪转移用 score-function 梯度，最终动作只用于评价 | DDPO/DPPO 已建立这条路径；普通 Gaussian policy 也有此性质 |
| 小范围 corruption、inpainting、共享前缀后分支 | 随机去噪链自然提供这些接口 | FM 可经条件训练、重启或加噪实现；分支采样本身已有工作，不能称 diffusion 独占 |
| 多模态与探索 | 随机 kernel 提供每步新随机性 | FM 的随机初始分布也能产生多模态；LSFlow 本身还有执行噪声。多模态或随机性不能单独解释优势 |
| 决策边界上的条件积分 | 最后一步对角 Gaussian + 已知解码阈值可以把一维条件积分精确算出 | 确定性 FM 的最后一步没有这个 Gaussian 积分，但 FM 加同样 Gaussian 尾部后也能使用；必须做这个对照 |

以上第一项已有明确反例：ReinFlow 的主要设计就是给 flow policy 加可学习噪声，转成有限 Gaussian Markov 链，并在很少步数下计算精确转移似然。不能用“flow 无法做 likelihood-based RL”作论文论据。[ReinFlow](https://arxiv.org/html/2505.22094v2)

因此，应研究“**随机去噪的结构被什么估计器利用了**”，并把 diffusion 与随机化 flow 的关系作为实证问题。若带相同 Gaussian 尾部的 flow 表现一样好，应把成果称作 solver-aware stochastic policy optimization。

## 2. LSFlow 当前实现中真正没有利用的接口

以下针对 pinned vendor 代码，不扩大为所有 flow 方法的缺陷：

- `vendor/lsflow/algos/dpmd_rf.py:230` 对 target actor 的终点 cost 计算 critic 权重；`:343` 把这些权重交给 weighted RFM。它没有给每个实际生成步骤分配单独的离散决策信用。
- `:276` 的 Bellman smoothing 在 actor 输出附近采样 tangent perturbations，然后用 latent-cost critic 评价。它不是“从中间去噪状态继续运行剩余生成过程”的条件价值。
- `vendor/lsflow/models/rfm/service.py:17` 的 perturb 是 Gaussian tangent approximation followed by normalization；其噪声与本提案的最后一步独立坐标 Gaussian 不是同一个分布。两者不能在数学推导中互换。
- native critic 接收 clean latent center，而实际执行经过扰动和解码。它可以作为定义好的平滑策略 critic；不能未经重定义就把它当作真实 binary action 的 Q，或当作任意噪声时刻的 continuation value。

值得测试的限制是“终点加权回归与固定 smoothing 是否丢失了可用的边界反事实信用”，不是“LSFlow 不能探索”或“FM 不可多模态”。改变 critic 语义、执行噪声、更新目标后，已经是新算法；必须另保留 native LSFlow 作为系统基线，并让主要机制消融共享这些改变。

## 3. 候选算法的完整定义，保留以便复核

### 实际有限采样链就是 policy

设可行动作为 \(\mathcal A=\{a\in\{0,1\}^{20}:\sum_i a_i\le10\}\)，\(D(c)\) 取 cost 中至多 10 个严格正的最大分量。这里采用 primitive 20 维 cost，避免把参数支持差异带入机制比较。任意可行 \(a\) 都能由 \(c=2a-1\) 唯一暴露。

采用固定、有限的生成链，不依赖连续时间极限：

\[
x_0\sim\mu,\qquad
x_{k+1}=m_{\theta,k}(s,x_k)+\sigma_k\epsilon_k,
\quad \epsilon_k\stackrel{iid}{\sim}\mathcal N(0,I),\quad k=0,\ldots,K-1,
\qquad a=D(x_K).
\]

仅有任意 \(m_{\theta,k}\) 加 Gaussian noise，只能定义一个普通 Markov policy，尚不足以称为 diffusion。这里进一步固定已知 VP forward corruption：令 \(T=1000\)，\(\beta_t\) 从 \(10^{-4}\) 线性增至 \(0.02\)，\(\alpha_t=1-\beta_t\)，\(\bar\alpha_t=\prod_{j=1}^t\alpha_j\)，\(\bar\alpha_0=1\)，则

\[
q(z_t\mid z)=\mathcal N(\sqrt{\bar\alpha_t}\,z,(1-\bar\alpha_t)I).
\]

学习 \(\epsilon_\theta(s,z_t,t)\)，并采用固定反向时间网格 \((\tau_0,\ldots,\tau_4)=(1000,750,500,250,0)\)。令 \(a_k=\bar\alpha_{\tau_k}/\bar\alpha_{\tau_{k+1}}\)，采样链的 mean 明确由 denoiser 给出：

\[
m_{\theta,k}(s,x_k)=\frac{1}{\sqrt{a_k}}\left[x_k-\frac{1-a_k}{\sqrt{1-\bar\alpha_{\tau_k}}}\epsilon_\theta(s,x_k,\tau_k)\right].
\]

首版 \(\mu=\mathcal N(0,I)\)，matched FM 也使用相同 prior。它不同于 native 单位球面 prior，是需要在共同实验接口中控制的改变。VP 的有限 \(q(z_T)\) 一般不精确等于 \(\mu\)；这里不把这种 warm-start approximation 当作分布恢复定理。

不需要专家数据：从同一份已有 policy/replay 的状态与可行动作构造 \(z=e(a)=(2a-1)/\sqrt{20}\)，以同一个冻结 critic 权重 \(w(s,a)\) 做 weighted denoising score matching：\(\mathbb E[w\|\epsilon-\epsilon_\theta(s,z_t,t)\|^2]\)。对照 flow 使用完全相同的状态、endpoint、权重、样本数和预算做 weighted FM warm-start；若复用 cost targets，也必须两边复用同一批，不能把 canonical embedding 和原始 latent endpoint 的差异混入机制比较。随后 RL 直接优化上述实际有限 kernels，而非用训练 loss 充当 endpoint likelihood。

中间 \(x_k\) 不做球面归一化；每一步实际条件分布就是上述 Gaussian。最终正比例归一化不改变此 decoder 的动作，可用于展示，但 likelihood 针对未归一化变量。所有 \(\sigma_k>0\)，包括最后一步；这是明确的 finite-policy 设计，不声称 \(\sigma_k^2\) 是精确 reverse posterior variance。标准实现中“最后一步关闭噪声”的习惯在这里必须改变。

第一轮仅固定 \(K=4\)，vector-RMS noise schedule 可取 \((0.50,0.35,0.20,0.10)\)，即每坐标 \(\sigma_k=\text{schedule}_k/\sqrt{20}\)。这是用于机制测试的一组声明值，不是调参结论。记录实际 sampler，训练和评估使用同一有限链；不要声称它精确反演某个连续扩散方程。\(K=1\) 对照用于检验是否根本不需要多步 diffusion。

### 外层 critic 必须评价真实执行动作

用实际环境数据 \((s,a,r,s',d)\) 学习 \(Q_\psi(s,a)\)，动作输入是 binary action，而非 40 维 cost。动作等价的 feature 表示只有在已经证明奖励和转移均等价时才可替代。普通回报 critic 的 TD target 为

\[
y=r+\gamma(1-d)\mathbb E_{a'\sim\pi_{\bar\theta}(\cdot\mid s')}\min_jQ_{\bar\psi,j}(s',a').
\]

不要把生成器所有中间向量当作环境动作，亦不要把枚举的反事实动作伪装成新增环境 transition。它们只由当前 learned Q 评价。

### 内层目标与 continuation value

每轮固定外层 target critic \(\bar Q\)、参考生成链 \(P_{\rm ref}\) 和 replay-state 分布 \(\rho\)。优化的是明确的 replay surrogate：

\[
J(\theta)=\mathbb E_{s\sim\rho,\,x_{0:K}\sim P_\theta}
\left[\bar Q(s,D(x_K))-\beta\sum_{k=0}^{K-1}
\log\frac{K_{\theta,k}(x_{k+1}\mid s,x_k)}{K_{{\rm ref},k}(x_{k+1}\mid s,x_k)}\right].
\]

这是一轮 actor 更新的 path-KL trust penalty；外层 critic 此处仍评价普通环境回报。它不是宣称在求解固定-reference maximum-entropy MDP。若要把路径惩罚纳入长期 RL 目标，外层 Bellman target 也必须相应改变。

定义实际剩余生成链的条件价值：

\[
C_K(s,x)=\bar Q(s,D(x)),\qquad
C_k(s,x)=\mathbb E_{x'\sim K_{\theta,k}}C_{k+1}(s,x')
-\beta\,\mathrm{KL}(K_{\theta,k}\Vert K_{{\rm ref},k}).
\]

用新鲜生成路径及 target networks 回归这些量。早期步骤可用 \(C_{k+1}\) 给 score-function update 提供信用，局部 Gaussian KL 解析计算；不用跨整条链反传，也不用对 solver 求导。**这个两层 actor–critic 结构已经是 BDPO/DIA 所覆盖的方向。** 它在这里仅承接最后一步的低噪声价值信息。[BDPO §4.2](https://arxiv.org/html/2502.04778v2), [DIA](https://arxiv.org/html/2609.12245v2)

### 唯一聚焦的机制：最后一步条件积分

条件于 \((s,x_{K-1})\)，最后一步 \(c_i\sim\mathcal N(m_i,\sigma_i^2)\) 相互独立。固定其余 19 个 scores，定义

\[
\tau_i=\max(0,\text{其余 scores 中第 10 大者}),\qquad
h_i=(m_i-\tau_i)/\sigma_i.
\]

忽略连续 Gaussian 下概率为零的 ties：\(c_i\le\tau_i\) 时输出固定动作 \(a_i^-\)，\(c_i>\tau_i\) 时输出固定动作 \(a_i^+\)。后者包含第 \(i\) 项，在预算已满时替换原来的最低入选项。令 \(\Delta Q_i=\bar Q(s,a_i^+)-\bar Q(s,a_i^-)\)，则

\[
\mathbb E_{c_i}[\bar Q(s,D(c))\mid c_{-i}]
=\bar Q(s,a_i^-)+\Phi(h_i)\Delta Q_i,
\]

\[
\partial_{m_i}\mathbb E\bar Q=\mathbb E_{c_{-i}}\left[\frac{\phi(h_i)}{\sigma_i}\Delta Q_i\right],
\qquad
\partial_{\log\sigma_i}\mathbb E\bar Q
=\mathbb E_{c_{-i}}[-h_i\phi(h_i)\Delta Q_i].
\]

第一轮固定 noise schedule，故只需要 mean 梯度。**被优先检查的是一个坐标、两个 Q evaluations 的完整向量估计器**：在观察终层 Gaussian noise 之前，均匀抽取 \(i\)。令 \(\bar F_i=\bar Q(s,a_i^-)+\Phi(h_i)\Delta Q_i\)，并选只依赖 prefix 的 baseline \(b(s,x_{K-1})\)。定义

\[
\widehat u_i=\frac{\phi(h_i)}{\sigma_i}\Delta Q_i,
\qquad
\widehat u_j=\frac{c_j-m_j}{\sigma_j^2}(\bar F_i-b),\quad j\ne i.
\]

整个向量 \(\widehat u\) 是 \((\bar Q(s,D(c))-b)\nabla_m\log\mathcal N(c;m,\mathrm{diag}(\sigma^2))\) 关于同一个 \(c_{-i}\) 的条件期望。它保持无偏，且固定 prefix 时具有完整的 covariance PSD reduction，进而通过固定 mean Jacobian 后也成立。最后一步 actor 梯度为

\[
\widehat g_{\rm last}=
\sum_j\widehat u_j\,\nabla_\theta m_{\theta,K-1,j}(s,x_{K-1})
-\beta\nabla_\theta\mathrm{KL}(K_{\theta,K-1}\Vert K_{{\rm ref},K-1}).
\]

这里 \(c_{-i}\)、阈值、离散候选动作、Q 值和 \(x_{K-1}\) 均在该局部 likelihood-gradient 项中 stop-gradient。其他步骤的 score terms 负责上游分布变化。对每个 conditional expectation 再沿其他 sampled coordinates 求导，会重复计算；只更新最后一步又不能称作整条链的精确策略梯度。

上面的 PSD 结论是相对同一个 prefix 下的一个原始 score-function sample；**不能由此断言等两个 Q evaluations 的 MSE 或 RL return 更好**，必须对比两个独立或 antithetic score samples。若根据已经采到的终层 \(c\) 自适应挑坐标，不能直接沿用上述无偏性结论。

另一个有效但较贵的选择，是对每个坐标分别构造 \(\phi(h_i)\Delta Q_i/\sigma_i\)，最多评价 40 个候选动作。每个分量分别降条件方差；由于不同分量使用不同 conditioning，不能自动推出它们经共享网络 Jacobian 相加后的 PSD covariance reduction。这与“两 Q、同一 conditioning、完整 score 向量”版本要区分。再一种选择是只保留抽中的解析坐标项并乘 20；它也无偏，但会增加坐标抽样方差。没有证据支持现在换到这些更复杂版本绕过已经观察到的负结果。

理论上的“无偏”首先是对冻结 \(\bar Q\) 的 finite-policy surrogate；它不意味着 learned Q 对真实 return 无偏。小 \(\sigma\) 会导致稀有边界大梯度和数值病态，解析积分不自动解决这个问题。

## 4. 最近文献把创新空间压缩到了哪里

| 已有工作 | 已覆盖的内容 | 本方向必须额外证明的部分 |
| --- | --- | --- |
| [DDPO](https://arxiv.org/abs/2305.13301), [DPPO](https://arxiv.org/abs/2409.00588) | 将去噪链作为决策过程，用单步可计算 likelihood 做 RL | 不能再以 denoising MDP 或逐步 policy gradient 为创新 |
| [BDPO](https://arxiv.org/html/2502.04778v2), [DIA](https://arxiv.org/html/2609.12245v2) | 中间生成状态的 value、两层信用分配、路径正则或 inner advantage | 不能仅增加一个 \(C_k\) 网络就声称新方法 |
| [DIPO](https://arxiv.org/abs/2305.13122), [Q-score matching](https://arxiv.org/abs/2312.11752), [RSM/DPMD](https://proceedings.mlr.press/v267/ma25d.html) | 在线 diffusion policy、Q 与 score 的联系、低成本加权训练 | 区别必须是离散 solver 下实际有限 kernel 的信用估计，而非泛称 Q-guided diffusion |
| [D3PO](https://arxiv.org/abs/2311.13231) | 去噪 policy 的直接偏好优化，不需单独训练 reward model | 不是当前有环境 Bellman Q 的方法，但阻止把逐步 preference credit 泛称新颖 |
| [ReinFlow](https://arxiv.org/html/2505.22094v2) | 为 FM 注入噪声、精确转移 likelihood、少步 RL | 同样 Gaussian 尾部的 flow 是必要机制对照 |
| [RL-D²](https://arxiv.org/abs/2509.22963) | combinatorial action 的 discrete diffusion 与 PMD distribution matching | discrete diffusion + PMD 本身已经有直接工作 |
| [CAM](https://arxiv.org/html/2605.30920v2), [DoobPO](https://arxiv.org/html/2610.28990v1) | discrete path-space control、生成策略的 Doob 优化 | 图、路径 KL、Doob potential 都不能单独作为新贡献 |
| [Rao–Blackwellized stochastic gradients](https://proceedings.mlr.press/v97/liu19c.html), [measure-valued policy gradients](https://arxiv.org/abs/2107.09359) | 条件求和减方差、无需可微 critic 的梯度 | 解析分支信用应定位为具体 solver/kernel 结构的利用 |
| [Perturbed optimizers](https://arxiv.org/abs/2002.08676), [stochastic softmax tricks](https://arxiv.org/abs/2006.08063) | 随机扰动后的组合 solver 与梯度估计 | 不能声称首次让组合优化器可训练 |
| [Gaussian top-K propensity](https://link.springer.com/article/10.1007/s10994-023-06444-4) | 利用 Gaussian 排名结构消除与奖励无关的随机性，并明确讨论 Rao–Blackwell 化 | 与这里的 coordinate-threshold Q-difference 不是同一公式，但属于必须正面区分的近邻 |
| [Decision-focused on-policy learning](https://arxiv.org/html/2606.01081v2) | Gaussian/flow/diffusion cost models、top-k、混合决策梯度 | 其 diffusion score 使用 ELBO proxy，明确不在相应精确理论内；这里直接用实际有限 Gaussian kernels 是一个差别，仍不足以单独确立创新 |

候选贡献的最窄表述是：**在组合策略中，利用可解析的终层 solver 分区，构造与真实有限去噪 kernel 一致的条件信用，并检验它能否改善多步 actor 优化的计算效率。** 还需要证明其价值不限于单步 Gaussian policy；否则应主动缩小题目。

## 5. 计算预算、偏差与能直接否证它的测试

若实施此候选，推理时只生成一条 \(K\) 步路径并调用一次 decoder，不增加 best-of-N 搜索。被检查的估计器在训练时每个 context 需要两个候选 Q evaluations，同时更新全部 mean coordinates。较贵的全坐标积分至多需要 40 个 Q evaluations，可将重复动作去重、一次 batch 调用，但必须记录原始候选数、unique action 数和实际 wall time。与 sampled policy gradient 比较时，应让对方用同样 Q 预算进行独立/antithetic noise repeats，而不是给它一个 sample。

本提案改变了 native 的 critic 输入、执行噪声和 actor 更新，不能直接拿改进结果归因于 diffusion。因此最小对照为：

1. 同一 finite Gaussian chain + sampled score gradient；先只比较终层估计器。
2. 同一 chain + 条件积分；其余全部相同。
3. \(K=1\) Gaussian actor + 条件积分；检验多步生成是否必要。
4. FM + 同样 Gaussian terminal kernel + 同样条件积分；参数量、NFE、Q 预算和执行噪声匹配。它若同样有效，说明收益来自随机边界信用。
5. 进入 RL 后再保留 native LSFlow，以及 primitive20、相同环境接口下的 flow/diffusion reference，区分新算法整体收益与机制收益。

**原计划的最小门槛是先不训练 RL：**固定 cardinality 问题与已知、包含交互项的真实动作价值；检查不同 means、boundary distances 和 noise levels，用高精度参考梯度验证 sampled score 与条件积分均值一致，报告等 Q 调用/等时间的 gradient MSE、cosine、一次小步更新后的真实目标变化。首轮已按文首范围完成，结果不支持继续。一个漂亮的二维图或零 bias 检查不算收益证据。

只有后续有新的、事先明确的使用条件通过成本门槛，才值得冻结真实 benchmark 的 action Q、检查从 known-Q 到 learned-Q 的 degradation；最后才是小规模闭环 RL。反事实 Q 差值若误差大，解析积分可能更稳定地优化错误方向；应报告 held-out 执行动作/可负担的反事实 rollout 校准，不能只看 critic training loss。

中间 continuation critic 的额外误差要单独列出：旧路径不等于当前 denoising occupancy；只用局部 importance ratio 不能修正整条路径分布。首版在 replay 环境状态上重新生成新鲜路径，冻结 critic/参考 actor 一轮，再做有限次更新。复用旧 paths、PPO clipping 或 bootstrap 均须标为近似。路径 KL 也只约束一个生成过程；它一般大于最终动作 KL，并可能惩罚动作不变的内部变化。

当前决定：停止把一维终层积分扩展成新 RL 系统，不立刻换成 block diffusion 或自适应 branching 继续堆组件。可以保留的研究假设只有“实际多尺度 continuation credit 可能含有终点加权回归未使用的信息”，但 BDPO/DIA 已研究它，不能直接当作创新。

若继续，下一步应是诊断而非新算法：在同一个 frozen actual-action Q 与同一批已有 actor prefixes 上，测各噪声时刻的真实 continuation advantage 的可预测部分，并把采集 suffixes、拟合 \(C_k\)、denoiser 和 Q 调用全部计入预算。若预测误差与成本不能优于直接终点信用，就没有充分理由继续工程化。若只在 \(K=1\) 有收益，应承认它是 Gaussian combinatorial actor 的改进；若 noisy FM 同样受益，应放弃“diffusion 独有优势”的表述。
