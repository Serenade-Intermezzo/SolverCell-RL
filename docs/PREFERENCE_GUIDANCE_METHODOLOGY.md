# 偏好可控生成：方法边界、最近先例与最小研究问题

2026-11-04。本文只做方法设计与 primary literature 核查，不修改 RL 核心代码，不启动训练。Benchmark 是否具有真实、冲突且可独立记录的目标，由并行 source audit 确认；这里不把已有单目标 reward 拆几个名字就当作多目标任务。

**用户提出的方向有明确价值：同一模型在部署时接受安全、速度、舒适度等偏好，并调整生成结果，无需重新训练。** 但“输入偏好 + 多步生成”已经同时存在于 diffusion 和 flow。当前推荐的聚焦方向是“决策一致的偏好引导”：改变 solver 最终动作的概率质量，同时保留每个动作 cell 内原来的条件表示；以真实动作效用定义目标，再研究随机去噪是否是更好的推理实现。不是把一套 SMC 技术栈当作新算法，也不是证明 diffusion 天生具有而 flow 不具有偏好控制能力。

## 已有工作已经覆盖了什么

| Primary source | 已有内容 | 对本项目的约束 |
| --- | --- | --- |
| [MODULI, ICML 2025](https://arxiv.org/html/2408.15501v2) | preference-conditioned diffusion planner，额外 slider adapter 改善偏好泛化 | 偏好输入、连续调整、未见偏好 generalization 都不是空白 |
| [PC-Flow, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37971) | 将 FM 的 ODE 转成相应 SDE，用轻量 preference classifier 对生成作引导 | “FM 不能偏好引导”不成立；这是图像 preference alignment，并非已实现的组合 RL 基线 |
| [ParetoSlider](https://arxiv.org/abs/2604.20816) 与 [作者项目页](https://shelley-golan.github.io/ParetoSlider-webpage/) | 单模型连续多目标偏好控制；实验明确包含 SD3.5、FluxKontext、LTX-2 等 flow-matching backbones | 论文标题中的 diffusion 不等于对 deterministic FM 的排他优势 |
| [Flow Matching for Offline RL with Discrete Actions](https://arxiv.org/html/2602.06138v3) | Appendix B.12 在 Resource-Gathering 上用单个 preference-conditioned CTMC policy，无需重训查询连续偏好 | 最接近“离散动作 + preference-conditioned flow”的直接先例；该实验不等于已经验证大规模 constrained combinatorial RL |
| [On the Guidance of Flow Matching, ICML 2025](https://arxiv.org/abs/2502.02150) | 一般 flow guidance，包含渐近精确的 Monte Carlo 方法、可学习 guidance 与近似方法 | 不能把任何 inference-time energy guidance 归为 diffusion 独有 |
| [UniGuide 的原始稿](https://openreview.net/pdf/0a1115b68be101e7a7db436a01cfd502ec736550.pdf) | preference-conditioned policy 为每个去噪步骤选择 guidance 参数 | “每一步按偏好决定引导强度”也已有直接先例；此处引用稿件内容，不推断最终接收状态 |
| [TDS](https://arxiv.org/abs/2306.17775), [TRI-TSMC](https://arxiv.org/abs/2605.25123) | diffusion 的有权重校正的条件采样、学习 twisting potentials 和匹配推理预算 | 下文的 Doob/SMC 框架是工具，不作为新贡献 |
| [UVFA](https://proceedings.mlr.press/v37/schaul15.html), [Successor Features](https://papers.neurips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html) | 共享 value 对目标/奖励变化泛化，改变奖励后复用知识 | 无需重训切换目标，本身不限于生成策略 |

PC-Flow 指 preference-classifier alignment；不要与其他同名图像 restoration 项目混淆。真正必要的强 baseline 是在同一 RL harness 内构造 **preference-conditioned FM/LSFlow**，共享 reward vector、critic、数据、执行噪声、参数规模与推理预算；不能只拿原来固定 reward 的 LSFlow 当唯一对手。

## 偏好接口先定义任务，再定义生成器

令环境返回可测量的向量 \(r(s,a,s')\in\mathbb R^m\)，偏好 \(w\in\Delta^{m-1}\)。若所有分量都按“越大越好”表示，固定偏好的普通目标是

\[
J_w(\pi)=\mathbb E_\pi\sum_{t\ge0}\gamma^t w^\top r_t.
\]

目标单位和归一化应由固定训练集或物理定义确定；每个 minibatch 重新标准化会改变同一个滑块的含义。速度、舒适度等名称必须对应真实可记录量；若舒适度依赖动作变化、jerk 或连续控制历史，状态/奖励必须保留必要历史。

需要区分三个不同的“安全”：

- 可行动作约束 \(a\in\mathcal A(s)\)：可以由组合 solver 精确执行。
- 期望违规成本预算，例如 \(\mathbb E\sum\gamma^tc_t\le d\)：是 constrained MDP，不能仅以用户减小安全权重保证满足。
- 尾部风险或事故概率：平均 reward vector 不包含足够信息。对 loss \(L\)，\(\mathrm{CVaR}_\alpha(L)=\inf_\eta[\eta+(1-\alpha)^{-1}\mathbb E(L-\eta)_+]\) 依赖分布。直接把平均安全 reward 加权不等于 CVaR。静态 trajectory-CVaR 与 time-consistent nested risk 也不同，可能需要 risk-to-go/augmented state。[CVaR MDP](https://arxiv.org/abs/1506.02188), [time-consistent risk control](https://arxiv.org/abs/1503.07461)

不要让用户的“更安全”偏好被解释成可以关闭硬约束。首个研究版本宜只研究明确的固定权重期望回报；风险约束应另定协议。

下文对 expected Q 取指数，只是在定义动作选择的 regularized target；它不等于对环境 trajectory return 取 exponential utility，更不自动实现 CVaR 风险控制。

线性 scalarization 也要正确表述：在给定的非凸可达回报集合中，它只恢复 supported optima。例如三个确定性选项 \((1,0),(0,1),(0.4,0.4)\)，中间选项非支配却不是正权重线性和的最优解。但是允许随机混合后，\((0.5,0.5)\) 可达并支配它；某些 MDP/policy-class 假设下期望回报集合本身是凸的。因此“所有 MORL 都因非凸 Pareto front 而需要非线性 scalarization”也是错的。先声明 deterministic/stochastic policies、期望回报还是逐次执行效用。相关严格分析见 [Lu, Herman & Yu, ICLR 2023](https://cs.uwaterloo.ca/~y328yu/publication/lu-hy-23/lu-hy-23.pdf)。

## 最干净的目标：动作分布上的偏好 tilt

先冻结一份有足够动作覆盖的 reference policy \(\pi_0(a\mid s)\)。用真实 binary action 学习向量价值

\[
\mathbf Q^{\pi_0}(s,a)=\mathbb E_{\pi_0}\left[\sum_{t\ge0}\gamma^t\mathbf r_t\mid s_0=s,a_0=a\right].
\]

这里每个目标 head 的未来 continuation 都是同一个 \(\pi_0\)。不能分别拿“最快 specialist 的 speed Q”和“最安全 specialist 的 safety Q”相加，因为它们假设了互不相同的未来行为。

部署时给定 \(w\)，定义

\[
\pi_w^+(a\mid s)=\frac{\pi_0(a\mid s)\exp(w^\top\mathbf Q^{\pi_0}(s,a)/\tau)}{Z(s,w)}.
\]

它是 \(\max_\pi\mathbb E_\pi[w^\top\mathbf Q^{\pi_0}]-\tau\mathrm{KL}(\pi\Vert\pi_0)\) 的解。相同 base 与 vector-Q 支持不重训查询多个 \(w\)，但这是相对 base 的一轮 regularized improvement，不是对每种偏好的全局最优保证。若将整个 \(\pi_w\) family 继续迭代优化，critic 也要变成 \(\mathbf Q(s,a,w)\)，其 future policy 对应 \(\pi_w\)。

此式相对于 old policy 的 PMD 与相对于 uniform/其他设计先验的 entropy regularization，是不同目标。Reference policy 的零支持不会通过 tilt 获得新动作；不能再把本来缺失的 solver 支持解释成 guidance 问题。

### 决策一致性：改变 cell 的概率，而非重新布置 cell 内样本

固定 deterministic decoder \(D_s\)，令 cell \(\mathcal C_a=\{c:D_s(c)=a\}\)，base cost density 为 \(p_0(c\mid s)\)。取任意真实动作效用 \(V_w(s,a)\)，例如 \(w^\top\mathbf Q^{\pi_0}\) 或与 \(\pi_w\) continuation 一致的 \(Q_w^{\rm old}\)，定义

\[
p_w(c\mid s)\propto p_0(c\mid s)\exp(V_w(s,D_s(c))/\tau).
\]

由于指数权重在每个 cell 内是常数，严格有

\[
\pi_w(a\mid s)\propto\pi_0(a\mid s)e^{V_w(s,a)/\tau},\qquad
p_w(c\mid a,s)=p_0(c\mid a,s).
\]

它也是给定目标动作分布的最小 KL lift，因为

\[
\mathrm{KL}(p\Vert p_0)=\mathrm{KL}(\pi\Vert\pi_0)
+\mathbb E_{a\sim\pi}\mathrm{KL}(p(c\mid a)\Vert p_0(c\mid a)).
\]

这给此前 SolverCell 的动机一个更准确的方向：应把概率质量移向偏好认可的**决策**，而不是假设把 endpoint 投影到另一处 cell 内就自动改善 policy。上述 identity 是标准 disintegration/KL chain rule，不单独构成新定理；真正待检验的是能否用低成本的噪声层级引导近似这个动作目标。

如果 native 执行还有固定噪声 \(\eta\)，实际动作是 \(a=D_s(c,\eta)\)，则上述 cell identity 适用于联合变量 \((c,\eta)\)。联合 tilt \(p_0(c,\eta)e^{V_w(D_s(c,\eta))/\tau}\) 一般改变 \(\eta\mid c\)。若坚持固定执行 kernel \(K(a\mid c)\) 而只改 center density，用 \(\exp(\mathbb E_K[V_w]/\tau)\) 会得到另一个受限 center-policy 目标，**不能**据此声称最终动作满足同样的精确 tilt。应将所有执行随机性纳入实际 sampler，或明确承认固定执行 kernel 的目标限制。

## 正确的噪声层级引导不是目标 score 的线性和

设 base generator 的**实际有限采样链**为

\[
P_0(x_{0:K}\mid s)=\mu(x_0)\prod_{k=0}^{K-1}K_{0,k}(x_{k+1}\mid x_k,s),\qquad a=D_s(x_K).
\]

所有执行噪声都应包含在这条链里；\(D_s\) 是原来的黑盒可行 solver。取 \(U_w(s,a)=w^\top\mathbf Q^{\pi_0}(s,a)/\tau\)，则正确 potential 为

\[
h_k^w(s,x)=\mathbb E_{P_0}\left[e^{U_w(s,D_s(X_K))}\mid X_k=x,s\right],\qquad \Psi_k^w=\log h_k^w.
\]

终端 \(h_K^w=e^{U_w\circ D_s}\)，并满足 \(h_k^w=K_{0,k}h_{k+1}^w\)。精确 Doob kernel 是

\[
K_{k}^{w}(x'\mid x,s)=K_{0,k}(x'\mid x,s)\frac{h_{k+1}^w(s,x')}{h_k^w(s,x)}.
\]

**两个不能省略的条件：**随机初始分布也要改成 \(\mu^w(x_0\mid s)=\mu(x_0)h_0^w(s,x_0)/Z(s,w)\)；有限 Gaussian base kernel 经任意 \(h\) tilt 后通常不再是 Gaussian。保留同一个 noise prior 然后只添加 drift，或做一次 Euler Gaussian mean shift，都不能默认称为精确 finite-step terminal tilt。

连续 generative-time SDE 若 diffusion covariance 为 \(A_t\)，Doob drift correction 为 \(A_t\nabla_x\log h_t^w\)，但这要求相应连续过程和精确 \(h\)。对 \(Q(s,D(x))\) 直接求 \(\nabla_x\) 通常得到几乎处处零/边界不可微，latent critic 的任意光滑梯度也不等于这个条件期望梯度。对黑盒 solver，更自然的是使用终端评价与条件采样，而不是对 solver 反传。

即便 \(U_w\) 线性，\(\Psi_k^w\) 也通常不是各目标势的线性和：

\[
\log\mathbb E[e^{\sum_j w_jQ_j/\tau}\mid X_k]
\ne\sum_j w_j\log\mathbb E[e^{Q_j/\tau}\mid X_k],
\quad
\log\mathbb E e^{U}\ne\mathbb E U.
\]

条件 cumulant expansion 含有 \(\tfrac1{2\tau^2}\sum_{ij}w_iw_j\mathrm{Cov}(Q_i,Q_j\mid X_k)\) 及更高阶项；目标冲突恰好会进入这些交互项。可训练一个 joint \(\Psi(s,x,k,w)\)，但不能仅分别训练 speed/safety scores 再相加并宣称 exact composition。即使干净终点的 product identity 正确，分别加噪后的 product 一般不是该目标的真实 noisy marginal。[Du et al. 的原始项目说明](https://energy-based-model.github.io/reduce-reuse-recycle/)

存在满足条件独立/互斥结构时的精确组合规则，但须验证那些条件。[LOGDIFF](https://proceedings.mlr.press/v306/alesiani26a.html) 正面研究了这种边界。速度与事故、平稳与时效一般不应默认条件独立。

## 最值得保留的算法假设：决策目标一致的偏好条件 continuation

建议先把它当作有明确 target 的控制实验，而非已经新颖的算法：

1. Base diffusion 使用已知 corruption、score/epsilon 网络与固定 finite sampler；在覆盖多个目标的共同数据上训练。共同 reference 与 solver 始终固定。强 FM 对照用同一数据训练。
2. 外层学习真实动作 vector-Q；首个实验冻结它。共享 \(\Psi_\phi(s,x,k,w)\) 预测上述 joint log-exponential continuation。训练样本来自真实 base suffixes 和终端 solver，只把 \(w\) 作为查询输入。普通 MSE 拟合 \(U\) 只得到 \(\mathbb E U\)，不是 \(\log\mathbb E e^U\)；log-mean-exp 有有限样本偏差，高权重 rare suffixes 必须监测。
3. 最小 inference candidate 用学到的 \(\Psi_k^w\) 对每步作明确标注的 approximate Doob guidance。例如 Gaussian base step \(\mathcal N(m_k,\Sigma_k)\) 对 \(\log h_{k+1}\) 在 \(m_k\) 处作一次线性近似，得到 mean \(m_k+\Sigma_k\nabla\Psi_{k+1}(s,m_k,w)\)。这里求导的是 learned noisy continuation，不是黑盒 decoder。除非 log-potential 确实仿射，它不等于精确 tilted kernel；若不校正初始 noise law，还多了一项初始分布误差。先检查 target fidelity，再讨论收益。
4. 粒子方法作为可校准的对照或预算允许时的候选实现：把 \(\widehat h_k=e^{\Psi_\phi}\) 当作 look-ahead potential，并保留 importance correction。一个无需 solver 梯度的简单 SMC：初始权重 \(G_0=\widehat h_0(X_0)\)，base propagation 后乘 \(G_{k+1}=\widehat h_{k+1}(X_{k+1})/\widehat h_k(X_k)\)，令最终 \(\widehat h_K=e^{U_w(D(X_K))}\)。路径乘积 telescopes 到终端权重；中途可以按 ESS resample。**有限粒子样本不是 exact samples**，consistency 是粒子数趋于无穷且满足相应条件；若使用 drift-guided proposal，还需加入 \(K_0/q_{\rm proposal}\)。Clipping 权重改变目标。

这些 guidance/Doob/SMC 的基础数学不新。随机 diffusion 的潜在实际收益在于：对有希望的 prefix 保留或复制后，多个 offspring 可以用独立后续 noise 产生不同完成结果。确定性 flow 对同一个 prefix 完全复制后，后续也完全相同；但它可以用多初始点、条件 guidance 或额外 stochastic kernels 弥补，因此必须做相同预算的 flow restart/rerank 与 noisy-flow 对照。首版的核心是决策目标与偏好响应的校准，不同时叠加完整的粒子学习系统。

运行时无需重训，指已训练模型能对其覆盖的偏好域作查询；不是承诺任意 OOD 权重或训练时从未出现的新目标。\(N\) 个粒子、\(K\) 步会花费 \(NK\) 次 denoiser evaluation 及 potential evaluation，最终通常有 \(N\) 次 solver/Q 调用。不能把它和只生成一个 flow sample 的系统直接比较后称为 diffusion 优势。

## 两种“中途切换偏好”不能混在一起

**在同一次生成链内切换。** 若旧 \(w\) 的当前粒子确实近似目标 marginal \(p_k^w\propto p_k^0h_k^w\)，切换到 \(w'\) 后应按 \(h_k^{w'}/h_k^w\) 重加权，再用新目标的 transition/weights。对单个已经选定的旧 prefix，仅替换 guidance 并继续，只实现“条件于该 prefix 的新偏好完成”，通常不是从 \(\pi_{w'}^+\) 重新采样。需要 restart、可校正的粒子群或明确接受这种 conditional repair 语义。早期剪掉的 modes 也不会凭一个新权重自动复活。

**在环境轨迹中切换。** 固定整段 trajectory 的 \(w\) 时，\(w^\top\mathbf Q^{\pi_w}(s,a)\) 与对应 scalar Q 一致；但若未来 \(w_t\) 会变化，需要指定 preference transition law 或可观察计划，把 \((s_t,w_t)\) 作为 augmented state。例如

\[
Q(s,a,w)=\mathbb E[w^\top r+\gamma Q(s',a',w')],\qquad a'\sim\pi(\cdot\mid s',w').
\]

未来权重变化时，不能把当前 \(w\) 提到整个未来 reward sum 外面。若未来偏好未知，应明确采用“每次查询假设当前偏好持续、下次再规划”的语义；不声称对任意未知 switch schedule 最优。偏好改变后，先前已发生的事故、能耗或时间损失也不可逆。

Replay hindsight relabeling 可以用保存的 reward vector 重算任意 \(w^\top r\)，但下一步 target 必须使用该 \(w\) 对应的 target policy，或使用规定的 \(w'\) 转移。把旧行为轨迹的累计 reward 重新加权，并不会把旧动作序列变成新偏好 policy 的 rollout；这是 off-policy evaluation 问题，不能靠 reward relabeling 消除。

## 创新门槛与最小判别实验

现有 LSFlow 固定 reward 的问题应先修成“偏好进入 actor、critic、replay target 和 evaluation 的一致接口”；这属于必要 baseline 工程。随后再判断 diffusion 的随机分支和多尺度条件价值是否增加了价值。若只改 actor 输入而不改 continuation，控制失配不能归因于 flow。

值得检验的更窄假设是：**在 objective interaction 明显、偏好改变且 oracle/推理预算有限的组合任务中，共享的 joint preference continuation 能否让随机 suffix repair 比从头采样或直接条件 FM 更省计算，同时保持可行性和可测量的偏好响应。** 它仍需与 MODULI、PC-Flow、discrete CTMC conditioning、TDS/TRI-TSMC 和通用 preference-conditioned RL 正面区分，不能现在就宣称足够创新。

首个测试应冻结一个小动作空间的准确 vector-Q 与共同 base，枚举真实 \(\pi_w^+\)；测试固定 \(w\)、未训练的插值 \(w\) 和明确时刻的 \(w\to w'\)。比较：直接 preference-conditioned FM；同预算 full-sample reranking；joint-potential SMC；naive sum-of-guidance；相同 stochastic kernel 的 noisy FM。记录 target TV/KL、实际各目标、偏好变化的方向与幅度、switch latency、solver/NFE/Q/potential 总调用及 wall time，而非只看 hypervolume。

Hypervolume 好不意味着滑块有效；已有工作直接指出 preference-conditioned policy 可能忽略输入偏好却取得良好的传统 MORL 指标。[Controllability in preference-conditioned MORL](https://arxiv.org/abs/2605.10585) 所以还必须报告控制响应、局部单调性失败、跨 seeds 变化、极端偏好下的退化。未证明期望单调的环境不应把每个 sample 都要求严格单调。

若共享 joint potential 不优于相同预算的终点重加权，或者 noisy FM 取得相同结果，就不能把收益归为 diffusion 特性。若 benchmark 根本没有足够冲突的目标、现有 base 不覆盖可行 trade-offs，先修任务/数据与基线；不要用更强 guidance 掩盖这个问题。
