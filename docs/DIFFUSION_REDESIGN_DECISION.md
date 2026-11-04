# 让 diffusion 的随机去噪过程承担真正的策略改进

日期：2026-11-04。研究决策与候选算法设计；不是新算法已经训练完成的报告。已完成的18个 native-structure runs 见 `../reports/native_revision_study_20261104/RESULTS.md`。本次不再扩展旧 cell-margin sweep。

## 决策

最值得检验的研究命题是：**在有不连续 solver decoder 的组合策略中，利用真实随机去噪核，把终点的离散决策边界解析边缘化，能否以更少的价值查询获得更可靠的策略改进信号？** 下述便宜核验已给出第一版估计器的负面答案；因此目前不将它列为应立即扩大训练的高效算法。

这比“diffusion 更能表示多峰”“投影后的运输距离短”“可行解全路径可行”更接近可证伪的 diffusion 算法问题。上述三个说法都不足以区别于 flow matching。候选核心是 solver 的条件分区与随机转移核相配合的梯度估计；continuation critic、路径 likelihood 和 proximal regularization 是已有基础设施，不能一起打包宣称新颖。

**更新：等 Q 查询的初步机制核验未通过。** 在3个固定的真实 late-replay states、4个事先指定坐标组成的12个 cases 上，使用已知即时奖励 oracle、固定 mean0/sigma0.5 的 primitive20 Gaussian 测试核，每 case20,000次估计。两查询单坐标条件积分的完整梯度 covariance trace，相对于平均两个独立 score-function samples，为1.473–1.862倍，12/12均更差。它确实优于一个 score sample，但未能补偿第二次 Q 查询成本。积分与有限差分核验通过，不代表效率通过。这不是已训练 diffusion 的实际 kernel，也不是 learned-Q /长期RL结果；它足以阻止把本版估计器直接扩大为训练 sweep。结果见 `../reports/continuation_branch_audit/summary.json`，推导见 `NOISE_CONTINUATION_CRITIC.md`。

必须保留 stochastic flow 的强对照。如果 SFM 加相同 Gaussian 末步、相同边缘化后获得全部收益，证据支持的是 solver-aware stochastic policy improvement，而不是 diffusion 训练范式具有独占优势。研究应接受这个结果。

## Diffusion 相对当前 LSFlow 参数化的可利用之处

| 特性 | 可以利用的算法接口 | 不能推出的结论 |
|---|---|---|
| 已知 corruption 与多噪声尺度 denoising | 给定部分受损候选，从声明的噪声尺度继续恢复；同一模型学习不同不确定性水平 | FM 不能修复、不能条件生成；它也可扩展成条件模型 |
| 非退化随机 reverse kernel | 计算有限步条件 log probability，给生成过程分配 RL 信号，无需 solver 可微 | 最终动作 marginal likelihood 可精确计算 |
| 局部随机分支 | 同一中间状态作配对反事实、条件积分或再采样 | 无代价增加探索；branch/Q/solver 成本都必须计入 |
| score 与 noise scale | 对真实生成分布作尺度一致的校正；评估噪声跨过哪些决策边界 | Gaussian 噪声天然符合组合动作语义，或 late noise 越大越好 |

FM 与 diffusion 在分布表达上没有这样简单的优劣关系。FM 可建模多峰；随机初值本来就产生随机动作。Score SDE 可有同 marginal 的 probability-flow ODE；FM 也可使用 diffusion probability paths。[Score SDE](https://arxiv.org/abs/2011.13456)、[Flow Matching](https://arxiv.org/abs/2210.02747)。

相对确定 ODE，随机 kernel 的概率接口很实用，但 [ReinFlow](https://arxiv.org/abs/2505.22094) 已给 FM 加可学习 Gaussian 转移并精确计算有限步路径 likelihood。因此比较对象必须写成具体算法与预算，不能把模型名称当成互斥能力清单。

## LSFlow 的优势应保留，局限应按证据排序

LSFlow 的优点是 latent generator + 可行 solver + 避免反复调用 solver 的 latent critic。这套分工仍然有价值。完整 source audit 见 `METHODOLOGY_AUDIT.md`。

1. **内部生成过程没有直接接受决策改进信号。** 当前 actor 用最终 latent 样本的权重训练 RFM；我们的现有 diffusion 替换也主要沿用 endpoint-weighted regression。额外 noise head 不等于已经利用了 reverse-kernel likelihood 或路径 credit。训练损失较低、运输较短都不能保证更好的动作分布。
2. **critic 的平滑核不是实际去噪的 continuation law。** Native critic 看到40维 cost center；执行/target smoothing 使用固定 tangent perturbation。Eval 又不加该执行扰动，并进行 best-of12。它是已定义的平滑 surrogate，但不能不加说明地等同于实际部署策略的 action-Q。真正的 diffusion critic 应明确噪声时钟、冻结的续程核和最终执行规则。
3. **continuous geometry 与 decision geometry 不一致。** 同一 clean cell 内的坐标变化可能不改变动作；但改动 center 会改变加噪后的决策概率。Nearest-cell relabeling 既不是 action-level KL 最小提升，也不保持 native noisy kernel。为旧点赋予恰当权重，在 deterministic decoder / cell-constant Q 条件下反而保留了 KL 最小 lift 的旧条件分布。
4. **当前 benchmark 的冗余和支持缺口需要独立控制。** 原40维 latent 的 decoder rank8，正体积可达效果268/9318。但 late learned replay 的512个状态，补全支持的最优即时奖励平均增量只有0.001042，p95为0，原奖励约16.20。不能把大计数缺口写成已证实的主要回报障碍。
5. **严重候选塌缩没有被本次诊断支持。** 原 quotient-sphere 的 K12 probe 平均约10.19个不同动作，且这些是各方法自己的 replay states，不是配对因果比较。Cell 更分散而回报更低。多样性数量不是核心成功标准。

## 一个集中的算法候选：随机末步的决策边界边缘化

先用 primitive20 costs，保留 latent-to-solver 范式，避免原 W 的表达限制混入机制验证。Decoder 为正分数 up-to-budget top-k。最终 costs 可归一化；正比例缩放不改变该 decoder 的动作。

策略是一条真正执行的有限 Gaussian reverse chain：

\[
c_{k-1}\sim P_{\theta,k}(\cdot\mid s,c_k),\qquad
a=D(c_0),\quad k=K,\ldots,1.
\]

Gaussian Markov chain 本身不构成 diffusion 的方法动机。真正的 diffusion 分支应绑定已声明的 forward corruption schedule，例如 VP/DDPM 的 alpha、beta，并由 epsilon-denoiser 参数化 reverse mean：

\[
m_{\theta,k}(s,c_k)=\alpha_k^{-1/2}
\left(c_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\epsilon_\theta(s,c_k,k)\right).
\]

可先用同一批可行 behavior/replay endpoint costs 作 denoising 初始化，无需专家示范；FM 对照用相同 endpoints、相同权重和初始化训练预算。随后通过实际 finite kernels 优化动作价值。任意时刻的 corruption/denoising 语义才是 diffusion 的结构，Gaussian kernel credit 是对它的利用方式。若只训练任意多步 Gaussian controller 而不建立这种关系，应明确称为 stochastic generative policy。

先固定正的 variance schedule；尤其最后一步 `sigma_1 > 0`。不在测试时关闭随机末步，不再另加未计入目标的 tangent execution noise。首次机制比较用 K candidates=1；如使用 best-of-M，应对所有方法一致启用，并把选择操作纳入目标/评估解释。

固定正的末步 variance 是有意的策略定义；它不等于通常为零的 DDPM posterior 最末步 variance，也没有 finite-step 精确恢复 endpoint law 的保证。所有梯度和评估均针对实际采样的这条 policy。

对最后一步，给定 `(s,c_1)`，设

\[
c_0=m_\theta(s,c_1)+\operatorname{diag}(\sigma)\epsilon,
\quad \epsilon\sim N(0,I).
\]

固定其余19个分数，只看第 i 个分数。它有两个可行动作：未选 i 的 `a_i^-`，以及选 i 并在必要时替换最弱已选项的 `a_i^+`。阈值是

\[
t_i=\max(0,\text{其他分数中的第 B 大值}),
\quad p_i=\Phi((m_i-t_i)/\sigma_i).
\]

边界处 ties 在非退化 Gaussian 下为零概率。对任何冻结的 bounded action-value function，条件期望和条件均值梯度为

\[
\bar Q_i=(1-p_i)Q(s,a_i^-)+p_i Q(s,a_i^+),
\]

\[
g_i=\frac{\phi((m_i-t_i)/\sigma_i)}{\sigma_i}
\,[Q(s,a_i^+)-Q(s,a_i^-)].
\]

对其他坐标取期望就得到真实 Gaussian-policy mean gradient。无需 `dD/dc`，也无需对 action-Q 求动作导数；Q 可以是非线性的。这里最有价值的是**同一决策边界两侧的价值差**，而不是在同一 cell 内移动任意 latent 点。

首先核验的版本使用**两次 Q 查询得到的完整条件 score gradient**。令父状态 baseline 为 beta。除上面的第 i 个分量外，其余分量为

\[
\bar g_j=\frac{c_j-m_j}{\sigma_j^2}(\bar Q_i-\beta),
\qquad j\ne i.
\]

整个向量是原 Gaussian mean-score estimator 在给定其他坐标后的条件期望，所以能直接乘以该步 mean-network Jacobian；其协方差矩阵不大于一个原始 score estimator。这里固定父状态，其他终点坐标与分支 Q 值在反向传播时都要 detach，不可同时加入它们的 pathwise derivative。更早步骤的 score 项也可使用同一个条件期望 bar-Q；这不需要通过采样 prefix 再反向传播。

另一种方案是从20个坐标中均匀选 i，使用 `d*g_i*e_i` 的无偏估计，但它丢弃了其他分量的信息，坐标采样方差仍需测量，不作为首选。如果选择概率不均匀，此坐标估计需要相应逆概率权重。计算全部坐标各自的解析梯度则需要至多2d个 Q 查询。可批量共享 top-k 排序和网络输入，缓存完全相同的动作效果，但缓存不会改变比较对象的数学定义。

这是已知 score-function estimator 的条件期望，属于 Rao–Blackwellization / weak-gradient 家族。完整条件向量对一个原始 estimator 具有条件方差优势；**不保证优于同样两次 Q 查询的任意 control-variate estimator，也不保证最终 return 更高**。动作价值学习误差仍会产生偏差，稀有动作的反事实 Q 尤其需要校验。

## 怎样连接到整个 RL 算法

**环境价值层。** 训练 `Q_env(s,a)`，或在本环境中严格等价的 `Q_env(s,Wa)`。TD target 对应实际执行的终点 policy；Twin critics/target network 可保留。不得将 noisy-center native Q 当成两个离散分支的真 action-Q。

**生成价值层。** 对冻结的旧链定义

\[
C_0(s,c_0)=Q_{env}(s,D(c_0)),\qquad
C_k(s,c_k)=E_{P_{old,k}} C_{k-1}(s,c_{k-1}).
\]

它给出真实续程下每个噪声尺度的价值；生成时钟内部 discount 为1，环境时钟仍为 gamma。训练拟合的 C 时应固定目标 actor/critic，标明 moving-target 近似。最后一步用上面的两分支条件积分降低 terminal target/gradient 的采样噪声；更早步骤使用实际 stochastic transition likelihood 与 inner critic。冻结 critic 的 kernel PG 可用；off-policy replay 下不做修正不能声称无偏。

**Actor 层。** 首版以可核查的 likelihood-ratio / local trust-region 更新为准，最后一步替换为解析条件积分；保持所有 stop-gradient、采样来源和 KL 参考明确。不要在 PPO clipped objective 中只替换 reward 而把 clipping/ratio 当常数：要么边缘化整个条件 integrand，要么验证使用的未裁剪梯度目标。路径 KL 一般强于终点 action KL，不应无条件宣称等价。

这里的 multiscale continuation critic 本身已有 [BDPO](https://arxiv.org/abs/2502.04778) 和更直接的 [DIA](https://arxiv.org/abs/2609.12245) 先例；路径 RL 也有 [DDPO](https://rl-diffusion.github.io/) / [DPPO](https://arxiv.org/abs/2409.00588)。候选贡献只能集中在 solver-conditioned marginalization 对复杂决策 credit 的可计算作用，而不是基础 two-time-scale MDP。

初版不增加 entropy controller、adaptive masks、Doob PDE、SMC、graph diffusion 等模块。先确认集中机制有效，再考虑把尺度/分支预算分配给高价值的不确定决策。

## 第二、第三个旧想法的保留位置

Feasible exchange 可以定义组合结构一致的 corruption/refinement kernel，适合需要协调多项编辑且完整 reverse conditional 可计算的任务。但 swap-only 不连通 up-to-budget 集合，受控 Gibbs/local search 也不能只因“多步随机修改”被称作新 diffusion。它是后续替换 Gaussian kernel 的候选，而不是本轮同时引入的第二主模块。

Geometry-prior correction 应改为明确声明 action/effect 参考测度及其 objective。它不能恢复零支持，不应每轮重复除以初始先验，也不是 diffusion 独有优势。Gaussian 末步的条件概率已经属于实际执行 law，先准确优化这个 law，再讨论是否要换 entropy/reference 目标。

## 最小、按顺序可停止的验证

1. **冻结价值的梯度机制检查。** 在同一真实 instance states 和 synthetic utility 上核验条件积分、score-function 的均值和方差；同时比较1-call与2-call预算。使用明确已知的 myopic reward 时，仅验证估计器，不能转成长期 RL 结果。
2. **训练目标与生成器的交叉控制。** Primitive20、同 critic/数据/终点随机核：Gaussian one-step actor；SFM + Gaussian 末步；diffusion + plain kernel PG；diffusion + boundary marginalization。对最强 stochastic-flow 对照也启用相同 marginalization。报告 Q调用、decoder调用、NFE、训练 wall-clock，不能只匹配 integration steps。
3. **顺序 RL 验证。** 第一轮用固定 critic 看真实 held-out improvement，再以相同 interaction budget 配对种子进行短 RL。比较完整曲线、AUC、last-window、决策尺度上的梯度 SNR 和实际 kernel 一致性。若收益仅来自 primitive20，归为表达修复；若仅来自末步 Gaussian，归为随机决策边界优化；若 Gaussian one-step actor 同样有效，不主张多步 diffusion 必要。

当前第1步的首版等查询 variance gate 未通过，所以不推进第2/3步训练。需要先找到能一次消去较大块无决策意义随机性的结构，或使分支 Q 查询充分复用，并重新与同预算 independent/antithetic sampling 比较；这属于待研究条件，不是已经实现的补救。只有第3步显示多尺度 denoising 在匹配成本下继续有增益，才值得扩大到更多 LSFlow benchmarks 或更长预算。现有18个 runs 并未检验这里的新候选。

更优先的立项诊断是：冻结 actual-action Q、环境状态与真实去噪 prefix，测各噪声尺度的续程价值差是否存在可预测、决策相关的信息，以及拟合该信息所需的 Q/NFE 成本。不能用同一个终点标签反复回归得到很低 loss 就声称找到了这种信息。若多个随机层没有比相同末步噪声的 flow 或单步 Gaussian 提供额外收益，现有任务上就缺少使用多步 diffusion 的方法动机。不在当前负结果之后直接加 block masks、learned temperature 或 SMC 来躲开这个问题。

## 新颖性边界

还必须对照 [Perturbed Optimizers](https://arxiv.org/abs/2002.08676)、[Measure-Valued Policy Gradients](https://arxiv.org/abs/2107.09359)、[Gaussian top-k policy gradients](https://link.springer.com/article/10.1007/s10994-023-06444-4) 和 [Decision-Focused On-Policy Learning](https://arxiv.org/abs/2606.01081)。后者涉及 Gaussian/generative cost policies 与混合梯度，说明“生成 costs + solver + RL gradient”也不是空白。

当前合理的创新假设是：**利用 solver 的条件决策分区，消去随机去噪中对动作无意义的采样变化，并将得到的价值差反馈给有限步生成策略；在明确成本下证明方差/样本效率和长期 RL 收益。** 这需要机制证据、可扩展性和最近邻文献比较。尚未证明达到论文级创新要求。
