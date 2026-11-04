# 从固定回报到决策一致的偏好可控生成

日期：2026-11-04。本文针对用户提出的多步生成偏好控制重新审查 LSFlow。它是已核源码的研究设计，不是已训练的偏好策略。没有修改原始环境、训练源码、历史配置或 checkpoint，也没有新增 RL 任务。

## 研究判断

偏好可控生成是比“更短的 endpoint transport”更有针对性的研究方向。目标应当变成：在相同环境和可行约束下，一套模型能否随部署偏好改变其**实际组合决策分布及长期行为**，而不为每个偏好重训完整 actor。

Diffusion 提供已知 corruption、多尺度 continuation、随机 suffix 的分叉和引导接口。但是 preference conditioning、classifier guidance 与多目标生成本身并非 diffusion 独占，也不是空白：[MODULI](https://arxiv.org/abs/2408.15501) 已做偏好条件 diffusion planner；[PC-Flow](https://ojs.aaai.org/index.php/AAAI/article/view/37971) 与 [PreferenceFlow](https://arxiv.org/abs/2609.36872) 已对 flow 做偏好引导。必须把贡献集中在组合 solver 与顺序 RL 中的决策一致性和实际成本，而不能只给 denoiser 增加一个权重向量。

## 原 benchmark 的目标与可研究的偏好

四个 synthetic RMAB domains 的原目标都是单一累计健康/engagement reward；Disease 是折扣的阳性发现数。下表中的新偏好都是明确的 benchmark 扩展，不能冒充论文原生多目标任务。

| Domain | 不改 dynamics 可定义的目标 | 不应直接宣称存在的偏好 | 判断 |
|---|---|---|---|
| Assignment | 健康收益与实际异质服务资源消耗；预先定义 cohort 的服务/健康收益 | 工人疲劳、动态容量、assignment 舒适度：现有状态/动作没有这些语义 | 最适合主要机制验证；存在精确同状态收益–成本冲突 witness |
| Intervention | 总收益、不同响应 cohort 的收益、intervention 数量/新声明的软成本 | good/bad response cohorts 不是人口统计标签；无已建模治疗副作用或安全风险 | 当前实现齐全，适合低成本机制原型；群体目标要先固定分组与尺度 |
| Scheduling | 健康收益与预定义群体的服务分配；可变服务人数下的资源使用 | 固定人数时总 assignment 数等于2倍人数，不能充当独立“舒适/工时”目标；具体工人分配被丢弃 | 适合结构化约束验证；工人层偏好需要保留完整排班解 |
| Routing | 完整 route 接口建立后，可比较节点收益与 route length | 当前 cost decoder 不是路线求解器；没有安全/舒适/行驶时间奖励，不能把节点选择当驾驶风格 | 原则上很匹配偏好叙事，当前发布路径必须先修复 |
| Disease testing | 不同预定义群体/组件的早期发现、有限时域覆盖分配 | 非传播/治疗仿真；未建模检测伤害；固定或最终穷尽检测数常使成本轴退化 | 顺序探索意义强，但 sampling/RNG/截断指标和数据可用性需先审计 |

Assignment、Scheduling 的详细源码及 witness 见 `PREFERENCE_BENCHMARK_SCHEDULING_ASSIGNMENT.md`；Intervention、Disease 见 `PREFERENCE_BENCHMARK_INTERVENTION_DISEASE.md`。

Intervention 还已有严格的同预算、native-decoder 可达冲突 witness：固定40个 arms 都在状态2，10个低响应组导向 interventions 给两组的归一化期望奖励约 `(0.808913,0.168500)`，10个高响应组导向 interventions 给 `(0.184000,0.953863)`。两者都能用显式 native costs 解码出来，权重改变会反转其排序。该有效状态的两动作比较不是完整 Pareto front，也不是典型访问频率或长期性能结论。分量与5000条既有 replay 的一致性核验见 `../reports/preference_component_audit/summary.json`。

### Routing 的实际接口问题

Pinned `RoutingRmabApproximator.solve_from_coeffs` 只根据40维分数选 top-B，再尝试并不存在于 RoutingRMAB 的 optional mask/projection。`RoutingRMAB.step` 也只检查数量，不验证这些点是否能沿给定长度的路线访问。单独存在的 MIP 路径约束不会自动保护这条 cost-decoder 路径。

`analysis/preference_routing_audit.py` 直接提取并执行该方法，使用发布的40节点/56边图、source10、B10、最大20条边，固定seed20261104抽32个独立Gaussian coefficient vectors。对选中集合求精确最短闭合 walk（允许重复节点，已经放松了 simple-cycle 约束），**30/32个集合仍需超过20条边**，最大31条边。DP另经四点全排列枚举核验。

这证明了该接口确实能输出不符合路径长度约束的节点集合；它不是已训练 LSFlow 的违规率。其余2个集合也未被认证具有满足条件的 simple cycle。结果见 `../reports/preference_benchmark_audit/routing_interface.json`。

若偏好依赖行程、工人 assignment 等辅助变量，必须保存完整 solver solution `(a,z)`，并让 reward/critic 评价它。当前只返回患者/节点集合 `a`，则同一 `a` 下多个不同 `z` 的风格被隐藏，任何 diffusion guidance 都无法控制已经从接口丢失的自由度。

## LSFlow 在偏好问题上的具体缺口

1. **训练对象固定为一个标量回报。** Actor 只接收状态与 flow time，critic 没有 preference 输入，scalar replay 将目标组件合并。已有 scalar critic 不能在部署时凭一个新权重恢复任意其他偏好的价值函数。
2. **提议和选优同时绑定原目标。** 原 best-of-K 仍由原 scalar Q 选样；只给 actor 增加条件、但保留原 selector，会把偏好引导结果再次按旧偏好筛掉。原 vMF/tangent exploration 与 guidance 后的执行法则也应统一。
3. **高单目标回报不要求保留互相冲突的模式。** 从一个已经只覆盖原目标高分动作的 base 出发，guidance 不能可靠找回缺失的资源节约型/不同群体模式。需要多偏好行为覆盖、明确参考分布和覆盖测量。
4. **动作和 latent 的等价关系未被用于偏好目标。** 本研究不再改变 cell 内代表点，而是只调整产生不同组合决策的概率质量；这样避免把“偏好”偷换成任意 latent 风格。
5. **顺序价值必须对应偏好下的未来策略。** 不同偏好的 Q heads 不能来自各自不同的固定 continuation policy 后再任意相加。应使用 `Q(s,a,w)` 或 `psi^{pi_w}(s,a)`，并在 next-state action 和 Bellman target 中沿用同一 w。
6. **历史依赖指标需要对应状态。** Switching cost 需要 previous action；累计公平/覆盖需要累计统计；有限时域需要 remaining horizon。CVaR 是轨迹风险目标，不是简单增加一个即时 reward head。

原本的 temperature/alpha/lambda schedule 和 vMF concentration 调节的是权重锐度、探索或更新幅度，不能替代健康收益与资源消耗之间的语义偏好。偏好 w 与采样温度 tau 必须分开。

这些缺口可以被 preference-conditioned FM 修复。因此核心对照应是“已正确扩展的条件 FM”，原始单目标 LSFlow只作外部系统参考。

## 更集中的候选：偏好作用于决策概率，diffusion 负责逐步实现

### 先保留目标向量

首轮只研究两项容易核查的目标：Assignment 的健康收益和异质资源消耗。所有方法使用相同原始 transition dynamics、相同 assignment feasibility；新增回报向量 `(r_health,-cost)`。成本不是被调大的硬容量，偏好也不能放松原可行约束。

给定部署偏好 w，使用共享的、偏好条件的长期价值估计；例如 `Q_w = w.T @ psi(s,a,w)`。每轮策略改进固定 critic、参考生成器和 w 取样协议。原始 vector rewards 要在 scalarization 前保存，归一化常数只由训练数据或已知界确定。

**不能为此丢掉 LSFlow 避免反复调用昂贵 solver 的优点。** 上式先规定理论 action-value 语义，不要求每个中间去噪状态都调用 MIP。噪声层的 joint potential 应摊销终点评价，只在生成/校准终点样本时解码。若在线学习直接改成 action critic，next-state TD target 的额外解码成本也要计入；Assignment 上不能用每个 minibatch 大量 MIP 调用换收益后与 native 的无重复解码更新宣称公平。

一个工程选项是保留偏好条件的 latent vector critic，并要求它近似 `Q_c(s,c,w)=Q_a(s,D(c),w)` 的语义，同时用已执行/共同终点样本核验决策一致性。近似 critic 可能仍在 cell 内产生虚假的偏好差异，因此 deterministic-cell 恒等式只对真实 action utility 精确，不能自动移植成神经 critic 的保证。第一轮冻结准确 utility 的实验将这类误差与生成误差分开，之后才决定哪种 critic 接口值得其 solver 成本。

### 决策一致的终点目标

在固定状态、固定可行 decoder D 下，令 `p_base(c|s)` 为具有多偏好覆盖的旧/参考生成器。对本轮冻结的实际 action utility U_w，定义

\[
p_w^+(c\mid s)=\frac{p_{base}(c\mid s)
\exp\{U_w(s,D(c))/\tau\}}{Z_w(s)}.
\]

对每个 cell 积分，得到

\[
\pi_w^+(a\mid s)=\frac{\pi_{base}(a\mid s)
\exp\{U_w(s,a)/\tau\}}{Z_w(s)},\qquad
p_w^+(c\mid s,D(c)=a)=p_{base}(c\mid s,D(c)=a).
\]

因此它调整的是动作质量而非 cell 内任意坐标；在相应支持集上，它也是指定动作分布相对 base 的最小 KL lift。这个结论来自标准 KL 链式分解，本身不作为新定理。

若 U 为本轮 `Q^{pi_old,w}`，这是明确的 KL-proximal actor improvement 目标；若始终对固定 base 做 tilt，则应声明固定-reference 正则目标及相应 Bellman 语义。不能混称。额外随机 execution noise、best-of-K selection 若保留，必须纳入终点 action law；不能套用 clean deterministic D 的公式后仍执行另一条 kernel。

特别地，固定执行核 `K(a|c)` 下，只对 center 使用 `exp(E_K[U]/tau)` 一般不实现所需的 action tilt。将执行随机数 eta 并入联合 latent `(c,eta)`，再按 `exp(U(D(c,eta))/tau)` 倾斜可恢复对应联合/action恒等式，但同时改变了 eta|c。若坚持 native K 原封不动，应明确接受受限策略族目标，并另外检查实际动作分布，而不是宣称上述 deterministic-cell exactness 仍然成立。

### 偏好沿实际随机续程传播

对固定 base 生成路径 `C_K -> ... -> C_0`，正确的软 continuation 是

\[
h_k(s,c,w)=\mathbb E_{base}
\left[\exp\{U_w(s,D(C_0))/\tau\}\mid C_k=c\right].
\]

不是 `exp(E[U|C_k]/tau)`；也一般不是分别训练各目标 guide 后把 scores 简单线性相加。噪声较大时，它评估当前尚未决定的续程中各偏好模式的质量；接近终点时，它分辨 solver 决策边界两侧的效用。

可学习共享偏好条件的 log-h，然后对实际 finite kernels 做近似引导或加权粒子校正。有限步 Doob 倾斜核通常不再是同方差 Gaussian；随机初始分布也一般需要 h_K 重加权。简单加 `grad log h` 不自动等于有限步精确终点采样。具体路径、初始 law、采样误差和再采样成本必须记录。详细推导与 primary prior art 见 `PREFERENCE_GUIDANCE_METHODOLOGY.md`。

一个更具体的机制假设是**联合续程效用的交互信息**。若同一 suffix pool 给出向量效用 V，则任意 w 的 target query 都可使用 `log E exp(w.T V/tau)`。在该层生成不确定性下，其形式展开含 `w.T E[V]/tau + w.T Cov(V) w/(2*tau^2)` 及高阶项；第二项显示了相互冲突/协同目标的信息为什么会被分别训练后相加的 guides 丢失。它是生成续程的不确定性，不应被解释为事故风险或 distributional RL 的完整 return law。

均值/协方差近似可以用作便宜的诊断，但强 guidance、稀有模式或高阶相关会使二阶截断失真；不能凭这个展开宣称已经有高效新算法。最直接的因果测试是 joint preference potential 对比 naive linear guide composition，保持 suffix 数据和计算预算一致；再与同样可用 joint potential 的 flow 比较。

先不叠加所有机制。Frozen-Q 小问题上先确认偏好目标与引导分布，再选择最便宜的实现。此前单坐标 Gaussian 条件积分的负面成本检查继续有效，不因改成偏好目标而重启它的训练 sweep。

## 什么才可能构成贡献

候选研究问题是：**在硬组合约束下，同一生成器能否用校准过的、多尺度偏好引导，保持决策级语义一致，并在未训练偏好及部署切换时，以匹配计算成本恢复高质量的行为权衡？**

其中的 preference conditioning、Doob transform、SMC 与 vector critic 都是已有工具。足够强的工作需要进一步给出：可验证的 solver/action-level distribution target；偏好控制与长期 Bellman 语义的一致性；固定成本下的显著可控性/泛化收益。仅将这些已知组件列在一起，尚不能确立论文创新。

## 最小验证与对照

1. **目标存在性。** 保留同状态可行解的收益–资源冲突 witness；检查第二目标是否恒定、与原奖励同序或只是硬约束重命名。Assignment 已有一步 witness，不据此宣称长期 Pareto frontier。
2. **冻结效用的偏好分布恢复。** 在可枚举子实例或可校验 oracle 上比较精确目标、conditional FM、终点 importance/reranking、逐步 diffusion guidance。共享 base support、utility 和总 NFE/Q/solver/wall-clock，保留 ESS、模式覆盖和误差。
3. **偏好控制的校准。** 对冻结的收益 H(a)、成本 C(a)，若 `pi_lambda ∝ pi_base exp((H-lambda*C)/tau)`，则 `d E_lambda[C]/d lambda = -Var_lambda(C)/tau <= 0`。该单调性提供便宜的目标一致性检查；当长期 Q 本身随lambda改变或策略未收敛时，不将它当作逐状态普适保证。
4. **顺序 RL。** 比较固定偏好专家、一个 preference-conditioned FM、条件 diffusion、guided diffusion；guidance 若可迁移给 noisy FM，也为其提供相同模块。原始 LSFlow 保留参考；不以原始单目标策略无法改偏好作为 diffusion 胜利。
5. **同模型部署。** 在预先划定的 train/heldout preference regions 上报告 expected utility、每轴原始指标、固定参考/尺度的 Pareto hypervolume、feasibility 和总成本。所有 actor/guide/critic 是否见过 heldout w 要披露。中途切换偏好先采用声明的“此后固定新w”协议；已发生的资源使用和健康变化不会被重置。

首轮的主张不包括 CVaR、安全约束、临床公平或舒适性；这些需要各自的环境语义、状态与风险验证。先用已有 Intervention 两响应群体任务核验偏好机制，再以 Assignment 的真实资源权衡作为主要结构化任务；Routing和Disease先完成上述接口/采样审计。
