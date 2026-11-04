# 修复结果与仍未解决的部分

本轮依次实现任务尺度/基线修复、实际末端核修复和共同固定数据的 actor 隔离实验。前三项的代码与限定实验已完成；固定分布拟合仍未过门槛，因此第四项新的逐步引导没有启动。不能把这份报告称作“完整算法问题已经解决”或新的在线RL胜利。

## 一、修正了对任务退化的判断

上一轮 learned-policy 的低响应回报平坦，不足以说明环境没有可控性。枚举268个native open-cell决策，并用独立arm有限时域DP的上界作lookahead，在真实可行动作上逐步选优，能够找到有效的对立参考策略。DP本身放松了联合预算，不是精确最优Q；最终动作始终从真实native catalog中选取。

尺度只用新校准种子1200000–1200031确定，两个参考极端的原始收益跨度为87.072和563.566。使用独立验证种子1300000–1300031，冻结尺度对应的参考策略族得到：

| 请求低响应权重 | 有效原始低响应权重 | 低响应组原始回报 | 高响应组原始回报 |
|---:|---:|---:|---:|
|0|0|119.94|639.52|
|.25|.6833|133.68|637.62|
|.5|.8662|158.78|462.75|
|.75|.9510|189.33|178.54|
|1|1|195.08|80.45|

这表明原任务有可达的权衡，原有条件策略未能学到它。新目标为`w · (raw_components / frozen_scales)`，按同一尺度计算critic TD与评估效用，原始replay分量和原始benchmark reward继续保存。新分数不能与旧的raw scalarization直接比较。校准只是让请求权重的意义更合理，不是学到有效前沿的保证；模型已知的参考策略不是model-free learned baseline。

这一步也暴露出benchmark范围限制：当前native quotient只有268类常规可达动作。这个小catalog使诊断可校验，但不足以展示大规模组合生成的必要性。后续必须包含适当的小规模直接分类/枚举参考，或转向更大/更丰富的可行集；不能用当前结果宣称扩展性。

## 二、末端修复已进入真正的训练/推理代码

`repaired_preference_config`是显式v4入口，默认不添加最后一次噪声，并声明冻结尺度。现有配置的默认值继续对应历史噪声/单位尺度，以便旧checkpoint在原语义下加载。变化包括：

- actual transition、suffix、sample_path和guide suffix labels使用同一末端核；最后仍消耗相同随机数draw，便于单因素比较。
- 确定性末端没有Lebesgue Gaussian密度，调用其Gaussian log_prob会明确拒绝。
- checkpoint spec绑定末端核、预测类型和尺度；不同语义不能静默加载旧guide。
- x0预测仅作为可选择的参数化消融，epsilon仍是v4默认；未因新增接口而宣称x0有效。
- critic从原始分量除以冻结尺度形成TD目标；评估同时给出原始向量、原始scalarization及新目标scalarization。myopic基线也使用同样尺度。

104项测试通过；新的完整训练/评估/checkpoint路径完成4-update工程预检，训练轨迹仍是12/20步的部分episode，不能当作学术训练结果。上一轮864个新episode证明了末端噪声对旧固定模型的负面影响；这轮不把它重复计作新数据，也没有重新声称在线RL提升。

## 三、固定数据隔离有结论，actor问题尚未解决

三个方法共享64个训练context、32个完全不重叠的validation context、268个相同单位endpoint、同一精确有限目标概率和相同训练endpoint抽样序列；网络参数均30408，8个actor sampling rows、2000步优化、batch64、lr4e-4，三个新拟合种子。只有训练几何/损失参数化不同。x0和epsilon损失的尺度/时间加权含义不同，这不是仅换变量名的等价优化。

每个最终case8192个样本，四个未用于拟合的偏好。按三个种子平均的heldout TV为sphere .2525、epsilon-clean .2770、x0-clean .2913。目标自己抽样的有限样本TV约.072，表明差距不能只用直方图噪声解释。epsilon略落后sphere，x0也未带来稳定改善。所有固定预算曲线、noise controls和种子均保留，没有取best checkpoint。

同时，分布TV与决策效用不同：模型可能更集中于高效用动作，导致期望效用略超过某个软目标，却没有恢复目标分布。不能据此认定正确的偏好分布已拟合。相反，在本固定分布上加末端噪声有时略降低TV，也不推翻旧模型上去掉噪声能提高episode回报的结果：目标/模型/指标不同，不能把噪声好坏作一概而论。

sphere也没有通过绝对TV门槛，因此目前不能把全部问题归结为diffusion独有。有限步离散化、共同endpoint代表点的多模态拟合、优化预算和几何仍未分开。共同数据结果说明critic/replay并不是全部解释，不证明在线RL的critic或探索不再重要。特别是原reward低状态非单调，真实长期收益的信用分配和稀有状态覆盖没有被这个离线拟合实验修复。

## 四、遵守门槛，暂不增加新的引导机制

预设条件是diffusion mean heldout TV<=.15，且比sphere差不超过.05。epsilon和x0都未通过第一个条件；二者相对sphere的差值分别.0245/.0388。`decision.json`明确拒绝进入新的staged guidance，旧对照/负结果不删除，不根据heldout重新调门槛。

下一节点是确定共同固定分布拟合误差的来源：加入适当的直接分类正对照，并用已知有限混合分布的解析score/反向核分离预测误差、先验law与有限步近似；这些是后续待做项，本轮没有实施。之后才有依据决定重新参数化、调整几何或改变步数。现有证据不足以展开新的在线RL sweep或把SMC复杂度加回来。

## 完成与复现边界

全部九个服务器任务exit0，完成预设2000更新，source/data/config/runner、实际worker、计数、有限checkpoint和相同训练endpoint RNG均通过审计；本地九个checkpoint字节hash与服务器一致。CPU24/25的单controller已完成，零运行/排队worker，没有GPU训练。老实验checkpoint/配置/原始source记录保留，原代码可以从其Git版本或服务器冻结snapshot恢复；用新代码重跑老manifest会被source guard拒绝，这是预期行为。

[完整结果](RESULTS.md) · [固定门槛](decision.json) · [任务审计](task_audit.json) · [全部context指标](study.json) · [完整直方图](histograms.npz) · [服务器最终状态](runtime/final_remote_state.json)
