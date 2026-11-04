# SolverCell-RL collaboration rules

## Scope
- Study solver-cell diffusion policies for sequential RL with constrained combinatorial actions.
- This is an independent workspace; preserve sibling 2048 experiments and all unrelated remote processes.
- Read this file and plan.md at the beginning of each research task.

## Scientific contract
- Preserve a pinned, unmodified upstream LSFlow source under vendor/lsflow.
- Distinguish upstream reproduction, adapted baselines, mechanism tests, and new method results.
- Compare fixed-budget complete curves, final-window return, normalized AUC, feasibility, mode coverage, and total compute including solver work.
- Initial tuning uses one declared training seed; it is not replication evidence.
- Change one declared factor at a time or predeclare a factorial. Retain degraded and failed arms.
- All new training paths support a small debug run, durable configs, source hashes, logs, checkpoints, and explicit completion status.
- Never call a queued/live worker a completed experiment or a numerical smoke test scientific validation.
- Do not modify source/configuration used by live workers. Deploy immutable snapshots.

## Remote execution
- Authorized target: wflow-server. Inspect live resources before launches; preserve unrelated jobs.
- Use a single controller per campaign and mark actual training workers as llmagent.
- Record worker PID/start time, host/device, source identity, config, progress, and exit status.
- Never print or retain credentials.

## Paths
- Local workspace: /home/serenade/AI/src/RL/SolverCell-RL
- Intended remote workspace: /home/li/SolverCell-RL
