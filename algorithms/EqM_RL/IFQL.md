# IFQL: how it trains, how its critic works, and how the critic is used

Written 2026-11-09, and corrected the same day after a second check against the sources below. IFQL ("implicit flow
Q-learning") is one of the baselines in the FQL paper (Park, Li, Levine 2025). It is the best-performing method in that paper on
our main task, `humanoidmaze-medium-navigate-singletask-v0` (69 % success on task 1), which is why it is written up here.

**Sources** (everything below was checked against these; anything that could not be checked is marked as such):
- FQL paper: `papers/FQL.pdf` (arXiv 2502.02538v2). Main text Section 4.1 (rejection sampling, Equation 11) and Section 5.1;
  Appendix B (implementation details), C (ablations), E.1 (environments and rewards), E.2 (how each baseline was run);
  Tables 2, 3, 4, 5, 6 and 7.
- Official code of the FQL paper, <https://github.com/seohongpark/fql> (MIT license): `agents/ifql.py`, `agents/iql.py`,
  `utils/networks.py`, `utils/flax_utils.py`, `utils/datasets.py`, `envs/env_utils.py`, `utils/evaluation.py`, `main.py`, and the
  exact IFQL command for humanoidmaze in `README.md`.
- IDQL paper (Hansen-Estruch et al. 2023, arXiv 2304.10573), Sections 4.1, 4.2 and 6.3. IFQL is "the flow counterpart of IDQL"
  (FQL paper, Section 5.1); the IDQL paper defines the critic-based weights described in Section 9.3.
- OGBench 1.2.2 (in the `ogbench` conda env): `ogbench/utils.py`, `ogbench/relabel_utils.py`, `ogbench/locomaze/maze.py`, and
  the humanoidmaze-medium dataset itself. Used for how the dataset's rewards and masks are set (Section 2).
- Our own runs (Sections 9.4 and 11): `sweeps/exp13_ogbench_aware.yaml`, `sweeps/exp14_ogbench_*.yaml`,
  `sweeps/exp15_ogbench_bestofn.yaml`, `sweeps/exp18_ogbench_aware_v.yaml`, the summaries in `sweep_runs/`, and `Aware/agents.py`.

## 1. Summary

1. IFQL trains three things **separately**, all from the first step, on the same batches of dataset rows:
   a value network V, a critic Q (two copies), and a flow network that copies the dataset's actions (behaviour cloning).
2. The critic is IQL's critic, unchanged. It is trained only at the dataset's own actions: during training, it never scores
   an action that a policy invented. It scores the flow network's proposals only when an action is chosen (point 4).
3. The flow network never sees Q or V. Every dataset row has weight 1.
4. The critic and the flow network are combined only when an action is chosen: the flow network proposes 32 actions,
   and the critic picks the one with the highest score.
5. So IFQL builds **no weights** from the critic during training. At decision time, the "weight" is 1 for the best candidate
   and 0 for the other 31.

## 2. Words used in this file

| word | exact meaning |
|---|---|
| state s | the numbers describing the situation at one time step; in humanoidmaze, 69 numbers (robot body and position) |
| action a | the command sent to the robot at one step; in humanoidmaze, 21 numbers (one per motor), each between −1 and 1 |
| reward r | always −1 or 0 (FQL appendix E.1: "rewards are always −1 or 0"). **In the dataset:** 0 on the rows whose state s is already inside the goal circle, −1 on all other rows, including the row whose next state s′ first enters the goal circle. **In the evaluation environment:** 0 on the step that reaches the goal, −1 on every step before it. See the note below this table |
| next state s′ | the state one step after s |
| mask | **in the dataset:** 0 on the rows whose state s is inside the goal circle (the same rows that have reward 0), 1 on all other rows. It stands in for the end of the episode: in the evaluation environment, the episode ends on the step that reaches the goal (appendix E.1) |
| dataset | 2,000,000 rows (s, a, r, s′, mask) recorded in advance (`humanoidmaze-medium-navigate-v0`, the training split). IFQL collects no new data in offline training |
| batch | 256 rows drawn uniformly at random from the dataset (with replacement) |
| discount γ | how much a later reward counts: a reward k steps later counts γᵏ times as much. Humanoidmaze: γ = 0.995 |
| return | the sum of all future rewards, each multiplied by its discount |
| floor of Q | the return of never reaching the goal: −1 / (1 − γ) = −200 for γ = 0.995. Every true Q value lies between −200 and 0 |
| Q(s, a), the critic | a network that takes a state and an action and outputs an estimate of the return after taking that action |
| Q1, Q2 | two critic networks with the same shape and different random starting weights; "min Q" = the smaller of their outputs |
| V(s), the value network | a network that takes only a state and outputs one number: how good that state is (defined exactly in Section 4, step 2) |
| target copy | a second set of weights that follows the trained weights slowly (Section 4, step 6). In IFQL, only Q1 and Q2 have one, and only V's training target is computed with it, so that this target does not jump around. Q's own target uses V, which has no copy |
| flow network v(s, x, t) | a network that takes a state s, a point x in action space (21 numbers) and a time t between 0 and 1, and outputs a direction (21 numbers) saying how to move x |
| behaviour cloning (BC) | training a network to reproduce the dataset's actions, with no notion of good or bad |
| expectile loss with τ = 0.9 | a squared error that counts with weight 0.9 when the prediction is too low and 0.1 when it is too high |
| GELU | a smooth curve applied to each number after a linear layer (like ReLU, without the sharp corner) |
| LayerNorm | rescales the 512 numbers of a layer to mean 0 and spread 1, then applies a learned scale and shift |
| IQL | Implicit Q-Learning (Kostrikov et al. 2022), the method IFQL takes its V and Q training from |
| IDQL | Implicit Diffusion Q-Learning (Hansen-Estruch et al. 2023): IQL's critic plus a diffusion BC model; IFQL is the same idea with a flow BC model |

**Note on the goal and the reward timing.** The goal circle is the set of x-y positions within 0.5 of the task's goal position
(task 1: (20, 20)). The dataset and the evaluation environment give reward 0 one step apart:
- OGBench sets the dataset's rewards and masks from the position in each row's state s, not in s′ (`relabel_dataset` in
  `ogbench/relabel_utils.py` reads `qpos`, and `ogbench/utils.py` aligns `qpos` with s).
- The evaluation environment checks for the goal after each step.
- Checked on the task 1 data: all 3,921 of the 2,000,000 rows with reward 0 have s inside the goal circle. All 209 rows that
  enter the circle (s outside, s′ inside) have reward −1 and mask 1.
- Effect: for every state s outside the goal circle, Q with the dataset's rewards = γ · (Q with the evaluation rewards) − 1.
  So the values learned from the data are lower by between 0 and 1: by 1 right before the goal, by almost 0 far from it.
  Both rank actions the same way.

## 3. The networks

| network | input | layers | output | copies |
|---|---|---|---|---|
| **Q1, Q2** (critic) | state and action joined: 69 + 21 = 90 numbers | 4 hidden layers of 512; each hidden layer is linear → GELU → LayerNorm; then linear → 1 | 1 number | 2 networks, different random starting weights |
| **target Q1, Q2** | same | same | same | 1 slow copy of Q1 and Q2; starts equal to them |
| **V** (value network) | state only: 69 numbers | same as Q: 4 × 512, GELU, LayerNorm | 1 number | 1 network, **no target copy** |
| **flow network** (actor) | state, point x (21 numbers), time t (one plain number, no sinusoidal embedding) joined | 4 hidden layers of 512, GELU, **no LayerNorm**; then linear → 21 | 21 numbers (a direction) | 1 network |

- Starting weights: Glorot-uniform (`variance_scaling(1.0, 'fan_avg', 'uniform')`), biases 0.
- States go into the networks as raw numbers: the code does not normalize states and does not scale rewards.
- Dataset actions are clipped to [−1 + 0.00001, 1 − 0.00001] when the data is loaded (`envs/env_utils.py`, `action_clip_eps=1e-5`).
- One Adam optimizer for all networks: learning rate 0.0003, optax defaults (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸), no gradient
  clipping, no weight decay, no learning-rate schedule.

## 4. Training, step by step

One training step is described below. It is repeated 1,000,000 times. All networks are trained from the first step;
there is no separate pre-training phase.

**Step 1. Draw a batch** of 256 rows (s, a, r, s′, mask).

**Step 2. Train V with the expectile loss.**
- For each row: q = min(Q1_target(s, a), Q2_target(s, a)), the target copies' score of the dataset's own action a.
- d = q − V(s).
- Loss = 0.9 · d² if d ≥ 0 (V is too low), and 0.1 · d² if d < 0 (V is too high). Averaged over the batch.
- Effect: a prediction that is too low costs 9 times more than one that is too high, so V(s) does not settle at the average
  Q of the actions taken at similar states. It settles between that average and the highest of them. Where exactly depends on
  how the Q values are spread, as the two examples show.
- Example 1: if the dataset's actions at similar states have Q = −120, −100 and −80, a plain squared loss puts V at −100 (the
  average); the 0.9 expectile puts V at about −85.5 (solve 0.9 · (−80 − V) = 0.1 · ((V + 120) + (V + 100))). Here V is
  closer to the highest Q.
- Example 2: if 99 actions have Q = 0 and 1 action has Q = 100, the average is 1 and the 0.9 expectile is about 8.3 (solve
  0.9 · (100 − V) = 0.1 · 99 · V). Here V is much closer to the average than to the highest Q.
- In plain words: V(s) is "how good s is if you act like the better actions in the data".

**Step 3. Train Q1 and Q2.**
- Target for each row: y = r + γ · mask · V(s′), with γ = 0.995 and V from the value network as it is now (not a copy).
- Loss = (Q1(s, a) − y)² + (Q2(s, a) − y)², averaged over the batch.
- V(s′) is used as a fixed number here: this loss does not change V.
- The target needs no action from any policy, only V(s′). Together with step 2 (V fitted at dataset actions only), this means
  Q is never evaluated during training at an action that is not in the dataset. This is what "implicit" refers to.

**Step 4. Train the flow network by flow matching (plain BC).**
- Draw x₀: 21 random numbers from a standard normal distribution. Let x₁ = the dataset action a.
- Draw a time t uniformly between 0 and 1.
- Take the point on the straight line from the noise to the action: xₜ = (1 − t) · x₀ + t · x₁.
- The correct direction at that point is x₁ − x₀.
- Loss = (v(s, xₜ, t) − (x₁ − x₀))², averaged over the 21 numbers and the batch.
- Every row counts equally. The flow network never sees Q or V.

**Step 5. One Adam step on the sum of the three losses** (V loss + Q loss + flow loss). Each loss depends only on its own
network (V's loss uses the target copies, which get no gradient; Q's loss uses V as a fixed number; the flow loss uses only the
flow network). So this is the same as three separate updates on the same batch, all computed from the weights before the step.

**Step 6. Move the target copies:** target = 0.995 × target + 0.005 × Q, for Q1 and Q2. (Exact code detail: the target moves
toward the Q weights from before this step's update.) V has no target copy.

**What never happens in training:** no step uses an action produced by the flow network. The critic and the flow network meet
only when an action is chosen (Section 5).

## 5. Choosing an action ("rejection sampling", FQL paper Equation 11)

At a state s, in every evaluation step:
1. Draw N = 32 noise vectors, each 21 numbers from a standard normal distribution.
2. Move each one 10 times: x ← x + v(s, x, t) / 10, with t = 0, 0.1, …, 0.9 (the Euler method with 10 steps). Each noise vector
   becomes one action proposed by the flow network.
3. Clip each proposed action to [−1, 1]. These are the 32 candidates.
4. Score each candidate with min(Q1, Q2)(s, candidate), using the trained critic (not the target copies).
5. Execute the candidate with the highest score.

Cost per environment step: 32 × 10 = 320 flow-network evaluations and 32 × 2 = 64 critic evaluations.

## 6. The critic in one place

| part | what it is |
|---|---|
| networks | Q1, Q2: 90 → 4 × 512 (linear → GELU → LayerNorm) → 1; V: 69 → same → 1; target copies of Q1, Q2 only |
| V's target | min(Q1_target, Q2_target)(s, a) at the dataset action |
| V's loss | expectile, τ = 0.9: 0.9 · d² if V too low, 0.1 · d² if V too high |
| Q's target | r + 0.995 · mask · V(s′); no policy action, no min, no max |
| Q's loss | (Q1 − y)² + (Q2 − y)² |
| where min(Q1, Q2) is used | only in V's target (with the target copies) and when choosing the action (with the trained Q) |
| target copies | Q only; updated every step with rate 0.005 |
| range of Q | true values lie in [−200, 0]; nothing in the code enforces this (no clipping of the target) |
| what the critic is used for | only to pick 1 of 32 candidates at decision time; never to weight or steer the actor in training |

## 7. Hyperparameters for humanoidmaze-medium

The exact command from the official README (everything not in it is a code default):

```
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent=agents/ifql.py --agent.discount=0.995 --agent.num_samples=32
```

| setting | value | source |
|---|---|---|
| discount γ | 0.995 | README command; paper Table 5 ("0.995 (antmaze-giant, humanoidmaze, antsoccer)") |
| expectile τ | 0.9 | `agents/ifql.py` default; appendix E.2 says IFQL uses IQL's defaults, and IQL uses 0.9 "for all environments" |
| V and Q networks | 4 × 512, GELU, LayerNorm (`layer_norm=True`) | code default; appendix B |
| flow network | 4 × 512, GELU, no LayerNorm (`actor_layer_norm=False`) | code default |
| number of Q networks | 2 | code |
| target update rate | 0.005 per step (Q only) | code; Table 5 |
| optimizer | Adam, learning rate 0.0003 | code; Table 5 |
| batch size | 256 | code; Table 5 |
| training steps | 1,000,000 | `main.py` (`offline_steps`); appendix B |
| candidates N | 32 (tried 32, 64 and 128 on task 1 and used for all five tasks) | README; Table 6; appendix E.2 |
| Euler steps of the flow | 10 | code; Table 5 |
| flow time distribution | uniform on [0, 1] | code; Table 5 |
| evaluation | every 100k steps, 50 episodes; score = mean of the evals at 800k, 900k and 1M | `main.py`; appendix B |
| seeds | 8 | paper Section 5.1 |
| **not used** | gradient clipping, reward scaling, state normalization, clipping of the TD target, weight decay, dropout, learning-rate schedule | checked in the code |

## 8. The tricks, and what each one does

1. **Expectile τ = 0.9 for V.** Makes V track the better dataset actions without ever taking a maximum over actions.
2. **Targets built only from dataset actions** (Q's target uses V(s′); V is fitted at dataset actions). Avoids the usual offline-RL
   failure where Q gives high values to actions that the data never tried.
3. **Two Q networks; min(Q1, Q2) in exactly two places:** V's target and the choice of action. Q's own target takes no min,
   because it comes from V.
4. **A slow target copy only for Q.** V has none, and Q's target uses the current V.
5. **LayerNorm in V and Q** (not in the flow network). Appendix B: "to further stabilize training".
6. **Large networks (4 × 512).** Appendix B: "using a large enough network is especially important in navigation environments".
7. **The mask.** Stops the target from adding value after the goal step.
8. **Dataset actions clipped to ±(1 − 0.00001)** when the data is loaded; **candidates clipped to [−1, 1]** before being scored.

**Two places in the FQL appendix that are easy to misread:**
- Appendix B ("Value learning") says the critic target uses the mean of Q1 and Q2 by default, and their min for adroit and
  antmaze-large/giant. This describes **FQL's own critic, not IFQL's**. IFQL's Q target is r + γ · mask · V(s′); its min is in V's
  target.
- Appendix E.2 writes "FIQL"; it is a typo for IFQL.

## 9. Weights from the critic

### 9.1 IFQL during training: none
The flow network copies every dataset action with weight 1. It never reads Q or V.

### 9.2 IFQL when acting: weight 1 for the best candidate, 0 for the rest
Section 5: score the 32 candidates with min(Q1, Q2) and execute the highest.

### 9.3 Original IDQL also defines softer weights (IFQL does not use them)
IDQL paper, Algorithm 2 and Equation 9. For each candidate aᵢ:
1. wᵢ = |τ − 1(Q(s, aᵢ) < V(s))|, which for τ = 0.9 is **0.9 if Q(s, aᵢ) ≥ V(s), and 0.1 if Q(s, aᵢ) < V(s)**.
2. Normalize: pᵢ = wᵢ / (w₁ + … + w_N).
3. Draw the executed action at random with probabilities pᵢ.

Example: if 8 of 32 candidates score above V, each of them gets probability 0.9 / (8 · 0.9 + 24 · 0.1) = 0.094 and each of the
other 24 gets 0.010; an above-V action is executed 75 % of the time.

The IDQL paper reports that taking the highest-scoring candidate works better at evaluation: "the expectile objective
performs the strongest with greedy extraction" (Section 6.3); taking the maximum "corresponds to selecting w(s, a) to be a
one-hot" (Section 4.2). That is why IFQL takes the maximum.

### 9.4 For comparison: methods that train the actor with weights from a critic

Results on humanoidmaze-medium task 1 (FQL paper Table 3, 8 seeds; our runs 3 seeds, except 2 for Exp 13's IQL critic).

| method | weight of each dataset row during actor training | success |
|---|---|---|
| IQL (Gaussian actor; same code base, same critic as IFQL) | min(exp(10 · (min(Q1, Q2)(s, a) − V(s))), 100); α = 10 (Table 6); uses the trained Q, not the target copies | 32 ± 7 % |
| FAWAC (flow actor with AWR weights) | exp(α · (Q(s, a) − V(s))) with α = 3 (Equation 10, Table 6). Its code is not public, so its V and any cap could not be checked | 6 ± 2 % |
| **IFQL** | **none (weight 1 for every row); best of 32 at decision time** | **69 ± 19 %** |
| our Aware with its AWAC critic (Experiment 13, AWAC runs; Experiment 18) | min(exp((Q − V)/T), 100) with T = 1/3, the same form as α = 3; Q = min(Q1, Q2)(s, a) from the trained critic; V = mean min Q over the actor's own samples (1 sample in Exp 13, 16 in Exp 18), not an expectile network | 0–2 % (stopped early; last eval at 100–200k in Exp 13, 300k in Exp 18) |
| our Aware with an IQL critic (Experiment 13, IQL runs) | the same formula with T = 1/3, but Q = min(Q1, Q2)(s, a) from the target copies, and V from a value network trained with the 0.9 expectile loss, as in IFQL | 0–2 % (2 seeds, the third run was never started; stopped early, last eval at 200k) |

## 10. Results on humanoidmaze-medium (FQL paper)

Success in %, 8 seeds, mean of the evals at 800k, 900k and 1M (Table 3; Table 2 for the average over the five tasks).
Exception: IDQL was trained for 3M steps, and its score is the mean of the evals at 2.5M, 2.75M and 3M (appendix E.2).

| task | BC | IQL | IDQL | CAC | FAWAC | FBRAC | **IFQL** | FQL |
|---|---|---|---|---|---|---|---|---|
| task 1 (default) | 1 ± 0 | 32 ± 7 | 1 ± 1 | 38 ± 19 | 6 ± 2 | 25 ± 8 | **69 ± 19** | 19 ± 12 |
| task 2 | 1 ± 0 | 41 ± 9 | 1 ± 1 | 47 ± 35 | 40 ± 2 | 76 ± 10 | **85 ± 11** | 94 ± 3 |
| task 3 | 6 ± 2 | 25 ± 5 | 0 ± 1 | 83 ± 18 | 19 ± 2 | 27 ± 11 | **49 ± 49** | 74 ± 18 |
| task 4 | 0 ± 0 | 0 ± 1 | 1 ± 1 | 5 ± 4 | 1 ± 1 | 1 ± 2 | **1 ± 1** | 3 ± 4 |
| task 5 | 2 ± 1 | 66 ± 4 | 1 ± 1 | 91 ± 5 | 31 ± 7 | 63 ± 9 | **98 ± 2** | 97 ± 2 |
| average of the 5 tasks | 2 ± 1 | 33 ± 2 | 1 ± 0 | 53 ± 8 | 19 ± 1 | 38 ± 5 | **60 ± 14** | 58 ± 5 |

- **Offline-to-online** (Table 4, task 1, success at 1M offline steps → after 1M more online steps): IFQL 56 ± 35 → 82 ± 20;
  FQL 12 ± 7 → 22 ± 12; IQL 21 ± 13 → 16 ± 8. The offline number (56) differs from Table 3 (69). One reason: Table 4 reports
  only the eval at 1M (appendix B), while Table 3 reports the mean of the evals at 800k, 900k and 1M. The paper does not say
  whether Table 4 uses the same runs as Table 3. For FQL it cannot: FQL used α = 30 in Table 6 and α = 100 in Table 7. IFQL
  used N = 32 in both tables.
- **The same idea with a different implementation scored 1 %:** IDQL (the official diffusion code, run by the FQL authors,
  appendix E.2) uses the same kind of critic (IQL, expectile loss) and also chooses among N = 32 samples (Table 6), but scored
  1 ± 1 % on task 1. The differences the appendix lists: value networks of 2 × 512 (4 layers were "unstable" in that code base), expectile 0.7 for locomotion
  (IFQL: 0.9), a diffusion model with residual layers instead of a flow MLP, and 3M training steps (1.5M for the value
  networks). The paper does not test which of these differences matters.

## 11. Comparison with our experiments on the same task

| | IFQL | our Experiment 15 (best of N) | our Aware with its AWAC critic (Exp 13 AWAC runs, Exp 18) | our Aware with an IQL critic (Exp 13 IQL runs) |
|---|---|---|---|---|
| actor proposing actions | flow BC, 10 Euler steps, 1M steps, never sees the critic | EqM-BC (Exp 14), K = 20 sampler steps, 500k steps, frozen | EqM actor trained with AWAC weights from the critic | EqM actor trained with AWAC weights from the critic |
| critic target | r + γ · V(s′); V by expectile 0.9 at dataset actions | QIPO softmax: average of Q over the dataset's next action and 16 EqM-BC samples at K = 10, weighted by softmax(3 · Q) | min Q at one next action from the current EqM actor | r + γ · V(s′); V by expectile 0.9 at dataset actions, as in IFQL |
| critic sees policy-made actions in training | no | yes (the 16 EqM-BC samples) | yes (the actor's next action) | no |
| critic network | 4 × 512, LayerNorm, γ = 0.995, no target clipping | 4 × 512, LayerNorm, γ = 0.995, target clipped to [−200, 0] | Exp 13: 4 × 256, γ = 0.995; Exp 18: 4 × 512, LayerNorm, γ = 0.999 | Q and V: 4 × 256, no LayerNorm, γ = 0.995 |
| how the critic is used | best of 32 at decision time | best of 16 / 32 / 64 at decision time | weights in actor training | weights in actor training |
| seeds | 8 | 3 | 3 | 2 (the third run was never started) |
| success, task 1 | 69 ± 19 % | 33 / 39 / 42 % (N = 16 / 32 / 64) | 0–2 % (stopped early; last eval at 100k–300k) | 0–2 % (stopped early; last eval at 200k) |

Experiment 15 and IFQL use the critic in the same way (choose one of N samples from a BC actor that the critic never changes).
They differ in how the critic is trained (IQL expectile V on dataset actions vs QIPO's softmax target over BC samples), in the
actor (EqM at K = 20 vs flow), in the actor's width (EqM-BC 4 × 256 vs flow 4 × 512), in BC training length (500k vs 1M
steps), in the evaluation (Exp 15: 100 episodes at the final checkpoints; FQL: the mean of 3 evals of 50 episodes each, at
800k, 900k and 1M), and in the number of seeds.

**Tested on 2026-11-10 as Experiment 19 (code and results in Section 12):** the IFQL recipe with our actor, i.e. train an IQL critic exactly as in Sections 4
and 7 (expectile 0.9, 4 × 512 with LayerNorm, γ = 0.995, 1M steps) and choose the best of 32 EqM-BC samples at K = 20. This would
separate the effect of the critic training from the effect of the actor type. Experiment 13 already trained an IQL critic
(expectile 0.9), but a different one, used differently: smaller networks (4 × 256, no LayerNorm), used only for the actor's
training weights and never to choose among samples, and stopped early (last eval at 200k, 0–2 %).

## 12. IEQM: the IFQL recipe with our EqM actor (Experiment 19)

Prepared and run on 2026-11-10 on branch `ieqm`. IEQM is IFQL with one change: IFQL's flow network is replaced by our EqM field.
The critic, its training and the way an action is chosen are IFQL's, as written in Sections 3 to 7.

| file | what it does |
|---|---|
| `Aware/train_ieqm.py` | trains V, Q1, Q2 and the EqM field on the same batches; at every eval, the critic picks the best of 32 actor samples |
| `sweeps/exp19_ogbench_ieqm.yaml` | the experiment: humanoidmaze-medium task 1, seeds 1–3, 1M steps |
| `run_exp19.sh` | starts the sweep, then writes the summary |
| `diagnostics/summarize_exp19.py` | the summary: scores per actor and seed, success over training, critic numbers |
| `run_sweep.py` | new target `train_ieqm_ogbench` |

### 12.1 What is the same as IFQL

Every item below is in `Aware/train_ieqm.py` and matches the section named.

| item | value | section |
|---|---|---|
| V, Q1, Q2 | 4 × 512; each hidden layer is linear → GELU → LayerNorm; then linear → 1; Glorot-uniform weights, biases 0 | 3 |
| target copies | Q1 and Q2 only; rate 0.005; the target moves toward the Q weights from before the step | 4, step 6 |
| V loss | d = min(Q1_target, Q2_target)(s, a) − V(s); 0.9 · d² if d ≥ 0, 0.1 · d² if d < 0 | 4, step 2 |
| Q loss | y = r + 0.995 · mask · V(s′) with the current V; (Q1 − y)² + (Q2 − y)² | 4, step 3 |
| one step | one Adam step (learning rate 0.0003) on V loss + Q loss + actor loss; batch 256; 1,000,000 steps | 4, step 5 |
| inputs of V and Q | raw states, rewards as given, dataset actions clipped to ±(1 − 0.00001), no clipping of the target | 3, 7 |
| choosing an action | 32 actor samples, clipped to [−1, 1]; the highest min(Q1, Q2) of the trained critic is executed | 5 |
| evaluation | every 100k steps, 50 episodes; score = mean of the evals at 800k, 900k and 1M | 7 |

As in IFQL, the critic never sees an action made by the actor during training, and the actor never sees Q or V.

### 12.2 What differs from IFQL

| | IFQL | IEQM (our code) |
|---|---|---|
| actor network | flow network: 4 × 512, GELU, inputs s, x and the time t | EqM field (`ConditionalMLP`): 4 × 512, Mish, inputs x and s, no time input |
| actor loss | flow matching (Section 4, step 4) | Aware's EqM loss with every row's weight 1, as in Experiment 14's EqM-BC (`scale_gradient` 8, `flag_type` trun, `flag_threshold` 0.8) |
| states given to the actor | raw | normalized, (s − mean) / std, as for all our EqM actors |
| making one sample | 10 Euler steps from noise | K = 20 NAG steps from noise (step size 0.01, momentum 0.3) |
| actor's starting weights | Glorot-uniform | PyTorch's default, as for all our EqM actors |
| seeds | 8 | 3 |
| start states of the eval episodes | not seeded | seeded: every eval of every run uses the same 50 start states |
| framework | JAX | PyTorch; the sweep sets `compile: true` (see 12.4) |

### 12.3 Two actors are scored by the same critic

The critic does not depend on the actor, so one critic per seed can score two actors. Every eval does both.

| name in the logs | actor | compare with | what differs in that comparison |
|---|---|---|---|
| `trained` (`eval/`) | the EqM field trained in the same run (4 × 512, as many steps as the critic) | IFQL: 69 ± 19 % | the actor (table 12.2) |
| `frozen` (`eval_frozen/`) | Experiment 14's EqM-BC of the same seed (4 × 256, 500k steps), loaded and not trained | Experiment 15, best of 32 at K = 20: 39 % | the critic: IQL here, QIPO there. Also the eval: 50 episodes at 800k, 900k and 1M here; 100 episodes at the final checkpoints there |

### 12.4 Checks done on 2026-11-10

- The V loss and the Q loss equal an independent numpy calculation on a random batch (difference below 0.00001).
- V's loss changes only V, and Q's loss changes only Q. The expectile weight is 0.9 where V is too low and 0.1 where it is
  too high. Layer order, Glorot limits and zero biases are as in Section 3. The target copy equals Q at the start.
- Two test runs of 6,000 steps on the real data, one with `compile: false` and one with `compile: true`. Both trained,
  evaluated both actors and saved a checkpoint that loads again.
- `compile: true` runs the three losses as one compiled graph (`torch.compile`). It is the same maths. After 3,000 steps
  the two test runs agree to 4 digits (Q loss 0.6727 and 0.6728, V on dataset rows −13.5390 and −13.5384). One training
  step takes 2.6 ms instead of 8.9 ms.
- **Not done:** running FQL's official JAX code side by side with ours (flax is not installed in the `ogbench` env).

### 12.5 How to run it

```
cd algorithms/EqM_RL
setsid nohup bash run_exp19.sh > /dev/null 2>&1 < /dev/null &
```

The 3 runs run together. One run alone takes about 1 hour: 1,000,000 steps of about 2.6 ms, plus 20 evals of about 1 minute.
Progress is in `sweep_runs/exp19_<date-time>/orchestrator.log`, the result in `sweep_runs/exp19_<date-time>/summary.md`.
It needs Experiment 14's EqM-BC checkpoints (`Exp14_BC_…humanoidmaze-medium…_seed1` to `_seed3`, `checkpoint_500000.pt`),
which are on disk.

### 12.6 Results (2026-11-10: 3 seeds, 1,000,000 steps each, about 67 minutes per run)

Summary file: `sweep_runs/exp19_20261110-000749/summary.md`. Success in %, 50 episodes of task 1 per eval.

| actor | score: mean of the evals at 800k, 900k and 1M, then mean ± std over the 3 seeds | seed 1 / 2 / 3 | reference |
|---|---|---|---|
| `trained` (EqM field of the run, 4 × 512) | **48.7 ± 32.9** | 2.7 / 78.0 / 65.3 | IFQL: 69 ± 19 (8 seeds) |
| `frozen` (Experiment 14's EqM-BC, 4 × 256) | **45.6 ± 30.5** | 2.7 / 70.7 / 63.3 | Experiment 15, same actors, QIPO critic: 39 (18 / 33 / 66) |

Success at every eval, written as `trained` / `frozen`:

| step | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 100k | 12 / 20 | 22 / 8 | 8 / 16 |
| 200k | 20 / 38 | 36 / 46 | 36 / 44 |
| 300k | 58 / 60 | 70 / 60 | 68 / 66 |
| 400k | 64 / 78 | 68 / 58 | 68 / 76 |
| 500k | 76 / 68 | 82 / 72 | 80 / 74 |
| 600k | 74 / 80 | 78 / 74 | 72 / 68 |
| 700k | 76 / 76 | 74 / 66 | 76 / 60 |
| 800k | 8 / 2 | 70 / 60 | 76 / 76 |
| 900k | 0 / 4 | 76 / 74 | 78 / 78 |
| 1M | 0 / 2 | 88 / 78 | 42 / 36 |

What these numbers show:
1. **From 300k to 700k, all 3 seeds are between 58 % and 82 %, with both actors.** The same frozen actors with the QIPO
   critic reached 18, 33 and 66 % in Experiment 15.
2. **Seed 1 falls from 76 % at 700k to 8 % at 800k, and to 0 % at 900k and 1M.** Both actors fall together. The frozen
   actor did not change, so the critic changed. Seed 1's V on dataset rows rose from −125.6 at 700k to −115.1 at 900k,
   about 0.5 per 10k steps. In seeds 2 and 3 it stayed between −125 and −127. Nothing blew up: Q on dataset rows stayed
   between −187 and −4, and the Q loss rose only from 2.7 to 3.2.
3. **Seed 3 falls at the last eval** (78 → 42 % `trained`, 78 → 36 % `frozen`). Its V rose by 1.0 over the last 100k
   steps. Whether this is the same change starting, or noise of one eval, is not known.
4. **The two actors score about the same with the same critic** (48.7 and 45.6). So with this critic, the actor's width
   (512 or 256) and its training length (as long as the critic, or 500k) change little.
5. **The score 48.7 ± 32.9 cannot be told apart from IFQL's 69 ± 19**: we have 3 seeds, and one of them is at 2.7. The
   FQL paper's own Table 4 gives IFQL at the 1M eval alone as 56 ± 35 (Section 10), so single seeds failing at single
   evals also happens there.

Why seed 1 fell is answered in Section 12.7.

### 12.7 Why seed 1 fell: the critic grew a false peak of V at maze cell (4,2)

Checked on 2026-11-10 with `diagnostics/exp19_critic_drift.py` (output: `diagnostics/out/exp19_critic_drift/summary.md`).
It reads the saved checkpoints (one per 100k steps and seed) and trains nothing.

The maze is a grid of cells of 4 × 4 length units. A cell is named (row, column); `####` is a wall:

```
  start   (1,2)    ####    ####   (1,5)   (1,6)
  (2,1)   (2,2)    ####   (2,4)   (2,5)   (2,6)
   ####   (3,2)   (3,3)   (3,4)    ####    ####
  (4,1)   (4,2)    ####   (4,4)   (4,5)   (4,6)
  (5,1)    ####   (5,3)   (5,4)    ####   (5,6)
  (6,1)   (6,2)   (6,3)    ####   (6,5)    goal
```

The route to the goal is start → (2,1) → (2,2) → (3,2) → (3,3) → (3,4) → (4,4) → (4,5) → (4,6) → (5,6) → goal.
Cell (4,2) lies below (3,2), off this route. It is the entrance of a side corridor: (4,2) → (4,1) → (5,1) → (6,1).

"V of a cell" below is V averaged over the dataset states whose x-y position is in that cell (300,000 sampled states).

1. **V changed in one region only.** Seed 1, from 700k to 800k: V rose by 25.7 at (4,2), by 24.2 at (4,1) and by 15.1 at
   (5,1). From 700k to 900k: by 44.3, 49.1 and 50.7. In the 13 cells on the goal side (columns 4 to 6), V changed by less
   than 5. In seed 2, between 700k and 900k, no cell changed by more than 2.2.
2. **This makes (4,2) a false peak.** Test: start in the start cell and always step to the neighbouring cell with the
   highest V, until no neighbour is higher. In seed 2 this walk ends in the goal cell at all 10 checkpoints. In seed 1 it
   ends at (4,2) at 800k, 900k and 1M. In seed 3 it ends at (4,2) at 1M. (Seed 1 also stopped once early, at (5,4) at 200k.)
3. **The policy walks to the false peak.** Rollouts of 50 episodes with the seed's frozen actor and the critic of one
   checkpoint, best of 32 as in the run:

   | seed | critic of | success | where the failed episodes spent their last 500 steps (share of those steps) |
   |---|---|---|---|
   | 1 | 700k | 70 % | (6,6) 41 %, (4,2) 13 %, (5,4) 10 %, (5,6) 10 % |
   | 1 | 800k | 8 % | (4,2) 34 %, (5,1) 28 %, (5,4) 18 %, (4,1) 11 % |
   | 1 | 900k | 0 % | (5,1) 71 %, (4,2) 19 %, (4,1) 7 %, (6,1) 2 % |
   | 3 | 900k | 92 % | (6,6) 89 %, (5,6) 11 % |
   | 3 | 1M | 38 % | (4,2) 64 %, (6,6) 7 %, (5,4) 6 %, (5,1) 6 % |
   | 2 | 1M | 62 % | (6,6) 29 %, (5,4) 26 %, (4,5) 9 %, (2,2) 6 % |

   These successes differ by up to 16 points from the run's own evals of the same checkpoints (70 vs 76, 92 vs 78,
   62 vs 78). The start states are the same; only the random candidates differ. So one eval of 50 episodes is this noisy.
4. **Seed 3's fall at the last eval (Section 12.6, point 3) is the same failure starting**, at the same cell: its V at
   (4,2) rose by 13.2 between 900k and 1M.
5. **An early sign, one checkpoint before the success fell.** The spread (standard deviation over the cell's dataset rows)
   of Q(s, a) − V(s) at (4,2): seed 1: 0.65 at 500k, 0.70 at 600k, then 1.58 at 700k, when success was still 76 %. Seed 3:
   0.71 at 800k, then 1.08 at 900k, when success was still 78 %. Seed 2: 0.59 at 700k and 0.70 at 900k. This is 3 seeds,
   so it is an observation, not a proven rule.
6. **The training losses did not show the failure.** In seed 1 the Q loss rose only from 2.7 to 3.2, and Q stayed between
   −187 and −4.
7. **The level of V is far more optimistic than the real returns, also in the good checkpoints.** V at the start cell is
   about −155, which is the return of reaching the goal in about 300 steps. Successful episodes need about 1,300 steps. So
   the policy is steered by which cell has the higher V, not by the level of V, and a local error of the shape is enough to
   break it.

Not known: why the peak starts, and why at (4,2) in both seeds.

### 12.8 An OOD score of the EqM actor does not help against the false peak

Checked on 2026-11-10 with `diagnostics/exp19_ood_check.py` (output: `diagnostics/out/exp19_ood_check/summary.md`), at
the user's request. Two scores of the EqM field f, as in the earlier OOD check. The sampler makes a candidate in K = 20
steps, x₁ … x₂₀:
- **last+1:** |f(x₂₀ | s)|, the norm of the field one evaluation after the last step (GeCO's score).
- **accumulated:** |f(x₁ | s)| + … + |f(x₂₀ | s)|, the sum of the norms over the 20 steps.

A state's score is the average over its 32 candidates. Rollouts of 50 episodes with the seed's frozen actor and the
critic of one checkpoint, as in Section 12.7.

| seed | critic of | success, no filter | success, only the 16 candidates with the lowest last+1 | success, only the 16 with the lowest accumulated |
|---|---|---|---|---|
| 1 | 700k (good) | 70 % | 64 % | 78 % |
| 1 | 800k (false peak) | 8 % | 2 % | 8 % |
| 1 | 900k (false peak) | 0 % | 0 % | 0 % |
| 3 | 900k (good) | 92 % | 72 % | 72 % |
| 3 | 1M (false peak) | 38 % | 46 % | 40 % |

1. **Filtering the candidates by either score does not rescue the bad critics** (8 → 2 and 8, 0 → 0 and 0, 38 → 46 and
   40; one eval of 50 episodes is noisy by about ±10 points). With the good critic of seed 3 it cost 20 points.
2. **The states in the corridor are not flagged.** Share of visited corridor states with a score above the 95th percentile
   of 5,000 held-out dataset states (5 % would be "like the data"): 5 to 9 % for last+1, 4 to 6 % for accumulated, 5 to
   10 % for the distance to the nearest dataset state.
3. **The critic does not pick candidates with unusual scores.** Position of the chosen candidate's score among the 32
   (0 = lowest, 1 = highest, 0.5 = no relation): 0.54 to 0.59 for last+1 and 0.47 to 0.51 for accumulated, the same for
   good and bad critics.

Reason: nothing in this failure is out of distribution. The wrong V is at dataset states (the corridor is in the
data), and the candidates are the actor's own samples. An OOD score measures distance from the data, so it cannot see a
critic that is wrong inside the data.
