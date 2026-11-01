# EqM Literature Notes

Short summaries of the papers in `papers/`.

---

## 1. EqM — Equilibrium Matching (Wang & Du, 2025)

Replaces time-conditioned flow velocity with one **time-invariant gradient field** of an implicit energy; data points are zeros of the field.

- **Loss:** `‖f(x_γ) − (ε − x)·c(γ)‖²`, with `x_γ = γx + (1−γ)ε`, γ \~ U(0,1) **not** given to the model, and `c(1)=0`.
- **Best c(γ):** truncated decay with `a=0.8` and multiplier `λ=4`: c=λ for γ≤0.8, then linear to 0. Only helps when the model is not noise-conditioned.
- **Sampling = gradient descent:** `x ← x − η f(x)`. Nesterov look-ahead (µ≈0.3–0.35) helps. Adaptive stop when `‖f‖ < g_min` (\~60% fewer steps). Robust to step size, unlike FM.
- **Explicit energy (EqM-E):** `E = x·f(x)` (dot product, preferred) or `−½‖f‖²`. Slightly worse samples, but enables **OOD detection** (energy) and **composition** (add gradients).
- **Theory (heuristic, high-dim):** `‖f‖≈0` at data; all local minima ≈ data points; GD gives `min‖f‖² ≤ 2(E₀−E_inf)/(ηK)`.
- ImageNet-256 FID 1.90 (beats SiT/FM).

## 2. GeCO — Generative Control as Optimization (Zhang et al., 2026)

EqM applied to robot action chunks as a **drop-in replacement for a flow-matching head**. For π0, they fix t=0 and fine-tune.

- **Adaptive inference:** fewer steps on easy states, more on precise ones. LIBERO 93.5% with an average of 11.6 NFE; 5 steps already beats 20-step rectified flow.
- **OOD score:** the final field norm `‖f(â, s)‖` after optimization. AUROC 0.93, vs 0.53 for a flow-matching-loss baseline. Moving-average or leaky-bucket filters give online alarms.
- **Truncation ablation:** a=0.8 is best; 0.6 is too smooth and slow, 0.9 too stiff.
- **Settings:** step sizes 0.1 → 0.05 → 0.02 → 0.01, stop at `‖f‖<0.4`, K_max=30.

## 3. π0-EqM (Liu et al., 2026)

Replaces π0's flow-matching action expert with EqM, solved with Nesterov.

- RoboTwin (19 tasks): 40.4% → 50.2% at a matched 300-step budget; LIBERO roughly unchanged.
- **Warm start:** reuse half of the previous chunk, fill the rest with fresh noise.
- **Stationarity–executability gap:** a lower residual does not always mean a better action. The best stopping threshold is task-dependent.
- ⚠ As printed, the target `w(γ)(A−ε)` and the update `A − ηf` have inconsistent signs. Follow EqM's convention.

## 4. EqC — Equilibrium Energy Control (Yang et al., 2026)

Explicit-energy EqM for control, plus **introspection**: knowing when the policy is failing.

- **Energy:** `E(x,z) = ⟨x, f(x,z)⟩`, trained by gradient matching. Inference is gradient **ascent**. Schedule `c(t)=min{4, 20(1−t)}`, i.e. EqM's λ=4, a=0.8.
- **Path identity:** `E(a) − E(x_t) = ‖a−ε‖²·C(t)` with `C(t)=∫_t^1 c`.
- **Consistency score** `U_ΔE`**:** the average gap between realized and predicted energy gain along the refinement path. A large value signals failure.
- **Energy-Gated DAgger:** m consecutive alarms → expert takeover → add the recovery data → retrain.
- **Results:** failure-detection AUROC 0.924 (vs 0.843 for Diff-DAgger). LIBERO-Plus 53.6% → 65.8%.
- **Argument:** explicit energy costs little in control because action distributions are concentrated given the observation.

---

## 5. AWAC — Advantage Weighted Actor Critic (Nair et al., 2020)

The classic **weighted-regression** actor-critic for offline pretraining followed by online fine-tuning. It is the template the later papers generalize.

- **Why existing methods fail at fine-tuning:**
  - On-policy fine-tuning (DAPG, AWR, MARWIL) is slow.
  - Off-policy SAC suffers bootstrap error on OOD actions and shows an initial performance dip, even with BC pretraining.
  - Constrained offline methods (BEAR, BCQ, BRAC) are **too conservative** online, because their explicit behavior model fits the growing, mixed replay data poorly.
- **Actor:** maximize `E_π[A(s,a)]` s.t. `KL(π‖π_β) ≤ ε`, whose solution is `π* ∝ π_β·exp(A/λ)`. Projecting with forward KL gives weighted max-likelihood: `θ ← argmax E_{(s,a)~buffer}[ log π_θ(a|s) · exp(A(s,a)/λ) ]`. This is an **implicit constraint**: no behavior model is learned; you just reweight buffer actions.
- **Critic:** off-policy TD on `Q^π`, with `A = Q(s,a) − V(s)` and V computed from current-policy actions. This is the key difference from AWR (Monte-Carlo / TD(λ) value of the behavior policy) and ABM (learned prior).
- **Practical notes:**
  - The per-state normalizer `Z(s)` is dropped; estimating it made results worse (door 95% → 0%).
  - λ = 0.3 for manipulation tasks, 1.0 for MuJoCo.
  - Batch 1024, 4×256 MLPs, lr 3e-4, τ = 5e-3. Q is clamped ≤ 0 when rewards are non-positive.
- **Results:** solves sparse-reward dexterous pen/door/relocate (pen in about 120K steps, \~20 min of real time) and three real robots. Fine-tunes from D4RL random/medium data where BEAR stalls.
- **For EqM:** replace `log π_θ` with the negative EqM loss and weight it by `exp(A/λ)`. QIPO (below) is the generative-model version of the same idea.

## 6. IQL — Implicit Q-Learning (Kostrikov, Nair & Levine, 2021)

An offline RL critic that **never queries out-of-dataset actions**, yet still does multi-step dynamic programming.

- **Value step (expectile regression):** `L_V = E_{(s,a)~D}[ L₂^τ(Q̂(s,a) − V(s)) ]`, with the asymmetric loss `L₂^τ(u) = |τ − 1(u<0)|·u²`.
  - With τ &gt; 0.5, V(s) tracks an **upper expectile** of Q over dataset actions.
  - As τ → 1 this approaches `max_{a ∈ support} Q(s,a)`.
- **Q step (plain MSE):** `L_Q = E[(r + γV(s') − Q(s,a))²]`.
  - Using V separately keeps "lucky" stochastic transitions from inflating the target.
  - τ = 0.5 gives SARSA; τ → 1 gives in-support Q-learning.
- **Policy extraction:** AWR-style weighted BC, `E_{(s,a)~D}[ exp(β·(Q̂(s,a) − V(s))) · log π(a|s) ]`.
  - The policy **does not affect** value learning, so extraction can run during or after critic training.
- **Theory:** V\_τ is monotone in τ and approaches the dataset-constrained optimal value as τ → 1.
- **Settings:**
  - AntMaze: τ = 0.9, β = 10, rewards shifted by −1.
  - Locomotion: τ = 0.7, β = 3, rewards normalized by the dataset's return range.
  - Kitchen/Adroit: τ = 0.7, β = 0.5, actor dropout 0.1.
  - Common: `exp(adv)` clipped at 100, min of two Q networks, 2×256 MLPs, lr 3e-4, cosine-decayed actor lr, τ\_target = 0.005.
- **Results:**
  - D4RL total 1070 (vs CQL 1002). AntMaze total 378 (vs CQL 304), where "stitching" partial trajectories needs a large τ.
  - Offline-to-online AntMaze 370 → 474.
  - About 20 min per 1M updates.
- **For EqM:** the critic is trained only on dataset actions, so it is independent of the actor. An EqM policy can be extracted with exp(βA)-weighted EqM loss, and the cost of EqM sampling never enters critic training.

## 7. QIPO — Energy-Weighted Flow Matching (Zhang, Zhang & Gu, ICLR 2025)

Learns `π ∝ µ·exp(βQ)` **exactly** by weighting the conditional FM loss, with no auxiliary guidance network.

- **Weights:** softmax of βQ over M=16 support actions sampled from the current policy, plus the dataset action.
- **Iteration:** refresh the support set every 10 epochs, so `π ∝ µ·exp((l+1)βQ)` and β effectively grows on its own.
- **Settings:** β=3, soft update τ=0.005. Q comes from QGPO's in-support softmax Q-learning.
- Note: t=0 = data in this paper.

## 8. DFQL — Direct Flow Q-Learning (Cao et al., ICML 2026)

Trains a multi-step flow policy with Q **without BPTT or distillation**.

- Compute `g = ∇_a Q(s, a_K)` at the final action only.
- **RL loss:** `−Σ_k w_k·Δt·gᵀ v_θ(s, a_k, t_k)` with `w_k = 1/(K−k)`. This is equivalent to regressing v onto `stopgrad(v + Δt·g)`.
- **Dense BC:** the FM loss is applied at every grid time t_k.
- **Settings:** K=10, α tuned per task, actor output capped with `5·tanh`. State of the art on 73 OGBench and D4RL tasks; works offline-to-online unchanged.

## 9. FFM — Fixed-point Flow Matching (Blessing et al., 2026)

Trains flow policies toward Boltzmann targets **without data**, using critic gradients.

- **Target flow identity:** `v = (α̇/α)x + η(t)/(λα)·E[∇Q(X₁) | X_t]`. In words, velocity is a conditional average of ∇Q taken at **clean** actions.
- **Loop:** sample from the current policy, label with ∇Q, noise, regress. The optimal policy is a fixed point.
- **Offline-to-online:** use `s_ref + ∇Q/λ` as the target score, with `s_ref` read off the pretrained velocity field. A velocity-space trust region (Lagrangian) keeps updates small.
- **Extras:** variance-optimal control variate `c*(t)=α²/(α²+σ²Ī)`, Hutchinson entropy for max-ent RL, OU-correlated noise for exploration.

## 10. FINO — Flow Matching with Injected Noise (Shin et al., ICLR 2026)

Offline-to-online RL on top of FQL.

- **Offline:** add noise to the FM input (σ=0.1, variance from 0 at t=0 to σ² at t=1). The mean path is unchanged, but action coverage is **wider**, which helps exploration.
- **Online:** sample `action_dim/2` candidate actions and pick one by `softmax(ξQ)`, with ξ auto-tuned to a target entropy. Greedy at evaluation.
- Noise in the training objective beats noise added to actions.

---

## 11. Fisher Information Metric for OOD (Dauncey et al., 2024)

OOD detection with gradients of the log-likelihood **with respect to the model's weights**.

- Likelihoods fail for OOD (Nalisnick). Weight-gradients are invariant to how the input is represented.
- The Fisher matrix is diagonal-dominant within each layer, so it is approximated per layer by a scaled identity.
- **Recipe:** for each layer compute `‖∇_{θ_j} ℓ(x)‖²` and take its log. Fit a Gaussian per layer on held-out data; the OOD score is the joint negative log-density. No hyperparameters.

---

## Quick reference

- **OOD signals:** EqM energy or `‖f(a|s)‖` at the candidate action (action-level); GeCO final `‖f‖` (state-level); EqC `U_ΔE` (failure-level); Fisher layer-gradient norms (weight space).
- **Critic choice:** AWAC uses an off-policy TD critic that queries policy actions; IQL uses an in-sample expectile critic that never queries policy actions and is independent of the actor.
- **Improving an EqM policy with Q:** weighted EqM loss with `exp(A/λ)` or `softmax(βQ)` weights (AWAC/IQL/QIPO), Q-gradient injection into f (DFQL), energy addition `f − ∇Q/λ` (FFM-style).
- **Conventions:** EqM/GeCO use target `(ε−x)c` and **descend**; EqC uses `c(a−ε)` and **ascends**. QIPO puts data at t=0; the others put data at t=1.
- **Shared hyperparameters:** truncated c(γ) with a=0.8 and λ=4 across EqM, EqC, and GeCO (GeCO's appendix says 0.1, which conflicts with its own ablation).