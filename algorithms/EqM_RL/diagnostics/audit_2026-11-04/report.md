**Verdict: the main RL performance results are substantially accurate, but the report is not fully accurate.** Experiment 2's original training table disagrees with W&B; its later reference table is correct. Several explanations turn suggestive diagnostics into stronger causal conclusions than the experiments establish.

Audit date: 2026-11-04. Project: [duck13/E](https://wandb.ai/duck13/E).

**What was checked**

Read all five existing Markdown files: `EqM_RL.md`, `EqM_2d_toy.md`, `EqM_literature.md`, `diagnostics/README.md`, and `sweeps/README.md`. Inventoried all 62 W&B runs: 54 finished, one killed, one crashed, and six failed startup attempts. Recomputed the final-20 evaluation averages for all 56 runs with evaluation data, covering 20 configurations. The killed/crashed preliminary runs are included as explicitly described in the report; failed startup attempts are excluded.

The connector's sampled-history endpoint returned every expected evaluation in the relevant windows. These values were independently checked against unsampled SDK `scan_history` reads. SDK pagination sometimes returned identical duplicate rows at the same `_step`; duplicates were checked for equality and removed before selecting the last 20 distinct evaluations. Every final window had 20 evaluations spaced 1,000 steps apart. All 56 SDK averages agree with the connector averages to numerical precision.

Recomputed the Experiment 2 diagnostic tables from the 15 saved diagnostic and 15 rollout JSON files in `/tmp/claude-0/-workspace-OfflineCORL/426464d2-aa60-4918-a2b4-fd0219e28b8f/scratchpad/`. Recomputed the follow-up tables from all 12 JSON files in `diagnostics/out/drift/`. Inspected sampling, clipping, critic, and diagnostic aggregation code. No training or simulation was rerun.

This checks the principal return tables, their seed statistics, run configurations, and saved sampler/rollout diagnostic tables. It is not an exhaustive independent verification of every dense training-metric average, raw dataset statistic, hardware-throughput measurement, or literature claim.

**1. Numerical correction: Experiment 2's original table**

[Original table](../../EqM_RL.md#experiment-2-λ-sweep-on-minari-simple-v0-and-medium-v0-3-seeds-2026-11-03), around line 313. Mean of the final 20 evaluations per seed, followed by mean and population SD across three seeds:

| Dataset / setting | Original report | Recomputed from W&B |
|---|---|---|
| simple, λ=2 | 305 ± 46 | **296 ± 37** |
| simple, λ=4 | 3,272 ± 55 | **3,269 ± 45** |
| simple, λ=8 | 7,389 ± 32 | **7,389 ± 21** |
| medium, λ=4 | 1,966 ± 135 | **1,969 ± 162** |
| medium, λ=8 | 1,273 ± 617 | **1,280 ± 487** |

The Experiment 3 reference rows, around line 437, already contain the corrected values. The discrepancy does not reverse the conclusions, but the two tables should use one reproducible aggregation. The original Experiment 2 within-run evaluation-noise values also differ: recomputed population SDs averaged across seeds are 111.8 for simple λ=8, 765.9 for medium λ=4, and 1,050.8 for medium λ=8, versus 114, 810, and 1,055 in the text.

**2. Main results that reproduce**

| Result | Report | Recomputed |
|---|---|---|
| Preliminary Gaussian AWAC / Aware | 7,470 / 7,465 | 7,470.27 / 7,465.01 |
| Minari simple, BC Aware | 7,097 ± 8 | 7,096.55 ± 7.72 |
| Minari simple, λ=2, η=.04 | 7,491 ± 16 | 7,490.85 ± 16.12 |
| Minari simple, Gaussian AWAC | 7,537 ± 18 | 7,537.12 ± 18.08 |
| Minari medium, BC Aware | 11,916 ± 248 | 11,915.63 ± 247.83 |
| Minari medium, λ=4, K=40 | 2,742 ± 222 | 2,741.68 ± 221.86 |
| Minari medium, λ=4, η=.02 | 3,149 ± 366 | 3,148.79 ± 366.08 |
| Minari medium, Gaussian AWAC | 7,207 ± 235 | 7,207.43 ± 235.36 |
| D4RL medium-expert, Aware / AWAC | 59.5 / 93.4 normalized | 59.5 / 93.4 |
| D4RL medium-replay, Aware / AWAC | 46.9 / 45.2 normalized | 46.9 / 45.2 |
| D4RL medium-expert, BC / T=1/3 Aware | 50.1 / 53.4 normalized | 50.1 / 53.4 |

The reported Experiment 3 population SDs reproduce. One immaterial rounding discrepancy: AWAC medium-expert's raw mean is 11,312.48, conventionally rounded to 11,312, while the report prints 11,313. Its normalized result is unchanged.

The follow-up's trained-setting rollout table reproduces: T=1 Aware 6,555 ± 1,467 with 34/60 flips (56.7%); T=1/3 Aware 6,040 ± 1,294 with 38/60 flips (63.3%); BC Aware 5,858 ± 348 with 1/60 flips; AWAC 11,203 ± 275 with 1/60 flips. The listed sampler-grid and dataset-state diagnostic values also reproduce at their displayed precision. These are saved checkpoint reevaluations, separate from W&B training averages.

**3. Reporting conventions need correction**

- Experiment 2's last 20 Aware evaluations are at `_step` 101,000–120,000, not an inclusive 100,000–120,000 window (which contains 21).
- Experiment 3 Aware uses `_step` 131,000–150,000; AWAC uses 131,999–150,999, corresponding to updates 132,000–151,000. The phrase “131k–151k” obscures this offset. Training lasts 151,000 updates, but Aware's last evaluation occurs roughly 1,000 updates before training ends.
- Preliminary Aware's final window is 99,000–118,000; AWAC's is 100,999–119,999. Their final-20 means are correctly reported, but “100k–120k” is only approximate.
- The Experiment 3 ± values use **population SD across the three seed means (`ddof=0`)**, not a confidence interval or standard error. State this explicitly. Sample SD (`ddof=1`) would be larger by sqrt(3/2).

**4. Factual correction: critics receive clipped actions**

Around `EqM_RL.md:567`, the report says V(s) and TD targets query the critic “at exactly these endpoints,” referring to endpoints outside the action box. [Actor.forward](../../Aware/actor_models.py) clamps generated actions before returning them (line 42). Both advantage estimation and TD targets consume this return value in [agents.py](../../Aware/agents.py), lines 201 and 245. The diagnostic also clips actions before calculating Q and nearest-neighbour action distance (`diagnostics/drift.py:181` and `:251`).

Correct wording: **raw sampler endpoints often exceed the bounds; the critic and environment receive the resulting clipped, potentially saturated actions.** Clipping does not guarantee that those actions are supported by the dataset at that state, but it changes the proposed mechanism.

**5. The causal interpretation needs weaker wording**

- **“Q / implied” is a rough proxy, not a measured overestimation factor.** The earlier tables compare Q on dataset states/actions with evaluation episode return divided by 1,000(1−γ). These use different state/action distributions and different return constructions. Therefore “the critic overestimates 14×” is too strong on this evidence alone. The follow-up's Q versus realized discounted return on the same visited states is more directly informative and gives 1.34–1.46 for the weighted policies.
- **Critic agreement does not rule out critic error.** Around lines 669–692, agreement between Aware's critic and AWAC's critic is presented as clearing the ranking. Both critics can share errors. In addition, AWAC's Q estimates continuation under AWAC's policy, so a favorable score for one Aware action does not establish the value of repeatedly following Aware afterward. Neither aggregate ΔQ nor nearest-neighbour distance establishes ranking correctness.
- **A nearest-neighbour ratio near 1 does not prove conditional support membership.** It shows comparable average distance to actions at nearby dataset states under this particular metric. Similarly, nearly pure medium/expert neighbourhoods demonstrate strong empirical separation, not a proof that the conditional action distribution has no multimodality.
- **Saturation causing flips remains a hypothesis.** The report correctly acknowledges this around line 683, but its final answer around line 691 treats the mechanism as established. The saved trained-setting T=1 episodes without flips actually have higher whole-episode pre-clip saturation (18.5%) than flipped episodes (12.6%). This does not disprove a saturation mechanism—flips change the subsequent gait—but illustrates why the cross-setting correlation is not sufficient causal evidence. Matched interventions and pre-flip measurements are needed.
- **“No critic can learn to avoid flips” is unsupported.** Sparse flipped states do not establish impossibility for every critic or learning method. Replace this with a statement about the critics actually evaluated.
- **“λ only matters through η·λ·K” and “K and η are interchangeable” overgeneralize a small ablation.** The tested settings support a useful budget heuristic. They do not establish universal equivalence under learning and finite-step sampling. The two medium results are 2,742 ± 222 and 3,149 ± 366; absence of a demonstrated difference would not establish equivalence.
- **The imitation/weighting decomposition is not a matched experiment.** Comparing Aware BC at 151k with published Gaussian BC at 1M cannot bound imitation's contribution to “at most a few points.” The report discloses the different training lengths, but still draws too strong a decomposition from them.

**6. Smaller diagnostic-statistics issue**

`summarize_drift.py` calculates ranks with `argsort(argsort(...))`, which breaks ties arbitrarily. Flip rates have many ties. Recomputing with tie-aware Spearman gives **0.5341** for T=1 and **0.6531** for T=1/3, rather than the script's displayed 0.54 and 0.65. This is a small numerical change and does not alter the qualitative association. The source code was not changed during this audit.

**7. Scope of the other Markdown files**

- `EqM_2d_toy.md`: no corresponding toy runs are present among the 62 runs in E. The SWD claim 0.519 → 0.038 therefore cannot be verified against this project. Its references to `main.py`, `sampling.py`, test files, and `Figures/...` also point to files absent from this folder. Label the original experiment location and retain its raw outputs before treating this as reproducible from the RL repository.
- `EqM_literature.md`: summarizes external papers rather than E runs. Reading it does not validate those papers' numerical or theoretical claims; a paper-by-paper source audit is separate work.
- `diagnostics/README.md`: the saved diagnostic tables are reproducible through the supplied aggregators. The Experiment 2 JSONs are currently in a temporary scratch directory, whereas the README's example uses `diagnostics/out/diag` and `diagnostics/out/roll`. Record durable source paths or archive those outputs.
- `sweeps/README.md`: operational documentation. W&B configurations and durations support the reported experiment setups, but do not independently verify its detailed throughput benchmarks.

**Supported conclusion**

Aware trains successfully on simple; BC Aware substantially outperforms weighted Aware on Minari medium; Gaussian AWAC strongly outperforms Aware on D4RL medium-expert; Aware has a smaller consistent advantage on medium-replay across the tested three seeds. The saved follow-up shows frequent flips in weighted medium-expert Aware and little continuing sampler movement after convergence. The source of the policy's instability remains incompletely isolated; saturated actions, weighted fitting, and critic-policy feedback are plausible explanations, not settled causes.

Evidence: [all training aggregates](all_training_results.md), [per-run CSV with W&B links](run_results.csv), [last-20 evaluation rows](evaluation_evidence.json), [recomputed drift tables](drift_recomputed.md), and [Experiment 2 diagnostic tables](experiment2_diagnostics.txt). Original reports and training code were left unchanged.
