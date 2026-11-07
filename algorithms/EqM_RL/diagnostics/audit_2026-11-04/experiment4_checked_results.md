# Checked Experiment 4 results

Verified from raw episode JSON, completed sweep manifests, and W&B histories in [duck13/E](https://wandb.ai/duck13/E). All 27 priority-2 training runs and all three 1M runs finished. The orchestrator reports all priority-1 jobs succeeded.

## Priority 1: confirmation

Three training seeds, 100 episodes per seed, environment seeds 1000–1099. Earlier Experiment-3 checkpoints; raw episode returns. Fall means the body-angle threshold was crossed.

### me_T1

| Treatment | Return | Falls |
|---|---:|---:|
| aware_b1_n1 | 7,159 | 50.7% |
| aware_b0.95_n1 | 7,637 | 38.7% |
| aware_b0.9_n1 | 7,419 | 38.0% |
| aware_b1_n4 | 7,983 | 43.3% |
| aware_b1_n16 | 8,039 | 36.7% |
| awac_b1 | 11,261 | 2.7% |
| awac_b0.95 | 11,098 | 4.3% |
| awac_b0.9 | 10,678 | 5.0% |

### me_T0.33

| Treatment | Return | Falls |
|---|---:|---:|
| aware_b1_n1 | 6,113 | 56.7% |
| aware_b0.95_n1 | 6,253 | 55.3% |
| aware_b0.9_n1 | 6,596 | 47.7% |
| aware_b1_n4 | 6,586 | 52.0% |
| aware_b1_n16 | 6,719 | 50.7% |
| awac_b1 | 11,261 | 2.7% |
| awac_b0.95 | 11,098 | 4.3% |
| awac_b0.9 | 10,678 | 5.0% |

### me_bc

| Treatment | Return | Falls |
|---|---:|---:|
| aware_b1_n1 | 5,784 | 1.7% |
| aware_b0.95_n1 | 5,889 | 0.3% |
| aware_b0.9_n1 | 5,713 | 0.3% |
| aware_b1_n4 | 6,200 | 0.7% |
| aware_b1_n16 | 6,242 | 0.0% |
| awac_b1 | 11,261 | 2.7% |
| awac_b0.95 | 11,098 | 4.3% |
| awac_b0.9 | 10,678 | 5.0% |

## Priority 2: actor-only fits

Means of the last 20 evaluations, steps 291k–310k, with 10 episodes per evaluation. ± is standard deviation across three training seeds. Starting seeds are reused across evaluations; these are not fresh 100-episode confirmation tests.

| Data / weights | Actor | Normalized score | Falls | Scores by seed |
|---|---|---:|---:|---|
| Expert-only BC | gaussian | 92.5 ± 0.1 | 0.0% | 92.6 / 92.5 / 92.6 |
| Expert-only BC | eqm | 92.6 ± 0.1 | 0.3% | 92.5 / 92.7 / 92.6 |
| Frozen IQL | gaussian | 86.6 ± 9.3 | 11.3% | 93.6 / 73.4 / 92.7 |
| Frozen IQL | eqm | 78.4 ± 13.9 | 25.0% | 91.6 / 59.2 / 84.2 |
| Frozen Aware | gaussian | 92.1 ± 0.8 | 2.2% | 93.1 / 91.4 / 91.8 |
| Frozen Aware | eqm | 86.2 ± 1.1 | 11.3% | 87.7 / 85.5 / 85.4 |
| Softer Aware (matched concentration) | gaussian | 93.2 ± 0.4 | 0.8% | 93.2 / 92.7 / 93.6 |
| Softer Aware (matched concentration) | eqm | 92.2 ± 0.2 | 2.2% | 92.1 / 92.0 / 92.4 |

IQL's own policy: 86.2 ± 2.2, averaging its last four evaluations. It uses different policy extraction settings and is a reference, not the same Gaussian refit.

## Weight concentration

| Seed | Aware effective share | IQL / matched Aware share | Matched temperature |
|---|---:|---:|---:|
| 1 | 4.29% | 46.54% | 3.709 |
| 2 | 4.28% | 14.94% | 2.330 |
| 3 | 4.58% | 34.28% | 3.937 |

Raw and matched Aware files have exactly identical advantage arrays. Effective share describes weight concentration, not how many rows were retained. Frozen sources trained approximately 310k updates, followed by approximately 310k actor updates.

## Separate longer-training experiment

100 confirmation episodes per seed, three seeds; one EqM sample, K=20, step size 0.01. Earlier 151k models are separate runs; 310k and 1M are checkpoints from the new long runs.

| Checkpoint | Return | Falls | Falls by seed |
|---|---:|---:|---|
| Earlier ~151k | 7,159 | 50.7% | 81% / 43% / 28% |
| New runs: 310k | 10,536 | 8.0% | 3% / 18% / 3% |
| New runs: 1M | 11,593 | 0.7% | 0% / 1% / 1% |

The 1M runs' last-20 training-evaluation normalized score is 95.5 ± 0.5. The earlier AWAC confirmation reference scored 11,261 with 2.7% falls. Different training budgets prevent an equal-budget superiority claim.

## Limits

- Clipping and averaging partly help the early policies; they do not rule out magnitude/noise contributing.
- Handoffs concern the tested upright states; later states exclude episodes that had already fallen.
- Expert-only BC demonstrates that EqM can learn the fast gait with the original action dimension and sampler.
- The successful frozen Aware weights come from 310k, not the original 151k checkpoints.
- Softening weights changes the weighted target distribution; matching effective share does not make two sources identical.
- These experiments do not directly isolate tangential drift or action-dimension effects.

## W&B provenance

- [exp4_aware_1m / eqm / seed 1](https://wandb.ai/duck13/E/runs/f06de0fa-d1f8-4278-9108-49705cdb700a)
- [exp4_aware_1m / eqm / seed 2](https://wandb.ai/duck13/E/runs/f71b0494-0671-4b76-b9bc-753a27d2bc8f)
- [exp4_aware_1m / eqm / seed 3](https://wandb.ai/duck13/E/runs/7e871525-1aa9-42c3-b2da-bee09c5befb3)
- [exp4_fit_aware / eqm / seed 1](https://wandb.ai/duck13/E/runs/f0df77ad-b2ce-4e8e-9399-e2c506fbefc6)
- [exp4_fit_aware / eqm / seed 2](https://wandb.ai/duck13/E/runs/2344118d-3a58-48ba-8346-3dd32f6f5b7d)
- [exp4_fit_aware / eqm / seed 3](https://wandb.ai/duck13/E/runs/e0219695-f603-48b4-9e6d-697c8be371df)
- [exp4_fit_aware / gaussian / seed 1](https://wandb.ai/duck13/E/runs/47aae115-90b0-4e6d-a947-2ce231ad0930)
- [exp4_fit_aware / gaussian / seed 2](https://wandb.ai/duck13/E/runs/d0b6f645-d02a-44c9-9c27-f24fe9fa6efe)
- [exp4_fit_aware / gaussian / seed 3](https://wandb.ai/duck13/E/runs/fac9d7c7-9977-4c82-9af2-1f582b49fe4f)
- [exp4_fit_aware_matched / eqm / seed 1](https://wandb.ai/duck13/E/runs/53a017b7-958e-4812-8626-d884c0d0ab70)
- [exp4_fit_aware_matched / eqm / seed 2](https://wandb.ai/duck13/E/runs/893fc2eb-7aad-4e71-8e2f-bfb1d31d484c)
- [exp4_fit_aware_matched / eqm / seed 3](https://wandb.ai/duck13/E/runs/213a8ddf-b6e5-4b4a-979e-480e2a1796bd)
- [exp4_fit_aware_matched / gaussian / seed 1](https://wandb.ai/duck13/E/runs/74709965-355c-4c31-803d-4beea613ee83)
- [exp4_fit_aware_matched / gaussian / seed 2](https://wandb.ai/duck13/E/runs/eeb5a385-11a5-4f73-a9f3-645130fc8646)
- [exp4_fit_aware_matched / gaussian / seed 3](https://wandb.ai/duck13/E/runs/95361ce7-39e3-4243-8833-b1ecf6bb4ac0)
- [exp4_fit_bc / eqm / seed 1](https://wandb.ai/duck13/E/runs/aac2eb4a-d18d-4ff8-9dc5-a65d54c5b73e)
- [exp4_fit_bc / eqm / seed 2](https://wandb.ai/duck13/E/runs/1d3619e5-26fa-4108-8213-78e2dddc6509)
- [exp4_fit_bc / eqm / seed 3](https://wandb.ai/duck13/E/runs/6b5ed6d4-8fd6-402c-a43b-920f0e015216)
- [exp4_fit_bc / gaussian / seed 1](https://wandb.ai/duck13/E/runs/41ca1202-5371-4332-b8e1-ead146fa5dc2)
- [exp4_fit_bc / gaussian / seed 2](https://wandb.ai/duck13/E/runs/99a6cf66-e7b8-4d4a-8c33-7251b43aa5d5)
- [exp4_fit_bc / gaussian / seed 3](https://wandb.ai/duck13/E/runs/00b999e5-9600-4f41-953d-a3b211e6e333)
- [exp4_fit_iql / eqm / seed 1](https://wandb.ai/duck13/E/runs/2c5cf6ca-2ed9-4c12-a282-f23bcd2c96b7)
- [exp4_fit_iql / eqm / seed 2](https://wandb.ai/duck13/E/runs/644f1be1-6e36-42d2-80d0-c5df49e84bef)
- [exp4_fit_iql / eqm / seed 3](https://wandb.ai/duck13/E/runs/3cc336d3-3f7a-4839-8460-94c7724701b2)
- [exp4_fit_iql / gaussian / seed 1](https://wandb.ai/duck13/E/runs/680f6b45-c952-44ee-b710-ae7347a1fb41)
- [exp4_fit_iql / gaussian / seed 2](https://wandb.ai/duck13/E/runs/8644a9ad-b350-42e3-bb3d-ffd1b0893791)
- [exp4_fit_iql / gaussian / seed 3](https://wandb.ai/duck13/E/runs/6c14f46c-5479-4e82-9729-2f20d6587fa1)
