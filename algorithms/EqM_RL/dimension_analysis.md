# Dimension analysis: understanding how EqM chooses actions

**Status: experiment plan.** The experiments described here have not been run as part of this document. The first stage uses existing checkpoints. Later stages add controlled synthetic tests and simulator comparisons.

This plan expands the visualization idea discussed after Experiment 4. It builds on the drift and dimension experiments in [EqM_RL.md](EqM_RL.md).

## 1. Objective

We want to understand how an EqM actor turns random starting noise into an action, and whether something about that process explains unreliable control.

The proposed starting point is to plot generated actions together with dataset actions using PCA, UMAP, or t-SNE. Those pictures can help us find patterns. To explain a failure, we also need to follow the actions during sampling, measure the learned field, and test the consequences in the simulator.

There are four main questions:

| Question | What we need to observe |
|---|---|
| Does the actor produce actions that differ from its training target? | Generated actions and a suitable data reference, compared at the same state. |
| Do actions slide along the data region during sampling? | The full sampling path, including movement before the usual 20-step stopping point. |
| Do conflicting training targets cancel and create bad stopping points? | Individual target vectors, their average, and whether the sampler actually gets trapped. |
| Does the number of action dimensions contribute to these effects? | Controlled tests in which dimension changes and changes in noise or signal scale are reported separately. |

The existing results give us important controls. Expert-only EqM BC works well. EqM also improves when Aware's weights are made less concentrated, and ordinary Aware works well after longer training. Therefore, the study should explain differences between successful and less successful actors. A pattern that appears equally in both is weaker evidence for the cause of falls.

We should keep three levels of evidence separate:

1. **A visible pattern:** a plot suggests that something happens.
2. **A measured mechanism:** measurements in the original action space confirm what happens.
3. **A control consequence:** changing that behavior improves or worsens the robot's performance.

A picture alone usually establishes only the first level.

## 2. Definitions used in this plan

| Term | Meaning here |
|---|---|
| State | The observation given to the policy. HalfCheetah uses 17 observation values. A simulator snapshot contains additional information needed to restore the environment. |
| Action | The six control values sent to HalfCheetah. |
| Sampler step | One refinement of a proposed action. The robot does not move during this calculation. |
| Environment step | Executing an action and advancing the robot in the simulator. |
| Training update | Changing the network parameters using a training batch. This is different from a sampler step. |
| Field | The vector predicted by the EqM network for a proposed action and a state. The sampler uses this vector to update the action. |
| Conditional action distribution | The range of actions associated with one particular state, rather than actions pooled across many different states. |
| Local reference | An estimate of that action distribution, made from data near the chosen state. It is an approximation when exact repeated states are unavailable. |
| Projection | A way to display six action coordinates using two plotting coordinates. The other four directions still exist. |
| Tangential movement | Movement along a reliably identified thin action region. When such a region cannot be identified, we will instead say movement along wide or narrow directions. |
| Cancellation | Individual training target vectors are substantial but point in opposing directions, so their average is small. |
| Stationary point | A point where the predicted field is zero or very small. |
| Trap | A region that attracts nearby sampler trajectories and keeps them there. A stationary point is not necessarily a trap. |

The field is produced by an unconstrained neural network. We should draw field arrows and field magnitudes, but should not label a plot as an exact energy landscape unless that interpretation has been established separately.

## 3. Shared design choices

### Use real states, not a grid over all state coordinates

A grid over 17 observation values would be enormous. It would also contain many combinations of body positions and velocities that are far from the training data.

Instead, we will build a fixed collection of real states. Every compared actor will be queried at the same states. This makes differences in their action choices easier to interpret.

A global action plot can still be useful as an overview, but it will not be used as evidence of cancellation or tangential drift. For example, two action groups might simply come from two different phases of running. They do not necessarily represent two choices available at the same state.

### Compare these actors

| Actor or checkpoint group | Purpose |
|---|---|
| EqM fitted with expert-only BC | A successful imitation control with no critic-dependent weights. |
| EqM fitted with the original frozen Aware weights | A less stable actor trained on concentrated weights. |
| EqM fitted with the softer, matched-concentration Aware weights | A more stable actor with the same underlying advantage estimates. |
| EqM fitted with IQL weights | A second weight source. Include all three seeds, including the weak second seed. |
| Corresponding fitted Gaussian actors | Reference actions learned from the same weights. Their executed mean actions will be shown explicitly. |
| Ordinary Aware at earlier and later checkpoints | A comparison of how the field changes as training progresses. |

For the frozen-weight fits, use the final 310k checkpoint as the primary checkpoint. Their previously reported last-20-evaluation averages describe a period of training, not a single averaged model.

For the training-progress comparison, prefer approximately 150k, 310k, and 1M checkpoints from the same long run, where available. Older Experiment-3 runs can be included, but must be labelled as separate runs rather than earlier checkpoints of a resumed run.

Use all three training seeds for the main analysis. A one-seed pilot is useful for checking the implementation, but is not the final result.

### Build a state bank before inspecting field plots

The initial pilot will use 32 states, with eight from each group below. The main analysis will aim for 256 states, with 64 from each group.

| State group | Why include it? |
|---|---|
| Medium dataset states | Show behavior away from the expert gait. |
| Expert dataset states | Show behavior around the fast gait represented in the data. |
| Stable rollout states | Show states that a working policy actually visits. |
| States before a fall | Concentrate some measurements around the failures we want to understand. |

For pre-fall states, start with observations 10, 30, and 50 environment steps before the first threshold crossing. Limit how many states come from one episode so that a single failure does not dominate the bank. Keep the source policy and time-to-fall attached to every state.

For comparisons involving pre-fall states, also select non-fall states at similar episode times and running speeds. Report the body angle as well. Otherwise, a difference could simply reflect the robot already being in a different physical condition.

These groups are deliberately balanced for diagnosis. Their combined average is not an estimate of how often each situation occurs during normal use. Report results separately for each group.

Rollout states require recorded observations. Simulator branches also require saved simulator snapshots. Existing files that contain only actions, angles, or speeds are not enough to reconstruct those snapshots. Collect short diagnostic rollouts when the needed information is missing.

For every state, save its source, raw observation, normalization, episode seed, environment step, and whether a fall had already happened. For branch tests, also save the simulator state and the subsequent recorded actions used to check restoration.

### Build the action reference carefully

The dataset usually contains only one recorded action at an exact state. We therefore need nearby states to estimate which actions are typical around the selected state.

Use the same state normalization as training. Start with 64 nearby dataset states, then repeat the analysis with 32 and 128 neighbors. Exclude the query transition itself. For dataset queries, also check a reference that excludes the query's episode, since adjacent frames can otherwise make the reference look artificially accurate.

State differences can widen the action cloud. To check this, show both the raw neighboring actions and an adjusted reference. The adjusted reference uses a small local regression to estimate how the action changes with the state, then removes that estimated change. Fit and check this regression on separate parts of the neighborhood, preferably separating episodes.

This adjustment is a sensitivity check, not a way to manufacture exact repeated-state data. If the raw and adjusted views give different conclusions, report that uncertainty.

Every comparison needs two references:

1. **The behavior reference:** nearby recorded actions with equal importance.
2. **The training reference:** those actions with the weights used by the particular actor.

For expert-only BC, the training reference must use the expert subset. For IQL or Aware fits, load the corresponding saved weights in the correct dataset row order.

Record how many different episodes contribute, how close their states are, and how concentrated their weights are. A neighborhood containing 64 rows may still have only a few effectively important examples. If there is too little information, mark the reference as unreliable rather than interpreting a thin-looking cloud as a true geometric structure.

## 4. Experiment A: compare action samples at fixed states

**Question:** Do the actors produce different action distributions when the physical situation is held fixed?

**Design:** For every state in the bank, generate 128 EqM samples using independent starting noise. Use the same noise vectors across the compared EqM actors. The pilot can use 32 samples per state. Keep each actor's trained sampler settings for this first comparison.

Show nearby data actions, the generated actions, and the Gaussian mean action. Show the data weights through point sizes or a separate panel. A small generated cloud inside a highly weighted part of the data is not automatically a failure to learn the target.

Keep raw sampler outputs separate from the clipped actions that would be executed. Clipping can create a pile of actions on a boundary even when the raw field has a different structure.

**Main plots:**

- A PCA view for each selected state.
- Views of pairs of original action coordinates.
- A UMAP view as an optional second look at group structure.
- A t-SNE view only when it adds useful information beyond those views.

Fit the PCA axes once from an independent, unweighted local data reference. Keep those axes and plot limits fixed across models and sampler steps for that state. Report the fraction of reference variation shown by the two axes.

For UMAP or t-SNE, use one shared embedding of a balanced collection of reference and model samples. Do not fit a separate embedding for each actor and then compare the positions. Repeat a small number of embedding settings and random seeds to see whether the visible pattern is stable.

UMAP and t-SNE positions are for exploration. Apparent gaps, distances, and point densities in those plots will not be treated as measurements of the original action distribution. UMAP can transform new points into a fitted map, but this does not establish that outlying points were placed faithfully.

**Measurements in the original six action dimensions:**

| Measurement | What it tells us |
|---|---|
| Distance from generated actions to nearby reference actions | Whether generated actions lie near the estimated action region. |
| Distance from reference actions to generated actions | Whether samples cover the reference, rather than merely reaching one small part of it. |
| Spread along each of the six reference directions | Whether variability has disappeared in particular directions. |
| Fraction of raw outputs outside the action bounds | Whether the sampler produces actions that require clipping. |

For the distance checks, compare against distances between independent parts of the reference itself. Keep sample counts comparable. More reference points alone can reduce nearest-neighbor distances.

**What this experiment can establish:** differences in action location, spread, and approximate coverage at matched states.

**What it cannot establish:** why those differences arose, whether they are caused by MSE cancellation, or whether they cause falls.

## 5. Experiment B: follow actions through the sampler

**Question:** Do samples approach the relevant action region and then continue moving along it? Does this differ between less stable and successful actors?

**Design:** Use the same states and starting noise as Experiment A. Record every sampler step up to 80. Display selected steps such as 0, 1, 2, 5, 10, 20, 40, and 80.

The first 20 steps are the primary interval because that is the usual sampler budget. Steps 21–80 test what happens with additional refinement. They cannot by themselves explain a policy that normally stops at 20.

At every step, record:

- The full six-dimensional proposed action.
- The actual change in action.
- The Nesterov look-ahead point and the vector evaluated there.
- The sampler's carried state, which affects the next look-ahead.
- The field magnitude, sample spread, reference distance, and clipping statistics.

Preserve the real sampler. In particular, do not add clipping after every refinement step if the trained sampler clips only the final executed action.

### The action-space meshgrid

At a fixed state, choose two local PCA directions and a reference action. Make a grid by varying those two coordinates while keeping the other four coordinates at the reference value.

Evaluate the six-dimensional field at every grid point. Plot its projected direction and its full magnitude. Draw arrows for `-f(action, state)`, labelled as field directions. Overlay the actual Nesterov trajectories, whose updates depend on their previous sampler state.

Do not force the trajectories to stay in the plotted plane. Measure how much they move outside it. A projected arrow can look small even when the full vector is large.

### Measuring movement along and across the action region

Estimate directions from the independent action reference, not from the EqM field being investigated.

If the reference has a clear thin shape, estimate directions along that shape and directions across it. Check whether those directions remain similar when the neighborhood size or reference subset changes.

If the reference has two separated action groups, do not automatically treat the line joining the groups as a tangent direction. Estimate local directions within a supported group. If there is no reliable thin structure, report movement along wide and narrow directions without calling them tangent and normal directions.

Define entry into the reference region using distances between held-out reference examples. For example, use a fixed upper percentile of those distances as the initial threshold. Set the rule before comparing model outcomes and check that nearby thresholds give similar conclusions.

After a sample enters this region, measure how far it continues to move along and across it. Report the accumulated movement as well as the final displacement; a path can travel far and return close to its starting point.

**Evidence for tangential movement:** substantial movement along a reliably estimated action region after the sample has reached it, confirmed in the full action coordinates.

**Evidence against a simple drift explanation:** very little such movement, or essentially the same behavior in successful and less successful actors at the same states.

**Important limit:** movement toward a heavily weighted part of the reference may be consistent with the target. Experiment C tests whether the movement actually worsens control.

## 6. Experiment C: test whether the movement harms control

This experiment has two parts. The first compares individual actions. The second changes the sampler throughout a rollout.

### C1. Earlier versus later refinement from the same simulator state

**Question:** Does additional refinement turn a useful action into a worse action?

Use states with valid simulator snapshots. For each state and starting noise, generate the actions at the selected refinement steps from one shared sampling path.

The initial comparison will be step 10 versus step 20. Secondary comparisons are 5 versus 10 and 20 versus 40. The last comparison tests extra refinement beyond the ordinary budget. Identify which comparisons occur after entry into the reference region; poor actions early in sampling do not establish drift.

For each action:

1. Restore the same simulator snapshot into a fresh environment.
2. Execute the action with the normal action bounds.
3. Let the same Gaussian AWAC continuation policy act for another 199 environment steps.
4. Record reward, speed, body angle, and whether a fall occurred.

This gives a total horizon of 200 steps, including the tested action. Also report the first 20 steps of that same continuation, since a strong continuation policy may repair an earlier mistake.

Use several predetermined noise starts per state, such as eight. Do not choose the best or worst sampled actions after seeing their outcomes. Keep already-fallen states separate.

Before any branch comparison, restore each snapshot and replay its recorded actions. Require finite observations and agreement within the existing `1e-6` tolerance. If restoration fails, improve the saved simulator state rather than quietly accepting the mismatch.

**Supporting result:** later actions repeatedly produce worse outcomes in the same states and noise starts, particularly during the interval in which Experiment B measured sliding.

**Limitation:** one action followed by a common controller does not reproduce the effects of executing the modified sampler throughout an episode.

### C2. Reduce movement in the suspected directions

**Question:** Does reducing that particular movement improve full rollouts?

Run this only where the reference directions are reliable. Otherwise, a directional intervention would have an unclear meaning.

| Treatment | Purpose |
|---|---|
| Ordinary sampler | Baseline. |
| Keep half of the proposed movement along the estimated tangent directions | A moderate intervention. |
| Remove that proposed movement | A stronger intervention. |
| Reduce the whole update | Test whether simply taking smaller steps explains an improvement. |
| Reduce movement in random directions of the same dimension | Test whether the estimated directions matter specifically. |

Apply the directional intervention only near the supported action region. Before that, use the ordinary sampler. Freeze this activation rule before confirmation tests.

Make the Nesterov implementation explicit. For the primary version, transform the vector returned at the look-ahead point, use that transformed vector for the action update, and carry it into the next look-ahead. Apply the same convention to the comparison treatments. Do not mix raw and modified carried vectors accidentally.

Match update sizes between the directional and comparison treatments as closely as practical. Record the achieved sizes. Repeat the random-direction comparison with several fixed random choices so that one lucky choice does not determine the conclusion.

Use a screening set of 20 episodes per training seed and a separate confirmation set of 100 episodes per seed. Proposed starting seeds are 2000+ for screening and 3000+ for confirmation. All treatments share episode seeds and sampler noise. Freeze the selected treatment before looking at confirmation results.

**Supporting result:** reducing the suspected movement lowers falls or improves return more than the size-matched and random-direction controls, while keeping a comparable running speed.

**Alternative reading:** if every smaller-update treatment helps equally, the result supports reducing update size, but does not identify tangential movement as the special cause. If the robot merely slows down substantially, report a speed–stability tradeoff.

## 7. Experiment D: test cancellation directly with a known distribution

**Question:** Can the EqM MSE objective itself create a misleading stopping point, or does the trained network introduce it through fitting error?

Real offline data makes this question difficult because the exact action distribution at one state is unknown. We will therefore start with a small synthetic distribution whose shape and probabilities are known.

### D1. Choose simple targets with different structures

Start in six action dimensions to match HalfCheetah. Hold the conditioning state fixed.

| Synthetic target | What it tests |
|---|---|
| One round Gaussian group | A simple reference with one central high-density region. |
| One group that is wide in some directions and narrow in others | Movement along wide directions without requiring two separate choices. |
| A thin curved band with a known shape | Movement along a curved region with an independently known tangent direction. |
| Two Gaussian groups | Cancellation between competing source actions. |

For the two-group target, vary the separation and imbalance. Initial settings can use center separations of 2, 4, and 8 times the component standard deviation, and group probabilities of 50/50, 80/20, and 95/5. Record the absolute action scale as well. These are starting settings, not claimed thresholds for failure.

When groups overlap, do not assume that the target has two visibly separate modes. Compare generated samples with fresh draws from the actual known target, using the same assignment rule for both.

Use equal weights first. Then introduce known changes in importance to test whether concentrated weighting changes the behavior. Keep the data distribution and the weighting rule separately recorded.

### D2. Compute the ideal MSE field

Use the same corruption and target construction as [Aware/agents.py](Aware/agents.py): mix a clean action with Gaussian noise, sample the mixing level uniformly, and use the current truncated target scaling. The model does not receive the mixing level as an input.

The ideal prediction at a proposed action is the weighted average of the training target vectors that could have produced that input. It is not an average over arbitrary training examples.

For readers implementing this calculation, the weighted MSE optimum is:

\[
f^*(z,s)=\frac{\mathbb{E}[w\,y\mid z,s]}{\mathbb{E}[w\mid z,s]},
\qquad y=c(\rho)(\epsilon-a).
\]

Here `a` is the clean action, `epsilon` is the starting noise, `rho` is the mixing level, and `c` includes the gradient multiplier. The averages include the possible mixing levels, because the actor does not observe them.

For Gaussian targets, use analytic conditional quantities where available and numerical integration over the mixing level. For other targets, use a checked numerical estimate. Increase the integration resolution or sample count to verify that the answer is stable. Mark poorly estimated locations as uncertain rather than interpreting numerical noise as cancellation.

This ideal field can be studied before training a network. Then train a small EqM network on the same synthetic task and compare its field with the ideal one. Use multiple network seeds and check field error on inputs that were not used for training.

### D3. Measure opposing instructions

At each candidate location, measure both:

- The magnitude of the average target vector.
- The average magnitude of the individual target vectors that contributed to it.

If individual vectors are large but their average is small, opposing instructions are cancelling. If the individual vectors are also small, label that separately; it is not evidence of strong opposing instructions.

A ratio of these two quantities can summarize the effect. Only interpret that ratio when the individual-vector magnitudes are large enough to measure reliably.

This concerns vectors in action space. It does not measure disagreement between neural-network parameter gradients from different minibatches.

### D4. Check whether a small field creates a real trap

A small field between two action groups does not establish a failure. It might be an unstable point that trajectories leave immediately.

For each candidate point with a small full-dimensional field:

1. Check how unlikely the point is under the known clean target distribution.
2. Perturb it in both directions along every action coordinate and in several random directions.
3. Use several perturbation sizes relative to the target's width.
4. Run the actual sampler and check whether the perturbed samples return, escape, or remain nearby.
5. Start many samples from the ordinary noise distribution and measure how often they enter this region.

Test both the standard sampler initialization and carried sampler states observed on incoming trajectories. With Nesterov look-ahead, the action alone does not describe the whole sampler state.

Start with 1,024 ordinary noise samples per synthetic case. Compare the usual 20-step budget with longer budgets such as 80 and 200. If samples have not settled by the measured horizon, report that rather than automatically calling them trapped.

| Result | Interpretation |
|---|---|
| The ideal field and trained field both attract samples to an unlikely region | The objective and sampling dynamics can produce this failure under these conditions. |
| Only the trained field creates the trap | Network approximation or optimization error contributes. |
| The midpoint has a small field but nearby samples escape | There is a stationary point, but no demonstrated attracting trap there. |
| A trap exists but ordinary noise starts almost never reach it | The mechanism exists, but its practical importance may be small. |
| The ideal field behaves well but the network has large field error | Improve or extend fitting before blaming the ideal objective. |

A synthetic result establishes what can happen under known conditions. It does not by itself establish what caused HalfCheetah's falls.

## 8. Experiment E: separate action dimension from other changes

**Question:** Do the measured effects change because of dimension, or because adding coordinates also changes the noise, target geometry, or learning difficulty?

This follows the six-dimensional synthetic study. The ideal-field calculation is particularly useful here because it removes neural-network fitting capacity as an explanation for differences between dimensions.

Run two different comparisons:

| Comparison | Design | Question answered |
|---|---|---|
| Change the number of actual action variables | Use targets with 2, 6, 12, and 24 varying coordinates. Extend to 48 if useful. | What happens when the underlying distribution itself has more dimensions? |
| Keep six actual action variables and add coordinates | Place the same six-dimensional target into 6, 12, 24, and 48 coordinates using a distance-preserving linear map. | What happens when the representation has more coordinates but the clean data still has six independent variables? |

The second comparison cannot include an embedding into two dimensions while preserving all six-dimensional distances.

More coordinates also mean more total noise if each coordinate receives the same amount. Run two explicitly labelled noise settings: one with the original noise scale per coordinate, and one with total expected noise magnitude held fixed. Neither setting should be presented as holding every aspect of the task unchanged.

For the first comparison, record group separation, group width, and total cloud radius in every dimension. Start with a fixed separation measured relative to a coordinate's width, then check a setting that holds separation relative to the whole cloud's radius fixed. This helps reveal whether an apparent dimension effect is actually a change in how easy the groups are to distinguish.

Report movement and errors both in original units and per coordinate. Measure unlikely stopping points, attraction, the fraction of ordinary starts reaching them, and how well samples reproduce the known target.

If networks are trained as well, report their parameter counts, training progress, and compute. A larger network or a longer effective training budget should not silently become the explanation for an apparent dimension benefit.

**Important caveat:** extra coordinates containing zero clean signal can still carry information about the corruption level once noise is added. Their effect is not necessarily a pure geometric benefit of higher dimension.

Fourier features and action chunking are later representation experiments. Fourier features change geometry and what can be inferred after corruption. Action chunking also changes the control problem by predicting several future actions. Success with either would need its own controls before being called evidence that low dimension caused the original failure.

## 9. Reading the real-policy and synthetic results together

| Observation | What we could reasonably say | What remains unresolved |
|---|---|---|
| A 2D plot shows samples between data groups | The display identifies a region worth inspecting. | Whether the separation and gap exist in six dimensions. |
| Full-dimensional measurements show movement along a supported action region | Tangential movement occurs in these cases. | Whether that movement is harmful. |
| Later actions worsen simulator outcomes and directional suppression helps beyond controls | The measured movement contributes to poor control in the tested setting. | Whether it explains all failures or other tasks. |
| Opposing target vectors produce an attracting low-density stopping point in the ideal field | Cancellation can create a practical sampling problem for that known target. | Whether the same target structure exists at relevant HalfCheetah states. |
| The pattern is similar in successful and less successful actors on common states | The pattern alone is a weak explanation of their performance difference. | Whether its interaction with particular states or action choices still matters. |
| Softening weights changes the field and improves control | The weighting transformation affects learning or sampling usefully. | Whether the benefit comes from easier fitting, a better target, less cancellation, or several effects together. |
| Dimension changes improve the controlled synthetic tests | Dimension-related changes matter under those stated conditions. | Whether they explain or improve the real RL task. |

The purpose is to let these explanations compete. We should not select only states or pictures that support tangential drift or cancellation.

## 10. Run order and expected outputs

| Stage | Work | Reason for this order |
|---|---|---|
| 1 | Build and validate the small state bank and local references. | Poor references can make every later plot misleading. |
| 2 | Run Experiments A and B on existing checkpoints. | Establish what differs between actors without new actor training. |
| 3 | Run the six-dimensional ideal-field cases in Experiment D. | Test cancellation directly without relying on an estimated real-data distribution. |
| 4 | Expand A and B across the full state bank and all training seeds. | Check whether the pilot patterns are repeatable. |
| 5 | Run the action branches in C1 and, where justified, the rollout interventions in C2. | Connect the measured movement to control outcomes. |
| 6 | Run the dimension comparisons in E. | Separate dimension-related effects after the measurements have been checked. |

The main analysis should produce an interactive state viewer and saved figures. Each state view should show the action reference, samples, a few readable sampling paths, and the corresponding full-dimensional measurements. Include original-coordinate views so that PCA is not the only view available.

Save the underlying numbers as well as the figures. Suggested outputs are:

- A manifest listing checkpoint paths and steps, weight files, dataset identity, state-selection rules, normalization, and random seeds.
- The state bank, simulator snapshots where available, and reference-row indices.
- Full action paths and sampler state, so that plots can be regenerated.
- A table with one row per state and model, plus summaries by state group and training seed.
- Separate branch-test and rollout results.
- Synthetic target definitions, ideal-field calculation settings, field-error checks, and trap measurements.
- A short written account of which questions the results answer and which remain open.

The implementation can reuse the checkpoint loading and sampling helpers in [diagnostics/common.py](diagnostics/common.py), the weight/data helpers in [Aware/offline_data.py](Aware/offline_data.py), and the restoration checks in [diagnostics/handoff.py](diagnostics/handoff.py). New diagnostics must keep the original actor loss and sampler conventions explicit.

Report variation across states, source episodes, and training seeds. Having 128 action samples at one state does not give 128 independent trained policies or 128 independent physical situations.

The first deliverable is a trustworthy comparison of the existing actors at fixed states. The stronger claims about harmful drift, cancellation, and dimension follow only when the corresponding measurements and interventions support them.

## Sources

- [Project experiment plan and results](EqM_RL.md).
- [EqM actor loss used in this project](Aware/agents.py).
- [Equilibrium Matching paper](https://arxiv.org/html/2510.02300v2), for the corruption process and MSE target.
- [t-SNE author's guidance](https://lvdmaaten.github.io/tsne/), including limitations of interpreting distances and mapping new points.
- [UMAP FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html), including limitations of density preservation.
- [Transforming new data with UMAP](https://umap-learn.readthedocs.io/en/latest/transform.html).
