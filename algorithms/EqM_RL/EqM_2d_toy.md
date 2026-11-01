# Equilibrium Matching (EqM) in 2D

**Paper:** Runqian Wang and Yilun Du, *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models*, arXiv:2510.02300 (v3, Oct 2025). PDF: papers/EqM.pdf.

Part I summarizes the paper and restates its appendix derivations. [Part II](#part-ii-1-step-no-chunking-eqm-in-low-dimension-the-drift-problem) describes the drift problem of 1-step (no chunking) EqM in low dimension, and the proposed Jacobian-gated samplers that address it. [Part III](#part-iii-extra-dimensions-against-mse-gradient-cancellation) attacks a second problem, gradient cancellation caused by the MSE average, by training on the data lifted to more dimensions. It also collects all experiments and results.

Notation: $x$ is a data sample, $\\epsilon \\sim \\mathcal N(0, I_d)$ is noise, $\\gamma \\sim U\[0,1\]$ is the interpolation factor, and $x\_\\gamma = \\gamma x + (1-\\gamma)\\epsilon$ is the corrupted sample ($\\gamma = 1$ is data, $\\gamma = 0$ is noise).

---

## 1. Idea in one paragraph

Diffusion and flow models learn **non-equilibrium** dynamics: a time-conditioned velocity $f(x_t, t)$ that has to be integrated along a fixed schedule from $t=0$ to $t=1$. EqM instead learns a single **time-invariant** gradient field $f(x)$ that does not see the noise level. The field is trained to be the gradient of an (implicit) energy landscape whose minima sit on the data. Sampling is then **optimization**: run gradient descent on the learned landscape. Any step size, optimizer (e.g. Nesterov) or stopping rule can be used, because no trajectory is prescribed.

## 2. Background: Flow Matching

Flow Matching (FM) with the linear path regresses the velocity:

$$ \\mathcal L\_{\\text{FM}} = \\big\\lVert f(x_t, t) - (x - \\epsilon) \\big\\rVert^2 . $$

The noise-unconditional variant (Sun et al., 2025) simply drops $t$:

$$ \\mathcal L\_{\\text{uncond-FM}} = \\big\\lVert f(x_t) - (x - \\epsilon) \\big\\rVert^2 . $$

This works but degrades sample quality. EqM argues the cause is the **target**: $x - \\epsilon$ does not vanish at the data, so it is incompatible with equilibrium dynamics.

## 3. EqM training objective

$$ \\boxed{;\\mathcal L\_{\\text{EqM}} = \\big\\lVert f(x\_\\gamma) - (\\epsilon - x), c(\\gamma) \\big\\rVert^2, \\qquad c(1) = 0;} $$

- **Direction** $\\epsilon - x$ points from data toward noise. It is a *gradient* (descending it moves toward data), which is the opposite sign of FM's velocity.
- **Magnitude** $c(\\gamma)$ must vanish at the data, $c(1) = 0$, so that real samples are stationary points (zero gradient) of the landscape.
- With $c \\equiv 1$ the objective is exactly the negated FM objective. FM instead needs $\\int_0^1 c(\\gamma),d\\gamma = 1$ to give a valid integration path.

Choices of $c$, each multiplied by a **gradient multiplier** $\\lambda$:

| name | $c(\\gamma)$ |
| --- | --- |
| linear | $1 - \\gamma$ |
| truncated | $1$ for $\\gamma \\le a$; $\\dfrac{1-\\gamma}{1-a}$ for $\\gamma &gt; a$ |
| piecewise | $b - \\dfrac{b-1}{a}\\gamma$ for $\\gamma \\le a$; $\\dfrac{1-\\gamma}{1-a}$ for $\\gamma &gt; a$ |

The paper's best setting is **truncated with $a = 0.8$ and $\\lambda = 4$**. This repo uses the same: `trun_interpolate: 0.8`, `training_scaler: 4.0`.

**Useful identity** (my derivation, not in the paper). Since $x\_\\gamma - x = (1-\\gamma)(\\epsilon - x)$, the target can be written in terms of $x\_\\gamma$:

$$ (\\epsilon - x),c(\\gamma) = \\frac{c(\\gamma)}{1-\\gamma},(x\_\\gamma - x). $$

For linear decay this is just $x\_\\gamma - x$. For truncated decay with multiplier $\\lambda$ it is $\\lambda,(x\_\\gamma - x)/\\max(1-\\gamma,, 1-a)$. Near the data ($\\gamma &gt; a$) the target is therefore $k,(x\_\\gamma - x)$ with $k = \\lambda/(1-a)$, which is $20$ in this repo.

## 4. Explicit energy (EqM-E)

Learn a scalar energy $g$ and match its gradient:

$$ \\mathcal L\_{\\text{EqM-E}} = \\big\\lVert \\nabla g(x\_\\gamma) - (\\epsilon - x), c(\\gamma) \\big\\rVert^2 . $$

The paper gives two ways to build $g$ from a vector network $f$ without new parameters. $J_f$ is the Jacobian of $f$.

| variant | energy | gradient |
| --- | --- | --- |
| dot product | $g(x) = x^\\top f(x)$ | $\\nabla g = f(x) + J_f(x)^\\top x$ |
| squared $L_2$ | $g(x) = -\\tfrac12 \\lVert f(x)\\rVert^2$ | $\\nabla g = -J_f(x)^\\top f(x)$ |

Both explicit variants lose FID relative to the implicit model (on EqM-B/4, FID 57.54 implicit vs 73.40 dot vs 75.53 $L_2$), but they give an energy value. That value is what the paper uses for OOD detection. The paper recommends the dot-product variant.

## 5. Sampling = optimization

With $\\nabla E = f$ (implicit) or $\\nabla E = \\nabla g$ (explicit):

- **Gradient descent:** $x\_{k+1} = x_k - \\eta, \\nabla E(x_k)$.
- **Nesterov (NAG-GD):** $x\_{k+1} = x_k - \\eta, \\nabla E\\big(x_k + \\mu (x_k - x\_{k-1})\\big)$. Since $x_k - x\_{k-1} = -\\eta, \\nabla E(\\cdot)*{k-1}$, the look-ahead point equals $x_k - \\mu\\eta, m*{k-1}$, which is how `sampling.py` implements it.
- **Adaptive compute:** stop each sample once $\\lVert \\nabla E(x_k) \\rVert_2 &lt; g\_{\\min}$.
- **ODE samplers** are a special case (Appendix D, below).

## 6. Theory (statements)

1. **Zero gradient at data.** With $c(1) = 0$ and perfect training, $\\lVert f(x^{(i)}) \\rVert_2 \\approx 0$ at every training sample, in high dimension.
2. **Local minima are data.** Any local minimum $\\hat x$ with $f(\\hat x) = 0$ satisfies $P(\\hat x \\in \\mathcal X) \\approx 1$, in high dimension.
3. **Convergence.** If $E$ is $L$-smooth and bounded below by $E\_{\\inf}$, gradient descent with $\\eta \\le 1/L$ gives $\\min\_{k&lt;K} \\lVert f(x_k) \\rVert^2 \\le 2(E(x_0) - E\_{\\inf})/(\\eta K)$, an $O(1/K)$ rate.

## 7. Main empirical results (short)

- **ImageNet 256×256, class-conditional:** EqM-XL/2 reaches **FID 1.90** (SiT-XL/2: 2.06, DiT-XL/2: 2.27). NAG-GD with 250 steps, $\\eta = 0.0017$, $\\mu = 0.3$.
- **Scaling:** EqM beats FM across training length, model size and patch size.
- **Step size:** EqM is robust across a wide range of $\\eta$. FM only works near $\\eta = 1/N$.
- **Adaptive compute:** stopping at gradient norm 10 uses about 40% of the fixed 250-step compute, with FID 33.79 vs 32.85 on B/2.
- **Ablations (B/2, 80 epochs, FID):**
  - $c(\\gamma)$: constant 40.81, linear 50.47, truncated $a=0.8$ 38.34 (the best).
  - $\\lambda = 4$ is best at 32.85.
  - The truncated target only helps *without* noise conditioning: 32.85 vs 41.89 with it.
- **Unique properties:**
  - Denoising partially noised inputs gets *better* as the input gets cleaner; it gets worse for FM.
  - OOD detection via energy: average AUROC 0.68, the best of the methods compared.
  - Composition: adding the gradients of two conditional models.

---

## Appendix C: derivations (restated)

### Lemma: the MSE minimizer is a conditional expectation

The paper uses this step without proof. For a target $T$ and input $Z$, let $m(Z) = \\mathbb E\[T \\mid Z\]$. Then

$$ \\mathbb E \\lVert f(Z) - T \\rVert^2 = \\mathbb E \\lVert f(Z) - m(Z) \\rVert^2 + \\mathbb E \\lVert T - m(Z) \\rVert^2

- 2,\\mathbb E\\big\[(f(Z) - m(Z))^\\top (m(Z) - T)\\big\]. $$

The cross term is zero by the tower property: condition on $Z$, and $\\mathbb E\[m(Z) - T \\mid Z\] = 0$. The second term does not depend on $f$. So the loss is minimized pointwise by $f = m$. For EqM:

$$ f^\*(z) = \\mathbb E\\big\[(\\epsilon - x), c(\\gamma) ,\\big|, x\_\\gamma = z\\big\]. \\tag{C.0} $$

### C.1 Statement 1: the learned gradient vanishes at training samples

**Setup.** The dataset $\\mathcal X = {x^{(1)}, \\dots, x^{(n)}} \\subset \\mathbb R^d$ is finite. Given $(x, \\gamma)$ with $\\gamma &lt; 1$, the corrupted sample is Gaussian:

$$ p(x\_\\gamma \\mid x, \\gamma) = \\mathcal N\\big(\\gamma x,\\ (1-\\gamma)^2 I_d\\big). $$

At $\\gamma = 1$, $x\_\\gamma = x$ exactly, a point mass on each training sample.

**Argument.**

1. By (C.0), $f^\*(x^{(i)}) = \\mathbb E\[(\\epsilon - x),c(\\gamma) \\mid x\_\\gamma = x^{(i)}\]$.

2. For $\\gamma &lt; 1$, the Gaussian density evaluated at an exact training point is exponentially small in $d$. The $\\gamma = 1$ branch keeps its mass on $x^{(i)}$. Therefore

   $$ P\\big(\\gamma = 1 \\mid x\_\\gamma = x^{(i)}\\big) \\to 1 \\quad (d \\to \\infty). $$

3. The posterior concentrates on $\\gamma = 1$ (and $x = x^{(i)}$), so

   $$ f^\*(x^{(i)}) \\approx (\\epsilon - x^{(i)}), c(1) = 0 . $$

Hence $\\lVert f(x^{(i)}) \\rVert_2 \\approx 0$. $\\square$

*Remark (my note).* At an exact training point the claim does not actually need $d \\to \\infty$. Write the target as $\\frac{c(\\gamma)}{1-\\gamma}(x\_\\gamma - x)$, as in §3. For the self term $x = x^{(i)} = z$ it is **exactly zero** for every $\\gamma$. Its likelihood $p(z \\mid x^{(i)}, \\gamma) \\propto (1-\\gamma)^{-d} e^{-\\lVert x^{(i)}\\rVert^2/2}$ is non-integrable as $\\gamma \\to 1$ for any $d \\ge 1$. So as $z \\to x^{(i)}$ the posterior concentrates on the self term. High dimension is what makes the *basins around* the training points dominate.

The argument relies on a finite dataset and an exact minimizer, i.e. memorization. A network that generalizes behaves like the population optimum for a continuous density. In 2D (this repo's toy) that optimum is a mean-shift field $f^\*(z) \\approx k,(z - \\mathbb E\[x \\mid z\])$ that is generally **not** zero on the data manifold. That is the source of the tangential drift studied in `sampling.py` (Idea-1/2/3).

### C.2 Statement 2: local minima are training samples

**Argument.** Let $\\hat x$ be a local minimum, so $f(\\hat x) = 0$. By (C.0):

$$ 0 = f^\*(\\hat x) = \\mathbb E\\big\[(\\epsilon - x),c(\\gamma) ,\\big|, x\_\\gamma = \\hat x\\big\]. $$

1. $c(\\gamma) \\ge 0$ everywhere, $c(1) = 0$ and $c(\\gamma) &gt; 0$ for $\\gamma &lt; 1$.
2. In high dimension, the paper argues that this vector expectation can only vanish if the posterior mass concentrates at $\\gamma = 1$.
3. As in C.1: if $\\hat x = x^{(i)}$ then $P(\\gamma = 1 \\mid x\_\\gamma = \\hat x) \\to 1$. For $\\gamma &lt; 1$ the density is exponentially small in $d$.
4. At $\\gamma = 1$ every $x\_\\gamma$ lies in $\\mathcal X$, so $P(\\hat x \\in \\mathcal X) \\approx 1$. $\\square$

*Remark (my note).* The derivation only uses $f(\\hat x) = 0$, i.e. *stationarity*. Stationary points off the data do exist through cancellation. For example, the midpoint of two training points has $f^\* = 0$ by symmetry. Such points are saddles, not minima ($\\operatorname{sym} J_f$ is not positive definite there). So it is the "local minimum" part of the statement that rules them out, not the derivation shown.

### C.3 Statement 3: convergence of gradient-based sampling

**Assumptions.** $\\nabla E = f$, $E$ is $L$-smooth, $E \\ge E\_{\\inf}$, and $\\eta \\in (0, 1/L\]$. The paper writes $\[0, 1/L\]$, but $\\eta = 0$ makes the bound divide by zero.

**Step 1: descent lemma.** $L$-smoothness gives, for all $x, y$,

$$ E(y) \\le E(x) + \\nabla E(x)^\\top (y - x) + \\tfrac{L}{2}\\lVert y - x \\rVert^2 . $$

**Step 2: one GD step.** Take $y = x\_{k+1} = x_k - \\eta f(x_k)$ and use $\\nabla E(x_k) = f(x_k)$:

$$ E(x\_{k+1}) \\le E(x_k) - \\eta \\lVert f(x_k)\\rVert^2 + \\tfrac{L\\eta^2}{2}\\lVert f(x_k)\\rVert^2 = E(x_k) - \\eta\\Big(1 - \\tfrac{L\\eta}{2}\\Big)\\lVert f(x_k)\\rVert^2 . $$

**Step 3: use $\\eta \\le 1/L$.** Then $1 - L\\eta/2 \\ge 1/2$, so

$$ E(x\_{k+1}) \\le E(x_k) - \\tfrac{\\eta}{2}\\lVert f(x_k)\\rVert^2 . $$

**Step 4: telescope over $k = 0, \\dots, K-1$.**

$$ E(x_K) - E(x_0) \\le -\\tfrac{\\eta}{2}\\sum\_{k=0}^{K-1}\\lVert f(x_k)\\rVert^2 \\le -\\tfrac{\\eta}{2}, K \\min\_{0 \\le k &lt; K}\\lVert f(x_k)\\rVert^2 . $$

**Step 5: use $E(x_K) \\ge E\_{\\inf}$ and rearrange.**

$$ \\min\_{0 \\le k &lt; K}\\lVert f(x_k)\\rVert^2 \\le \\frac{2\\big(E(x_0) - E\_{\\inf}\\big)}{\\eta K}. \\qquad \\square $$

*Typos in the paper's version.*

- Step 2 says "recall $\\nabla E(x_k) = -f(x_k)$". It must be $+f(x_k)$, as in the statement; with $-f$ the first-order term would have the wrong sign.
- Step 3 is written with "$\\ge$". The correct direction is "$\\le$", which is what the telescoping step needs.

*Scope.* The bound needs $f$ to be a true gradient field. That holds for EqM-E. For the implicit model it is an assumption: an unconstrained network output is generally not conservative. This repo's toy shows it directly: $J_f$ is far from symmetric.

---

## Appendix D: integration vs optimization sampling

### D.1 The ODE sampler is a special case of gradient descent

Take $\\dot x = v(x)$ and its forward Euler discretization on a uniform grid over $\[0,1\]$:

$$ x\_{k+1} = x_k + h, v(x_k), \\qquad h = \\tfrac1N,; k = 0, \\dots, N-1 . $$

If the velocity is conservative, $v = -\\nabla E$, this is

$$ x\_{k+1} = x_k - h, \\nabla E(x_k), $$

which is exactly gradient descent with $\\eta = h = 1/N$. So $N$-step Euler on a unit horizon is GD with $\\eta = 1/N$.

### D.2 General integrators

With $v = -\\nabla E$, any time integrator defines an optimization update rule. ODE samplers fix a uniform grid $h = 1/N$ over a unit horizon. Under the optimization view that constraint goes away: the direction field stays the same, and the step size (or the whole optimizer) can be chosen freely. That is what makes adaptive step sizes, NAG and early stopping legitimate for EqM.

---

# Part II: 1-step (no chunking) EqM in low dimension: the drift problem

## II.1 Setting

In control, EqM can generate either an **action chunk** $x \\in \\mathbb R^{H d_a}$ (H future actions at once) or a **single action** $x \\in \\mathbb R^{d_a}$ (1 step, no chunking). With chunking the sample space is high-dimensional, which is closer to the regime the paper's Statements 1–2 assume. With 1 step, $d_a$ is small, typically 2–14. This repo's 2D toy (moons, heart) is the minimal version of the 1-step case.

## II.2 Why low dimension produces drift

Statements 1–2 rely on high-dimensional posterior concentration on a finite, memorized dataset. In low dimension a network that generalizes approaches the **population** optimum (C.0) for a continuous density. Near the data ($\\gamma &gt; a$), the identity in §3 gives

$$ f(z) \\approx k,\\big(z - \\mathbb E\[x \\mid z\]\\big), \\qquad k = \\frac{\\lambda}{1-a}\\quad(= 20 \\text{ here}), $$

a **mean-shift field**. With $\\sigma$ the effective local noise scale, Tweedie's formula gives $\\mathbb E\[x \\mid z\] \\approx z + \\sigma^2 \\nabla \\log p\_\\sigma(z)$, so

$$ f(z) \\approx -k\\sigma^2, \\nabla \\log p\_\\sigma(z). $$

Split $f$ at a point near the data manifold $\\mathcal M$:

- **Normal component** $f_n$: pulls points onto $\\mathcal M$. This is wanted.
- **Tangential component** $f_t \\approx -k\\sigma^2, \\nabla_T \\log p$: a small net drift along $\\mathcal M$ toward higher density.

$f_t$ never vanishes on $\\mathcal M$, so the paper's "data = equilibrium" picture fails. Every gradient step also slides the sample along the manifold. Over many steps, or with a large $\\eta$, the slide merges samples into a single mode.

In the toy, plain NAG with $\\eta = 10^{-3}$ gives SWD 0.110 after 250 steps (not yet converged). After 1000 steps every sample lies on one curve that piles up at the inner moon's tip (SWD 0.624). With $3\\eta$ the samples collapse to a short arc (SWD 1.259).

## II.3 Key structural fact: the Jacobian separates normal from tangent

Differentiating the mean-shift field, with $\\partial, \\mathbb E\[x \\mid z\] / \\partial z \\approx \\operatorname{Cov}\[x \\mid z\]/\\sigma^2$:

$$ J_f(z) \\approx k\\Big(I - \\frac{\\operatorname{Cov}\[x \\mid z\]}{\\sigma^2}\\Big). $$

- **Across the manifold**, the posterior is narrow ($\\operatorname{Cov} \\ll \\sigma^2$), so the eigenvalue is about $k$: these directions are **stiff**.
- **Along the manifold**, the posterior is wide ($\\operatorname{Cov} \\approx \\sigma^2$), so the eigenvalue is about $0$: these directions are **soft**.
- **Far from the data**, the posterior is a near-point mass in every direction, so $J_f \\approx kI$ is isotropic with **no eigengap**.

So the spectrum of $\\operatorname{sym}(J_f) = \\tfrac12(J_f + J_f^\\top)$ is a local manifold detector, and its eigengap is a natural gate. The symmetrization is needed because an implicit $f$ is not conservative.

## II.4 Proposed samplers (no retraining)

Per particle and per step, at the evaluation point (the look-ahead point for NAG):

1. Compute $f$ and $J_f$, then $\\operatorname{sym}(J_f) = V, \\operatorname{diag}(\\lambda_1 \\le \\dots \\le \\lambda_D), V^\\top$.
2. **Normal space** $N = {v_i : \\lambda_i \\ge \\rho, \\lambda\_{\\max}}$ (the top eigenvector is always in $N$). **Tangent space** $T$ = the rest. Default $\\rho = 0.5$.
3. **Gate:** on if $T \\neq \\emptyset$ and $\\lambda\_{\\max} / (\\max\_{i \\in T} |\\lambda_i| + \\varepsilon) &gt; \\tau$. Default $\\tau = 5$. Optionally **sticky**: once gated, a particle stays gated.
4. Projectors $P_N = V_N V_N^\\top$ and $P_T = I - P_N$. Split $f_n = P_N f$ and $f_t = f - f_n$.

Ungated particles always take the plain step $x \\leftarrow x - \\eta f$. For gated particles the three variants are:

| sampler | gated update | intent |
| --- | --- | --- |
| **Idea-1**: normal projection | $x \\leftarrow x - \\eta f_n$ | remove the slide entirely |
| **Idea-2**: tangential Langevin | $x \\leftarrow x - \\eta f_n - \\eta f_t + \\sqrt{2\\eta T}, P_T \\xi$ | sample the density along $\\mathcal M$ |
| **Idea-3**: tangential diffusion | $x \\leftarrow x - \\eta f_n + \\sqrt{2\\eta T}, P_T \\xi$ | spread uniformly along $\\mathcal M$ |

Here $\\xi \\sim \\mathcal N(0, I)$. For NAG, $f$ and $J_f$ are evaluated at the look-ahead point, and the momentum stores only the deterministic direction actually used (never the noise).

**Idea-1.** Every point of $\\mathcal M$ is a fixed point ($P_N f = 0$ there), so samples stop where they land. *Limitation:* the density along $\\mathcal M$ is set by where particles land, not by $p\_{\\text{data}}$.

**Idea-2.** Along the manifold the dynamics are the Langevin SDE

$$ dx_T = k\\sigma^2, \\nabla_T \\log p; dt + \\sqrt{2T}, dW_T, $$

whose stationary density is $\\propto p^{,k\\sigma^2 / T}$. Choosing $T \\approx k\\sigma^2$ recovers $p$ along $\\mathcal M$. $T$ must stay $&gt; 0$: annealing it to 0 brings back the collapse. The normal direction stays deterministic, so samples are not blurred off $\\mathcal M$.

**Idea-3** (variant). This drops $f_t$ and keeps only tangential noise, so samples random-walk along $\\mathcal M$ and their density becomes uniform along it. That is correct when the data density is uniform along the manifold, as for moons. Rule of thumb: tangential noise on a curve of radius $R$ adds an outward (Itô) drift of about $T/R$, which the normal stiffness $k_N$ balances at an offset of about $T/(R,k_N)$. So $T$ should stay moderate.

**Cost.** $D$ batched forward-mode JVPs for $J_f$ plus a batched $D \\times D$ eigendecomposition per step. That is cheap for a single low-dimensional action, but grows quickly for long chunks. So the method fits exactly the regime where the drift appears. For explicit models (EqM-E), $J_f$ is the Hessian of $E$ and is already symmetric. Option: recompute $J$ only every $m$ steps and reuse the eigenvectors in between.

## II.5 Checks and results in this repo

**Sanity test** ([test_sanity.py](test_sanity.py)), on the analytic field $f(z) = (\\varepsilon_t z_x,; k z_y)$ with $k = 20$ and $\\varepsilon_t = 0.5$, whose manifold is the x-axis:

- Idea-1 leaves $x$ unchanged and sends $y \\to 0$.
- Idea-2's spread in $x$ matches the expected variance $T/\\varepsilon_t$ (0.1001 vs 0.1003).
- Idea-3's $x$-variance grows like a random walk, $+2T\\eta n$.
- An isotropic field is never gated.

**2D moons**: NAG, $\\eta = 10^{-3}$, 1000 steps, 5000 samples, SWD to the test set (lower is better):

| sampler | model trained 100 epochs | model trained 300 epochs |
| --- | --- | --- |
| plain | 0.624 | 0.523 |
| Idea-1 ($\\tau = 5$) | 0.210 | 0.176 |
| Idea-2 (best $T \\le 0.2$) | 0.576 | – |
| **Idea-3** ($T = 0.3$) | **0.147** | **0.111** |

**Takeaways.**

- **Idea-1** removes the collapse: SWD stays near 0.2 from 250 to 1000 steps and at $3\\eta$, where plain reaches 1.259.
- **Idea-2** does not help here. Its premise $f_t \\approx -k\\sigma^2 \\nabla_T \\log p$ fails: moons have uniform density along each moon, so the true $\\nabla_T \\log p \\approx 0$. The strong drift is learned by the network, and no temperature turns it into the right density (best SWD 0.536, at $T = 1$).
- **Idea-3** works best, because it ignores the learned drift entirely.
- **Remaining gap:** the learned equilibrium curve never covers the left half of the inner moon (at most about 10% of samples there, vs 25% in the data). This holds for every training variant tried: 300 epochs helps, while width 256/512 and scaler 1/14 do not. It is a property of the learned field, which a sampler cannot fix.
- **Update:** changing what the model is trained on does close this gap. With extra fourier dimensions ([Part III](#part-iii-extra-dimensions-against-mse-gradient-cancellation)), the plain sampler covers the left half of the inner moon.

---

# Part III: Extra dimensions against MSE gradient cancellation

## III.1 The problem: the MSE average cancels the gradient

By the lemma (C.0), a well-trained network does not learn one target per training pair. It learns the **average target** over all the ways a noisy point $z$ could have been made:

$$ f^\*(z) = \\mathbb E\\big\[c(\\gamma),(\\epsilon - x) ,\\big|, x\_\\gamma = z\\big\] = \\mathbb E\\Big\[\\tfrac{c(\\gamma)}{1-\\gamma},(z - x) ,\\Big|, z\\Big\]. $$

Each possible source $x$ "votes" with a vector that points from $x$ to $z$, weighted by how likely it is that $x$ produced $z$. When the likely sources lie **on different sides** of $z$, their votes point in opposite directions and the average is small. This happens between the two moons and inside the outer moon. The learned field is then weak, or exactly zero, at places that are not data. Samples slow down there, stop on curves that bridge the moons, or merge into one curve.

This is a different problem from the tangential drift of Part II. Drift is a small slide *along* the data. Cancellation is the field becoming weak *off* the data because opposite votes average out.

In the 2D toy, the plain sampler at 1000 steps collapses onto one curve that runs from the top of the outer moon to the right tip of the inner moon (SWD 0.624). Training longer barely changes this: 0.523 after 300 epochs and 0.519 after 1000.

## III.2 The idea: give each point more coordinates

Train on the lifted point $\\tilde x = (x, \\phi(x)) \\in \\mathbb R^{2+m}$ instead of $x \\in \\mathbb R^2$. Here $\\phi$ is a fixed function that computes $m$ extra numbers from the 2D position. Training is unchanged: noise is added to all $2+m$ coordinates, and the network maps $\\mathbb R^{2+m} \\to \\mathbb R^{2+m}$. To sample, start from noise in $\\mathbb R^{2+m}$, run the same sampler there, and keep only the first two coordinates.

**Why it helps.** Given a noisy lifted point $\\tilde z = (z\_{2D}, z_e)$, the weight of a candidate source $x$ is

$$ p(x \\mid \\tilde z, \\gamma) ;\\propto; p(x); \\mathcal N\\big(z\_{2D};, \\gamma x,, (1-\\gamma)^2 I\\big); \\prod\_{j=1}^{m} \\mathcal N\\big(z\_{e,j};, \\gamma,\\phi_j(x),, (1-\\gamma)^2\\big). $$

Each extra coordinate is **another noisy measurement of where the point came from**. If $\\phi$ gives different values to points on different moons, a candidate on the wrong moon gets a tiny weight. Fewer candidates remain, fewer votes point in opposite directions, and the average stops cancelling.

**What $\\phi$ must satisfy:**

- **It must depend on $x$.** With $\\phi = 0$ the extra factor does not depend on $x$. It only reveals the noise level, through $\\lVert z_e \\rVert^2 \\approx m(1-\\gamma)^2$.
- **It must be nonlinear.** For a linear $\\phi(x) = Ax$, the lifted data is a flat 2D sheet. After a rotation that is just the 2D data stretched, plus zeros, which is the same as rescaling the data.
- **It must be fixed:** the same function at every call, so training and the residual check below agree.
- **It must be smooth enough** for the network to learn the surface ${(x, \\phi(x))}$.

## III.3 The three extra types

Config keys, in `config.yaml` and `config_explicit.yaml`:

| key | meaning | default |
| --- | --- | --- |
| `data.extra_dims` | $m$, the number of extra coordinates. `0` is the plain 2D model. | `0` |
| `data.extra_type` | `zeros`, `fourier` or `label` | `zeros` |
| `data.extra_freq` | $w$, the frequency of the fourier waves (fourier only) | `1.0` |

`zeros` **(control).** $\\phi(x) = 0$. This adds coordinates and network inputs but no information about $x$. It tests whether "more dimensions" alone helps.

`fourier` **(treatment).** For each extra coordinate $j$:

$$ \\phi_j(x) = \\sin\\big(x^\\top W_j + b_j\\big), \\qquad W_j \\sim \\mathcal N(0, w^2 I_2), \\quad b_j \\sim U\[0, 2\\pi\]. $$

$W$ and $b$ are drawn once from `training.random_state` (42), so every run with the same $m$ uses the same waves.

`label` **(strong treatment, moons only).** Every extra coordinate is $+1$ if the point is on the outer moon and $-1$ if it is on the inner moon, so 32 dims means 32 copies of the moon label. The label comes from the closest noise-free moon curve, so it can also be computed for generated samples. On the data it equals `make_moons`' own label.

### What a fourier coordinate is, concretely

**One extra coordinate is one wave over the plane.** $x^\\top W_j + b_j$ is constant along straight lines perpendicular to $W_j$. So $\\phi_j$ is constant along those lines and repeats every $2\\pi / \\lVert W_j \\rVert$ units across them. Colored over the plane, it looks like parallel stripes. The stripes do not know where the moons are: it is the same fixed formula everywhere. What matters is only that points on different moons get different values.

![One fourier coordinate for w = 1 to 12](Figures/fourier_waves_w1-12.png)**The frequency $w$ sets how narrow the stripes are.** $W_j$ is drawn as $w$ times a fixed random vector (the seed is the same for every $w$). So changing $w$ keeps the direction and phase of every wave and only squeezes it: doubling $w$ halves the stripe width. Median stripe period of the 32 waves actually used: 6.5 at $w=1$, 2.2 at $w=3$, 1.1 at $w=6$ and 0.54 at $w=12$. For scale, the moons span about $3 \\times 1.5$ units, the closest gap between them is 0.5, and each band is about 0.1 thick (noise 0.05).

**Example with real numbers.** Take point A = (0, 1), the top of the outer moon, and point B = (0, 0.5), the left tip of the inner moon. They are 0.5 apart, on different moons. A2 = (0.05, 1) is A moved 0.05 along its own moon. Below are the first 4 of the 32 extra numbers, and distances measured over all 32:

|  | A (first 4) | B (first 4) | distance A to B | distance A to A2 |
| --- | --- | --- | --- | --- |
| plain 2D, distance in the plane | – | – | 0.50 | 0.05 |
| $w=1$ | 0.14, 0.89, 0.72, −0.67 | −0.12, 1.00, 0.48, −0.17 | 1.51 | 0.15 |
| $w=3$ | 0.92, −0.51, 0.89, −0.15 | 0.39, 0.64, 0.90, −0.96 | 3.45 | 0.59 |
| $w=6$ | 0.43, −0.17, −0.68, −0.09 | 0.92, −0.51, 0.89, −0.15 | 4.76 | 1.12 |
| $w=12$ | −0.49, −1.00, 0.96, −0.55 | 0.43, −0.17, −0.68, −0.09 | 6.30 | 1.41 |

Near the data the training noise has a standard deviation of at most 0.2 per coordinate. Over 32 coordinates that is a vector of length about $0.2\\sqrt{32} \\approx 1.1$.

- **$w = 1$:** A and B differ by 1.5. After noise they are hard to tell apart, so their votes still mix.
- **$w = 3$ to $6$:** they differ by 3.5 to 4.8. The noise no longer hides which moon a point came from.
- **$w = 12$:** the moons are even further apart, but a 0.05 step *along the same moon* already changes the numbers by 1.4. Each moon becomes a strongly crumpled surface in 34 dimensions, which is harder for the network to learn.

**Rule of thumb for "how much information".** For a small step $\\delta$ in the plane, the lifted point moves by $(\\delta,, J\_\\phi \\delta)$ with $J\_\\phi = \\operatorname{diag}(\\cos(\\cdot)), W^\\top$. Averaging $\\cos^2 \\to \\tfrac12$ and $W W^\\top \\approx m w^2 I$ gives

$$ \\lVert \\delta\_{\\text{lifted}} \\rVert \\approx s, \\lVert \\delta \\rVert, \\qquad s = \\sqrt{1 + m w^2 / 2}. $$

The noise per coordinate stays the same, so locally the noise looks $s$ times smaller and the candidate set shrinks by the same factor. This only holds below one stripe period. Because $\\sin$ is bounded, points far apart end up about $\\sqrt m$ apart (about 5.7 for $m=32$), whatever $w$ is. The value of $s$ is listed for every run in III.5.

## III.4 Code changes

- [**utils.py**](utils.py)**:**

  - `Loader.extra_features(X, extra_dims, extra_type, extra_freq, seed)` computes $\\phi$. Fourier uses its own `default_rng(seed)`, so the global RNG is untouched.
  - New helpers:
    - `extra_phi(cfg, X)`: $\\phi$ from the config. It raises an error for `label` on non-moons data.
    - `lift(cfg, X)` returns $(X, \\phi(X))$.
    - `extra_residual(cfg, X)` returns the mean $\\lVert x_e - \\phi(x\_{2D}) \\rVert$. It is 0 when a sample lies exactly on the lifted data surface.
    - `extra_tag(cfg)`.
  - Checkpoint names get a suffix such as `_Xfourier32w6`, `_Xzeros32` or `_Xlabel32`. With `extra_dims = 0` names are unchanged.

- [**metrics.py**](metrics.py)**:** `nearest_on_moons` is split out of `moons_coverage`, so the label code can reuse it. `moons_coverage` output is unchanged.

- [**main.py**](main.py)**:**

  - Lifts the training data and gives the network $2+m$ inputs and outputs.
  - Draws initial noise with the 2D part first, so the 2D part of $z$ is identical to the 2D runs and only the extra noise coordinates are added.
  - Computes SWD and MMD on the 2D part, and prints the extra residual.
  - The plot filename now includes the temperature (idea2/idea3), width × layers, epochs and the extra tag. For example: `Eqm_none_plain_NumSteps_1000_Fino_False__Scaler_4.0_H128x3_E1000_Xfourier32w6_results.png`.

- [**sweep.py**](sweep.py)**, [train_sweep.py](train_sweep.py):** stop with an error if `extra_dims > 0`, since they assume 2D.

- [**extra_dims_sweep.py**](extra_dims_sweep.py) (new): trains or loads each variant (2D, +32 zeros, +32 fourier $w=1$, +32 fourier $w=3$, +32 label), then runs the plain sampler at 250 and 1000 steps, plus 1000 steps at $3\\eta$. It also measures cancellation directly, from (noisy point, target) pairs drawn exactly as in training:

  - `energy_ratio` $= \\sum \\lVert f\_{2D} \\rVert^2 / \\sum \\lVert t\_{2D} \\rVert^2$, per noise-level bin and as a map over $z\_{2D}$. Over any set of points defined by $z$ it is at most 1: 1 means the votes agree, and 0 means they cancel completely.
  - `cos_agreement` is the mean cosine between the field and the individual targets.

- [**test_extra_dims.py**](test_extra_dims.py) (new), 7 tests:

  - 2D runs are unchanged.
  - Zeros lift correctly.
  - Fourier is deterministic and leaves the global RNG alone.
  - Label matches `make_moons`.
  - The 2D part of the initial noise is shared.
  - Model and sampler shapes are correct.
  - The cancellation metric gives 0 for a zero field.

  Together with [test_sanity.py](test_sanity.py), all 14 tests pass.

Commands (in the `basic_rl` env):

```bash
# train (1000 epochs) and sample
python main.py data.extra_dims=32 data.extra_type=fourier data.extra_freq=6.0 training.epochs=1000
# re-sample from the saved model without retraining
python main.py data.extra_dims=32 data.extra_type=fourier data.extra_freq=6.0 \
  training.load_checkpoint=checkpoints/eqm_moons_none_Fino_False_Scaler_4.0_H128x3_E1000_Xfourier32w6.pt
# controls
python main.py data.extra_dims=32 data.extra_type=zeros
python main.py data.extra_dims=32 data.extra_type=label
# full extra-dims sweep with cancellation metrics
python extra_dims_sweep.py
```

## III.5 Experiments and results

**Common setup** (all runs):

- **Data:** moons, 20 000 points with noise 0.05; 4000 of them form the test set.
- **Model:** implicit EqM with an MLP of width 128 × 3 layers; scaler $\\lambda = 4$, $a = 0.8$, learning rate $3 \\cdot 10^{-4}$.
- **Sampler:** plain NAG with $\\eta = 10^{-3}$, $\\mu = 0.3$, 1000 steps and 5000 samples, all from the same initial noise.
- **Metric:** SWD between the 2D part of the samples and the test set; lower is better.

Every number is **one training run**: the network seed is fixed at 42.

### Experiment A: zeros control (100 epochs, `extra_dims_sweep.py`)

Columns: SWD after 250 steps, after 1000 steps, and after 1000 steps at $3\\eta$. Then two columns at 1000 steps: the extra residual, and the share of samples that are not on any moon.

| variant | 250 steps | 1000 steps | 1000 steps, $3\\eta$ | residual | off the moons |
| --- | --- | --- | --- | --- | --- |
| 2D | 0.110 | 0.624 | 1.259 | – | 6% |
| +8 zeros | 0.099 | 0.625 | 0.797 | 0.01 | 56% |
| +32 zeros | 0.163 | 1.092 | 3.369 | 0.10 | 77% |
| +128 zeros | 0.309 | 0.846 | 3.090 | 5.40 | 75% |

![Zeros control: field map and samples](Figures/extra_dims_zeros_E100.png)*Left column: $\\log\_{10} \\lVert f\_{2D} \\rVert$ on the data sheet (extra coordinates = 0); dark means a weak field. Red: samples. Blue: data.*

- **Zeros never help, and more zeros make things worse.**
  - **+8 zeros:** the field has isolated dark spots (single equilibrium points) instead of a dark valley along the moons.
  - **+32 zeros:** one basin in the middle, so the samples end on a diagonal band.
  - **+128 zeros:** the field is almost flat. The samples have not converged: the residual of 5.4 means the extra coordinates were never pulled back to 0.
- **Adding dimensions, inputs or parameters is not what helps.** The zeros carry no information about $x$, so cancellation over $x$ is unchanged, and the network gets a harder input.
- **The 250-step SWD is misleading.** At 250 steps many samples are still moving (25% are off the moons for 2D), so a low SWD there does not mean the field is good. All comparisons below use 1000 steps.

### Experiment B: fourier at equal training (100 epochs)

| model | SWD |
| --- | --- |
| 2D | 0.624 |
| +32 fourier, $w=1$ | 0.289 |
| **+32 fourier, $w=3$** | **0.145** |
| 2D + Idea-3 sampler ($T=0.3$), for reference | 0.147 |

- At the same 100 epochs and with the plain sampler, $w=3$ is **4× better** than 2D. It matches the best special sampler from Part II.
- **What the samples show** (figure below, top-right):
  - **Bridges:** curves where samples stop between the moons, such as a vertical streak at $x \\approx 0$ from the outer moon to the inner moon.
  - **Undertraining:** the loss was still falling at epoch 100 (6.5, against 5.2 after 1000 epochs).
- **Interpretation:** with 34 inputs and a curved surface to learn, the model needs longer training. That motivated the 1000-epoch runs.
- The 100-epoch runs for +32 label and +256 fourier were trained, but their plots were overwritten by the 1000-epoch runs before epochs were added to the filename. So their numbers are lost.

### Experiment C: frequency and number of dims (1000 epochs)

$s = \\sqrt{1 + m w^2/2}$ is the local stretch from III.3 (larger means more information about position).

| model | $m$ | $w$ | $s$ | SWD | what the samples look like |
| --- | --- | --- | --- | --- | --- |
| 2D | 0 | – | 1 | 0.519 | collapse: one curve from the outer moon's top to the inner moon's right tip; inner-left half empty |
| +32 label | 32 | – | – | 0.565 | outer moon roughly right (a bit too long); inner moon a thin curve that overshoots both tips; a stray streak across |
| +256 fourier | 256 | 3 | 34 | 0.392 | no structure: samples scattered over the whole plane |
| +1 fourier | 1 | 6 | 4.4 | 0.298 | broken pieces and crossing streaks |
| +32 fourier | 32 | 1 | 4.1 | 1.943 | thin wrong curves joining the moons |
| +4 fourier | 4 | 6 | 8.5 | 0.083 | both moons recognizable, but broken into segments with gaps |
| +32 fourier | 32 | 3 | 12 | 0.065 | both moons, mostly continuous; small streak at the inner moon's left tip |
| +12 fourier | 12 | 6 | 15 | 0.057 | both moons continuous, slightly wavy |
| **+32 fourier** | 32 | **6** | 24 | **0.038** | both moons fully covered, including the inner-left half; wavy, clumpy bands |
| +32 fourier | 32 | 12 | 48 | 0.089 | moon shapes visible but noisy: scattered clumps, bands too thick |

![Generated samples for all extra-dims runs](Figures/extra_dims_samples.png)### Experiment D: Idea-3 sampler in the lifted space (not finished)

Idea-3 already works in $2+m$ dims: a 2-step check on an untrained 34-dim model runs. A run on +32 fourier $w=3$ (1000 epochs) with $T = 0.3$ was started but stopped before it finished, so there is no result yet.

- **Cost:** each step needs $2+m = 34$ JVPs and a 34 × 34 eigendecomposition per sample. That is about 1.5 s per step on the CPU, or about 25 minutes for 1000 steps. `sampler.jacobian_every=10` cuts this to a few minutes.
- **Temperature:** the tangential noise spreads over all 34 coordinates, but most of a step along the lifted surface goes into the wave coordinates. So the 2D spread is roughly $T / s^2$, and $T \\approx 0.3 \\cdot 145 \\approx 40$ should match the 2D setting. This is an estimate, not a measurement.

**Not run yet:** the cancellation metrics (`energy_ratio`, `cos_agreement`). The sweep that produced Experiment A ran before they were added.

## III.6 Interpretation

1. **Information about position is what fixes the collapse; extra dimensions alone do not.** Zeros (no information) never help. Fourier (information) helps at equal training: 0.624 → 0.145 at 100 epochs, and 0.519 → 0.038 at 1000 epochs. Longer training alone barely helps the 2D model: 0.624, 0.523 and 0.519 at 100, 300 and 1000 epochs. This fits the cancellation story: the extra coordinates shrink the set of candidate sources, so fewer votes cancel. The direct check (the cancellation maps of `extra_dims_sweep.py`) has not been run yet.

2. **What counts is $m w^2$, not $m$ alone.** Sorted by $s$, the results line up:

   - $s \\approx 4$ ($w=1$, or a single extra dim) fails;
   - $s \\approx 8$ to $15$ works;
   - $s \\approx 24$ ($m=32$, $w=6$) is best;
   - $s \\approx 48$ ($w=12$) gets worse again.

   So 4 dims at $w=6$ carry more information than 32 dims at $w=1$. About 4 extra dims already give most of the gain, but leave gaps; 12 to 32 dims give continuous moons.

3. **Too high a frequency hurts.** At $w=12$ a small step along a moon changes the extra numbers by more than the noise does, so each moon becomes a strongly crumpled surface in $\\mathbb R^{34}$. A 128 × 3 MLP probably cannot fit it well; MLPs learn high-frequency functions slowly. This is a hypothesis: the waviness of the $w=6$ and $w=12$ samples may be the network's fitting error, and it has not been measured.

4. **The label type is worse than fourier.** The label tells the model which moon, but not where on the moon. So votes from different parts of the same moon still average, and the inner moon comes out as a thin curve that overshoots both tips, much like the 2D collapse. The label also jumps from +1 to −1 between the moons, which is hard for a smooth network. Fourier encodes both which moon and where along it, and it changes smoothly.

5. **256 extra dims fails.** With $w=3$ its $s$ (34) is in the good range, so the amount of information is not the issue. Likely causes, untested: 256 noise coordinates dominate the input, the field in the extra dims is learned poorly, or 1000 sampling steps are not enough to converge (the residual was not recorded).

6. **The inner-left gap from II.5 is closed** by training on lifted data ($m \\ge 12$, $w = 3$–$6$), with the plain sampler. Part II concluded that no sampler could close it, and that holds: the fix had to change the learned field. Coverage percentages were not computed for these runs, since `main.py` prints only SWD and MMD, so this is visual.

7. **Still open:**

   - The bands are wavy and clumpy instead of a smooth band of the right thickness.
   - Differences between 0.038, 0.057 and 0.065 are within what one could expect from seed noise; only the big jumps (0.5 → 0.08) are clearly real.
   - The SWD of +32 fourier $w=1$ at 1000 epochs (1.943) is probably inflated by a few samples far outside the plot.

## III.7 Next steps (proposed, not run)

- **Seeds:** add a `training.seed` option. The network seed is currently hard-coded to 42; `training.random_state` changes the test split and the waves, but not the initial weights. Then repeat $m = 2, 4, 8$ at $w=6$ with 3 seeds each.
- **Confirm the mechanism:** run the cancellation metrics on the saved checkpoints. The prediction is that `energy_ratio` between the moons is clearly higher for fourier than for 2D or zeros.
- **Frequency:** fill in $w = 4$–$8$ at $m = 32$, and $m = 8, 16$ at $w = 6$.
- **Idea-3 in the lifted space:** with `jacobian_every=10` and $T$ around 40, for the remaining tangential problems (clumps).
- **Robot actions:** $\\phi$ needs only the sample itself (no labels), so the same fourier lift applies directly to a single low-dimensional action. The sampler keeps the first $d_a$ coordinates.