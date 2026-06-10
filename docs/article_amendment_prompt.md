# Amendment prompt for `article.tex`
## "A stabilized finite element method for incompressible, inertial flows in inhomogeneous porous media"

You are an expert LaTeX editor and numerical analyst. Your task is to apply a set of
corrections to `article.tex` (SIAM format, `siamart190516` class). Every correction below
was found and verified during a full mathematical review of the paper (all key identities
re-derived symbolically). Apply them exactly as specified.

### Ground rules

1. **Verbatim matching.** Each amendment provides an OLD block copied character-for-character
   from `article.tex` and a NEW block to replace it with. Match the OLD string exactly
   (including spacing and line breaks). If a string does not match, stop and report it
   rather than guessing.
2. **Do not rename or renumber any `\label{...}`.** Other files (`shared.tex`,
   `elemental_matrices_appendix.tex`, the supplement) may reference them.
3. **Do not modify any table data** (`tab:*` environments) unless explicitly instructed.
4. **Preserve the draft-annotation macros** `\Guillermo{...}` and `\Joaquin{...}` unless an
   amendment explicitly edits their content; the authors use them to track changes.
5. After all edits, if the full project (with `shared.tex`, `references.bib`, `figures/`)
   is available, recompile with `pdflatex` + `bibtex` and confirm zero errors and no
   undefined references. `article.tex` alone does **not** compile (it `\input`s `shared`
   and `elemental_matrices_appendix.tex`).
6. Items marked **[VERIFY WITH AUTHOR]** change a scientific claim or describe what was
   done numerically; apply the edit but flag it prominently in your final report so the
   authors can cross-check against their simulation setup/data.

---

# PART A — Mandatory corrections (mathematical errors)

## A1. The smallness condition on ε must be a two-constant condition (eq:UpperBoundOnEpsilon)

**Why.** As stated, `ε < c₁ inf_K{α_K²τ_{1,K}/h²}` gives **no uniform margin**: when σ = 0
one has exactly `ετ₂ < 1`, so the coercivity coefficient `ε(1−ετ₂)` cannot be bounded
below by `Cε` with a constant `C` independent of the data, and (eq:PressureTermStabilityBound)
does not follow. The condition must fix a constant `C₂ < 1`. Also, `h` inside the infimum
should be the local `h_K`. (The numerical practice in the paper already complies: the choice
`ε = 10⁻⁴ε_ref` in Section 7.2 gives effectively `C₂ = 10⁻²`; see eq:EpsilonRef.)

**OLD:**
```latex
We have mentioned that $\varepsilon$ must be small. In particular, we will require that
\begin{equation} \label{eq:UpperBoundOnEpsilon}
    \varepsilon < c_1 \inf_K{\left\{\frac{\alpha_K^2 \tau_{1,K}}{h^2}\right\}}.
\end{equation}

From \cref{eq:UpperBoundOnEpsilon}, we have that the coefficient of the norm of the pressure term is
\begin{equation} \label{eq:PressureTermStabilityBound}
    \varepsilon (1 - \varepsilon \tau_2) > C \varepsilon,
\end{equation}
with $C>0$, so as to make sure that the compressibility term does not switch, from adding, to removing stability.
```

**NEW:**
```latex
We have mentioned that $\varepsilon$ must be small. In particular, we will require that
\begin{equation} \label{eq:UpperBoundOnEpsilon}
    \varepsilon \leq C_2\, c_1 \inf_K{\left\{\frac{\alpha_K^2 \tau_{1,K}}{h_K^2}\right\}},
\end{equation}
for some fixed constant $0 \leq C_2 < 1$, independent of the mesh and of the physical parameters.

From \cref{eq:UpperBoundOnEpsilon}, since $\varepsilon \tau_2 \leq C_2 < 1$ on every element, the coefficient of the norm of the pressure term satisfies
\begin{equation} \label{eq:PressureTermStabilityBound}
    \varepsilon (1 - \varepsilon \tau_2) \geq (1 - C_2)\, \varepsilon \eqqcolon C \varepsilon,
\end{equation}
with $C>0$, so as to make sure that the compressibility term does not switch, from adding, to removing stability. Note that the strict inequality $\varepsilon \tau_2 < 1$ alone would not suffice here, as it provides no uniform margin when $\sigma = 0$.
```

**Verification of the new claim** (for the editor's benefit): with
`τ₂ = h²/(c₁α_Kτ_{1,NS} + εh²)` one has `ετ₂ ≤ εh²/(c₁α_Kτ_{1,NS}) ≤ εh_K²/(c₁α_K²τ_{1,K}) ≤ C₂`,
where `τ_{1,K} ≤ α_K τ_{1,NS}` was used (valid since `τ_{1,K}⁻¹ = α_Kτ_{1,NS}⁻¹ + σ ≥ α_Kτ_{1,NS}⁻¹`).

## A2. The broken norm must be the ℓ² combination of elemental norms (line ~800)

**Why.** As defined (`Σ_K ‖·‖_{L²(K)}`, a plain sum), the quantities `‖τ₁^{1/2}αX(U_h)‖_h²`
etc. appearing in (eq:StabilityEstimate) would not equal the sum of squares used everywhere
in the estimates (e.g., line (eq:StabilityEstimateFinal) manipulates them as
`Σ_K ‖·‖²_{L²(K)}`). The standard definition is the square root of the sum of squares.

**OLD:**
```latex
where $\| \bullet \|_h \coloneqq \sum_K{\| \bullet \|_{L^2(K)}}$ and $X(U_h) \coloneqq \boldsymbol{a} \cdot \nabla \boldsymbol{u}_h + \nabla p_h$.
```

**NEW:**
```latex
where $\| \bullet \|_h \coloneqq \big( \sum_K{\| \bullet \|_{L^2(K)}^2} \big)^{1/2}$ and $X(U_h) \coloneqq \boldsymbol{a} \cdot \nabla \boldsymbol{u}_h + \nabla p_h$.
```

## A3. The Neumann (natural) boundary operator is missing a factor 2 (eq:neumann_bc)

**Why.** Integrating the weak form by parts, the natural flux is
`n_i K_{ij}∂_j U − n_i A_{f,i} U = 2αν Π̃∇u·n − αp n` on the momentum rows (the viscous
term in the strong equation carries the factor 2: `−2∇·(ανΠ̃∇u)`). With the bilinear form
(eq:BilinearForm) and `B(u,U,V)=L(V)` (eq:WeakFormCompact), consistency forces
`D_N U = α(2νΠ̃∇u − pI)·n`. As printed, the operator drops the 2 on the viscous part.

**OLD:**
```latex
    \mathcal{D}_{\text{N},U} \colon \Gamma_\text{N} &\rightarrow \mathbb{R}^3 \\
    \boldsymbol{x} &\mapsto \alpha \big ( \nu \ViscProj \nabla \boldsymbol{u} \small\vert_{\Gamma}(\boldsymbol{x}) - p \small\vert_{\Gamma}(\boldsymbol{x})  \mathbb{I} \big ) \cdot \boldsymbol{n},
```

**NEW:**
```latex
    \mathcal{D}_{\text{N},U} \colon \Gamma_\text{N} &\rightarrow \mathbb{R}^3 \\
    \boldsymbol{x} &\mapsto \alpha \big ( 2 \nu \ViscProj \nabla \boldsymbol{u} \small\vert_{\Gamma}(\boldsymbol{x}) - p \small\vert_{\Gamma}(\boldsymbol{x})  \mathbb{I} \big ) \cdot \boldsymbol{n},
```

**[VERIFY WITH AUTHOR]** only insofar as the implementation should be checked to apply the
traction with the factor 2 (the natural BC of the discrete form automatically does).

## A4. The viscous stabilization parameter has |k₀|², not |k₀| (eq:StabilizationParameters)

**Why.** The Fourier symbol of `u ↦ −2∇·(ανΠ̃∇u)` is `αν(|k|²I + (1−2/d)k⊗k)` with
eigenvalues `αν|k|²` (multiplicity d−1) and `(4/3)αν|k|²` for d = 3 (verified symbolically).
With the h-normalized wavenumber, the spectral radius is `(4/3)αν|k₀|²/h²`. This is also
the only version consistent with `c₁ := |k₀|²` and (eq:TauNavierStokes) downstream.

**OLD:**
```latex
  \tau_{\nu, 1}^{-1} &= \frac{4}{3} \alpha \nu \frac{|\boldsymbol{k}_0|}{h^2}, & \tau_{\nu, 2}^{-1} &= 0, \\
```

**NEW:**
```latex
  \tau_{\nu, 1}^{-1} &= \frac{4}{3} \alpha \nu \frac{|\boldsymbol{k}_0|^2}{h^2}, & \tau_{\nu, 2}^{-1} &= 0, \\
```

## A5. The inequality in (eq:FTOfDifferentialOperator) needs a factor equal to the number of terms

**Why.** `|Σᵢ τᵢ⁻¹|²_Λ ≤ Σᵢ |τᵢ⁻¹|²_Λ` is false in general (take two equal terms:
`|2A|² = 4|A|² > 2|A|²`). By the Cauchy–Schwarz inequality, `(Σᵢ aᵢ)² ≤ n Σᵢ aᵢ²` with
n = 5 contributions here. Since the argument is a heuristic motivation, the constant is
harmless, but the displayed inequality must be correct.

**OLD:**
```latex
\begin{align} \label{eq:FTOfDifferentialOperator}
  | \boldsymbol{\tau}_K^{-1} |_{\Lambda}^2 &\le | \boldsymbol{\tau}_{\nu}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{c}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{b}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{\sigma}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{\nabla \alpha}^{-1} |_{\Lambda}^2\\
  & = |\widehat{ \mathcal{L}}_{\nu}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{c}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{b}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{\sigma}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{\nabla \alpha}(\boldsymbol{k}_0) |_{\Lambda}^2
\end{align}
```

**NEW:**
```latex
\begin{align} \label{eq:FTOfDifferentialOperator}
  | \boldsymbol{\tau}_K^{-1} |_{\Lambda}^2 &\le 5 \left( | \boldsymbol{\tau}_{\nu}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{c}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{b}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{\sigma}^{-1} |_{\Lambda}^2 + | \boldsymbol{\tau}_{\nabla \alpha}^{-1} |_{\Lambda}^2 \right) \notag \\
  & = 5 \left( |\widehat{ \mathcal{L}}_{\nu}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{c}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{b}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{\sigma}(\boldsymbol{k}_0) |_{\Lambda}^2 + |\widehat{ \mathcal{L}}_{\nabla \alpha}(\boldsymbol{k}_0) |_{\Lambda}^2 \right)
\end{align}
```
(the factor 5, by the Cauchy--Schwarz inequality, is immaterial for the design argument and
can be absorbed in the algorithmic constants; the `\notag` keeps a single equation number).

## A6. Robustness section: corrected asymptotics and dominant-reaction estimates **[VERIFY WITH AUTHOR]**

All of the following were verified symbolically from the exact definitions
`τ₁ = (α_Kτ_{1,NS}⁻¹+σ)⁻¹`, `τ₂ = h²τ_{1,NS}⁻¹/(c₁α_K)`, `σ̃_α = στ_{1,NS}⁻¹/(τ_{1,NS}⁻¹+σ/α_K)`,
with `Re_h = |a|h/ν`, `Da_h = σh²/(α_Kν)`.

### A6.1 `σ̃_α` is missing a factor `α_K` in the general asymptotics (eq:GeneralAsymptoticBehaviourOfParameters)

The exact ratio of `σ̃_α` to the printed expression is `α_K` (symbolically exact with
c₁ = c₂ = 1). The printed version also contradicts the paper's own dominant-reaction limit
(which correctly shows `σ̃_α ~ α(1+Re_h)ν/h²`).

**OLD:**
```latex
        \widetilde{\sigma}_{\alpha} &\sim \frac{(1 + \Reyn_h)\Damk_h}{1 + \Reyn_h + \Damk_h} \, \frac{\nu}{h^2},
```

**NEW:**
```latex
        \widetilde{\sigma}_{\alpha} &\sim \alpha_K \frac{(1 + \Reyn_h)\Damk_h}{1 + \Reyn_h + \Damk_h} \, \frac{\nu}{h^2},
```

### A6.2 Same missing `α_K` in the dominant-viscosity limit

**OLD:**
```latex
    \tau_1 &\sim \frac{h^2}{\alpha_K \nu}, \\
    \tau_2 &\sim \frac{\nu}{\alpha_K}, \\
    \widetilde{\sigma}_{\alpha} &\sim \Damk_h \frac{\nu}{h^2}.
```

**NEW:**
```latex
    \tau_1 &\sim \frac{h^2}{\alpha_K \nu}, \\
    \tau_2 &\sim \frac{\nu}{\alpha_K}, \\
    \widetilde{\sigma}_{\alpha} &\sim \alpha_K \Damk_h \frac{\nu}{h^2}.
```

### A6.3 Same missing `α` in the dominant-convection limit

**OLD:**
```latex
    \tau_2 &\sim \frac{h \lVert \boldsymbol{a} \rVert_{\infty, K}}{\alpha}, \\
    \widetilde{\sigma}_{\alpha} &\sim \Damk_h \frac{\nu}{h^2},
```

**NEW:**
```latex
    \tau_2 &\sim \frac{h \lVert \boldsymbol{a} \rVert_{\infty, K}}{\alpha}, \\
    \widetilde{\sigma}_{\alpha} &\sim \alpha \Damk_h \frac{\nu}{h^2},
```

### A6.4 Dominant reaction: `τ₁ ~ 1/σ`, not `α/σ`

`τ₁ = (α_Kτ_{1,NS}⁻¹+σ)⁻¹ → 1/σ` as σ → ∞ (verified: lim τ₁σ = 1). The paper's own general
formula (eq:GeneralAsymptoticBehaviourOfParameters) gives the same limit. The spurious α
propagates into the displayed error bounds below.

**OLD:**
```latex
    \tau_1 &\sim \frac{\alpha}{\sigma}, \\
    \tau_2 &\sim \frac{1 + \Reyn_h}{\alpha} \nu, \\
    \widetilde{\sigma}_{\alpha} &\sim \alpha(1 + \Reyn_h) \frac{\nu }{h^2}.
```

**NEW:**
```latex
    \tau_1 &\sim \frac{1}{\sigma}, \\
    \tau_2 &\sim \frac{1 + \Reyn_h}{\alpha} \nu, \\
    \widetilde{\sigma}_{\alpha} &\sim \alpha(1 + \Reyn_h) \frac{\nu }{h^2}.
```

### A6.5 Dominant-reaction error bound: `α₀ → α₀^{1/2}` (three occurrences)

With the corrected `τ₁ ~ 1/σ`, the working-norm component
`‖τ₁^{1/2}α∇e_p‖ ≥ (α₀/(σν))^{1/2}·(να₀)^{1/2}‖∇e_p‖/(να₀)^{1/2}` and the right-hand side
weight `τ₁^{1/2}/(\nu α₀)^{1/2}` both carry `α₀^{1/2}/(σν)^{1/2}`, not `α₀/(σν)^{1/2}`.

**OLD:**
```latex
\begin{align*}
    \lVert \ViscProj \nabla \boldsymbol{e}_u \rVert + (1 + \Reyn_h)^{1/2} \frac{\lVert \boldsymbol{e}_u \rVert}{h} + \frac{\alpha_0}{\sigma^{1/2} \nu^{1/2}} \lVert \nabla e_p \rVert_h + \frac{(1 + \Reyn_h)^{1/2}}{\alpha_0^{1/2}} \lVert \nabla \cdot(\alpha \boldsymbol{e}_u)\rVert_h, \\
    \lesssim \frac{1}{\alpha_0} \left( (1 + \Reyn_h)^{1/2} \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h} + \frac{\alpha_0}{\sigma^{1/2} \nu^{1/2}} \frac{\mathrm{E}_\text{int}(p)}{h} \right).
\end{align*}
```

**NEW:**
```latex
\begin{align*}
    \lVert \ViscProj \nabla \boldsymbol{e}_u \rVert + (1 + \Reyn_h)^{1/2} \frac{\lVert \boldsymbol{e}_u \rVert}{h} + \frac{\alpha_0^{1/2}}{\sigma^{1/2} \nu^{1/2}} \lVert \nabla e_p \rVert_h + \frac{(1 + \Reyn_h)^{1/2}}{\alpha_0^{1/2}} \lVert \nabla \cdot(\alpha \boldsymbol{e}_u)\rVert_h \\
    \lesssim \frac{1}{\alpha_0} \left( (1 + \Reyn_h)^{1/2} \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h} + \frac{\alpha_0^{1/2}}{\sigma^{1/2} \nu^{1/2}} \frac{\mathrm{E}_\text{int}(p)}{h} \right).
\end{align*}
```
(note the stray comma at the end of the first line has also been removed).

### A6.6 Velocity-gradient estimate in the dominant-reaction case

The same `α₀^{1/2}` propagates, and the final coefficient becomes `Da^{1/2}/(α₀α_∞)^{1/2}`
(verified symbolically with `P ~ Da Uν/L`). Note also: the second display line was missing
its alignment, the equation ended with a period followed by a lowercase "where", and the
P-scaling in the following sentence must use the macroscopic length `L`, not `h`
(this is what makes the printed coefficient `Da^{1/2}/α_∞^{1/2}` come out at all).

**OLD:**
```latex
\begin{align}
    \lVert \ViscProj \nabla \boldsymbol{e}_u \rVert \lesssim \frac{1}{\alpha_0} \left( (1 + \Reyn_h)^{1/2} + \frac{\alpha_0}{\sigma^{1/2} \nu^{1/2}} \frac{P}{U}\right) \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h} \notag \\ \label{eq:DominantReactionVelocityGradientEstimate}
    \sim \left(\frac{1}{\alpha_0} (1 + \Reyn_h)^{1/2} + \frac{\Damk^{1/2}}{\alpha_\infty^{1/2}}\right) \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h}.
\end{align}
where we have used that $P \sim \Damk U \nu / h$ as $\Damk_h \rightarrow \infty$.
```

**NEW:**
```latex
\begin{align}
    \lVert \ViscProj \nabla \boldsymbol{e}_u \rVert &\lesssim \frac{1}{\alpha_0} \left( (1 + \Reyn_h)^{1/2} + \frac{\alpha_0^{1/2}}{\sigma^{1/2} \nu^{1/2}} \frac{P}{U}\right) \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h} \notag \\ \label{eq:DominantReactionVelocityGradientEstimate}
    &\sim \left(\frac{1}{\alpha_0} (1 + \Reyn_h)^{1/2} + \frac{\Damk^{1/2}}{(\alpha_0\alpha_\infty)^{1/2}}\right) \frac{\mathrm{E}_\text{int}(\boldsymbol{u})}{h},
\end{align}
where we have used that $P \sim \Damk U \nu / L$ as $\Damk_h \rightarrow \infty$.
```

### A6.7 Pressure-gradient estimate in the dominant-reaction case and its interpretation

With the corrected weights, the transfer coefficient evaluates (symbolically verified) to
`(1+Re_h)^{1/2}(α_∞/α₀)^{1/2} Da^{−1/2}` inside the parenthesis, with `Da^{−1/2} = Da_h^{−1/2} h/L`.
The printed version `Da_h^{−1/2}(L/h)` is incorrect on two counts (the α-power chain and an
inverted `h/L`), and — importantly — the corrected coefficient is **independent of the mesh
size** (`Da_h^{−1/2}h ∝ const` since `Da_h ∝ h²`), so the closing caveat about "the very
finest meshes" no longer applies. **This changes the narrative**: the Da-mitigation of the
pressure error does *not* degrade under refinement. Please cross-check with the convergence
tables (the reported high-Da pressure rates being stable across meshes is in fact consistent
with the corrected statement).

**OLD:**
```latex
\begin{equation}
    \lVert \nabla e_p \rVert_h \lesssim \frac{1}{\alpha_0} \left( \frac{(1 + \Reyn_h)^{1/2}}{\alpha_0} \Damk_h^{-1/2} \frac{L}{h} + 1 \right) \frac{\mathrm{E}_\text{int}(p)}{h}. \label{eq:DominantReactionPressureGradientEstimate}
\end{equation}
Here the situation is reversed with respect to the dependence on $\Damk_h$. The loss of one order in the pressure accuracy is greatly mitigated by the presence of $\Damk_h^{-1/2}$ in reaction-dominated flows, except for the very finest meshes.
```

**NEW:**
```latex
\begin{equation}
    \lVert \nabla e_p \rVert_h \lesssim \frac{1}{\alpha_0} \left( (1 + \Reyn_h)^{1/2} \Bigl(\frac{\alpha_\infty}{\alpha_0}\Bigr)^{1/2} \Damk^{-1/2} + 1 \right) \frac{\mathrm{E}_\text{int}(p)}{h}. \label{eq:DominantReactionPressureGradientEstimate}
\end{equation}
Here the situation is reversed with respect to the dependence on $\Damk$. The loss of one order in the pressure accuracy is greatly mitigated by the factor $\Damk^{-1/2} = \Damk_h^{-1/2}\, h/L$ in reaction-dominated flows. Note that this mitigation factor is independent of the mesh size, since $\Damk_h \propto h^2$.
```

Also check the Conclusions sentence mentioning "the mechanism of pressure error improvement
with growing $\Damk_h$" — with the corrected estimate the improvement is governed by the
(mesh-independent) macroscopic $\Damk$. Apply:

**OLD:**
```latex
such as the mechanism of pressure error improvement with growing $\Damk_h$ or the (very weak in practice) deterioration of the velocity error with $\Damk$.
```

**NEW:**
```latex
such as the mechanism of pressure error improvement with growing $\Damk$ or the (very weak in practice) deterioration of the velocity error with $\Damk$.
```

## A7. The forcing-term scaling is inverted (Section 6, nondimensionalization)

**Why.** Multiplying the dimensional momentum equation through by `L²/(α_∞νU)` produces the
dimensionless equation (eq:DimensionlessMomentumEquation) (verified term by term). Hence
`f* = L²/(α_∞νU) f`, i.e. `f = (α_∞νU/L²) f*`; the printed relation is its inverse. The
footnote's multiplication direction is inverted in the same way.

**OLD:**
```latex
$\boldsymbol{u} = U\boldsymbol{u}^*$, $\alpha = \alpha_\infty\alpha^*$, $\nabla = L^{-1}\nabla^*$ and $\boldsymbol{f} = L^2/(\alpha_\infty \nu U) \boldsymbol{f}^*$\footnote{\Guillermo{One can check that substituting back into \eqref{eq:DimensionlessMomentumEquation} the various definitions of the dimensionless variables and multiplying through by $L^2 / (\alpha_\infty \nu U)$ yields the original momentum conservation equation.}}.
```

**NEW:**
```latex
$\boldsymbol{u} = U\boldsymbol{u}^*$, $\alpha = \alpha_\infty\alpha^*$, $\nabla = L^{-1}\nabla^*$ and $\boldsymbol{f} = (\alpha_\infty \nu U / L^2) \boldsymbol{f}^*$\footnote{\Guillermo{One can check that substituting back into \eqref{eq:DimensionlessMomentumEquation} the various definitions of the dimensionless variables and multiplying through by $\alpha_\infty \nu U / L^2$ yields the original momentum conservation equation.}}.
```

## A8. Da_h decreases by 64² ≈ 4096, not 64, across the mesh sequence (Section 7.1)

**Why.** `Da_h ∝ h²` and the nodal spacing shrinks by a factor 64 from the 10×10 to the
640×640 grid, so `Da_h` shrinks by `64² ≈ 4.1·10³` (consistent with the paper's own numbers:
`10⁶/10² = 10⁴` coarsest vs `10⁶/640² ≈ 2.44` finest).

**OLD:**
```latex
becoming about \num{64} times smaller when passing from the coarsest discretization to the finest one.
```

**NEW:**
```latex
becoming about $64^2 \approx \num{4.1e3}$ times smaller when passing from the coarsest discretization to the finest one.
```

## A9. The inverse estimate applied to `∇·(ανΠ̃∇u_h)` needs a porosity-aware constant

**Why.** The inverse estimate (eq:InverseEstimateFiniteOrderNorm) holds for finite element
functions, but `ανΠ̃∇u_h` is **not** piecewise polynomial (α is a general W^{1,∞} field).
The estimate used at (eq:BoundOfCrossedTerm) is still valid, but with the modified constant
`C̄_inv := √(d δ_α) C_inv + C_α`, where `h‖∇α‖_{L^∞(K)} ≤ C_α α_{0,K}` is the resolution
condition (eq:SmallPorosityGradient) and `δ_α := 1 + C_α` (so `α_{∞,K} ≤ δ_α α_{0,K}`); see
the companion appendix (Lemma "weighted inverse estimates") for the proof. Insert the
following remark right after the paragraph ending with the citation `\cite{brenner2008mathematical}`
(i.e., after the sentence "…under the assumption that the sequence of mesh refinements is
non-degenerate (see, e.g., \cite{brenner2008mathematical}).") and replace `C_\text{inv}` by
`\overline{C}_\text{inv}` in the four places where the weighted estimate is used:

**INSERT after that paragraph:**
```latex
Note that, since $\alpha$ is not piecewise polynomial, \cref{eq:InverseEstimateFiniteOrderNorm} does not directly apply to quantities of the form $\nabla \cdot (\alpha \nu \ViscProj \nabla \boldsymbol{u}_h)$. Expanding the divergence and using \cref{eq:SmallPorosityGradient}, the same elemental estimates hold with $C_\text{inv}$ replaced by a constant $\overline{C}_\text{inv}$ that additionally depends on the constant in \cref{eq:SmallPorosityGradient}. With a slight abuse of notation, we keep the symbol $C_\text{inv}$ for this modified constant in what follows.
```

(The last sentence makes the minimal-change option explicit; alternatively, replace
`C_\text{inv}` by `\overline{C}_\text{inv}` in (eq:BoundOfCrossedTerm), the two displays that
follow it, (eq:conditions_on_num_param) and \cref{lemma:Stability} — choose one option and
apply it consistently.)

## A10. Complete the hypotheses of Lemma 1 (stability) and Lemma 2 (continuity)

**Why.** As stated, the lemmas omit hypotheses that their proofs use, reference the
pre-simplification parameter definitions, use the undefined space `X_h`, and Lemma 2 has a
grammar slip. Specifically: the stability proof uses (i) the simplified parameters
(eq:Tau1Final)–(eq:Tau2Final) (not eq:Tau1), (ii) the ε-condition (eq:UpperBoundOnEpsilon),
(iii) `∇·(αa)=0` and homogeneous Dirichlet conditions; the continuity result additionally
requires (iv) the resolution condition (eq:SmallPorosityGradient), (v) trace regularity of
`a`, and — when σ > 0 — (vi) a jump condition on τ₁ across interior faces (the analogue of
Eq. (39) in codina2001stabilized).

**OLD:**
```latex
\begin{lemma} \label{lemma:Stability}
  Assume that $\tau_1$ is defined as in \cref{eq:Tau1} and that $c_1 > 2\xi C_\text{inv}^2$, with $\xi > 2$. Then there exists a positive constant C such that for any $U_h = [\boldsymbol{u}_h; p_h] \in \mathcal{X}_h$ it holds that
```

**NEW:**
```latex
\begin{lemma} \label{lemma:Stability}
  Assume that $\tau_1, \tau_2$ are defined as in \cref{eq:Tau1Final,eq:Tau2Final}, that $\varepsilon$ satisfies \cref{eq:UpperBoundOnEpsilon}, that $\nabla \cdot (\alpha \boldsymbol{a}) = 0$, that all boundary conditions are of Dirichlet type, and that $c_1 > 2\xi C_\text{inv}^2$, with $\xi > 2$. Then there exists a positive constant $C$ such that for any $U_h = [\boldsymbol{u}_h; p_h] \in \mathcal{X}_{h0}$ it holds that
```

**OLD:**
```latex
\begin{lemma} \label{lemma:Continuity}
    Assume that $\tau_1, \tau_2$ are defined as in \cref{eq:Tau1,eq:Tau2} and that all the algorithmic constants involved are positive. Assume also that the field $\alpha\boldsymbol{a}$ is (weakly) divergence-free and $\nabla\alpha$ is uniformly bounded in $\Omega$. Then, there exist a positive constant $C$, such that
```

**NEW:**
```latex
\begin{lemma} \label{lemma:Continuity}
    Assume that $\tau_1, \tau_2$ are defined as in \cref{eq:Tau1Final,eq:Tau2Final}, that the hypotheses of \cref{lemma:Stability} hold, that the porosity field satisfies the resolution condition \cref{eq:SmallPorosityGradient}, and that $\boldsymbol{a}$ is continuous. Assume in addition, when $\sigma > 0$, that $\tau_1$ satisfies a jump condition across interior faces analogous to Eq.~(39) of~\cite{codina2001stabilized}. Then, there exists a positive constant $C$, such that
```

**OLD:**
```latex
    for all $U_h, V_h \in \mathcal{X}_h$.
\end{lemma}
```

**NEW:**
```latex
    for all $U_h, V_h \in \mathcal{X}_{h0}$.
\end{lemma}
```

Also fix the dangling reference to the pre-simplification τ in the coefficient expansion
(this fragment is independent of whether A1 has been applied):

**OLD (fragment):**
```latex
Using \cref{eq:Tau1}, the coefficient of
```

**NEW (fragment):**
```latex
Using \cref{eq:Tau1Final}, the coefficient of
```

## A11. The manufactured solution does not satisfy null Dirichlet conditions **[VERIFY WITH AUTHOR]**

**Why.** The chosen field (eq:ManufacturedProblem) has
`u₂ = U(α₀/α)cos(πx₁)cos(πx₂)`, which does not vanish on any side of the unit square
(e.g., `u₂(x₁,0) = U(α₀/α)cos(πx₁) ≠ 0`). The MMS therefore requires non-homogeneous
Dirichlet data given by the trace of the manufactured solution. (The field does satisfy
`∇·(αu) = 0` exactly, as required — verified.)

**OLD:**
```latex
We consider the unit square $(0,1) \times (0,1)$ as the domain with null Dirichlet boundary conditions on all sides.
```

**NEW:**
```latex
We consider the unit square $(0,1) \times (0,1)$ as the domain, with Dirichlet boundary conditions on all sides given by the trace of the manufactured velocity field.
```

Flag this for the authors: confirm that the implementation indeed imposed the manufactured
trace (anything else would not have produced the reported optimal rates).

---

# PART B — Minor corrections (typos, notation, prose)

## B1. Missing elemental subscript on the dual pairing in the definition of B_S

**OLD:**
```latex
  B_\text{S}(\boldsymbol{w}, U, V) &\coloneqq B(\boldsymbol{w}, U, V) - \sum_K{ \dualpairing{\mathcal{L}_{\boldsymbol{w}}^*V, \boldsymbol{\tau}_K(\boldsymbol{w}) \mathcal{L}_{\boldsymbol{w}} U}}, \\
```
**NEW:**
```latex
  B_\text{S}(\boldsymbol{w}, U, V) &\coloneqq B(\boldsymbol{w}, U, V) - \sum_K{ \dualpairing{\mathcal{L}_{\boldsymbol{w}}^*V, \boldsymbol{\tau}_K(\boldsymbol{w}) \mathcal{L}_{\boldsymbol{w}} U}_K}, \\
```

## B2. The convective stabilization parameter is a spectral radius, hence non-negative

**OLD:**
```latex
  \tau_{c, 1}^{-1} &= \alpha \frac{\boldsymbol{w} \cdot \boldsymbol{k}_0}{h}, & \tau_{c, 2}^{-1} &= 0, \\
```
**NEW:**
```latex
  \tau_{c, 1}^{-1} &= \alpha \frac{|\boldsymbol{w} \cdot \boldsymbol{k}_0|}{h}, & \tau_{c, 2}^{-1} &= 0, \\
```

## B3. "Equations \cref{...}" duplicates the word produced by cleveref

**OLD:**
```latex
Using these definitions, the boundary value problem defined by Equations \cref{eq:StrongMassEquation,eq:StrongMomentumEquation}, together with appropriate boundary conditions can be cast in the following standard form:
```
**NEW:**
```latex
Using these definitions, the boundary value problem defined by \cref{eq:StrongMassEquation,eq:StrongMomentumEquation}, together with appropriate boundary conditions, can be cast in the following standard form:
```

## B4. "i.e," missing the second period (inside a \Guillermo annotation)

**OLD (fragment):**
```latex
\Guillermo{(i.e, dominated by the term that is proportional to the fluid velocity by analogy with classical convection-diffusion-reaction systems)}
```
**NEW (fragment):**
```latex
\Guillermo{(i.e., dominated by the term that is proportional to the fluid velocity by analogy with classical convection-diffusion-reaction systems)}
```

## B5. Missing sentence-final period after the DBF footnote

**OLD (fragment):**
```latex
we recover the DBF equations~\cite{cocquet2021error}\footnote{Although in other works we find $\scriptstyle{\DPi} \displaystyle{=} \scriptstyle{\SPi} \displaystyle{\equiv} \mathbb{I}$~\cite{skrzypacz}, even though the name used for the equations is also DBF.}
```
**NEW (fragment):**
```latex
we recover the DBF equations~\cite{cocquet2021error}\footnote{Although in other works we find $\scriptstyle{\DPi} \displaystyle{=} \scriptstyle{\SPi} \displaystyle{\equiv} \mathbb{I}$~\cite{skrzypacz}, even though the name used for the equations is also DBF.}.
```

## B6. Spelling consistency (SIAM uses American English)

Replace `analyse` → `analyze` (one occurrence, Section 7: "will allow us to analyse the
performance"). Optionally also `modelled` → `modeled` (Introduction) and
`modelling` → `modeling` (Section 4.2 heading "Finite element discretization \& modelling of
SGSs"), and `realised` → `realized` (Section 7.1). Apply uniformly; do not change words
inside citation keys or quoted titles.

## B7. "fluid and pressure fields" → "velocity and pressure fields"

**OLD (fragment):**
```latex
Our pick for the fluid and pressure fields are
```
**NEW (fragment):**
```latex
Our picks for the velocity and pressure fields are
```

## B8. Missing comma after the A_b matrix

**OLD:**
```latex
      \delta_{i1} & \delta_{i2} & \delta_{i3} & 0
    \end{bmatrix} \\
    \mathbf{S}_{\nabla \alpha} &=
```
**NEW:**
```latex
      \delta_{i1} & \delta_{i2} & \delta_{i3} & 0
    \end{bmatrix}, \\
    \mathbf{S}_{\nabla \alpha} &=
```

## B9. Matrix norm `|·|_Λ` of an operator used before it is defined

In (eq:BoundProjectionOfLBySubscales) the quantity `|L̂|_Λ` denotes the operator norm from
`|·|_{Λ⁻¹}` to `|·|_Λ`; this is only made precise later (via the generalized spectral
radius). Insert one clarifying sentence immediately after the display
(eq:BoundProjectionOfLBySubscales), i.e. after the sentence ending
"…must be $|\boldsymbol{k}_0| \gtrsim 1$.":

**INSERT:**
```latex
Here, $| \widehat{\mathcal{L}} |_{\Lambda}$ denotes the operator norm of the matrix $\widehat{\mathcal{L}}(\boldsymbol{k})$ from $(\mathbb{C}^n, |\bullet|_{\Lambda^{-1}})$ to $(\mathbb{C}^n, |\bullet|_{\Lambda})$, which coincides with $\rho_{\Lambda^{-1}}\bigl(\widehat{\mathcal{L}}^\dag \Lambda \widehat{\mathcal{L}}\bigr)^{1/2}$ in the notation introduced below.
```

## B10. The second equality in (eq:BoundProjectionOfLBySubscales) should be an inequality

Dropping the projection `Π̃` is valid as an upper bound (an orthogonal projection is a
contraction), not an equality. In the first line of the display, change the second `=`
(the one before the integral) to `\le`:

**OLD (fragment):**
```latex
\| \compositeaccents{\widehat}{\widetilde{\Pi}[\mathcal{L}\widetilde{U}]} \|_{L^2_\Lambda(\mathbb{R}^d)}^2 = \int_{\mathbb{R}^d}{ | \widehat{\mathcal{L}} \compositeaccents{\widehat}{\widetilde{U}}|_{\Lambda}^2 \, \mathrm{d} \boldsymbol{k}} \notag \\
```
**NEW (fragment):**
```latex
\| \compositeaccents{\widehat}{\widetilde{\Pi}[\mathcal{L}\widetilde{U}]} \|_{L^2_\Lambda(\mathbb{R}^d)}^2 \le \int_{\mathbb{R}^d}{ | \widehat{\mathcal{L}} \compositeaccents{\widehat}{\widetilde{U}}|_{\Lambda}^2 \, \mathrm{d} \boldsymbol{k}} \notag \\
```

## B11. Notation: the convective field in the analysis section is `a`, not `w`

In the two coefficient expansions following (eq:StabilityEstimateFinal), replace
`|\boldsymbol{w}|_{\infty, K}` by `\lVert \boldsymbol{a} \rVert_{\infty, K}` (two
occurrences: the display giving the coefficient of the viscous norm and the display giving
the coefficient of `‖u_h‖²_K`); the convective field was introduced as `a` at the start of
the section.

## B12. "α = 0.5" should be "α₀ = 0.5" (Section 7.1, convection-dominated discussion)

**OLD (fragment):**
```latex
we were unable to get the nonlinear iterations to converge with biquadratic elements on the finest mesh when $\alpha = 0.5$
```
**NEW (fragment):**
```latex
we were unable to get the nonlinear iterations to converge with biquadratic elements on the finest mesh when $\alpha_0 = 0.5$
```

## B13. Resistance-tensor parenthetical is dimensionally imprecise (optional)

`σu` must have the dimensions of `ν∇²u`, so σ is the kinematic viscosity times the inverse
permeability, not the inverse permeability itself.

**OLD (fragment):**
```latex
where $\boldsymbol{\sigma}$ represents a viscous resistance tensor (the inverse of the permeability tensor)
```
**NEW (fragment):**
```latex
where $\boldsymbol{\sigma}$ represents a viscous resistance tensor (the kinematic viscosity times the inverse of the permeability tensor)
```

## B14. Trace operator codomain: prefer `\mathbb{R}^d` (optional, two occurrences)

In (eq:TraceOperator) and (eq:dirichlet_bc)/(eq:neumann_bc) the codomain is written
`\mathbb{R}^3` although the section is otherwise dimension-agnostic; replace by
`\mathbb{R}^d` in the three `\rightarrow \mathbb{R}^3` occurrences of that subsection, or
add "(for $d=3$)" once. Apply consistently with the authors' preference.

## B15. Cleanup of commented duplicate labels (optional, hygiene)

The commented-out blocks at the lines containing `%\end{equation} \label{eq:AdjointFlux}`
(weak-form section) and `% \begin{align} \label{eq:VMSWeakFormSystem}` (stabilized-problem
section) contain duplicate `\label`s that will clash if ever uncommented. Either delete the
dead blocks or rename the labels inside the comments.

## B16. Pending items inside draft annotations (report, do not fix)

Report to the authors (do not attempt to resolve): `\Guillermo{Add figures}` in
Section 7.3 (also missing a space: `in\Guillermo{Add figures}` → `in \Guillermo{Add figures}`
— apply the space fix), and the literature-example subsection ends without the announced
figures. The 16 `\Guillermo{}` and 3 `\Joaquin{}` wrappers must be accepted/flattened before
submission.

## B17. Non-bold component symbol in the inflow profile (literature example)

In the subsection "An example from the literature", the inlet profile component is written
with a bold `u` although it is a scalar component. Replace the fragment
`\boldsymbol{u}_\text{in,1}` by `u_\text{in,1}` (one occurrence), and add the missing
final period at the end of the porosity-profile equation that follows (the display ends
without punctuation and the next sentence starts a new paragraph).

---

# PART C — Recommended additions (strengthen the paper; optional but advised)

## C1. Cite the companion appendix for the continuity lemma

A complete, self-contained proof of \cref{lemma:Continuity} (adapting Lemma 2 of
codina2001stabilized step by step to the porous setting, including the weighted inverse
estimates of A9 and the precise hypotheses of A10) is available in the companion document
`continuity_appendix.tex`. Replace the hand-waving sentence:

**OLD:**
```latex
It is also straightforward to follow an analogous process to that used in the proof of Lemma 2 in~\cite{codina2001stabilized} to prove a certain continuity of the bilinear form $B_\text{S}$. In particular, it is possible to show that
```
**NEW:**
```latex
Following the process used in the proof of Lemma 2 in~\cite{codina2001stabilized}, adapting every estimate to the presence of the porosity field, one can prove the following continuity result for the bilinear form $B_\text{S}$ (a complete proof is provided in the supplementary material):
```
(and add the appendix as supplementary material or an appendix section, per the authors'
choice).

## C2. State the regularity required by the convergence theorem

In \cref{th:Convergence}, "under the assumptions of \cref{lemma:Stability,lemma:Continuity}"
should be supplemented with the interpolation regularity: insert
", and assuming $\boldsymbol{u} \in H^{k_u+1}(\Omega)^d$ and $p \in H^{k_p+1}(\Omega)$"
after "under the assumptions of \cref{lemma:Stability,lemma:Continuity}".

## C3. Optional sharper estimate remark

The continuity proof yields the sharper, locally $\alpha_K$-weighted bound
`B_S(a;U_h,V_h) ≤ C(‖α_K τ₂^{1/2}h⁻¹u_h‖_h + ‖α_K τ₁^{1/2}h⁻¹p_h‖_h)|||V_h|||`, from which
the article's form follows using `α_K ≤ 1`. The corresponding convergence estimate carries
the local weight `α_K` inside the sum, which mildly *strengthens* the robustness discussion
in regions of low porosity. Consider adding a remark after \cref{th:Convergence} (text
available in the companion appendix, Remarks 3.2 and 4.4).

---

# PART D — Review summary: what was checked and found CORRECT

For the authors' record, the following were explicitly verified during this review (symbolic
verification where applicable) and require **no change**:

1. **CDR-form matrices** (eq:matrices_stationary_strong_problem): the `K_ij` block exactly
   reproduces `−2∇·(ανΠ̃∇u)` for d = 3 (including the 1/3 and 2/3 entries); `A_c`, `A_f`,
   `S` reproduce convection, pressure gradient, reaction, and the split
   `∇·(αu) = α∇·u + u·∇α` plus `εp` in the mass row.
2. **Adjoint operators** (eq:AdjointFlux, eq:AdjointDifferentialOperator): both correct,
   including the elementwise boundary flux `n_iK_{ji}ᵀ∂_jV + n_iA_{c,i}ᵀV`.
3. **B_S/L_S sign conventions** (eq:OSGSProblem and the definitions that follow): consistent
   with `Ũ = τ(RU_h − π_h)` and with the ASGS recovery `π_h = 0`.
4. **Galerkin stability identity** and the full **stability chain**
   (eq:StabilityEstimate) → (eq:BoundOfCrossedTerm) → (eq:StabilityEstimateFinal), including
   the Young-inequality weights and both coefficient expansions (verified algebraically;
   the constants `C = min{2−4C²inv/c₁, 2(1−2/ξ)}` and `C = 1−ξC²inv/c₁` are correct).
   Side note: `c₁ > 2ξC²inv` is slightly stronger than strictly needed (`c₁ > ξC²inv` with
   `ξ > 2` suffices) — conservative, not an error.
5. **Fourier design**: the `τ_b`, `τ_σ`, `τ_∇α` entries (eq:StabilizationParameters) are
   correct (the `√λ`/`1/√λ` pair was re-derived from the generalized eigenproblem); the
   choice `λ = h²/(|k₀|²τ²_{1,NS})` recovers (eq:Tau1)–(eq:Tau2) exactly, including
   `C_α = α + (h/|k₀|)|∇α|` (verified by symbolic summation of the five contributions).
6. **The viscous symbol eigenvalues** are `αν|k|²` (×2) and `(4/3)αν|k|²` (sympy), which is
   precisely why A4 is needed and why `c₁ = |k₀|²` is the consistent identification.
7. **eq:EpsilonRef chain** (Section 7.2): both inequalities verified; the practical choice
   `ε = 10⁻⁴ε_ref` satisfies the corrected condition A1 with `C₂ = 10⁻²`.
8. **Manufactured solution** (eq:ManufacturedProblem): satisfies `∇·(αu) = 0` identically
   (the `α₀/α` factor does its job); the plateau-bump porosity (eq:PlateauBumpFunction)–
   (eq:Gamma) is smooth and monotone as claimed (`dγ/dη = (2η²−2η+1)/(η(1−η))² > 0`,
   correct limits at `r₁`, `r₂`).
9. **Literature example**: the Cocquet et al. coefficients `a(α) = (150/Re)((1−α)/α)²`,
   `b(α) = 1.75(1−α)/α` and the exponential porosity profile match the reference.
10. **Label/environment hygiene**: all live `\cref` targets resolve within
    `article.tex` except `app:ElementalMatrices` (defined in the input appendix file —
    expected); all environments balanced; the only duplicate labels are inside comments
    (see B15).

## Reporting

After applying the amendments, produce a short report listing: (i) each amendment applied
with a one-line confirmation, (ii) any OLD string that failed to match verbatim (with the
closest match found), (iii) the items flagged **[VERIFY WITH AUTHOR]** (A3 implementation
check, A6 narrative change vs. numerics, A11 boundary conditions), and (iv) compile status
if the full project was available.
