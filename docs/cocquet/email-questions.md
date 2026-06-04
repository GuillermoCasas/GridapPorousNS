# Draft follow-up email to Cocquet et al. (re: J. Comput. Appl. Math. 2021)

> **ARTIFACT — draft email (status: unsent as of 2026-06-04).** Not a status doc. Findings context: canonical [cocquet_investigation_synthesis.md](investigation-synthesis.md). Some questions below may be obviated by the later ~2× mesh-recipe result (synthesis header).

This is a **follow-up** to the earlier mesh-focused email (still awaiting response). The follow-up:
- Acknowledges the prior message in case it got lost.
- Adds value before the second batch of questions: a controlled experiment (AllDirichlet) now empirically supports the corner-singularity intuition that motivated the first email.
- Does NOT re-ask the mesh recipe / `a(2N)+b(N)+c(2N)+d(N)` question (already covered).
- Asks the seven non-mesh questions that remain open.

Suggested email subject: *Re: Reproducing Figure 2 of your DBF paper — a few additional questions*

---

## Draft email

Dear Professor Cocquet,

I am writing to follow up on my earlier message about the mesh used in Figure 2 of your 2021 DBF paper. In case it did not arrive, I have consolidated below: a brief update on what I have observed since, and a small set of additional technical questions that — if you have a few minutes — would help me close the comparison. Please do not feel obliged to answer all of them.

**An empirical update.** Since writing to you, I have run a controlled experiment that supports the corner-singularity intuition I raised earlier. Replacing your traction-free outlet with a parabolic Dirichlet (same profile as the inlet) lifts our P₂/P₂ ASGS slope from ~1.27 to **2.46** on a uniform structured mesh — i.e., the Dirichlet/Neumann corner mismatch at (2,0) and (2,1) is empirically the dominant convergence-cap mechanism in our setup. On an unstructured FreeFem-style mesh with the original mixed BCs the slope recovers further to ~2.05 (matching your reported ~2.18). So the mesh / corner-singularity story explains essentially all of the *slope* difference.

**However**, even with the corner singularity eliminated by the BC swap, our absolute L²(u) at N=80 is still about **12× larger** than the value Figure 2 reports (4.7×10⁻⁶ vs ~4×10⁻⁷). On the unstructured mesh with the original mixed BCs, where the slope matches yours, the absolute gap is roughly **600× at N=80**. Independently, our manufactured-solution test on the same operators reaches the optimal P₂ rate (L²(u) ~ 2.85×10⁻⁷ at h=1/320 with the unstabilized P₂/P₁ Galerkin pathway), so the discretization itself appears to be correctly implemented. The mesh seems to fully account for the slope match, but there appears to be an additional source of magnitude difference that the mesh recipe alone may not explain.

With that in mind, the additional questions:

### Highest impact

1. **Raw data / mesh files.** Would you be willing to share the raw L²(u) values plotted in Figure 2 right panel (or Figure 3), or any of the .msh files used? Even a single (N, L²(u)) datapoint, e.g. at N=10, would be enormously informative.

2. **Pressure gauge.** Equations (1)–(2) do not fix the pressure mean for the natural-outlet case. Did you use the εη·p·q penalty with η=10⁻⁷ (Sec. 5, p. 30 of the paper), a Lagrange multiplier, or a zero-mean projection? Just confirming the gauge you actually used in the runs that produced Figure 2.

3. **Reference solution.** Is the N=200 reference solved with the same P₂/P₁ Taylor–Hood pair as the coarse comparisons, or with a higher-order method (e.g., P₃/P₂) to make the reference closer to the continuous truth?

### Useful but lower priority

4. **Quadrature degree.** The β(ε)·|u|·u term is non-polynomial in u. What quadrature order did you use? We use 4k_v+⌊k_v/2⌋=9 for k_v=2, and have verified that bumping to 21 changes our residual by only ~5×10⁻⁹.

5. **Linear solver.** Direct (UMFPACK / MUMPS / PARDISO) or iterative? What tolerance for the linear solve inside Newton? We use a direct LU with Newton ftol/xtol = 10⁻¹¹ and reach a residual of ~10⁻¹³.

6. **Forchheimer–Ergun constants.** We use the standard 150 and 1.75 (your Eq. 49). Some groups use 180 / 1.8 — just confirming you used 150 / 1.75 in the runs behind Figure 2.

7. **Outlet boundary integral.** We obtain ε·σ·n=0 naturally by integrating the pressure-divergence term by parts. Did you add any explicit boundary integral on Γ_out (e.g., a backflow-stabilization term of the form `+∫_{Γ_out, u·n<0} …` of the kind commonly added in open-flow problems), or rely purely on the natural condition?

Any answer to any of the above would help me close the comparison. I am happy to share our code, raw results, and field plots in return if it would be useful.

Thank you again for considering this, and apologies for the second email if my first one is simply still working its way through.

Best regards,

Guillermo Casas González

---

## Why each question matters (internal record — not for the email)

| # | Question | If they say… | Closes / informs hypothesis |
|---|---|---|---|
| 1 | Raw data / .msh files | Any numbers or files | Lets us directly check whether our remaining gap is per-N constant (suggests global mode S3) or grows with N (suggests slope cap from mesh / corner). Best single answer. |
| 2 | Pressure gauge | "η-penalty 1e-7" | Confirms H8 retrofit was correct. Closes O6. |
| 2 | Pressure gauge | "Lagrange multiplier / zero-mean" | Opens a concrete fix for our code (add gauge). Could close the residual ~12× magnitude factor. |
| 3 | Reference solution order | "Same P₂/P₁" | Strengthens our O1 symmetry argument. |
| 3 | Reference solution order | "Higher order (P₃/P₂)" | Their reference is closer to truth than ours; the apparent 845× gap shrinks as a true measure of discretization error. Could substantially reduce the apparent gap. |
| 4 | Quadrature | Any answer | Sanity check on O5. We've ruled this out as a 5×10⁻⁹ effect. |
| 5 | Linear solver | Any answer | Sanity check. We verified Newton already converges to machine precision. |
| 6 | Forchheimer constants | "150, 1.75" | Confirms H4. |
| 6 | Forchheimer constants | "180, 1.8" | New hypothesis — re-run with their constants, predict ~20% drag shift, irrelevant to magnitude gap. |
| 7 | Outlet boundary integral | "No, natural only" | Confirms H5. |
| 7 | Outlet boundary integral | "Yes, backflow term" | New hypothesis — could affect outlet-corner solution structure. |

## Notes on tone

- The follow-up adds **value** (a confirming experiment) before re-asking; this reduces the "another batch of questions" feel.
- "Apologies for the second email" at the end softens the re-contact. If the recipient never received the first, they get full context. If they did, they will read the new content.
- The opening line names the original email by its subject thread; if you sent it via a different subject, edit accordingly.
