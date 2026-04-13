# Lessons Learned – Porous VMS Solver

This file serves as the authoritative, persistent memory bank for historical mistakes, anti-patterns, and mathematical regressions previously corrected in the Porous Navier-Stokes solver. 

**Rules for this file:**
* append only
* one row per lesson
* concise but precise
* no vague wording
* "Why it was wrong" must be mathematical or numerical, not emotional
* "Correct action taken" must describe the concrete fix or policy
* include tags like `ASGS`, `OSGS`, `adjoint-sign`, `Jacobian`, `Newton`, `MMS`, `projection`, `stopping-criteria`, `legacy-branch`, `verification`

| Date | Error / Anti-pattern | Why it was wrong | Correct action taken | Related files | Tags |
| ---- | -------------------- | ---------------- | -------------------- | ------------- | ---- |
| 2026-04-13 | Using raw infinity norm for staggered OSGS state drift | $\ell^\infty$ drift relies on interleaved DOF layouts, heavily skewing multi-physics (velocity vs pressure) scaling and corrupting projection limits. | Replaced discrete array slicing with continuous functional $L^2$ mapping via identical finite element operator `∫(e_u ⋅ e_u)dΩ`. | `porous_solver.jl` | `OSGS`, `stopping-criteria`, `projection` |
| 2026-04-13 | Passing implicit or unbracketed projection state during stabilization initialization | Passing zeroed residuals into an iterative OSGS staggered loop falsely mimics ASGS projection-free boundaries, corrupting algorithmic continuity. | OSGS natively bootstraps prior iteration $\Pi(\mathcal{R})$ before starting the Newton sub-solves inside the discrete loop. | `porous_solver.jl` | `OSGS`, `ASGS`, `projection` |
| 2026-04-13 | Coupling outer OSGS tolerance checks loosely against static inner Newton boundaries | Allows the inner solver to over-iterate wildly past floating point boundaries when the outer projection drift converges much earlier, exploding iteration costs. | Implemented dynamic Eisenstat-Walker coupling `tau_inner_m = max(ftol, min(0.5, 0.1 * prev_osgs_drift))`. | `porous_solver.jl` | `OSGS`, `stopping-criteria`, `Newton` |
| 2026-04-13 | Silently dropping analytical derivative terms evaluating Exact Newton components | Fails to incorporate the $\partial_{\boldsymbol{u}} \tau$ and $\partial_{\boldsymbol{u}} \Pi$ functional evaluations, truncating true quadratic convergence to linear steps. | Mapped analytical discrete variations through explicit `ExactNewtonMode()` typing dispatch across stabilization kernels. | `viscous_operators.jl` | `Jacobian`, `Newton`, `adjoint-sign` |
| 2026-04-13 | Replacing canonical viscous operator formulations silently with simplified legacy mapping | Fails to account for non-constant viscosity distributions resulting in invalid Laplacian simplifications $\nabla \cdot (\nu \nabla u)$ rather than $\nabla \cdot (2\nu \nabla^s u)$. | Enforced analytical deviatoric mapping $\varepsilon(u)$ evaluation for strictly valid momentum transport. | `viscous_operators.jl` | `ASGS`, `legacy-branch` |
| 2026-04-13 | Reversing the adjoint coefficient sign in stabilization evaluation | Flips the mathematical properties natively switching the continuous VMS bounds from bounded diffusion to exponentially anti-diffusive state explosions. | Adjoint symmetry explicitly evaluated across standard `c_1` and `c_2` mapped bounds dynamically in code variables. | `reaction.jl`, `tau.jl` | `adjoint-sign`, `Jacobian` |
| 2026-04-13 | Artificially weakening MMS iterative plateaus to circumvent poor solver iterations | Falsely accepts initial noisy residuals as analytically sound "convergence," masking underlying physical formulation mapping regressions. | Mandate strict multi-pass sequence evaluation $r_{\text{max}} \le \tau_{\text{error}}$ across iterative mesh refinements. | `ManufacturedSolutions` | `MMS`, `verification` |
| 2026-04-13 | Bounding Newton limits blindly using scalar dimensionless values | Initializing ALM bounds unconditionally truncates natural quadratic steps. | Implemented strict Trust-Region evaluations mapping element-wise scalar checks validating step reductions before committing. | `nonlinear.jl` | `Newton`, `stopping-criteria` |
