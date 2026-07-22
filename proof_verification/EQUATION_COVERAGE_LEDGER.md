# Equation-by-equation proof-verification coverage ledger

_Generated 2026-07-21 from an automated coverage audit (15 region agents over all 4 `.tex` files, 369 displayed equations, plus an adversarial cross-check that re-derived every uncovered checkable equation in SymPy). **Zero algebra errors were found** beyond the corrected `eq:DominantPressureGradientXTermEstimate` (M03). Coverage tags: **COQ** (abstract a-priori chain), **SYMPY** (concrete-algebra check), **MMS** (numerical convergence tables), **DEFINITIONAL** (notation/definition, no algebra to check), **PARTIAL** (inputs/limits checked, display not reconstructed), **UNCHECKED**→ verified-correct gap._

> **STATUS 2026-07-21 — gaps closed.** Of the 23 `CONFIRMED_UNCHECKED` equations the cross-check confirmed
> correct-but-unverified, **21 now carry a permanent SymPy check** (five new `sympy/coverage_*_verification.py`
> modules). The **2 remaining** — `eq:ExplicitExactSubscales` and
> `eq:NonlinearStabilizedEquation` — were re-derived and confirmed correct but need the full discrete residual
> model for a self-contained check, so they stay documented-PARTIAL (see `sympy/README.md`). The rows below
> that show `UNCHECKED`/`PARTIAL` for those 21 reflect the *pre-closure* audit snapshot; consult the
> `coverage_*` modules for their now-encoded checks._
>
> **STATUS 2026-07-22 — appendix display coverage added.** The 2026-07-22 audit of the harmonized `article_v2`
> exposed that the body↔appendix display-consistency net stopped at §4/App. A. Two scripts now extend it to the
> appendices: `sympy/osgs_display_consistency_verification.py` (App. D `oa:cor:Ltwo` L²-velocity display, the
> `(1+c₁/Da_h)` factor) and `sympy/continuity_grouping_verification.py` (App. C `eq:groupstep` `T₁₃/N`
> term-accounting; the `|T₁₃|+|N|→|T₁₃^c|+|N|` fix). Suite now **253/253 across 19 scripts**. See
> `verification-gap-coverage.md` (addendum 2026-07-22) and `docs/ChatGPT audit/latest_audit_response.md`._

## Summary

| Coverage | Count |
|---|---|
| SYMPY | 126 |
| DEFINITIONAL | 101 |
| COQ | 64 |
| PARTIAL | 42 |
| UNCHECKED | 26 |
| MMS | 10 |
| **Total** | **369** |

## A1-model  (10 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:StrongMomentumEquation | 251 | DEFINITION | SYMPY | cdr_operator_verification.py check("momentum rows of L_w U match -2div(..)+alpha w.grad u+alpha grad p+sigma u |
| eq:StrongMassEquation | 252 | DEFINITION | SYMPY | cdr_operator_verification.py check("mass row of L_w U equals eps p + div(alpha u) = eps p + alpha div u + u.gr |
| unnumbered@258 | 258 | IDENTITY | PARTIAL | [CONFIRMED_UNCHECKED] Re-derived from first principles in sympy (/tmp/sympy_venv/bin/python, verify258 |
| eq:DBFResistanceTerm | 266 | DEFINITION | DEFINITIONAL |  |
| unnumbered@275 | 275 | NOTATION | DEFINITIONAL |  |
| eq:CDR_equation | 280 | DEFINITION | DEFINITIONAL |  |
| eq:generic_differential_equation | 292 | DEFINITION | DEFINITIONAL |  |
| eq:generic_boundary_conditions | 293 | BC | DEFINITIONAL |  |
| unnumbered@300 | 300 | NOTATION | DEFINITIONAL |  |
| eq:matrices_stationary_strong_problem | 311 | IDENTITY | SYMPY | cdr_operator_verification.py: check("K_ij reproduces -2 div(alpha nu Pi grad u) for all 3 components (varying  |

## A2-weakform  (12 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:generic_differential_equation | 292 | DEFINITION | DEFINITIONAL |  |
| eq:generic_boundary_conditions | 293 | BC | DEFINITIONAL |  |
| unnumbered@298 | 298 | DEFINITION | DEFINITIONAL |  |
| eq:matrices_stationary_strong_problem | 311 | IDENTITY | SYMPY | proof_verification/sympy/cdr_operator_verification.py: check('K_ij reproduces -2 div(alpha nu Pi grad u) for a |
| eq:TraceOperator | 346 | DEFINITION | DEFINITIONAL |  |
| eq:dirichlet_bc | 359 | BC | DEFINITIONAL |  |
| eq:neumann_bc | 364 | BC | SYMPY | proof_verification/sympy/cdr_operator_verification.py: check('n_i K_ij d_j u == 2 alpha nu (Pi grad u).n   (al |
| eq:RHS_neumann | 371 | BC | DEFINITIONAL |  |
| eq:WeakForm | 462 | DEFINITION | DEFINITIONAL |  |
| eq:BilinearForm | 470 | IDENTITY | PARTIAL | [CONFIRMED_UNCHECKED] Independently re-derived eq:BilinearForm (article.tex:470-473) from the strong o |
| eq:RHSLinearForm | 477 | DEFINITION | DEFINITIONAL |  |
| eq:WeakFormCompact | 485 | DEFINITION | DEFINITIONAL |  |

## A3-vms  (30 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:RHSLinearForm | 478 | DEFINITION | DEFINITIONAL |  |
| eq:WeakFormCompact | 485 | DEFINITION | DEFINITIONAL |  |
| unnumbered@494 | 494 | DEFINITION | DEFINITIONAL |  |
| eq:WeakFormResolvedScales | 499 | DEFINITION | DEFINITIONAL |  |
| eq:WeakFormSubscales | 500 | DEFINITION | DEFINITIONAL |  |
| eq:weak_form_resolved | 505 | DEFINITION | DEFINITIONAL |  |
| eq:weak_form_subscales | 506 | DEFINITION | DEFINITIONAL |  |
| eq:weak_form_eliminated_subscales | 512 | IDENTITY | SYMPY | display_consistency_verification.py: check("eliminated form with PLUS is exact: A U_h + <L Ltilde^{-1} R U_h>  |
| eq:ExplicitExactSubscales | 516 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Independently re-derived the Hughes fine-scale Green operator from the VMS subsc |
| eq:simplified_weak_form_resolved | 524 | IDENTITY | PARTIAL | [NOT_AN_ERROR] The equation is B(u,U_h,V_h) + B(u,Ũ,V_h) = L(V_h) (eq:weak_form_resolved) with  |
| eq:AdjointFlux | 530 | IDENTITY | SYMPY | cdr_operator_verification.py: check("int_box (V.L_w U - (L*V).U) == int_bdry (U.D*_N V - V.D_N U)   (D*_N eq:A |
| eq:AdjointDifferentialOperator | 531 | IDENTITY | SYMPY | cdr_operator_verification.py: Ladj defined line 160 exactly matching the printed L*, and check("int_cell ( V.L |
| eq:weak_form_subscales_with_assumption | 545 | DEFINITION | DEFINITIONAL |  |
| eq:weak_form_subscales_as_projection | 549 | DEFINITION | DEFINITIONAL |  |
| eq:simplified_weak_form_subscales | 557 | DEFINITION | DEFINITIONAL |  |
| eq:VMSWeakFormSystem | 566 | IDENTITY | PARTIAL | [NOT_AN_ERROR] Re-derived eq:VMSWeakFormSystem (article.tex line 566) from upstream equations.  |
| eq:OSGSProjection | 576 | DEFINITION | DEFINITIONAL |  |
| eq:TauInnerProduct | 580 | DEFINITION | DEFINITIONAL |  |
| eq:L2InnerProduct | 584 | DEFINITION | DEFINITIONAL |  |
| eq:NonlinearStabilizedEquation | 597 | IDENTITY | PARTIAL | [CONFIRMED_UNCHECKED] Re-derived from first principles and verified with sympy. eq:NonlinearStabilized |
| eq:ElementwiseSubscaleEquation | 598 | DEFINITION | DEFINITIONAL |  |
| eq:NonlinearResidualProjection | 599 | DEFINITION | DEFINITIONAL |  |
| eq:DefinitionOfNonlinearParameter | 600 | DEFINITION | DEFINITIONAL |  |
| unnumbered@606 | 606 | DEFINITION | PARTIAL | a3b.py: B_S with the minus sign reproduces the VMS system; the '+' variant fails. |
| unnumbered@608 | 608 | DEFINITION | PARTIAL | a3b.py: with this L_S, B_S - L_S equals B - L + <L*V,tau Pitilde R>. |
| eq:AlgebraicNonlinearSystem | 617 | DEFINITION | DEFINITIONAL |  |
| eq:DiscretizedNonlinearStabilizedEquat | 627 | DEFINITION | DEFINITIONAL |  |
| eq:DiscretizedElementwiseSubscaleEquat | 628 | DEFINITION | DEFINITIONAL |  |
| eq:DiscretizedNonlinearResidualProject | 629 | DEFINITION | DEFINITIONAL |  |
| eq:DiscretizedDefinitionOfNonlinearPar | 630 | DEFINITION | DEFINITIONAL |  |

## A4-fourier-tau  (39 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| unnumbered@702 | 702 | DEFINITION | DEFINITIONAL |  |
| unnumbered@706 | 706 | NOTATION | DEFINITIONAL |  |
| unnumbered@716 | 716 | DEFINITION | DEFINITIONAL |  |
| eq:BoundProjectionOfLBySubscales | 722 | ESTIMATE | SYMPY | subscale_norm_verification.py: check("n={n}: \|Lhat u\|^2_Lambda <= \|Lhat\|^2_Lambda \|u\|^2_(Lambda^-1)") an |
| eq:DesignConditionOnTau | 728 | OTHER | DEFINITIONAL |  |
| eq:DifferentialOperatorExpressionForFo | 734 | DEFINITION | DEFINITIONAL |  |
| eq:DefinitionDifferentialOperatorDecom | 738 | DEFINITION | DEFINITIONAL |  |
| unnumbered@746 | 746 | DEFINITION | DEFINITIONAL |  |
| eq:DesignConditionOnTauWeak | 772 | DEFINITION | DEFINITIONAL |  |
| eq:FTOfDifferentialOperator | 776 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from first principles using upstream eq:DesignConditionOnTauWeak (tau |
| unnumbered@782 | 782 | IDENTITY | COQ | TauDesign.v Theorem viscous_symbol_closed_form (Mvisc = k_i k_j K_ij closed form); also fourier_tau_verificati |
| unnumbered@783 | 783 | IDENTITY | COQ | TauDesign.v Theorem convective_symbol_scalar; fourier_tau_verification.py check("Lhat_c == i (a/h)(w.k0) I_3 o |
| unnumbered@784 | 784 | IDENTITY | COQ | TauDesign.v Theorems coupling_eigen_plus/minus, coupling_spectrum, coupling_spectral_radius; fourier_tau_verif |
| unnumbered@785 | 785 | IDENTITY | DEFINITIONAL |  |
| unnumbered@786 | 786 | IDENTITY | DEFINITIONAL |  |
| eq:StabilizationParameters | 791 | IDENTITY | COQ | TauDesign.v: parallel_factor_d3/d2 & viscous_parallel_eigenpair (viscous), convective_spectral_radius (conv),  |
| unnumbered@802 | 802 | DEFINITION | COQ | TauDesign.v Definition lam_choice and Theorem sqrt_lam_choice (sqrt lam_choice = h/(k0 tau1NS)); fourier_tau_v |
| eq:Tau1 | 812 | IDENTITY | COQ | TauDesign.v Theorem tau1_inv_assembled; fourier_tau_verification.py check("tau_1^{-1} == C_alpha tau_{1,NS}^{- |
| eq:Tau2 | 813 | IDENTITY | COQ | TauDesign.v Theorem tau2_assembled; fourier_tau_verification.py check("tau_2 == h^2 / (c1 a tau_{1,NS} + eps h |
| eq:CAlpha | 817 | IDENTITY | COQ | TauDesign.v Definition C_alpha + Theorem tau1_inv_assembled; fourier_tau_verification.py C_alpha = alpha + (h/ |
| eq:TauNavierStokes | 818 | DEFINITION | DEFINITIONAL |  |
| eq:SmallPorosityGradient | 823 | ESTIMATE | UNCHECKED | [NOT_AN_ERROR] Re-derived from eq:CAlpha (C_alpha = alpha + (h/\|k0\|)\|grad alpha\|) and the mesh- |
| eq:Tau1Final | 835 | IDENTITY | SYMPY | stability_estimate_verification.py line 42 tau1 = 1/(alphaK*tau1NS_inv + sigma); StabilityAlgebra.v Definition |
| eq:Tau2Final | 836 | IDENTITY | SYMPY | stability_estimate_verification.py line 43 tau2 = h**2/(c1*alphaK/tau1NS_inv); StabilityAlgebra.v Definition t |
| unnumbered@846 | 846 | DEFINITION | DEFINITIONAL |  |
| unnumbered@850 | 850 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived the Galerkin coercivity identity B(a,U_h,U_h)=2nu\|\|alpha^{1/2}Pi grad |
| eq:StabilityEstimate | 860 | IDENTITY | UNCHECKED | [ALREADY_COVERED] eq:StabilityEstimate (article.tex line 858-860) is algebraically CORRECT and is  |
| eq:BoundOfCrossedTerm | 865 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Independently reconstructed the full chain of article.tex:865 (eq:BoundOfCrossed |
| unnumbered@873 | 873 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived the align block at article.tex lines 873-879 from first principles wi |
| eq:InverseEstimateFiniteOrderNorm | 882 | ESTIMATE | DEFINITIONAL |  |
| eq:StabilityEstimateFinal | 889 | ESTIMATE | SYMPY | stability_estimate_verification.py: visc_final = nu*tau1*(2/tau1 - 4*Cinv**2*alphaK*nu/h**2 - 4*sigma/xi) and  |
| eq:UpperBoundOnEpsilon | 896 | ESTIMATE | COQ | StabilityAlgebra.v Definition eps_max = C2*c1*alphaK^2*tau1/h^2 and Theorem eps_tau2_le_C2 (eps<=eps_max -> ep |
| eq:PressureTermStabilityBound | 902 | ESTIMATE | COQ | StabilityAlgebra.v Theorem pressure_term_coercive; stability_estimate_verification.py check("eps(1 - eps tau2) |
| eq:ViscousCoefficientBound | 906 | ESTIMATE | COQ | StabilityAlgebra.v Theorem viscous_coefficient_expansion (visc_final=visc_847) and viscous_coefficient_lower_b |
| unnumbered@910 | 910 | DEFINITION | COQ | StabilityAlgebra.v Definition C_visc := Rmin (2 - 4*Cinv^2/c1) (2*(1 - 2/xi)) and Theorem viscous_coefficient_ |
| eq:VelocityCoefficientBound | 914 | ESTIMATE | COQ | StabilityAlgebra.v Theorem velocity_coefficient_expansion (u_final=u_855) and velocity_slack_identity; stabili |
| eq:SigmaAlpha | 918 | DEFINITION | COQ | StabilityAlgebra.v Theorems sigt_form_phi, sigt_form_tau1, sigt_form_key (three equivalent forms); stability_e |
| unnumbered@922 | 922 | DEFINITION | COQ | StabilityAlgebra.v Definition C_u := 1 - xi*Cinv^2/c1 and velocity_slack_identity |
| eq:conditions_on_num_param | 927 | ESTIMATE | COQ | StabilityAlgebra.v Theorem stability_constants_positive_sharp and C_stab_margin / elemental_coercivity (positi |

## A5-theorems  (12 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:PressureTermStabilityBound | 902 | ESTIMATE | COQ | StabilityAlgebra.v:425 Theorem pressure_term_coercive (proves eps*(1-eps*tau2) >= (1-C2)*eps under eps<=eps_ma |
| eq:ViscousCoefficientBound | 906 | IDENTITY | COQ | StabilityAlgebra.v:212 Theorem viscous_coefficient_expansion (visc_final = visc_847) and StabilityAlgebra.v:27 |
| unnumbered@910 | 910 | VALUE | COQ | StabilityAlgebra.v:276 Definition C_visc := Rmin (2 - 4*Cinv^2/c1) (2*(1-2/xi)); used in viscous_coefficient_l |
| eq:VelocityCoefficientBound | 914 | IDENTITY | COQ | StabilityAlgebra.v:233 Theorem velocity_coefficient_expansion (u_final = u_855) and StabilityAlgebra.v:242 The |
| eq:SigmaAlpha | 918 | DEFINITION | DEFINITIONAL |  |
| unnumbered@922 | 922 | VALUE | COQ | StabilityAlgebra.v:231 Definition C_u := 1 - xi * Cinv^2 / c1; consumed by velocity_coefficient/velocity_slack |
| eq:conditions_on_num_param | 927 | ESTIMATE | COQ | StabilityAlgebra.v:321 Theorem stability_constants_positive_sharp (proves C_visc>0 ∧ C_u>0 under the SHARP c1> |
| lemma:Stability@940 | 940 | ESTIMATE | COQ | AbstractStability.v:577 Theorem abstract_stability : 0 < C_stab c1 Cb xi C2 /\ BS >= C_stab c1 Cb xi C2 * Norm |
| unnumbered@945 | 945 | NOTATION | DEFINITIONAL |  |
| lemma:Continuity@957 | 957 | ESTIMATE | COQ | AbstractContinuity.v:2493 Theorem abstract_continuity : BS <= Ctot * ((BrU + BrP) * NV); sharp variant abstrac |
| eq:ConvergenceResult | 967 | ESTIMATE | COQ | AbstractConvergence.v:952 Theorem abstract_convergence : NErr <= Cconv * Psi, with Psi (AbstractConvergence.v: |
| eq:InterpolationError | 972 | DEFINITION | DEFINITIONAL |  |

## A6-robustness  (26 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:DimensionlessMomentumEquation | 983 | IDENTITY | SYMPY | robustness_asymptotics_verification.py checks 'convection coefficient -> Re', 'viscous coefficient -> 1 (the 2 |
| eq:DimensionlessParameters | 986 | DEFINITION | SYMPY | robustness_asymptotics_verification.py lines 88-89 encode Re = U*L/nu and Da = sigma_dim*L**2/(alpha_inf*nu) a |
| eq:PressureScale | 992 | DEFINITION | SYMPY | robustness_asymptotics_verification.py line 90 'P = (1 + Re + Da)*U*nu/L  # eq:PressureScale' used in check 'p |
| eq:GeneralAsymptoticBehaviourOfParamet | 999 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'tau1 = h^2/(alpha_K nu (1+Re_h+Da_h))' (exact, c1=c2=1). |
| eq:GeneralAsymptoticBehaviourOfParamet | 1000 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'tau2 = (1+Re_h) nu/alpha_K'. |
| eq:GeneralAsymptoticBehaviourOfParamet | 1001 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'sigtilde = alpha_K (1+Re_h)Da_h/(1+Re_h+Da_h) * nu/h^2   [A6.1:  |
| unnumbered@1009 (dom-viscosity tau1) | 1009 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. viscosity: tau1 -> h^2/(alpha_K nu),  tau2 -> nu/alpha_K' ( |
| unnumbered@1010 (dom-viscosity tau2) | 1010 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. viscosity: tau1 -> h^2/(alpha_K nu),  tau2 -> nu/alpha_K' ( |
| unnumbered@1011 (dom-viscosity sigtild | 1011 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. viscosity: sigtilde ~ alpha_K Da_h nu/h^2   [A6.2 correctio |
| eq:ConvergenceResultDominantViscosity | 1015 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Independently re-derived eq 1015 from first principles in sympy (scratchpad/veri |
| eq:DominantViscosityVelocityGradientEs | 1023 | ESTIMATE | SYMPY | robustness_isolation_verification.py check 'eq 1023  factor = a0^{-1/2}(1 + P h/(U nu))'. |
| eq:DominantViscosityPressureGradientEs | 1027 | ESTIMATE | SYMPY | robustness_isolation_verification.py check 'eq 1027  factor = a0^{-1/2}(U nu/(P h) + 1)   [CORRECTED display]' |
| unnumbered@1033 (dom-convection tau1) | 1033 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. convection: tau1 ~ h/(alpha_K \|a\|)  and  tau2 ~ h\|a\|/al |
| unnumbered@1034 (dom-convection tau2) | 1034 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. convection: tau1 ~ h/(alpha_K \|a\|)  and  tau2 ~ h\|a\|/al |
| unnumbered@1035 (dom-convection sigtil | 1035 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. convection: sigtilde ~ alpha_K Da_h nu/h^2   [A6.3 correcti |
| eq:ConvergenceResultDominantConvection | 1039 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:ConvergenceResultDominantConvection (article.tex:1039) from the pa |
| eq:DominantConvectionEstimate | 1043 | ESTIMATE | SYMPY | robustness_isolation_verification.py check 'eq 1043  factor = a0^{-1/2}(U + P/\|\|a\|\|)   (coupled, in E*_int |
| eq:DominantConvectionXTermEstimate | 1048 | ESTIMATE | SYMPY | robustness_isolation_verification.py check 'eq 1048  pre-sub factor = a0^{-1/2}(1 + P/(U\|\|a\|\|)); with P~U^ |
| eq:DominantPressureGradientXTermEstima | 1052 | ESTIMATE | SYMPY | robustness_isolation_verification.py check 'eq 1052  factor = a0^{-1/2}(\|\|a\|\| U/P + 1)   [CORRECTED; follo |
| unnumbered@1057 (dom-reaction tau1) | 1057 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. reaction: tau1 ~ 1/sigma   [A6.4 correction: NOT alpha/sigm |
| unnumbered@1058 (dom-reaction tau2) | 1058 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. reaction: tau2 ~ (1+Re_h) nu/alpha_K  and  sigtilde ~ alpha |
| unnumbered@1059 (dom-reaction sigtilde | 1059 | IDENTITY | SYMPY | robustness_asymptotics_verification.py check 'dom. reaction: tau2 ~ (1+Re_h) nu/alpha_K  and  sigtilde ~ alpha |
| unnumbered@1063 (coupled dom-reaction  | 1063 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Independently re-derived the coupled dominant-reaction bound (article.tex 1064-1 |
| eq:DominantReactionVelocityGradientEst | 1068 | ESTIMATE | SYMPY | robustness_isolation_verification.py checks 'eq 1068  factor = a0^{-1/2}((1+Re_h)^{1/2} + (1/(sig^{1/2}nu^{1/2 |
| eq:DominantReactionPressureGradientEst | 1075 | ESTIMATE | SYMPY | robustness_isolation_verification.py checks 'eq 1075  factor = a0^{-1}( sig^{1/2} nu^{1/2}(1+Re_h)^{1/2} U/P + |
| eq:ManufacturedProblem | 1094 | DEFINITION | SYMPY | manufactured_solution_verification.py verifies div(alpha u)=0 for this field (2D and 3D extruded) per README ( |

## A7-num2d3d  (13 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:ManufacturedProblem | 1094 | DEFINITION | COQ | Coq Theorem manufactured_mass_conservation_2d and manufactured_mass_conservation_3d in proof_verification/coq- |
| eq:PlateauBumpFunction | 1101 | DEFINITION | COQ | Coq PlateauBump.v Theorems am_range (a0<am e<1) and am'_pos (0<am' on (0,1)); sympy checks 'limit alpha(eta->0 |
| eq:Gamma | 1111 | DEFINITION | COQ | Coq PlateauBump.v Theorem gam_deriv (gamma') and gam'_pos (0<gam' on (0,1)); sympy check 'd gamma/d eta == (2  |
| unnumbered@1116 | 1116 | NOTATION | DEFINITIONAL |  |
| unnumbered@1132 | 1132 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from upstream eq:DimensionlessParameters (line 986-988): Re=UL/nu, Da |
| tab:Linear2DL2 | 1225 | OTHER | MMS | Numerical convergence-rate table; exercised by the MMS harness test/extended/ManufacturedSolutions (README not |
| tab:Linear2DH1 | 1291 | OTHER | MMS | Numerical convergence-rate table; MMS harness test/extended/ManufacturedSolutions (sympy README 'out of scope  |
| tab:Quadratic2DL2 | 1357 | OTHER | MMS | Numerical convergence-rate table; MMS harness test/extended/ManufacturedSolutions. |
| tab:Quadratic2DH1 | 1423 | OTHER | MMS | Numerical convergence-rate table; MMS harness test/extended/ManufacturedSolutions. |
| footnote@1429 | 1429 | VALUE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from first principles using the proven Proposition (elementwise coerc |
| eq:EpsilonRef | 1435 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Independent first-principles re-derivation confirms the printed form is algebrai |
| tab:3DL2 | 1489 | OTHER | MMS | Numerical convergence-rate table; MMS harness test/extended/ManufacturedSolutions (3D sibling); run at c1=16k^ |
| tab:3DH1 | 1539 | OTHER | MMS | Numerical convergence-rate table; MMS harness test/extended/ManufacturedSolutions (3D sibling). |

## A8-dbf-concl  (8 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| tab:3DL2 | 1489 | ESTIMATE | MMS | 3D MMS convergence tables produced by test/extended/ManufacturedSolutions3D harness (results/k*/TET/structured |
| tab:3DH1 | 1539 | ESTIMATE | MMS | 3D MMS harness convergence output (test/extended/ManufacturedSolutions3D); no sympy/coq reconstruction of the  |
| eq:CocquetMMSReaction | 1546 | DEFINITION | SYMPY | check("a(alpha), b(alpha) >= 0 for alpha in (0,1)  (=> sigma = a + b\|u\| is PSD)", ...) at manufactured_solut |
| inline@1550 | 1550 | VALUE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from first principles with sympy. From sigma = a(alpha)+b(alpha)\|u\| w |
| tab:CocquetMMSL2 | 1596 | ESTIMATE | MMS | test/extended/CocquetFormMMS harness (config-driven Cocquet MMS); convergence-rate/FME numerics only, no algeb |
| tab:CocquetMMSH1 | 1641 | ESTIMATE | MMS | test/extended/CocquetFormMMS harness convergence output; numeric slopes/FME only, no sympy/coq reconstruction. |
| inline@1644 | 1644 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from first principles using the paper's own L2 pressure FME table (ta |
| inline@1659 | 1659 | VALUE | UNCHECKED | [NOT_AN_ERROR] Re-derived from first principles with sympy (/tmp/sympy_venv/bin/python). The co |

## CA1-setup  (26 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:strongop | 24 | DEFINITION | DEFINITIONAL |  |
| eq:XG | 38 | DEFINITION | DEFINITIONAL |  |
| eq:brokennotation | 61 | NOTATION | DEFINITIONAL |  |
| eq:Bstab | 74 | IDENTITY | PARTIAL | [NOT_AN_ERROR] Re-derived eq:Bstab (continuity_appendix.tex:74) from first principles. Upstream |
| eq:consistency | 94 | OTHER | DEFINITIONAL | Encoded as hypothesis Horth in AbstractConvergence.v ('Galerkin orthogonality (eq:consistency)', lines 14/397) |
| eq:taus | 100 | DEFINITION | DEFINITIONAL | Main-article eq:Tau1/eq:Tau2 forms recovered in fourier_tau_verification.py [6]; here it is a restatement/def  |
| eq:phi1 | 113 | DEFINITION | DEFINITIONAL | phi1 = alphaK*tau1NS_inv defined and used in stability_estimate_verification.py line 44. |
| eq:sigmatilde | 118 | IDENTITY | SYMPY | stability_estimate_verification.py: check("sigtilde = sigma phi1/(phi1+sigma)") L51, check("sigtilde = sigma - |
| eq:triplenorm | 129 | DEFINITION | DEFINITIONAL |  |
| eq:epscond | 150 | BC | DEFINITIONAL | Concrete instance checked in manufactured_solution_verification.py L121 (eps=1e-4 eps_ref <= C2 c1 alpha_K^2 t |
| eq:resolved | 160 | DEFINITION | DEFINITIONAL |  |
| eq:inverse | 188 | ESTIMATE | DEFINITIONAL | Transcribed as Coq hypothesis Cinv_pos and consumed via winv_est schema (AbstractStability.v L24/L228, Abstrac |
| eq:jumpcond | 202 | DEFINITION | DEFINITIONAL | Transcribed as Coq hypotheses cJ_pos/cJ_le_1/one_le_cJ' (AbstractConvergence.v L436, AbstractStability.v). |
| P:basic | 223 | ESTIMATE | UNCHECKED | [ALREADY_COVERED] Equation P:basic (continuity_appendix.tex L224, label L223) asserts, with tau1=1 |
| P:components | 225 | ESTIMATE | UNCHECKED | [ALREADY_COVERED] Re-derived from eq:taus/eq:phi1 (continuity_appendix.tex L100-117): tauNSinv=c1  |
| P:27 | 229 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived P:27 (continuity_appendix.tex:229-233) from upstream eq:taus and eq:p |
| P:sigmatilde | 233 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived P:sigmatilde (continuity_appendix.tex line 233-235) from upstream def |
| P:eps | 236 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived P:eps (continuity_appendix.tex line 236-239) from first principles us |
| unnumbered@248 | 248 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived P:27 (continuity_appendix.tex L248-252) from upstream defs: eq:phi1 ( |
| unnumbered@258 | 258 | ESTIMATE | UNCHECKED | [ALREADY_COVERED] The equation is at continuity_appendix.tex lines 259-260 (the "undefined" path i |
| unnumbered@266 | 266 | ESTIMATE | PARTIAL | [ALREADY_COVERED] First-principles re-derivation of continuity_appendix.tex:266-268 from eq:taus/e |
| eq:winv-divvisc | 283 | ESTIMATE | PARTIAL | [CONFIRMED_UNCHECKED] Re-derived eq:winv-divvisc from first principles via its own proof (continuity_a |
| eq:winv-grad | 287 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:winv-grad (continuity_appendix.tex line 287) from first principles |
| eq:winv-divu | 289 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived eq:winv-divu (continuity_appendix.tex line 288-290) from first princi |
| eq:winv-conv | 293 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:winv-conv (continuity_appendix.tex:291-294) from first principles. |
| eq:winv-gradp | 297 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived eq:winv-gradp (continuity_appendix.tex:296) from first principles: \|\| |

## CA2-stability  (45 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:winv-divvisc | 283 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] The equation lives in continuity_appendix.tex:283 (label eq:winv-divvisc, C.14), |
| eq:winv-grad | 287 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:winv-grad (continuity_appendix.tex:284-287) from first principles  |
| eq:winv-divu | 290 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:winv-divu (continuity_appendix.tex:290) from first principles via  |
| eq:winv-conv | 294 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:winv-conv (continuity_appendix.tex:291-294) from first principles. |
| eq:winv-gradp | 297 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] eq:winv-gradp (continuity_appendix.tex:295-297) prints \|\|alpha grad r_h\|\|_K <= ( |
| unnumbered@310 | 310 | ESTIMATE | COQ | ContinuityAlgebra.v winv_ratio (L953: aInf/sqrt a0 <= sqrt(1+Ca) sqrt aInf) and delta_alpha_bound (L950: aInf  |
| unnumbered@321 | 321 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived the second-contribution bound at continuity_appendix.tex line 321 fro |
| eq:jumpest | 358 | ESTIMATE | COQ | ContinuityAlgebra.v jump_formula (L711), jump_bound_A (L791), sigt_id_tau (L158); coq_coverage.tex L623 maps l |
| unnumbered@367 | 367 | IDENTITY | COQ | ContinuityAlgebra.v jump_formula (L711) proves exactly this closed form for the tau1 jump |
| eq:continuity | 384 | ESTIMATE | COQ | AbstractContinuity.v abstract_continuity (L2493: BS <= Ctot*((BrU+BrP)*NV)); coq_coverage.tex L675 |
| eq:sharpcont | 393 | ESTIMATE | COQ | AbstractContinuity.v abstract_continuity_sharp (L2164); coq_coverage.tex L673 (Step 8 sharp assembly) |
| eq:T1 | 415 | DEFINITION | DEFINITIONAL |  |
| eq:T2 | 416 | DEFINITION | DEFINITIONAL |  |
| eq:T3 | 417 | DEFINITION | DEFINITIONAL |  |
| eq:T4 | 418 | DEFINITION | DEFINITIONAL |  |
| eq:T5 | 419 | DEFINITION | DEFINITIONAL |  |
| eq:T6 | 420 | DEFINITION | DEFINITIONAL |  |
| eq:T7 | 422 | DEFINITION | DEFINITIONAL |  |
| eq:T8 | 424 | DEFINITION | DEFINITIONAL |  |
| eq:T9 | 426 | DEFINITION | DEFINITIONAL |  |
| eq:T10 | 428 | DEFINITION | DEFINITIONAL |  |
| eq:T11 | 429 | DEFINITION | DEFINITIONAL |  |
| eq:T12 | 430 | DEFINITION | DEFINITIONAL |  |
| eq:T13 | 432 | DEFINITION | DEFINITIONAL |  |
| eq:T14 | 433 | DEFINITION | DEFINITIONAL |  |
| eq:T15 | 434 | DEFINITION | DEFINITIONAL |  |
| eq:T16 | 435 | DEFINITION | DEFINITIONAL |  |
| eq:T17 | 436 | DEFINITION | DEFINITIONAL |  |
| eq:T18 | 437 | DEFINITION | DEFINITIONAL |  |
| eq:easystep | 447 | ESTIMATE | COQ | AbstractContinuity.v bound_T1 (L967: <=2*(NU*NV)), bound_T10 (L1180), bound_T18 (L1199); coq_coverage.tex L667 |
| eq:reactivestep | 462 | IDENTITY | COQ | ContinuityAlgebra.v sigt_id_sub (L161: sigt = sigma - sigma^2*tau1); AbstractContinuity.v bound_T3_T14 (L995). |
| eq:penaltystep | 472 | ESTIMATE | COQ | AbstractContinuity.v bound_T5 (L1034: <=1*(NU*NV)), bound_T15 (L1059: <=C2*(NU*NV)); ContinuityAlgebra.v P5_ep |
| eq:keyvisc | 483 | ESTIMATE | COQ | ContinuityAlgebra.v keyvisc_sqrt (L484), keyvisc_sq (L473: nu tau1 alphaK c1 <= h^2); AbstractContinuity.v key |
| eq:T6bound | 496 | ESTIMATE | COQ | AbstractContinuity.v keyvisc_coef (L1221, used at L1329/L1381 for the T6/T7-type bounds); constant is the squa |
| eq:T7bound | 503 | ESTIMATE | COQ | AbstractContinuity.v keyvisc_coef (L1221, applied at L1329) provides the eq:keyvisc factor; triple-norm identi |
| eq:doubleinv | 514 | ESTIMATE | COQ | InverseEstimates.v winv_compose (L59: composes two winv_est into C1 C2 (W1 W2)/h^2); coq_coverage.tex L567 |
| eq:T8bound | 520 | ESTIMATE | COQ | ContinuityAlgebra.v T8_chain (L505: sigma*(c1 nu alphaK/h^2)*tau1 <= sigt), visc_le_phi1 (L218), sigt_id_tau ( |
| eq:viscstep | 535 | ESTIMATE | COQ | Aggregate of AbstractContinuity.v keyvisc_coef (T6,T7,T9) and ContinuityAlgebra.v T8_chain (T8,T12); assembled |
| eq:T13conv | 544 | ESTIMATE | COQ | ContinuityAlgebra.v T13_chain (L516: sigma*tau1*(c2 alphaK amag/h)<=sigt), conv_le_phi1 (L227), sigt_id_tau (L |
| eq:skew | 570 | IDENTITY | COQ | AbstractContinuity.v Hypothesis H_skew (L141, trusted diagonal Green identity) consumed by step6a (L1635); coq |
| eq:globalibp | 576 | IDENTITY | COQ | AbstractContinuity.v Hypotheses H_ibp_vp (L146), H_ibp_qu (L149) (trusted Green identities) consumed by step6a |
| eq:elemibp | 584 | IDENTITY | PARTIAL | [CONFIRMED_UNCHECKED] Re-derived eq:elemibp (continuity_appendix.tex:584) from first principles and co |
| unnumbered@591 | 591 | IDENTITY | COQ | AbstractContinuity.v step6a (L1635, rewrites using H_skew, H_ibp_vp, H_ibp_qu at L1672) |
| unnumbered@595 | 595 | IDENTITY | COQ | AbstractContinuity.v step6a (L1635) consumes the elemental IBP to produce this rewriting; coq_coverage.tex L66 |
| eq:fiveterms | 602 | IDENTITY | COQ | AbstractContinuity.v step6a (L1635: Rterm+T2+T4+T11 = Iterm+IIterm+IIIterm+IVterm+Vterm); coq_coverage.tex L66 |

## CA3-continuity  (26 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:T13conv | 553 | ESTIMATE | COQ | Theorem T13_chain (sigma*tau1*(c2*alphaK*amag/h) <= sigt) in proof_verification/coq-formal/ContinuityAlgebra.v |
| eq:skew | 570 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived from first principles with sympy (/tmp/skew_check.py). The upstream p |
| eq:globalibp | 576 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Independently re-derived both identities at continuity_appendix.tex:576 from fir |
| eq:elemibp | 584 | IDENTITY | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived eq:elemibp (continuity_appendix.tex:584) from first principles. The i |
| unnumbered@592 | 592 | IDENTITY | UNCHECKED | [ALREADY_COVERED] The equation T2+T4 = -(u,αX(V))-(p,∇·(αv)) is algebraically correct (my sympy sn |
| unnumbered@596 | 596 | IDENTITY | UNCHECKED | [ALREADY_COVERED] Re-derived from first principles: T11 (line 429) = -sigma(tau1 alpha X(U), v)_h  |
| eq:fiveterms | 610 | IDENTITY | UNCHECKED | [ALREADY_COVERED] Independent first-principles sympy re-derivation (representing each distinct bil |
| eq:IandIV | 621 | IDENTITY | PARTIAL | [NOT_AN_ERROR] Re-derived eq:IandIV from first principles with a concrete 1D two-element discre |
| eq:volpart | 641 | ESTIMATE | COQ | Theorem volpart_scalar (phi1*tau1 <= sqrt phi1 * sqrt tau1) + Theorem P3_sqrt (sqrt phi1 * h = sqrt c1 * alpha |
| eq:jumpsplit | 656 | ESTIMATE | COQ | Theorem jumpsplit_AB and Theorem jump_bound_A (sigma*\|tA-tB\| <= (CJ/cJ)(sgA*tA)) in ContinuityAlgebra.v Sect |
| eq:jumppart | 671 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:jumppart (continuity_appendix.tex:671, NOT a file 'undefined') fro |
| eq:jumptotal | 679 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:jumptotal (continuity_appendix.tex line 679) from upstream eq:jump |
| eq:IIVbound | 688 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived from upstream eqs with sympy (/tmp/check_iiv.py, all checks pass). eq |
| eq:IIVterm | 709 | ESTIMATE | COQ | Theorem sqrtphi_tau_le (sqrt phi1 * tau1 <= sqrt tau1) + P3_sqrt in ContinuityAlgebra.v |
| unnumbered@717 | 717 | IDENTITY | UNCHECKED | [ALREADY_COVERED] Re-derived the [III] reassembly from eq:fiveterms (continuity_appendix.tex:605-6 |
| eq:IIIface | 729 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] I re-derived eq:IIIface (continuity_appendix.tex:729) from first principles and  |
| eq:IIIbound | 737 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived from eq:IIIface. The facewise integrand bound C·Σ_{i,j∈{1,2}} (σ̃^{1/ |
| eq:groupstep | 745 | ESTIMATE | PARTIAL | [ALREADY_COVERED] Independently re-derived eq:groupstep (continuity_appendix.tex:745) as a triangl |
| eq:crosspenalty | 756 | ESTIMATE | COQ | Theorem step7_scalar (eps*sqrt tau2 <= sqrt C2 * sqrt eps) in ContinuityAlgebra.v Section EpsilonBound; uses P |
| eq:assembly | 767 | ESTIMATE | COQ | Theorem abstract_continuity (BS <= Ctot*((BrU+BrP)*NV)) in proof_verification/coq-formal/AbstractContinuity.v  |
| eq:absorb1 | 783 | ESTIMATE | COQ | Theorem absorb1_core (nu <= alphaK*tau2), absorb1_sqrt, and P3_sqrt in ContinuityAlgebra.v |
| eq:absorb2 | 786 | ESTIMATE | COQ | Theorem absorb2_sqrt (sqrt sigt <= sqrt phi1) and Corollary absorb2_full (sqrt sigt * h <= sqrt c1 * alphaK *  |
| eq:absorb3 | 789 | ESTIMATE | COQ | Theorem P5_sqrt (sqrt eps * h <= sqrt c1 * alphaK * sqrt tau1) in ContinuityAlgebra.v Section EpsilonBound |
| eq:absorb4 | 795 | ESTIMATE | COQ | Theorem absorb4_vel (sqrt tau1*(c2 alphaK amag/h) <= sqrt phi1) and Corollary absorb4_full + P3_sqrt in Contin |
| eq:absorb5 | 798 | ESTIMATE | UNCHECKED | [CONFIRMED_UNCHECKED] Re-derived eq:absorb5 (continuity_appendix.tex:797-799) from first principles. I |
| eq:normconv | 809 | ESTIMATE | COQ | Theorem norm5_absorption (five squared elemental pieces each <= k*A or k*B aggregate to sqrt(...)*(A+B)) in Co |

## CA4-convergence  (15 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:normconv | 809 | ESTIMATE | COQ | AbstractContinuity.v Lemma step9 (line 2399): NU <= Kabs*(BrU+BrP); Lemma absorb_elem (line 2215) with comment |
| eq:regularity | 844 | DEFINITION | DEFINITIONAL |  |
| eq:Eint | 858 | DEFINITION | DEFINITIONAL |  |
| eq:interp | 864 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] eq:interp (continuity_appendix.tex:864) prints the standard Lagrange interpolati |
| eq:interpinfty | 873 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived from first principles via affine scaling K̂→K: L∞ is scale-invariant  |
| eq:psih | 881 | DEFINITION | PARTIAL | Psi is Definition in AbstractConvergence.v (line 410) / AbstractInterpolation.KabsI framework; the weighted->u |
| eq:continterp | 896 | ESTIMATE | COQ | AbstractInterpolation.v Theorem abstract_continterp (line 2900) and Theorem abstract_continterp_sharp (line 25 |
| eq:interpdivvisc | 926 | ESTIMATE | PARTIAL | [CONFIRMED_UNCHECKED] Re-derived from first principles. Line 1 (Leibniz + triangle): ∇·(αΠ∇e_u)=α∇·(Π∇ |
| eq:interpgrad | 932 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived eq:interpgrad (continuity_appendix.tex line 932-935) from first princ |
| eq:interpfirst | 936 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] Re-derived both bounds of eq:interpfirst (continuity_appendix.tex lines 936-940) |
| eq:interpzero | 941 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] eq:interpzero (continuity_appendix.tex:941-946) asserts three bounds. Re-derivin |
| eq:interpinftyE | 947 | ESTIMATE | PARTIAL | [NOT_AN_ERROR] eq:interpinftyE (continuity_appendix.tex lines 947-950; the file the task called |
| eq:Enorm | 958 | ESTIMATE | COQ | AbstractInterpolation.v interp_triple_norm (line 2781: NU <= sqrt2*KaggI*(PsU+PsP)) for the first (squared-sum |
| eq:convergence | 991 | ESTIMATE | COQ | AbstractConvergence.v Theorem abstract_convergence (line 952: NErr <= Cconv*Psi), Cconv:=Kb+Ct/Cst (line 950); |
| unnumbered@1005 | 1005 | ESTIMATE | COQ | AbstractConvergence.v Lemma stab_W (line 573: BSWW >= Cst*NW^2, from prop:stability), Lemma cont_EW (line 593: |

## FA-fourier  (1 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:ftSplit | 18 | IDENTITY | PARTIAL | [CONFIRMED_UNCHECKED] eq:ftSplit (fourier_appendix.tex line 18-22) asserts A_{c,i}(w) + A_{f,i} = A_{v |

## EA1-matrices  (61 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:ViscousTerm | 7 | IDENTITY | SYMPY | cdr_operator_verification.py check 'K_ij reproduces -2 div(alpha nu Pi grad u)...' (PASS); I re-derived the de |
| eq:VelocityDerivativeTerm | 8 | IDENTITY | SYMPY | cdr_operator_verification.py check 'momentum rows of L_w U match ... alpha w.grad u ...' and 'mass row ... alp |
| eq:FluxTerm | 9 | IDENTITY | SYMPY | cdr_operator_verification.py A_f (pressure gradient alpha delta_{ia}) in the verified full operator; I re-ran  |
| eq:ReactionTerm | 10 | IDENTITY | SYMPY | cdr_operator_verification.py Smat (sigma velocity block, grad alpha in mass row, eps p) inside 'mass row of L_ |
| eq:StabilizationLVLU | 17 | DEFINITION | SYMPY | elemental_bilinear_form_verification.py builds L*_mom/L_mom/L*_mass/L_mass from scratch and every family check |
| eq:GalerkinForceTerm | 23 | DEFINITION | DEFINITIONAL |  |
| eq:NeumannBCsTerm | 24 | BC | DEFINITIONAL |  |
| eq:StabilizationLVF | 30 | DEFINITION | SYMPY | elemental_bilinear_form_verification.py checks 'F_V = A_F+L_F+C_F+G_betaF+D_betaF+R_sigmaF+D_phi+V_phi' and 'F |
| eq:DefinitionOfMassSourceTerm | 33 | DEFINITION | DEFINITIONAL |  |
| unnumbered@45 | 45 | NOTATION | DEFINITIONAL |  |
| unnumbered@59 | 59 | NOTATION | DEFINITIONAL | assembly_consistency_verification.py enforces every named block is defined and every defined block used (4/4 P |
| unnumbered@90 | 90 | NOTATION | DEFINITIONAL |  |
| eq:FEfunction | 106 | DEFINITION | DEFINITIONAL |  |
| eq:GenericMatrixVU | 112 | DEFINITION | DEFINITIONAL | elemental_matrices_verification.py implements exactly this Gateaux-derivative framework (Dvel) |
| eq:GSComponents | 118 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G_S = alpha nu (d_l N^a d_l N^b delta_ij + d_j N^a d_i N^b)' (PASS,  |
| eq:DDComponents | 124 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'D_nuD = -2/3 alpha nu d_i N^a d_j N^b' (PASS) |
| eq:VComponents | 128 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'V = alpha N^a a_l d_l N^b delta_ij (convection)' (PASS) |
| eq:ReactionTermComponents | 132 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'R_sigma = N^a N^b sigma_ij (reaction)' (PASS) |
| eq:AALHSStabilizationTerm | 139 | IDENTITY | SYMPY | elemental_matrices_verification.py 'A_A = ...' (PASS) and elemental_bilinear_form_verification.py family 'A_•  |
| eq:ALLHSStabilizationTerm | 141 | IDENTITY | SYMPY | elemental_matrices_verification.py 'A_L = ...' (PASS) and elemental_bilinear_form_verification.py family 'A_•' |
| eq:ACLHSStabilizationTerm | 142 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'A_• (A_A,A_L,A_C,A_Gbeta,A_Dbeta,A_sigma)' (PASS) contains A_C |
| eq:AGBetaLHSStabilizationTerm | 144 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'A_•' (PASS); transcribed A_Gb = -tau1 alpha nu aNa (d(Nb,i)d(a |
| eq:ADBetaLHSStabilizationTerm | 145 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'A_•' (PASS); A_Db = (2/3)tau1 alpha nu aNa d(alpha,i)d(Nb,j) |
| eq:ASigmaLHSStabilizationTerm | 146 | IDENTITY | SYMPY | elemental_matrices_verification.py 'A_sigma = tau1 alpha (a.grad N^a) N^b sigma_ij' (PASS) and family 'A_•' (P |
| eq:LALHSStabilizationTerm | 148 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_• (L_A,L_L,L_C,L_Gbeta,L_Dbeta,L_sigma)' (PASS); L_A transcr |
| eq:LLLHSStabilizationTerm | 151 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_•' (PASS); L_L (4-term) transcribed and family-verified |
| eq:LCLHSStabilizationTerm | 153 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_•' (PASS) |
| eq:LGBetaLHSStabilizationTerm | 155 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_•' (PASS); L_Gb transcribed matches |
| eq:LDBetaLHSStabilizationTerm | 157 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_•' (PASS) |
| eq:LSigmaLHSStabilizationTerm | 159 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'L_•' (PASS) |
| eq:CALHSStabilizationTerm | 160 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_• (C_A,C_L,C_C,C_Gbeta,C_Dbeta,C_sigma)' (PASS) |
| eq:CLLHSStabilizationTerm | 162 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_•' (PASS) |
| eq:CCLHSStabilizationTerm | 163 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_•' (PASS); C_C = -(4/9)tau1 a^2 nu^2 S(d^2_ki Na d^2_kj Nb) |
| eq:CGBetaLHSStabilizationTerm | 165 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_•' (PASS) |
| eq:CDBetaLHSStabilizationTerm | 166 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_•' (PASS) |
| eq:CSigmaLHSStabilizationTerm | 167 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'C_•' (PASS) |
| eq:GBetaALHSStabilizationTerm | 169 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_• (G_bA,G_bL,G_bC,G_bG,G_bD,G_bsigma)' (PASS) |
| eq:GBetaLLHSStabilizationTerm | 171 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_•' (PASS) |
| eq:GBetaCLHSStabilizationTerm | 173 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_•' (PASS) |
| eq:GBetaGLHSStabilizationTerm | 176 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_•' (PASS); Gb_G transcribed and family-verified |
| eq:GBetaDLHSStabilizationTerm | 178 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_•' (PASS) |
| eq:GBetaSigmaLHSStabilizationTerm | 180 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Gbeta_•' (PASS) |
| eq:DBetaALHSStabilizationTerm | 181 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_• (D_bA,D_bL,D_bC,D_bG,D_bD,D_bsigma)' (PASS) |
| eq:DBetaLLHSStabilizationTerm | 183 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_•' (PASS) |
| eq:DBetaCLHSStabilizationTerm | 184 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_•' (PASS) |
| eq:DBetaGLHSStabilizationTerm | 186 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_•' (PASS); Db_G = (4/3)tau1 nu^2 d(Na,i) S(d(alpha,l)d(N |
| eq:DBetaDLHSStabilizationTerm | 187 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_•' (PASS) |
| eq:DBetaSigmaLHSStabilizationTerm | 188 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'Dbeta_•' (PASS) |
| eq:RSigmaALHSStabilizationTerm | 189 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'R_• (R_sA,R_sL,R_sC,R_Gbeta,R_Dbeta,R_Rsigma)' (PASS) |
| eq:RSigmaLLHSStabilizationTerm | 191 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'R_•' (PASS); R_sL = tau1 alpha nu Na (S d^2_ll Nb sig_ij + S s |
| eq:RSigmaCLHSStabilizationTerm | 192 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'R_•' (PASS) |
| eq:RGBetaLHSStabilizationTerm | 194 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'R_•' (PASS); R_Gb transcribed |
| eq:RDBetaLHSStabilizationTerm | 195 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'R_•' (PASS) |
| eq:RRSigmaLHSStabilizationTerm | 196 | IDENTITY | SYMPY | elemental_matrices_verification.py 'R_Rsigma = -tau1 sigma_ik N^a sigma_kj N^b' (PASS) and family 'R_•' (PASS) |
| eq:DULHSStabilizationTerm | 197 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py check 'mass_• (G_U,G_D,D_U,D_D) [found+fixed an alpha typo in D_U]' (P |
| eq:DAlphaDLHSStabilizationTerm | 198 | IDENTITY | SYMPY | elemental_matrices_verification.py 'D_D = tau2 alpha^2 d_i N^a d_j N^b' (PASS) and family 'mass_• (G_U,G_D,D_U |
| eq:GULHSStabilizationTerm | 199 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'mass_• (G_U,G_D,D_U,D_D)' (PASS); G_U = tau2 Na Nb d(alpha,i)d |
| eq:GAlphaDLHSStabilizationTerm | 200 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py family 'mass_• (G_U,G_D,D_U,D_D)' (PASS); G_D = tau2 alpha Na d(alpha, |
| eq:GenericMatrixQU | 205 | DEFINITION | DEFINITIONAL |  |
| eq:QComponents | 209 | IDENTITY | SYMPY | elemental_matrices_verification.py 'Q_D = alpha N^a d_j N^b (alpha div u)' (PASS) |
| eq:GAlphaComponents | 213 | IDENTITY | SYMPY | elemental_matrices_verification.py 'G_alphaD = N^a N^b d_j alpha (u . grad alpha)' (PASS) |

## EA2-matrices  (45 equations)

| Label | Line | Kind | Coverage | Evidence / verdict |
|---|---|---|---|---|
| eq:GAlphaDLHSStabilizationTerm | 200 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py check 'mass_•  (G_U,G_D,D_U,D_D)   [found+fixed an alpha typo in D_U]' |
| eq:GenericMatrixQU | 205 | DEFINITION | DEFINITIONAL |  |
| eq:QComponents | 209 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'Q_D    = alpha N^a d_j N^b  (alpha div u)' (lines 102-104); RAN 19/1 |
| eq:GAlphaComponents | 213 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G_alphaD = N^a N^b d_j alpha  (u . grad alpha)' (lines 106-108); RAN |
| eq:GALHSStabilizationTerm | 218 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G_A  = tau1 alpha^2 d_j N^a (a.grad N^b)' (line 156) AND elemental_b |
| eq:GLLHSStabilizationTerm | 220 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py G_family G_L (line 197), check 'G_•  (G_A,G_L,G_C,G_Gbeta,G_Dbeta,G_R) |
| eq:GCLHSStabilizationTerm | 221 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py G_family G_C (line 198); check 'G_•  (...)' RAN PASS |
| eq:GGBetaLHSStabilizationTerm | 223 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py G_family G_Gb (line 199); check 'G_•  (...)' RAN PASS |
| eq:GDBetaLHSStabilizationTerm | 224 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py G_family G_Db (line 200); check 'G_•  (...)' RAN PASS |
| eq:GRLHSStabilizationTerm | 225 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G_R  = tau1 alpha d_k N^a sigma_kj N^b' (line 160) AND elemental_bil |
| eq:QepsilonDBetaLHSStabilizationTerm | 226 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py massQU Q_eD (line 210), check 'mass_•  (Q_epsD, Q_U)' RAN PASS |
| eq:QULHSStabilizationTerm | 227 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py massQU Q_U (line 211), check 'mass_•  (Q_epsD, Q_U)' RAN PASS |
| eq:GenericMatrixVP | 232 | DEFINITION | DEFINITIONAL |  |
| eq:PComponents | 237 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'P    = -alpha d_i N^a N^b  (pressure gradient)' (lines 115-117); RAN |
| eq:PComponents | 238 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G_P  = -d_i alpha N^a N^b' (lines 119-121; integrand -d_i alpha N^a  |
| eq:AGLHSStabilizationTerm | 242 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'A_G  = tau1 alpha^2 (a.grad N^a) d_i N^b' (line 170) AND elemental_b |
| eq:LGLHSStabilizationTerm | 244 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPfamily LG (line 227), check 'A_G,L_G,C_G,G_G,D_G,R_sigmaG = tau1 a^2 |
| eq:CGLHSStabilizationTerm | 245 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPfamily CG (line 228); check RAN PASS |
| eq:GGLHSStabilizationTerm | 247 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPfamily GG (line 229); check RAN PASS |
| eq:DGLHSStabilizationTerm | 248 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPfamily DG (line 230); check RAN PASS |
| eq:RGLHSStabilizationTerm | 249 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPfamily RsG (line 231); check RAN PASS |
| eq:DPLHSStabilizationTerm | 250 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPmass D_P (line 239), check 'mass_•  (D_P, G_P)   [found+fixed a typo |
| eq:GPLHSStabilizationTerm | 251 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py VPmass G_P (line 240, comment '[a N^a/alpha typo here was found and fi |
| eq:GenericMatrixQP | 256 | DEFINITION | DEFINITIONAL |  |
| eq:PQLHS | 262 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'P_Q  = eps N^a N^b  (compressibility)' (lines 123-124); RAN PASS |
| eq:QPLHSStabilizationTerm | 266 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'Q_P  = -tau2 eps^2 N^a N^b' (lines 178-179) AND elemental_bilinear_f |
| eq:GLHSStabilizationTerm | 270 | IDENTITY | SYMPY | elemental_matrices_verification.py check 'G    = tau1 alpha^2 d_k N^a d_k N^b  (pressure Laplacian, K_QP)' (li |
| eq:GenericVFTerm | 275 | DEFINITION | DEFINITIONAL |  |
| eq:GTerm | 280 | DEFINITION | DEFINITIONAL |  |
| eq:AFTerm | 286 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed A_F (line 272), check 'F_V = A_F+L_F+C_F+G_betaF+D_betaF+R_ |
| eq:LFTerm | 287 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed L_F (line 273); check 'F_V = ...' RAN PASS |
| eq:CFTerm | 288 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed C_F (line 274); check 'F_V = ...' RAN PASS |
| eq:GBetaFTerm | 289 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed G_bF (line 276, comment 'appendix prints ... f_k; bilinear  |
| eq:DBetaFTerm | 290 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed D_bF (line 277); check 'F_V = ...' RAN PASS |
| eq:RSigmaFTerm | 291 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed R_sF (line 278); check 'F_V = ...' RAN PASS |
| eq:DPhiTerm | 296 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed D_phi (line 279); check 'F_V = ...' RAN PASS |
| eq:VPhiTerm | 297 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FV_printed V_phi (line 280); check 'F_V = ...' RAN PASS |
| eq:GenericQFTerm | 302 | DEFINITION | DEFINITIONAL |  |
| eq:QAlphaFTerm | 307 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FQ_printed Q_aF (line 290), check 'F_Q = Q_alphaF + Q_phi  = tau1 alph |
| eq:QPhiTerm | 308 | IDENTITY | SYMPY | elemental_bilinear_form_verification.py FQ_printed Q_phi (line 291), check 'F_Q = Q_alphaF + Q_phi ... [Q_phi  |
| eq:VTerm | 312 | BC | DEFINITIONAL |  |
| unnumbered@318 | 318 | OTHER | SYMPY | assembly_consistency_verification.py checks 'no matrix is NAMED in the assembly without being DEFINED' and 'no |
| unnumbered@323 | 323 | OTHER | SYMPY | assembly_consistency_verification.py structural invariant (lines 217-220) — V_F and V_T both defined and prese |
| eq:GeneralStabilizationMatrices | 331 | OTHER | SYMPY | assembly_consistency_verification.py slice_general_assembly parses exactly this display; checks 'no matrix NAM |
| eq:FirstOrdertabilizationMatrices | 357 | OTHER | DEFINITIONAL | Explicitly EXCLUDED from assembly_consistency_verification.py (slice_general_assembly ends at 'If second-order |
