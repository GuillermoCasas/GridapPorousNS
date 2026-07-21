(* ========================================================================= *)
(*  Asymptotics.v                                                            *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the robustness   *)
(*  analysis of Section 6 of *A stabilized finite element method for         *)
(*  incompressible, inertial flows in inhomogeneous porous media*.           *)
(*                                                                           *)
(*  Mirrors theory/verification scripts/robustness_asymptotics_verification. *)
(*  With the exact parameters (equal-order constants c1 = c2 = 1),           *)
(*      |a|    = Re_h nu / h,          sigma = Da_h alpha_K nu / h^2,        *)
(*      tau1NS^{-1} = nu/h^2 + |a|/h = (nu/h^2)(1 + Re_h),                   *)
(*      tau1   = (alpha_K tau1NS^{-1} + sigma)^{-1}     (eq:Tau1Final)       *)
(*      tau2   = h^2 tau1NS^{-1} / alpha_K              (eq:Tau2Final)       *)
(*      sigt   = tau1NS^{-1} sigma/(tau1NS^{-1}+sigma/alpha_K) (eq:SigmaAlpha)*)
(*  it proves:                                                               *)
(*   (1) the general closed forms (eq:GeneralAsymptoticBehaviourOfParameters,*)
(*       amendment A6.1: the alpha_K factor on sigma-tilde);                 *)
(*   (2) every dominant-regime limit of Section 6, as genuine limits         *)
(*       (Re_h -> 0/oo, Da_h -> 0/oo), including the A6.2-A6.4 corrections;  *)
(*   (3) the nondimensionalization of eq:DimensionlessMomentumEquation with  *)
(*       the pressure scale eq:PressureScale, incl. the corrected forcing    *)
(*       scaling f = (alpha_inf nu U / L^2) f*  (amendment A7).              *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

From PNSFormal Require Import Limits.

(* ========================================================================= *)
(*  The exact parameters as functions of (Re_h, Da_h).                       *)
(* ========================================================================= *)

Section Robustness.

Variables (nu h alphaK : R).
Hypothesis nu_pos     : 0 < nu.
Hypothesis h_pos      : 0 < h.
Hypothesis alphaK_pos : 0 < alphaK.

(* |a| = Re_h nu/h and sigma = Da_h alpha_K nu/h^2 (elemental numbers). *)
Definition amag  (Re : R) : R := Re * nu / h.
Definition sigma (Da : R) : R := Da * alphaK * nu / h^2.

Definition tau1NS_inv (Re : R) : R := nu / h^2 + amag Re / h.
Definition tau1 (Re Da : R) : R := / (alphaK * tau1NS_inv Re + sigma Da).
Definition tau2 (Re : R) : R := h^2 * tau1NS_inv Re / alphaK.
Definition sigt (Re Da : R) : R :=
  tau1NS_inv Re * sigma Da / (tau1NS_inv Re + sigma Da / alphaK).

Lemma h2_pos : 0 < h^2.
Proof. apply pow_lt; exact h_pos. Qed.

(* ========================================================================= *)
(*  (1)  General closed forms (c1 = c2 = 1),                                 *)
(*       eq:GeneralAsymptoticBehaviourOfParameters.                          *)
(* ========================================================================= *)

Theorem tau1_general_form : forall Re Da, 0 <= Re -> 0 <= Da ->
  tau1 Re Da = h^2 / (alphaK * nu * (1 + Re + Da)).
Proof.
  intros Re Da HRe HDa.
  unfold tau1, tau1NS_inv, amag, sigma.
  assert (0 < alphaK * nu) by nra.
  assert (0 < alphaK * nu * (1 + Re + Da)) by nra.
  field. repeat split; nra.
Qed.

Theorem tau2_general_form : forall Re, 0 <= Re ->
  tau2 Re = (1 + Re) * nu / alphaK.
Proof.
  intros Re HRe.
  unfold tau2, tau1NS_inv, amag.
  field. repeat split; nra.
Qed.

(* Amendment A6.1: the alpha_K factor is present on sigma-tilde. *)
Theorem sigt_general_form : forall Re Da, 0 <= Re -> 0 < Da ->
  sigt Re Da = alphaK * ((1 + Re) * Da / (1 + Re + Da)) * (nu / h^2).
Proof.
  intros Re Da HRe HDa.
  unfold sigt, tau1NS_inv, amag, sigma.
  field. repeat split; nra.
Qed.

(* ========================================================================= *)
(*  (2)  Dominant-regime limits.                                             *)
(* ========================================================================= *)

(* ---- Dominant viscosity: Re_h, Da_h -> 0. ------------------------------- *)

(* Exact values at the limit point (the functions are rational and defined   *)
(* there):  tau1 -> h^2/(alpha_K nu),  tau2 -> nu/alpha_K. *)
Theorem dom_viscosity_tau1 : tau1 0 0 = h^2 / (alphaK * nu).
Proof.
  rewrite tau1_general_form by lra.
  field; repeat split; nra.
Qed.

Theorem dom_viscosity_tau2 : tau2 0 = nu / alphaK.
Proof. rewrite tau2_general_form by lra. field; nra. Qed.

(* Amendment A6.2:  sigt(0, Da) ~ alpha_K Da nu / h^2  as Da -> 0+,          *)
(* stated as: the ratio tends to 1. *)
Theorem dom_viscosity_sigt :
  tendsto_at_0plus
    (fun Da => sigt 0 Da / (alphaK * Da * (nu / h^2))) 1.
Proof.
  apply tendsto_at_0plus_ext_pos with (g := fun Da => 1 / (1 + Da)).
  - intros Da HDa.
    rewrite sigt_general_form by lra.
    field; repeat split; nra.
  - exact (lim_A_over_Apx_at0 1 Rlt_0_1).
Qed.

(* ---- Dominant convection: Re_h -> +oo (Da_h fixed). --------------------- *)

Section DominantConvection.
Variable Da : R.
Hypothesis Da_pos : 0 < Da.

(* tau1 ~ h/(alpha_K |a|):  tau1 * (alpha_K |a| / h)  ->  1. *)
Theorem dom_convection_tau1 :
  tendsto_at_top (fun Re => tau1 Re Da * (alphaK * amag Re / h)) 1.
Proof.
  apply tendsto_at_top_ext_pos with (g := fun Re => Re / ((1 + Da) + Re)).
  - intros Re HRe.
    rewrite tau1_general_form by lra.
    unfold amag. field. repeat split; nra.
  - exact (lim_x_over_Cpx (1 + Da) ltac:(lra)).
Qed.

(* tau2 ~ h |a| / alpha_K:  tau2 / (h |a| / alpha_K)  ->  1. *)
Theorem dom_convection_tau2 :
  tendsto_at_top (fun Re => tau2 Re / (h * amag Re / alphaK)) 1.
Proof.
  apply tendsto_at_top_ext_pos with (g := fun Re => (1 + Re) / Re).
  - intros Re HRe.
    rewrite tau2_general_form by lra.
    unfold amag. field. repeat split; nra.
  - exact (lim_Cpx_over_x 1 Rlt_0_1).
Qed.

(* Amendment A6.3:  sigt ~ alpha_K Da nu / h^2. *)
Theorem dom_convection_sigt :
  tendsto_at_top (fun Re => sigt Re Da / (alphaK * Da * (nu / h^2))) 1.
Proof.
  apply tendsto_at_top_ext_pos
    with (g := fun Re => (1 + Re) / ((1 + Da) + Re)).
  - intros Re HRe.
    rewrite sigt_general_form by lra.
    field. repeat split; nra.
  - exact (lim_Apx_over_Bpx 1 (1 + Da)).
Qed.

End DominantConvection.

(* ---- Dominant reaction: Da_h -> +oo (Re_h fixed). ----------------------- *)

Section DominantReaction.
Variable Re : R.
Hypothesis Re_nonneg : 0 <= Re.

(* Amendment A6.4:  tau1 ~ 1/sigma (NOT alpha_K/sigma):                      *)
(*   tau1 * sigma  ->  1  as Da -> +oo. *)
Theorem dom_reaction_tau1 :
  tendsto_at_top (fun Da => tau1 Re Da * sigma Da) 1.
Proof.
  apply tendsto_at_top_ext_pos with (g := fun Da => Da / ((1 + Re) + Da)).
  - intros Da HDa.
    rewrite tau1_general_form by lra.
    unfold sigma. field. repeat split; nra.
  - exact (lim_x_over_Cpx (1 + Re) ltac:(lra)).
Qed.

(* tau2 does not depend on Da_h at all: exact statement. *)
Theorem dom_reaction_tau2 : tau2 Re = (1 + Re) * nu / alphaK.
Proof. apply tau2_general_form; lra. Qed.

(* sigt ~ alpha_K (1 + Re) nu / h^2. *)
Theorem dom_reaction_sigt :
  tendsto_at_top (fun Da => sigt Re Da / (alphaK * (1 + Re) * (nu / h^2))) 1.
Proof.
  apply tendsto_at_top_ext_pos with (g := fun Da => Da / ((1 + Re) + Da)).
  - intros Da HDa.
    rewrite sigt_general_form by lra.
    field. repeat split; nra.
  - exact (lim_x_over_Cpx (1 + Re) ltac:(lra)).
Qed.

End DominantReaction.

End Robustness.

(* ========================================================================= *)
(*  (3)  Nondimensionalization (eq:DimensionlessMomentumEquation,            *)
(*       eq:PressureScale), including the corrected forcing scaling          *)
(*       f = (alpha_inf nu U / L^2) f*   (amendment A7).                     *)
(*  Multiplying the dimensional momentum equation by                         *)
(*       mult := L^2 / (alpha_inf nu U)                                      *)
(*  must turn each term's dimensional prefactor into its dimensionless       *)
(*  coefficient.                                                             *)
(* ========================================================================= *)

Section Nondimensionalization.

Variables (nu U L alpha_inf sigma_dim : R).
Hypothesis nu_pos    : 0 < nu.
Hypothesis U_pos     : 0 < U.
Hypothesis L_pos     : 0 < L.
Hypothesis ainf_pos  : 0 < alpha_inf.
Hypothesis sigd_pos  : 0 < sigma_dim.

Definition Re : R := U * L / nu.
Definition Da : R := sigma_dim * L^2 / (alpha_inf * nu).
Definition Pscale : R := (1 + Re + Da) * U * nu / L.       (* eq:PressureScale *)
Definition mult : R := L^2 / (alpha_inf * nu * U).

Theorem convection_coefficient : (alpha_inf * U^2 / L) * mult = Re.
Proof. unfold mult, Re. field. repeat split; nra. Qed.

Theorem viscous_coefficient : (alpha_inf * nu * U / L^2) * mult = 1.
Proof.
  unfold mult. pose proof (pow_lt L 2 L_pos).
  field. repeat split; nra.
Qed.

Theorem pressure_coefficient : (alpha_inf * Pscale / L) * mult = 1 + Re + Da.
Proof. unfold mult, Pscale. field. repeat split; nra. Qed.

Theorem reaction_coefficient : (sigma_dim * U) * mult = Da.
Proof. unfold mult, Da. field. repeat split; nra. Qed.

(* Amendment A7: with f = (alpha_inf nu U / L^2) f*, the forcing             *)
(* coefficient is exactly 1. *)
Theorem forcing_coefficient : (alpha_inf * nu * U / L^2) * mult = 1.
Proof. exact viscous_coefficient. Qed.

End Nondimensionalization.

(* ========================================================================= *)
(*  (4)  Per-term ISOLATION displays of Section 6.                           *)
(*                                                                           *)
(*  The tau-limit theorems above (parts 1-2) verify the INPUTS of the        *)
(*  robustness estimates (the tau closed forms and their regime limits) but  *)
(*  NOT the per-term isolation displays -- the steps that take a coupled     *)
(*  regime bound, isolate a single left-hand term, and re-express it through  *)
(*      E_int(u) = U E*_int,   E_int(p) = P E*_int   (equal order).           *)
(*  That is the exact blind spot in which the external audit (2026-07) found  *)
(*  an ALGEBRAIC ERROR in eq:DominantPressureGradientXTermEstimate (~l.1052): *)
(*      printed   ||a||_inf / sqrt(P) + 1     (WRONG, does not follow)        *)
(*      correct   ||a||_inf * U / P + 1       (follows from eq:1043).         *)
(*  Both reduce to O(1) under P ~ U^2, so the final "~" is unchanged, but the *)
(*  intermediate display was wrong.  These theorems close the Coq-side gap    *)
(*  (mirror of sympy/robustness_isolation_verification.py).  The isolation    *)
(*  mechanic: a coupled bound  m_coup * ||T|| <~ (1/a0^q) * RHScoef * Es/h,   *)
(*  printed as  m_prt * ||T|| <~ (1/a0^q) * factor * E_target/h, gives        *)
(*      factor = (m_prt/m_coup) * RHScoef / target_scale.                     *)
(*  Below we prove each printed 'factor' (the bracketed part; the common      *)
(*  a0^{-1/2} prefactor is carried unchanged, a0^{-1} for eq:1075).           *)
(* ========================================================================= *)

Section Isolation.

(* U,P: velocity/pressure scales; amag = ||a||_inf; s_sig,s_nu,rr stand for   *)
(* sqrt(sigma), sqrt(nu), sqrt(1+Re_h) (kept opaque -- no sqrt reasoning       *)
(* needed for the pure coefficient algebra).                                   *)
Variables (U P amag nu h s_sig s_nu rr : R).
Hypothesis U_pos    : 0 < U.
Hypothesis P_pos    : 0 < P.
Hypothesis amag_pos : 0 < amag.
Hypothesis nu_pos'  : 0 < nu.
Hypothesis h_pos'   : 0 < h.
Hypothesis ssig_pos : 0 < s_sig.
Hypothesis snu_pos  : 0 < s_nu.
Hypothesis rr_pos   : 0 < rr.

(* --- Dominant viscosity.  Coupled RHS coefficient of Es/h is (U + P h/nu). --- *)
(* eq:DominantViscosityVelocityGradientEstimate (~l.1023): m_coup=m_prt=1, scale U. *)
Theorem iso_1023 : (U + P*h/nu)/U = 1 + P*h/(U*nu).
Proof. field. repeat split; nra. Qed.

(* eq:DominantViscosityPressureGradientEstimate (~l.1027): m_coup=h/nu,m_prt=1, scale P. *)
Theorem iso_1027 : (U + P*h/nu)/(h/nu)/P = U*nu/(P*h) + 1.
Proof. field. repeat split; nra. Qed.

(* --- Dominant convection.  Coupled RHS coefficient of Es/h is (U + P/amag). --- *)
(* eq:DominantConvectionXTermEstimate (~l.1048): 1/||a|| kept on both sides, scale U. *)
Theorem iso_1048 : (U + P/amag)/U = 1 + P/(U*amag).
Proof. field. repeat split; nra. Qed.

(* eq:DominantPressureGradientXTermEstimate (~l.1052), CORRECTED: m_coup=1/amag,      *)
(* m_prt=1 (multiply through by ||a||), scale P  =>  factor = ||a|| U/P + 1.          *)
Theorem iso_1052_correct : (U + P/amag)/(1/amag)/P = amag*U/P + 1.
Proof. field. repeat split; nra. Qed.

(* eq:1052 DISCRIMINATING: the printed factor ||a||/sqrt(P)+1 does NOT equal the       *)
(* correct ||a|| U/P + 1 in general.  Concrete counterexample U=1,P=4,amag=1:          *)
(* printed = 1/sqrt 4 + 1 = 3/2, correct = 1*1/4 + 1 = 5/4. *)
Theorem iso_1052_printed_differs : 1 / sqrt 4 + 1 <> 1 * 1 / 4 + 1.
Proof.
  assert (H4 : sqrt 4 = 2).
  { replace 4 with (Rsqr 2) by (unfold Rsqr; lra). apply sqrt_Rsqr; lra. }
  rewrite H4. lra.
Qed.

(* eq:1052 under the stated scaling P = U^2: printed and correct COINCIDE (= amag/U+1), *)
(* which is why the final "~ a0^{-1/2} E_int(p)/h" conclusion is unaffected. *)
Theorem iso_1052_agree_under_PU2 :
  amag * U / (U^2) + 1 = amag / U + 1
  /\ amag / sqrt (U^2) + 1 = amag / U + 1.
Proof.
  split.
  - field; lra.
  - assert (Hs : sqrt (U^2) = U).
    { replace (U^2) with (Rsqr U) by (unfold Rsqr; ring). apply sqrt_Rsqr; lra. }
    rewrite Hs. field; lra.
Qed.

(* --- Dominant reaction.  Coupled RHS coef of Es/h is (rr U + P/(s_sig s_nu)). --- *)
(* eq:DominantReactionVelocityGradientEstimate (~l.1068): m_coup=m_prt=1, scale U. *)
Theorem iso_1068 : (rr*U + P/(s_sig*s_nu))/U = rr + P/(U*s_sig*s_nu).
Proof. field. repeat split; nra. Qed.

(* eq:DominantReactionPressureGradientEstimate (~l.1075): m_coup=a0^{1/2}/(s_sig s_nu), *)
(* m_prt=1, scale P.  The bracketed factor (the a0^{-1} prefactor is carried apart) is: *)
Theorem iso_1075 : (rr*U + P/(s_sig*s_nu))*(s_sig*s_nu)/P = s_sig*s_nu*rr*U/P + 1.
Proof. field. repeat split; nra. Qed.

End Isolation.
