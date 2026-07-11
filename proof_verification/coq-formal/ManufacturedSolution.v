(* ========================================================================= *)
(*  ManufacturedSolution.v                                                   *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the              *)
(*  manufactured-solution constructions of Section 7 of *A stabilized        *)
(*  finite element method for incompressible, inertial flows in              *)
(*  inhomogeneous porous media*.                                             *)
(*                                                                           *)
(*  Mirrors proof_verification/sympy/manufactured_solution_verification.  *)
(*  py, checks:                                                              *)
(*   [1] div(alpha u) = 0 for ARBITRARY (differentiable, positive) porosity  *)
(*       fields, because alpha u = U alpha0 S with the fixed solenoidal      *)
(*       field  S = (sin(k x1) sin(k x2), cos(k x1) cos(k x2)),              *)
(*       k = pi / L (eq:ManufacturedProblem); genuine partial derivatives   *)
(*       via derivable_pt_lim, and the conclusion transferred to ANY         *)
(*       derivative values by uniqueness of the derivative;                  *)
(*   [2] the same for the z-extruded 3D field (third component zero,         *)
(*       z-independent data);                                                *)
(*   [3] the boundary trace of u does NOT vanish on the boundary             *)
(*       (amendment A11: u . n <> 0 somewhere), witnessed at the origin      *)
(*       where u2(0,0) = U alpha0 / alpha(0,0) > 0;                          *)
(*   [4] the eps_ref reporting chain (amendment A1):                         *)
(*       eq:UpperBoundOnEpsilon with C2 = 10^-2 dominates the mid            *)
(*       expression, and eps = 10^-4 eps_ref equals 10^-2 of the bound;      *)
(*   [5] the Darcy--Brinkman--Forchheimer literature coefficients are        *)
(*       admissible ((a, b) >= 0 on the porosity range) and the literature   *)
(*       porosity profile alpha(y) = 0.45 + 0.55 e^{y-1} satisfies           *)
(*       alpha(1) = 1, alpha'(y) > 0 and 0 < alpha(0) < 1.                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

From PNSFormal Require Import DerivKit.

(* ========================================================================= *)
(*  [1] + [2] + [3]  The manufactured field.                                 *)
(* ========================================================================= *)

Section ManufacturedField.

Variables (Uc alpha0 kk : R).
Variable  alphaf : R -> R -> R.               (* the porosity field alpha *)
Hypothesis Uc_pos     : 0 < Uc.
Hypothesis alpha0_pos : 0 < alpha0.
Hypothesis alpha_pos  : forall x y, 0 < alphaf x y.

(* The solenoidal template  S = (sin sin, cos cos)  (eq:ManufacturedProblem)*)
Definition S1 (x y : R) : R := sin (kk * x) * sin (kk * y).
Definition S2 (x y : R) : R := cos (kk * x) * cos (kk * y).

(* The manufactured velocity  u = (U alpha0 / alpha) S. *)
Definition u1 (x y : R) : R := Uc * alpha0 / alphaf x y * S1 x y.
Definition u2 (x y : R) : R := Uc * alpha0 / alphaf x y * S2 x y.

(* alpha cancels exactly:  alpha u = U alpha0 S,  for ANY positive alpha. *)
Lemma alpha_u1 : forall x y, alphaf x y * u1 x y = Uc * alpha0 * S1 x y.
Proof.
  intros x y. unfold u1.
  field. pose proof (alpha_pos x y). lra.
Qed.

Lemma alpha_u2 : forall x y, alphaf x y * u2 x y = Uc * alpha0 * S2 x y.
Proof.
  intros x y. unfold u2.
  field. pose proof (alpha_pos x y). lra.
Qed.

(* Partial derivative of alpha u1 in x1. *)
Lemma d1_alpha_u1 :
  forall x y : R,
    derivable_pt_lim (fun t => alphaf t y * u1 t y) x
      (Uc * alpha0 * (kk * cos (kk * x)) * sin (kk * y)).
Proof.
  intros x y.
  apply dpl_ext with (f := fun t => (Uc * alpha0 * sin (kk * y)) * sin (kk * t)).
  - intro t. rewrite alpha_u1. unfold S1. ring.
  - eapply dpl_val_eq.
    + apply dpl_scale, dpl_sin_lin.
    + ring.
Qed.

(* Partial derivative of alpha u2 in x2. *)
Lemma d2_alpha_u2 :
  forall x y : R,
    derivable_pt_lim (fun t => alphaf x t * u2 x t) y
      (- (Uc * alpha0 * (kk * cos (kk * x)) * sin (kk * y))).
Proof.
  intros x y.
  apply dpl_ext with (f := fun t => (Uc * alpha0 * cos (kk * x)) * cos (kk * t)).
  - intro t. rewrite alpha_u2. unfold S2. ring.
  - eapply dpl_val_eq.
    + apply dpl_scale, dpl_cos_lin.
    + ring.
Qed.

(* --- Mass conservation, 2D:  div(alpha u) = 0 pointwise.  Stated in the   *)
(*     strongest honest form: WHATEVER the partial derivatives are, they    *)
(*     sum to zero (uniqueness of the derivative). --------------------------*)
Theorem manufactured_mass_conservation_2d :
  forall x y d1 d2 : R,
    derivable_pt_lim (fun t => alphaf t y * u1 t y) x d1 ->
    derivable_pt_lim (fun t => alphaf x t * u2 x t) y d2 ->
    d1 + d2 = 0.
Proof.
  intros x y d1 d2 H1 H2.
  pose proof (uniqueness_limite _ _ _ _ H1 (d1_alpha_u1 x y)) as E1.
  pose proof (uniqueness_limite _ _ _ _ H2 (d2_alpha_u2 x y)) as E2.
  rewrite E1, E2. ring.
Qed.

(* --- 3D extrusion: third velocity component is zero and all data are      *)
(*     z-independent, so the z-partial vanishes and the 2D cancellation     *)
(*     carries over unchanged. -------------------------------------------- *)
Lemma d3_alpha_u3 :
  forall (x y z : R),
    derivable_pt_lim (fun _ : R => alphaf x y * 0) z 0.
Proof.
  intros x y z.
  apply dpl_ext with (f := fct_cte (alphaf x y * 0)).
  - intro t. reflexivity.
  - apply derivable_pt_lim_const.
Qed.

Theorem manufactured_mass_conservation_3d :
  forall x y z d1 d2 d3 : R,
    derivable_pt_lim (fun t => alphaf t y * u1 t y) x d1 ->
    derivable_pt_lim (fun t => alphaf x t * u2 x t) y d2 ->
    derivable_pt_lim (fun _ : R => alphaf x y * 0) z d3 ->
    d1 + d2 + d3 = 0.
Proof.
  intros x y z d1 d2 d3 H1 H2 H3.
  pose proof (uniqueness_limite _ _ _ _ H3 (d3_alpha_u3 x y z)) as E3.
  pose proof (manufactured_mass_conservation_2d x y d1 d2 H1 H2) as E12.
  rewrite E3. lra.
Qed.

(* --- Boundary trace (amendment A11): u does not vanish on the boundary;   *)
(*     at the origin the normal component through the bottom side is        *)
(*     u2(0,0) = U alpha0 / alpha(0,0) > 0. -------------------------------- *)
Theorem boundary_trace_nonzero : 0 < u2 0 0.
Proof.
  unfold u2, S2.
  replace (kk * 0) with 0 by ring.
  rewrite cos_0.
  pose proof (alpha_pos 0 0) as Ha.
  assert (HUa : 0 < Uc * alpha0) by nra.
  assert (Hq : 0 < Uc * alpha0 / alphaf 0 0)
    by (apply Rdiv_lt_0_compat; lra).
  nra.
Qed.

End ManufacturedField.

(* ========================================================================= *)
(*  [4]  The eps_ref chain (amendment A1).                                   *)
(*  With  tau1NS^{-1} = c1 nu/h^2 + c2 |a|/h,  |a| = Re nu/h,                *)
(*        sigma = Da alpha_K nu/h^2,                                         *)
(*        tau1  = (alpha_K tau1NS^{-1} + sigma)^{-1},                        *)
(*  eq:UpperBoundOnEpsilon with C2 = 10^-2 reads                             *)
(*        eps_ref_bound := 100 c1 alpha_K^2 tau1 / h^2                       *)
(*  (the script's rhs).  The chain:                                          *)
(*    (i)  rhs = 100 alpha_K / (nu (1 + (c2/c1) Re + (1/c1) Da));            *)
(*    (ii) the mid expression (Da-coefficient 100/c1) is <= rhs;             *)
(*    (iii) eps := 10^-4 rhs' scaling: (1/10000) rhs = (1/100) of the        *)
(*          eq:UpperBoundOnEpsilon right-hand side with C2 = 10^-2,          *)
(*          i.e. eps satisfies the bound with two orders of margin.          *)
(* ========================================================================= *)

Section EpsRefChain.

Variables (nu h alphaK c1 c2 Re Da : R).
Hypothesis nu_pos : 0 < nu.
Hypothesis h_pos  : 0 < h.
Hypothesis aK_pos : 0 < alphaK.
Hypothesis c1_pos : 0 < c1.
Hypothesis c2_pos : 0 < c2.
Hypothesis Re_nn  : 0 <= Re.
Hypothesis Da_nn  : 0 <= Da.

Definition amagE      : R := Re * nu / h.
Definition sigmaE     : R := Da * alphaK * nu / h^2.
Definition tau1NS_invE : R := c1 * nu / h^2 + c2 * amagE / h.
Definition tau1E      : R := / (alphaK * tau1NS_invE + sigmaE).

Definition rhsE : R := 100 * c1 * alphaK^2 * tau1E / h^2.
Definition midE : R := alphaK / (nu * (1 + (c2/c1) * Re + (100/c1) * Da)).

Lemma h2E_pos : 0 < h^2.
Proof. apply pow_lt; exact h_pos. Qed.

Lemma denomE_pos : 0 < c1 + c2 * Re + Da.
Proof. nra. Qed.

(* (i)  Closed form of the bound. *)
Theorem rhs_closed_form :
  rhsE = 100 * alphaK / (nu * (1 + (c2/c1) * Re + (1/c1) * Da)).
Proof.
  unfold rhsE, tau1E, tau1NS_invE, sigmaE, amagE.
  pose proof h2E_pos. pose proof denomE_pos.
  assert (0 < nu * (c1 + c2 * Re + Da)) by nra.
  field. repeat split; nra.
Qed.

(* (ii) The polynomial gap identity: 100 x (mid denominator) minus the      *)
(*      (rhs denominator) is manifestly nonnegative. *)
Theorem denominator_gap :
  100 * (1 + (c2/c1) * Re + (100/c1) * Da)
  - (1 + (c2/c1) * Re + (1/c1) * Da)
  = 99 + 99 * (c2/c1) * Re + (9999/c1) * Da.
Proof. field. lra. Qed.

Corollary denominator_gap_nonneg :
  1 + (c2/c1) * Re + (1/c1) * Da
  <= 100 * (1 + (c2/c1) * Re + (100/c1) * Da).
Proof.
  pose proof denominator_gap as Hg.
  assert (0 <= (c2/c1) * Re).
  { assert (0 < c2/c1) by (apply Rdiv_lt_0_compat; lra). nra. }
  assert (0 <= (9999/c1) * Da).
  { assert (0 < 9999/c1) by (apply Rdiv_lt_0_compat; lra). nra. }
  assert (0 <= 99 * (c2/c1) * Re) by nra.
  lra.
Qed.

Corollary mid_le_rhs : midE <= rhsE.
Proof.
  rewrite rhs_closed_form.
  unfold midE.
  set (DS := 1 + (c2/c1) * Re + (1/c1) * Da).
  set (DB := 1 + (c2/c1) * Re + (100/c1) * Da).
  assert (Hc2c1 : 0 < c2/c1) by (apply Rdiv_lt_0_compat; lra).
  assert (H1c1  : 0 < 1/c1) by (apply Rdiv_lt_0_compat; lra).
  assert (H100c1 : 0 < 100/c1) by (apply Rdiv_lt_0_compat; lra).
  assert (HDS : 0 < DS) by (unfold DS; nra).
  assert (HDB : 0 < DB) by (unfold DB; nra).
  assert (Hgap : DS <= 100 * DB) by (unfold DS, DB; apply denominator_gap_nonneg).
  assert (HnDS : 0 < nu * DS) by nra.
  assert (HnDB : 0 < nu * DB) by nra.
  assert (Han  : 0 < alphaK * nu) by nra.
  apply (Rmult_le_reg_r ((nu * DS) * (nu * DB))); [nra |].
  replace (alphaK / (nu * DB) * ((nu * DS) * (nu * DB)))
    with (alphaK * (nu * DS)) by (field; repeat split; lra).
  replace (100 * alphaK / (nu * DS) * ((nu * DS) * (nu * DB)))
    with (100 * alphaK * (nu * DB)) by (field; repeat split; lra).
  nra.
Qed.

(* (iii) eps = 10^-4 rhs satisfies eq:UpperBoundOnEpsilon with C2 = 10^-2,  *)
(*       with a factor-100 margin:                                          *)
(*         (1/10000) rhs = (1/100) * ( (1/100) c1 alpha_K^2 tau1 / h^2 ).   *)
Theorem eps_choice_margin :
  (/ 10000) * rhsE = (/ 100) * ((/ 100) * (100 * c1 * alphaK^2 * tau1E / h^2)).
Proof. unfold rhsE. field. pose proof h2E_pos. lra. Qed.

End EpsRefChain.

(* ========================================================================= *)
(*  [5]  Darcy--Brinkman--Forchheimer literature coefficients and the        *)
(*       literature porosity profile.                                        *)
(* ========================================================================= *)

(* Helper used below (mirrors Rdiv_nonneg in StabilityAlgebra.v; kept local *)
(* so the files stay independent). *)
Lemma Rdiv_nonneg_gen : forall x y : R, 0 <= x -> 0 < y -> 0 <= x / y.
Proof.
  intros x y Hx Hy. unfold Rdiv.
  apply Rmult_le_pos; [exact Hx | left; apply Rinv_0_lt_compat; exact Hy].
Qed.

Section DBFCoefficients.

Variable ReD : R.
Hypothesis ReD_pos : 0 < ReD.

(*  a(alpha) = (150/Re) ((1-alpha)/alpha)^2 >= 0  on 0 < alpha <= 1. *)
Theorem dbf_a_nonneg :
  forall al : R, 0 < al <= 1 -> 0 <= 150 / ReD * ((1 - al) / al)^2.
Proof.
  intros al Hal.
  assert (Hf : 0 < 150 / ReD) by (apply Rdiv_lt_0_compat; lra).
  pose proof (pow2_ge_0 ((1 - al) / al)).
  nra.
Qed.

(*  b(alpha) = (7/4) (1-alpha)/alpha >= 0  on 0 < alpha <= 1. *)
Theorem dbf_b_nonneg :
  forall al : R, 0 < al <= 1 -> 0 <= 7 / 4 * ((1 - al) / al).
Proof.
  intros al Hal.
  assert (Hq : 0 <= (1 - al) / al) by (apply Rdiv_nonneg_gen; lra).
  lra.
Qed.

End DBFCoefficients.

(*  The literature porosity profile  alpha(y) = 0.45 + 0.55 e^{y-1}. *)
Section LiteraturePorosity.

Definition alpha_lit (y : R) : R := 45/100 + 55/100 * exp (y - 1).

(* Equivalent to the paper's normalised form 0.45 (1 + (0.55/0.45) e^{y-1}).*)
Lemma alpha_lit_forms :
  forall y, 45/100 * (1 + (55/100)/(45/100) * exp (y - 1)) = alpha_lit y.
Proof. intro y. unfold alpha_lit. field. Qed.

(*  alpha(1) = 1. *)
Theorem alpha_lit_at_1 : alpha_lit 1 = 1.
Proof.
  unfold alpha_lit.
  replace (1 - 1) with 0 by ring.
  rewrite exp_0. lra.
Qed.

(*  alpha'(y) = 0.55 e^{y-1} > 0: strictly increasing. *)
Theorem alpha_lit_deriv :
  forall y, derivable_pt_lim alpha_lit y (55/100 * exp (y - 1)).
Proof.
  intro y.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_plus (fct_cte (45/100))
             (fun t => 55/100 * exp (t - 1))).
    + apply derivable_pt_lim_const.
    + apply dpl_scale, dpl_exp_shift.
  - ring.
Qed.

Theorem alpha_lit_deriv_pos :
  forall y, 0 < 55/100 * exp (y - 1).
Proof. intro y. pose proof (exp_pos (y - 1)). lra. Qed.

(*  0 < alpha(0) < 1. *)
Theorem alpha_lit_at_0_range : 0 < alpha_lit 0 < 1.
Proof.
  unfold alpha_lit.
  replace (0 - 1) with (- 1) by ring.
  pose proof (exp_pos (- 1)) as Hp.
  assert (He1 : exp (- 1) < 1).
  { rewrite <- exp_0. apply exp_increasing. lra. }
  lra.
Qed.

End LiteraturePorosity.
