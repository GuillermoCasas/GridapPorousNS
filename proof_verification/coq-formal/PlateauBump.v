(* ========================================================================= *)
(*  PlateauBump.v                                                            *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the              *)
(*  plateau-bump porosity construction of Section 7 of *A stabilized         *)
(*  finite element method for incompressible, inertial flows in              *)
(*  inhomogeneous porous media* (eq:PlateauBumpFunction, eq:Gamma).                            *)
(*                                                                           *)
(*  Mirrors theory/verification scripts/manufactured_solution_verification.  *)
(*  py (bump section).  With                                                 *)
(*      gamma(eta)  = (2 eta - 1) / (eta (1 - eta)),                         *)
(*      alpha_mid   = 1 - (1 - alpha0) / (1 + e^{gamma(eta)}),               *)
(*  it proves:                                                               *)
(*   [1]  gamma'(eta) = (2 eta^2 - 2 eta + 1) / (eta (1 - eta))^2 > 0        *)
(*        on (0,1), as a genuine derivative (derivable_pt_lim);              *)
(*   [2]  alpha_mid'(eta)                                                    *)
(*          = (1 - alpha0) e^{gamma} gamma' / (1 + e^{gamma})^2 > 0          *)
(*        on (0,1): the transition is strictly monotone;                     *)
(*   [3]  alpha0 < alpha_mid(eta) < 1: the bump stays inside the physical    *)
(*        porosity range;                                                    *)
(*   [4]  alpha_mid -> alpha0 as eta -> 0+ and alpha_mid -> 1 as             *)
(*        eta -> 1-, as genuine one-sided limits (the joins are continuous); *)
(*   [5]  alpha_mid' -> 0 at both ends: the joins are C^1.  (Higher          *)
(*        derivatives vanish by the same e^{-1/x}-type mechanism; the        *)
(*        first-order case formalized here is the one the SymPy script       *)
(*        also verifies symbolically.)                                       *)
(*                                                                           *)
(*  Every limit is proved through fully explicit epsilon-delta bounds        *)
(*  derived from the elementary inequalities  t < e^t  and                   *)
(*  t^3 / 27 <= e^t (t >= 0);  no logarithms are needed.                     *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
From PNSFormal Require Import Limits DerivKit.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  Elementary exponential bounds.                                           *)
(* ========================================================================= *)

(*  t < e^t  for every real t. *)
Lemma lt_exp : forall t : R, t < exp t.
Proof.
  intro t.
  destruct (Req_dec t 0) as [-> | Hne].
  - rewrite exp_0. lra.
  - pose proof (exp_ineq1 t Hne). lra.
Qed.

(*  e^{-x} = 1 / e^x  (self-contained; avoids depending on Rpower). *)
Lemma exp_opp : forall x : R, exp (- x) = / exp x.
Proof.
  intro x.
  pose proof (exp_pos x) as Hp.
  apply (Rmult_eq_reg_l (exp x)); [| lra].
  rewrite <- exp_plus.
  replace (x + - x) with 0 by ring.
  rewrite exp_0, Rinv_r by lra.
  reflexivity.
Qed.

(*  For z < 0:  e^z < 1 / (-z). *)
Lemma exp_neg_upper : forall z : R, z < 0 -> exp z < / (- z).
Proof.
  intros z Hz.
  assert (Hpos : 0 < - z) by lra.
  pose proof (lt_exp (- z)) as Hlt.
  pose proof (exp_pos (- z)) as Hep.
  assert (Hinv : / exp (- z) < / (- z)).
  { apply Rinv_lt_contravar; [nra | exact Hlt]. }
  assert (E : exp z = / exp (- z)).
  { rewrite <- (Ropp_involutive z) at 1. apply exp_opp. }
  rewrite E. exact Hinv.
Qed.

(*  For t >= 0:  t^3 / 27 <= e^t   (cube of  1 + t/3 <= e^{t/3}). *)
Lemma exp_cube_lower : forall t : R, 0 <= t -> t^3 / 27 <= exp t.
Proof.
  intros t Ht.
  assert (H13 : 0 <= t / 3) by lra.
  assert (Hle2 : 1 + t / 3 <= exp (t / 3)).
  { destruct (Req_dec (t / 3) 0) as [E | Hne].
    - rewrite E, exp_0. lra.
    - left. apply exp_ineq1, Hne. }
  assert (Hcube : (t / 3)^3 <= (exp (t / 3))^3).
  { apply pow_incr. split; [exact H13 | lra]. }
  replace ((exp (t / 3))^3) with (exp t) in Hcube.
  2:{ replace ((exp (t / 3))^3)
        with (exp (t / 3) * exp (t / 3) * exp (t / 3)) by ring.
      rewrite <- !exp_plus. f_equal. lra. }
  assert (Ht3 : t^3 / 27 = (t / 3)^3) by field.
  lra.
Qed.

(*  For z > 0:  e^{-z} <= 27 / z^3. *)
Lemma exp_neg_cube_upper : forall z : R, 0 < z -> exp (- z) <= 27 / z^3.
Proof.
  intros z Hz.
  pose proof (exp_cube_lower z (Rlt_le _ _ Hz)) as Hc.
  assert (Hz3 : 0 < z^3) by (apply pow_lt; lra).
  assert (Hq : 0 < z^3 / 27) by lra.
  pose proof (exp_pos z) as Hep.
  assert (Hinv : / exp z <= / (z^3 / 27))
    by (apply Rinv_le_contravar; [exact Hq | exact Hc]).
  replace (/ (z^3 / 27)) with (27 / z^3) in Hinv by (field; lra).
  rewrite exp_opp. exact Hinv.
Qed.

(* ========================================================================= *)
(*  Division comparison helpers.                                             *)
(* ========================================================================= *)

Lemma div_nonneg : forall x y : R, 0 <= x -> 0 < y -> 0 <= x / y.
Proof.
  intros x y Hx Hy. unfold Rdiv.
  apply Rmult_le_pos; [exact Hx | left; apply Rinv_0_lt_compat; exact Hy].
Qed.

(*  Nonnegative numerator: bigger denominator, smaller quotient. *)
Lemma div_le_denom :
  forall N D1 D2 : R, 0 <= N -> 0 < D1 -> D1 <= D2 -> N / D2 <= N / D1.
Proof.
  intros N D1 D2 HN HD1 HDD. unfold Rdiv.
  apply Rmult_le_compat_l; [exact HN |].
  apply Rinv_le_contravar; assumption.
Qed.

(*  Nonpositive numerator: bigger denominator, bigger quotient. *)
Lemma div_neg_denom :
  forall N D1 D2 : R, N <= 0 -> 0 < D1 -> D1 <= D2 -> N / D1 <= N / D2.
Proof.
  intros N D1 D2 HN HD1 HDD. unfold Rdiv.
  apply Rmult_le_compat_neg_l; [exact HN |].
  apply Rinv_le_contravar; assumption.
Qed.

Lemma div_num_mono :
  forall N1 N2 D : R, 0 < D -> N1 <= N2 -> N1 / D <= N2 / D.
Proof.
  intros N1 N2 D HD HN. unfold Rdiv.
  apply Rmult_le_compat_r;
    [left; apply Rinv_0_lt_compat; exact HD | exact HN].
Qed.

Lemma div_le_of_ge_one : forall X Y : R, 0 <= X -> 1 <= Y -> X / Y <= X.
Proof.
  intros X Y HX HY.
  pose proof (div_le_denom X 1 Y HX Rlt_0_1 HY) as H.
  unfold Rdiv in *. rewrite Rinv_1, Rmult_1_r in H. exact H.
Qed.

(*  For X > 0:  X / (1 + X)^2 <= 1 / X. *)
Lemma frac_sq_le_inv :
  forall X : R, 0 < X -> X / ((1 + X) * (1 + X)) <= / X.
Proof.
  intros X HX.
  assert (HXX : 0 < X * X) by nra.
  assert (Hsq : X * X <= (1 + X) * (1 + X)) by nra.
  assert (Hstep : X / ((1 + X) * (1 + X)) <= X / (X * X))
    by (apply div_le_denom; [lra | exact HXX | exact Hsq]).
  replace (X / (X * X)) with (/ X) in Hstep by (field; lra).
  exact Hstep.
Qed.

(* ========================================================================= *)
(*  The plateau bump.                                                        *)
(* ========================================================================= *)

Section PlateauBump.

Variable a0 : R.
Hypothesis Ha0 : 0 < a0 < 1.

Definition gam  (e : R) : R := (2*e - 1) / (e * (1 - e)).
Definition gam' (e : R) : R :=
  (2*e^2 - 2*e + 1) / ((e * (1 - e)) * (e * (1 - e))).

Definition am  (e : R) : R := 1 - (1 - a0) / (1 + exp (gam e)).
Definition am' (e : R) : R :=
  (1 - a0) * exp (gam e) * gam' e
  / ((1 + exp (gam e)) * (1 + exp (gam e))).

Lemma denom_pos : forall e, 0 < e < 1 -> 0 < e * (1 - e).
Proof. intros e He. nra. Qed.

Lemma one_plus_exp_pos : forall z, 0 < 1 + exp z.
Proof. intro z. pose proof (exp_pos z). lra. Qed.

(* ------------------------------------------------------------------------ *)
(*  [1]  gamma' as a genuine derivative, and its positivity.                 *)
(* ------------------------------------------------------------------------ *)

Theorem gam_deriv :
  forall e, 0 < e < 1 -> derivable_pt_lim gam e (gam' e).
Proof.
  intros e He.
  pose proof (denom_pos e He) as Hd.
  unfold gam, gam'.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_div (fun t => 2*t - 1) (fun t => t * (1 - t))).
    + apply (derivable_pt_lim_minus (fun t => 2*t) (fct_cte 1)).
      * apply dpl_linear.
      * apply derivable_pt_lim_const.
    + apply (derivable_pt_lim_mult (fun t => t) (fun t => 1 - t)).
      * apply derivable_pt_lim_id.
      * apply (derivable_pt_lim_minus (fct_cte 1) (fun t => t)).
        -- apply derivable_pt_lim_const.
        -- apply derivable_pt_lim_id.
    + cbv beta. lra.
  - cbv beta. unfold Rsqr. field. lra.
Qed.

Theorem gam'_pos : forall e, 0 < e < 1 -> 0 < gam' e.
Proof.
  intros e He. unfold gam'.
  pose proof (denom_pos e He) as Hd.
  assert (Hnum : 0 < 2*e^2 - 2*e + 1).
  { pose proof (pow2_ge_0 (2*e - 1)). nra. }
  assert (Hdd : 0 < (e * (1 - e)) * (e * (1 - e))) by nra.
  apply Rdiv_lt_0_compat; assumption.
Qed.

(* ------------------------------------------------------------------------ *)
(*  [2]  alpha_mid' as a genuine derivative, and monotonicity.               *)
(* ------------------------------------------------------------------------ *)

Lemma exp_gam_deriv :
  forall e, 0 < e < 1 ->
    derivable_pt_lim (fun t => exp (gam t)) e (exp (gam e) * gam' e).
Proof.
  intros e He.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_comp gam exp).
    + apply gam_deriv, He.
    + apply derivable_pt_lim_exp.
  - reflexivity.
Qed.

Lemma q_deriv :
  forall e, 0 < e < 1 ->
    derivable_pt_lim (fun t => (1 - a0) / (1 + exp (gam t))) e
      (- ((1 - a0) * exp (gam e) * gam' e)
         / ((1 + exp (gam e)) * (1 + exp (gam e)))).
Proof.
  intros e He.
  pose proof (one_plus_exp_pos (gam e)) as Hg.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_div (fct_cte (1 - a0))
             (fun t => 1 + exp (gam t))).
    + apply derivable_pt_lim_const.
    + apply (derivable_pt_lim_plus (fct_cte 1) (fun t => exp (gam t))).
      * apply derivable_pt_lim_const.
      * apply exp_gam_deriv, He.
    + cbv beta. lra.
  - cbv beta. unfold Rsqr, fct_cte. field. lra.
Qed.

Theorem am_deriv :
  forall e, 0 < e < 1 -> derivable_pt_lim am e (am' e).
Proof.
  intros e He.
  pose proof (one_plus_exp_pos (gam e)) as Hg.
  unfold am.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_minus (fct_cte 1)
             (fun t => (1 - a0) / (1 + exp (gam t)))).
    + apply derivable_pt_lim_const.
    + apply q_deriv, He.
  - unfold am'. field. lra.
Qed.

Theorem am'_pos : forall e, 0 < e < 1 -> 0 < am' e.
Proof.
  intros e He. unfold am'.
  pose proof (exp_pos (gam e)) as He1.
  pose proof (gam'_pos e He) as Hg'.
  pose proof (one_plus_exp_pos (gam e)) as Hd.
  assert (H1 : 0 < 1 - a0) by lra.
  assert (P1 : 0 < (1 - a0) * exp (gam e)) by nra.
  assert (P2 : 0 < (1 - a0) * exp (gam e) * gam' e) by nra.
  assert (PD : 0 < (1 + exp (gam e)) * (1 + exp (gam e))) by nra.
  apply Rdiv_lt_0_compat; assumption.
Qed.

(* ------------------------------------------------------------------------ *)
(*  [3]  Range:  alpha0 < alpha_mid < 1.                                     *)
(* ------------------------------------------------------------------------ *)

Theorem am_range : forall e : R, a0 < am e < 1.
Proof.
  intro e. unfold am.
  pose proof (exp_pos (gam e)) as Hp.
  pose proof (one_plus_exp_pos (gam e)) as Hd.
  assert (Hinvlt : / (1 + exp (gam e)) < 1).
  { assert (H := Rinv_lt_contravar 1 (1 + exp (gam e))).
    rewrite Rinv_1 in H. apply H; lra. }
  assert (Hinvpos : 0 < / (1 + exp (gam e)))
    by (apply Rinv_0_lt_compat; lra).
  assert (Hq1 : (1 - a0) / (1 + exp (gam e)) < 1 - a0).
  { unfold Rdiv. nra. }
  assert (Hq0 : 0 < (1 - a0) / (1 + exp (gam e)))
    by (apply Rdiv_lt_0_compat; lra).
  lra.
Qed.

(* ------------------------------------------------------------------------ *)
(*  Quantitative bounds near the joins.                                      *)
(* ------------------------------------------------------------------------ *)

(*  On (0, 1/4]:  gamma <= 2 - 1/eta. *)
Lemma gam_upper_small :
  forall e, 0 < e <= /4 -> gam e <= 2 - /e.
Proof.
  intros e He.
  assert (He1 : e < 1) by lra.
  assert (Hd1 : 0 < e * (1 - e)) by nra.
  assert (HN : 2*e - 1 <= 0) by lra.
  assert (Hdd : e * (1 - e) <= e) by nra.
  assert (Hstep : (2*e - 1) / (e * (1 - e)) <= (2*e - 1) / e)
    by (apply div_neg_denom; assumption).
  replace ((2*e - 1) / e) with (2 - /e) in Hstep by (field; lra).
  exact Hstep.
Qed.

(*  On (0, 1/4]:  1/(2 eta) <= - gamma. *)
Lemma neg_gam_lower_small :
  forall e, 0 < e <= /4 -> / (2*e) <= - gam e.
Proof.
  intros e He.
  pose proof (gam_upper_small e He) as Hg.
  assert (Hinv : 4 <= / e).
  { apply (Rmult_le_reg_l e); [lra |].
    rewrite Rinv_r by lra. lra. }
  rewrite Rinv_mult. lra.
Qed.

(*  On (0, 1/4]:  e^{gamma} <= 2 eta. *)
Lemma exp_gam_small :
  forall e, 0 < e <= /4 -> exp (gam e) <= 2*e.
Proof.
  intros e He.
  pose proof (neg_gam_lower_small e He) as Hlow.
  assert (Hinv2e : 0 < / (2*e)) by (apply Rinv_0_lt_compat; lra).
  assert (Hgneg : gam e < 0) by lra.
  pose proof (exp_neg_upper (gam e) Hgneg) as Hupper.
  assert (Hstep : / (- gam e) <= / (/ (2*e)))
    by (apply Rinv_le_contravar; [exact Hinv2e | exact Hlow]).
  rewrite Rinv_inv in Hstep. lra.
Qed.

(*  On (0, 1/4]:  e^{gamma} <= 216 eta^3  (cubic decay for the derivative). *)
Lemma exp_gam_cubic_small :
  forall e, 0 < e <= /4 -> exp (gam e) <= 216 * e^3.
Proof.
  intros e He.
  pose proof (neg_gam_lower_small e He) as Hlow.
  assert (Hinv2e : 0 < / (2*e)) by (apply Rinv_0_lt_compat; lra).
  assert (Hgpos : 0 < - gam e) by lra.
  pose proof (exp_neg_cube_upper (- gam e) Hgpos) as Hcu.
  rewrite Ropp_involutive in Hcu.
  assert (Hc3 : (/ (2*e))^3 <= (- gam e)^3)
    by (apply pow_incr; split; [left; exact Hinv2e | exact Hlow]).
  assert (Hd3 : 0 < (/ (2*e))^3) by (apply pow_lt; exact Hinv2e).
  assert (Hstep : 27 / (- gam e)^3 <= 27 / (/ (2*e))^3)
    by (apply div_le_denom; lra).
  replace (27 / (/ (2*e))^3) with (216 * e^3) in Hstep
    by (field; repeat split; lra).
  lra.
Qed.

(*  On [3/4, 1):  1/(2(1 - eta)) <= gamma. *)
Lemma gam_lower_big :
  forall e, 3/4 <= e < 1 -> / (2*(1 - e)) <= gam e.
Proof.
  intros e He.
  assert (Hd1 : 0 < e * (1 - e)) by nra.
  assert (HN : 0 <= 2*e - 1) by lra.
  assert (Hdd : e * (1 - e) <= 1 - e) by nra.
  assert (Hstep : (2*e - 1) / (1 - e) <= (2*e - 1) / (e * (1 - e)))
    by (apply div_le_denom; lra).
  assert (Hhalf : / (2*(1 - e)) <= (2*e - 1) / (1 - e)).
  { rewrite Rinv_mult.
    assert (H12 : /2 <= 2*e - 1) by lra.
    unfold Rdiv.
    apply Rmult_le_compat_r;
      [left; apply Rinv_0_lt_compat; lra | exact H12]. }
  unfold gam. lra.
Qed.

(*  On [3/4, 1):  e^{-gamma} <= 2 (1 - eta). *)
Lemma exp_neg_gam_big :
  forall e, 3/4 <= e < 1 -> exp (- gam e) <= 2*(1 - e).
Proof.
  intros e He.
  pose proof (gam_lower_big e He) as Hlow.
  assert (Hi : 0 < / (2*(1 - e))) by (apply Rinv_0_lt_compat; lra).
  assert (Hneg : - gam e < 0) by lra.
  pose proof (exp_neg_upper (- gam e) Hneg) as Hup.
  rewrite Ropp_involutive in Hup.
  assert (Hstep : / gam e <= / (/ (2*(1 - e))))
    by (apply Rinv_le_contravar; [exact Hi | exact Hlow]).
  rewrite Rinv_inv in Hstep. lra.
Qed.

(*  On [3/4, 1):  e^{-gamma} <= 216 (1 - eta)^3. *)
Lemma exp_neg_gam_cubic_big :
  forall e, 3/4 <= e < 1 -> exp (- gam e) <= 216 * (1 - e)^3.
Proof.
  intros e He.
  pose proof (gam_lower_big e He) as Hlow.
  assert (Hi : 0 < / (2*(1 - e))) by (apply Rinv_0_lt_compat; lra).
  assert (Hgpos : 0 < gam e) by lra.
  pose proof (exp_neg_cube_upper (gam e) Hgpos) as Hcu.
  assert (Hc3 : (/ (2*(1 - e)))^3 <= (gam e)^3)
    by (apply pow_incr; split; [left; exact Hi | exact Hlow]).
  assert (Hd3 : 0 < (/ (2*(1 - e)))^3) by (apply pow_lt; exact Hi).
  assert (Hstep : 27 / (gam e)^3 <= 27 / (/ (2*(1 - e)))^3)
    by (apply div_le_denom; lra).
  replace (27 / (/ (2*(1 - e)))^3) with (216 * (1 - e)^3) in Hstep
    by (field; repeat split; lra).
  lra.
Qed.

(*  On (0, 1/4]:  gamma' <= 16 / (9 eta^2). *)
Lemma gam'_upper_small :
  forall e, 0 < e <= /4 -> gam' e <= 16 / (9 * e^2).
Proof.
  intros e He.
  assert (He1 : e < 1) by lra.
  assert (Hd : 0 < e * (1 - e)) by nra.
  assert (Hnum_le : 2*e^2 - 2*e + 1 <= 1) by nra.
  assert (Hnum_pos : 0 < 2*e^2 - 2*e + 1).
  { pose proof (pow2_ge_0 (2*e - 1)). nra. }
  assert (Hee : 3/4 * e <= e * (1 - e)) by nra.
  assert (H34e : 0 < 3/4 * e) by lra.
  assert (HAB : 0 < 3/4 * e + e * (1 - e)) by lra.
  assert (Hden_low :
      (3/4 * e) * (3/4 * e) <= (e * (1 - e)) * (e * (1 - e))) by nra.
  assert (Hden_pos : 0 < (3/4 * e) * (3/4 * e)) by nra.
  unfold gam'.
  assert (S1 : (2*e^2 - 2*e + 1) / ((e * (1 - e)) * (e * (1 - e)))
               <= 1 / ((e * (1 - e)) * (e * (1 - e))))
    by (apply div_num_mono; nra).
  assert (S2 : 1 / ((e * (1 - e)) * (e * (1 - e)))
               <= 1 / ((3/4 * e) * (3/4 * e)))
    by (apply div_le_denom; lra).
  replace (1 / ((3/4 * e) * (3/4 * e))) with (16 / (9 * e^2)) in S2
    by (field; lra).
  lra.
Qed.

(*  On [3/4, 1):  gamma' <= 16 / (9 (1 - eta)^2). *)
Lemma gam'_upper_big :
  forall e, 3/4 <= e < 1 -> gam' e <= 16 / (9 * (1 - e)^2).
Proof.
  intros e He.
  assert (Hd : 0 < e * (1 - e)) by nra.
  assert (Hnum_le : 2*e^2 - 2*e + 1 <= 1) by nra.
  assert (Hnum_pos : 0 < 2*e^2 - 2*e + 1).
  { pose proof (pow2_ge_0 (2*e - 1)). nra. }
  assert (Hee : 3/4 * (1 - e) <= e * (1 - e)) by nra.
  assert (H34e : 0 < 3/4 * (1 - e)) by lra.
  assert (HAB : 0 < 3/4 * (1 - e) + e * (1 - e)) by lra.
  assert (Hden_low :
      (3/4 * (1 - e)) * (3/4 * (1 - e))
      <= (e * (1 - e)) * (e * (1 - e))) by nra.
  assert (Hden_pos : 0 < (3/4 * (1 - e)) * (3/4 * (1 - e))) by nra.
  unfold gam'.
  assert (S1 : (2*e^2 - 2*e + 1) / ((e * (1 - e)) * (e * (1 - e)))
               <= 1 / ((e * (1 - e)) * (e * (1 - e))))
    by (apply div_num_mono; nra).
  assert (S2 : 1 / ((e * (1 - e)) * (e * (1 - e)))
               <= 1 / ((3/4 * (1 - e)) * (3/4 * (1 - e))))
    by (apply div_le_denom; lra).
  replace (1 / ((3/4 * (1 - e)) * (3/4 * (1 - e))))
    with (16 / (9 * (1 - e)^2)) in S2 by (field; lra).
  lra.
Qed.

(* ------------------------------------------------------------------------ *)
(*  [4]  Continuity of the joins:  alpha_mid -> alpha0 (eta -> 0+),          *)
(*       alpha_mid -> 1 (eta -> 1-).                                         *)
(* ------------------------------------------------------------------------ *)

Theorem am_limit_0 : tendsto_at_0plus am a0.
Proof.
  intros eps Heps.
  assert (H1a0 : 0 < 1 - a0) by lra.
  assert (H2a : 0 < 2 * (1 - a0)) by lra.
  assert (Hd2 : 0 < eps / (2 * (1 - a0)))
    by (apply Rdiv_lt_0_compat; lra).
  set (delta := Rmin (/4) (eps / (2 * (1 - a0)))).
  assert (Hdpos : 0 < delta) by (apply Rmin_pos; lra).
  exists delta. split; [exact Hdpos |].
  intros e [He0 Hed].
  assert (Hdl : delta <= /4) by apply Rmin_l.
  assert (Hdr : delta <= eps / (2 * (1 - a0))) by apply Rmin_r.
  assert (He4 : e <= /4) by lra.
  pose proof (one_plus_exp_pos (gam e)) as Hopep.
  pose proof (exp_pos (gam e)) as Hep.
  assert (Hid : am e - a0 = (1 - a0) * (exp (gam e) / (1 + exp (gam e)))).
  { unfold am. field. lra. }
  assert (Hfrac : exp (gam e) / (1 + exp (gam e)) <= exp (gam e))
    by (apply div_le_of_ge_one; lra).
  assert (Hfrac0 : 0 <= exp (gam e) / (1 + exp (gam e)))
    by (apply div_nonneg; lra).
  pose proof (exp_gam_small e (conj He0 He4)) as Hexp.
  assert (Hpos : 0 <= am e - a0) by (rewrite Hid; nra).
  rewrite Rabs_right by lra.
  assert (Hb1 : am e - a0 <= (1 - a0) * exp (gam e)).
  { rewrite Hid. apply Rmult_le_compat_l; [lra | exact Hfrac]. }
  assert (Hb2 : (1 - a0) * exp (gam e) <= (1 - a0) * (2*e))
    by (apply Rmult_le_compat_l; lra).
  assert (Hb3 : (1 - a0) * (2*e) < (1 - a0) * (2*delta)) by nra.
  assert (Hfield : eps / (2 * (1 - a0)) * (2 * (1 - a0)) = eps)
    by (field; lra).
  assert (Hb4 : (1 - a0) * (2*delta) <= eps) by nra.
  lra.
Qed.

Theorem am_limit_1 : tendsto_at_1minus am 1.
Proof.
  intros eps Heps.
  assert (H1a0 : 0 < 1 - a0) by lra.
  assert (H2a : 0 < 2 * (1 - a0)) by lra.
  assert (Hd2 : 0 < eps / (2 * (1 - a0)))
    by (apply Rdiv_lt_0_compat; lra).
  set (delta := Rmin (/4) (eps / (2 * (1 - a0)))).
  assert (Hdpos : 0 < delta) by (apply Rmin_pos; lra).
  exists delta. split; [exact Hdpos |].
  intros e [He0 He1].
  assert (Hdl : delta <= /4) by apply Rmin_l.
  assert (Hdr : delta <= eps / (2 * (1 - a0))) by apply Rmin_r.
  assert (He34 : 3/4 <= e) by lra.
  pose proof (one_plus_exp_pos (gam e)) as Hopep.
  pose proof (exp_pos (gam e)) as Hep.
  assert (Hid : 1 - am e = (1 - a0) / (1 + exp (gam e))).
  { unfold am. field. lra. }
  pose proof (am_range e) as Hrange.
  rewrite Rabs_left1 by lra.
  replace (- (am e - 1)) with (1 - am e) by ring.
  assert (Hb1 : (1 - a0) / (1 + exp (gam e)) <= (1 - a0) * exp (- gam e)).
  { rewrite exp_opp. unfold Rdiv.
    apply Rmult_le_compat_l; [lra |].
    apply Rinv_le_contravar; lra. }
  pose proof (exp_neg_gam_big e (conj He34 He1)) as Hexp.
  assert (Hb2 : (1 - a0) * exp (- gam e) <= (1 - a0) * (2*(1 - e)))
    by (apply Rmult_le_compat_l; lra).
  assert (Hb3 : (1 - a0) * (2*(1 - e)) < (1 - a0) * (2*delta)) by nra.
  assert (Hfield : eps / (2 * (1 - a0)) * (2 * (1 - a0)) = eps)
    by (field; lra).
  assert (Hb4 : (1 - a0) * (2*delta) <= eps) by nra.
  lra.
Qed.

(* ------------------------------------------------------------------------ *)
(*  [5]  C^1 joins: alpha_mid' -> 0 at both ends.                            *)
(* ------------------------------------------------------------------------ *)

Theorem am'_limit_0 : tendsto_at_0plus am' 0.
Proof.
  intros eps Heps.
  assert (H1a0 : 0 < 1 - a0) by lra.
  assert (H384 : 0 < 384 * (1 - a0)) by lra.
  assert (Hd2 : 0 < eps / (384 * (1 - a0)))
    by (apply Rdiv_lt_0_compat; lra).
  set (delta := Rmin (/4) (eps / (384 * (1 - a0)))).
  assert (Hdpos : 0 < delta) by (apply Rmin_pos; lra).
  exists delta. split; [exact Hdpos |].
  intros e [He0 Hed].
  assert (Hdl : delta <= /4) by apply Rmin_l.
  assert (Hdr : delta <= eps / (384 * (1 - a0))) by apply Rmin_r.
  assert (He4 : e <= /4) by lra.
  assert (He01 : 0 < e < 1) by lra.
  pose proof (one_plus_exp_pos (gam e)) as Hopep.
  pose proof (exp_pos (gam e)) as Hep.
  pose proof (gam'_pos e He01) as Hg'p.
  pose proof (am'_pos e He01) as Ham'.
  rewrite Rabs_right by lra.
  replace (am' e - 0) with (am' e) by ring.
  (* am' <= (1-a0) e^gamma gamma'  (the squared denominator is >= 1) *)
  assert (Hden1 : 1 <= (1 + exp (gam e)) * (1 + exp (gam e))) by nra.
  assert (Hnum1 : 0 <= (1 - a0) * exp (gam e)) by nra.
  assert (Hnum0 : 0 <= (1 - a0) * exp (gam e) * gam' e) by nra.
  assert (Hb1 : am' e <= (1 - a0) * exp (gam e) * gam' e).
  { unfold am'. apply div_le_of_ge_one; assumption. }
  (* product of the two decay bounds *)
  pose proof (exp_gam_cubic_small e (conj He0 He4)) as Hcube.
  pose proof (gam'_upper_small e (conj He0 He4)) as Hgam'.
  assert (Hprod : exp (gam e) * gam' e <= (216 * e^3) * (16 / (9 * e^2))).
  { apply Rmult_le_compat; [lra | lra | exact Hcube | exact Hgam']. }
  assert (Hsimp : (216 * e^3) * (16 / (9 * e^2)) = 384 * e)
    by (field; lra).
  assert (Hb2 : (1 - a0) * exp (gam e) * gam' e <= (1 - a0) * (384 * e)).
  { replace ((1 - a0) * exp (gam e) * gam' e)
      with ((1 - a0) * (exp (gam e) * gam' e)) by ring.
    apply Rmult_le_compat_l; [lra | lra]. }
  assert (Hb3 : (1 - a0) * (384 * e) < (1 - a0) * (384 * delta)) by nra.
  assert (Hfield : eps / (384 * (1 - a0)) * (384 * (1 - a0)) = eps)
    by (field; lra).
  assert (Hb4 : (1 - a0) * (384 * delta) <= eps) by nra.
  lra.
Qed.

Theorem am'_limit_1 : tendsto_at_1minus am' 0.
Proof.
  intros eps Heps.
  assert (H1a0 : 0 < 1 - a0) by lra.
  assert (H384 : 0 < 384 * (1 - a0)) by lra.
  assert (Hd2 : 0 < eps / (384 * (1 - a0)))
    by (apply Rdiv_lt_0_compat; lra).
  set (delta := Rmin (/4) (eps / (384 * (1 - a0)))).
  assert (Hdpos : 0 < delta) by (apply Rmin_pos; lra).
  exists delta. split; [exact Hdpos |].
  intros e [He0 He1].
  assert (Hdl : delta <= /4) by apply Rmin_l.
  assert (Hdr : delta <= eps / (384 * (1 - a0))) by apply Rmin_r.
  assert (He34 : 3/4 <= e) by lra.
  assert (He01 : 0 < e < 1) by lra.
  pose proof (one_plus_exp_pos (gam e)) as Hopep.
  pose proof (exp_pos (gam e)) as Hep.
  pose proof (gam'_pos e He01) as Hg'p.
  pose proof (am'_pos e He01) as Ham'.
  rewrite Rabs_right by lra.
  replace (am' e - 0) with (am' e) by ring.
  (* am' = ((1-a0) gamma') * ( e^gamma / (1+e^gamma)^2 )
         <= ((1-a0) gamma') * e^{-gamma} *)
  assert (Hid : am' e
      = ((1 - a0) * gam' e)
        * (exp (gam e) / ((1 + exp (gam e)) * (1 + exp (gam e))))).
  { unfold am'. field. lra. }
  pose proof (frac_sq_le_inv (exp (gam e)) Hep) as Hfrac.
  assert (Hb1 : am' e <= ((1 - a0) * gam' e) * exp (- gam e)).
  { rewrite Hid, exp_opp.
    apply Rmult_le_compat_l; [nra | exact Hfrac]. }
  pose proof (exp_neg_gam_cubic_big e (conj He34 He1)) as Hcube.
  pose proof (gam'_upper_big e (conj He34 He1)) as Hgam'.
  assert (Hprod : gam' e * exp (- gam e)
                  <= (16 / (9 * (1 - e)^2)) * (216 * (1 - e)^3)).
  { apply Rmult_le_compat;
      [lra | left; apply exp_pos | exact Hgam' | exact Hcube]. }
  assert (Hsimp : (16 / (9 * (1 - e)^2)) * (216 * (1 - e)^3)
                  = 384 * (1 - e)) by (field; lra).
  assert (Hb2 : ((1 - a0) * gam' e) * exp (- gam e)
                <= (1 - a0) * (384 * (1 - e))).
  { replace (((1 - a0) * gam' e) * exp (- gam e))
      with ((1 - a0) * (gam' e * exp (- gam e))) by ring.
    apply Rmult_le_compat_l; [lra | lra]. }
  assert (Hb3 : (1 - a0) * (384 * (1 - e)) < (1 - a0) * (384 * delta))
    by nra.
  assert (Hfield : eps / (384 * (1 - a0)) * (384 * (1 - a0)) = eps)
    by (field; lra).
  assert (Hb4 : (1 - a0) * (384 * delta) <= eps) by nra.
  lra.
Qed.

End PlateauBump.
