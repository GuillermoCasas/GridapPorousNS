(* ========================================================================= *)
(*  ContinuityAlgebra.v                                                      *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the algebraic    *)
(*  core of Appendix B (app:Continuity) of *A stabilized finite element      *)
(*  method for incompressible, inertial flows in inhomogeneous porous        *)
(*  media*: the continuity proof for the stabilized bilinear form B_S.       *)
(*                                                                           *)
(*  Contents (crosswalk to the appendix):                                    *)
(*   [1]  lem:parameters (P1)--(P5), including the square-root forms         *)
(*        actually used downstream (P3: phi1^(1/2) h = c1^(1/2) alpha_K      *)
(*        tau2^(1/2); P5: eps^(1/2) h <= c1^(1/2) alpha_K tau1^(1/2), and    *)
(*        the Step-7 scalar  eps tau2^(1/2) <= C2^(1/2) eps^(1/2));          *)
(*   [2]  the elemental scalar cores of the proof of lem:continuity:         *)
(*        eq:keyvisc (Step 4), the T8 and T13 chains (Steps 4--5),           *)
(*        eq:volpart's  phi1 tau1 <= (phi1 tau1)^(1/2)  (Step 6b),           *)
(*        phi1^(1/2) tau1 <= tau1^(1/2)  (Step 6c),                          *)
(*        and the coefficient chains of eq:absorb1--eq:absorb5 (Step 9);     *)
(*   [3]  lem:jump in full, with explicit constants in place of the          *)
(*        appendix's generic C: the jump formula, the bound                  *)
(*        sigma |[tau1]| <= (CJ/cJ) sigmatilde tau1, the elementwise         *)
(*        comparabilities across a face, the second form of eq:jumpsplit,    *)
(*        and the Step-6d absorption  tau1^(1/2)|_i phi1^(1/2)|_j <= C;      *)
(*   [4]  the scalar chain behind H:porosity / lem:winv:                     *)
(*        alpha_inf <= (1+C_alpha) alpha_0 and                               *)
(*        alpha_inf / alpha_0^(1/2) <= delta^(1/2) alpha_inf^(1/2);          *)
(*   [5]  the finite Cauchy--Schwarz inequalities used to assemble the       *)
(*        elementwise estimates (Steps 1, 8) and the Step-9 aggregation      *)
(*        lemma (five squared pieces, each dominated by multiples of two    *)
(*        norms A and B, give a triple-norm bound by A + B).                 *)
(*                                                                           *)
(*  The genuinely functional-analytic ingredients (elementwise inverse       *)
(*  estimates, integration by parts, facewise assembly, interpolation        *)
(*  theory) are outside the reach of the bare standard library and are       *)
(*  audited by hand instead; see the README scope ledger.                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  General helpers.                                                         *)
(* ========================================================================= *)

Lemma div_nonneg : forall x y : R, 0 <= x -> 0 < y -> 0 <= x / y.
Proof.
  intros x y Hx Hy. unfold Rdiv.
  apply Rmult_le_pos; [exact Hx | left; apply Rinv_0_lt_compat; exact Hy].
Qed.

(*  Compare nonnegative reals through their squares. *)
Lemma nonneg_le_of_sqr :
  forall x y : R, 0 <= x -> 0 <= y -> x^2 <= y^2 -> x <= y.
Proof.
  intros x y Hx Hy Hs.
  destruct (Rle_or_lt x y) as [H | H]; [exact H |].
  exfalso.
  assert (Hxy : 0 < x + y) by lra.
  assert (Hp : 0 < (x - y) * (x + y)) by nra.
  nra.
Qed.

Lemma nonneg_eq_of_sqr :
  forall x y : R, 0 <= x -> 0 <= y -> x * x = y * y -> x = y.
Proof.
  intros x y Hx Hy He.
  apply Rsqr_inj; assumption.
Qed.

(*  0 <= x <= 1  ==>  x <= sqrt x   (used for  phi1 tau1 <= (phi1 tau1)^(1/2)). *)
Lemma x_le_sqrt : forall x : R, 0 <= x <= 1 -> x <= sqrt x.
Proof.
  intros x [H0 H1].
  apply nonneg_le_of_sqr; [exact H0 | apply sqrt_pos |].
  replace ((sqrt x)^2) with (sqrt x * sqrt x) by ring.
  rewrite sqrt_sqrt by exact H0.
  nra.
Qed.

(*  0 <= x <= M  ==>  x <= sqrt M * sqrt x   (used in eq:jumpsplit). *)
Lemma prod_sqrt_bound :
  forall x M : R, 0 <= x -> x <= M -> x <= sqrt M * sqrt x.
Proof.
  intros x M Hx HM.
  apply nonneg_le_of_sqr; [exact Hx | |].
  - apply Rmult_le_pos; apply sqrt_pos.
  - replace ((sqrt M * sqrt x)^2)
      with ((sqrt M * sqrt M) * (sqrt x * sqrt x)) by ring.
    rewrite !sqrt_sqrt by lra.
    nra.
Qed.

(* ========================================================================= *)
(*  [1] + [2]  Single-element parameter inequalities and step scalar cores.  *)
(*  Notation as in the appendix (eq:taus, eq:phi1, eq:sigmatilde):           *)
(*    tauNSinv = c1 nu/h^2 + c2 |a|/h,   phi1 = alpha_K tauNSinv,            *)
(*    tau1 = 1/(phi1 + sigma),           tau2 = h^2 tauNSinv/(c1 alpha_K),   *)
(*    sigt = sigma phi1/(phi1 + sigma).                                      *)
(*  Note sigma >= 0 (the appendix allows sigma = 0).                         *)
(* ========================================================================= *)

Section ContinuityParameters.

Variables (nu h alphaK sigma amag c1 c2 : R).
Hypothesis nu_pos       : 0 < nu.
Hypothesis h_pos        : 0 < h.
Hypothesis alphaK_pos   : 0 < alphaK.
Hypothesis alphaK_le_1  : alphaK <= 1.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis amag_nonneg  : 0 <= amag.
Hypothesis c1_pos       : 0 < c1.
Hypothesis c2_pos       : 0 < c2.

Definition tauNSinv : R := c1 * nu / h^2 + c2 * amag / h.
Definition phi1     : R := alphaK * tauNSinv.
Definition tau1     : R := / (phi1 + sigma).
Definition tau2     : R := h^2 * tauNSinv / (c1 * alphaK).
Definition sigt     : R := sigma * phi1 / (phi1 + sigma).

(* ---------- Positivity ---------------------------------------------------- *)

Lemma h2_pos : 0 < h^2.
Proof. apply pow_lt; exact h_pos. Qed.

Lemma tauNSinv_pos : 0 < tauNSinv.
Proof.
  unfold tauNSinv.
  assert (H1 : 0 < c1 * nu / h^2)
    by (apply Rdiv_lt_0_compat; [nra | exact h2_pos]).
  assert (H2 : 0 <= c2 * amag / h) by (apply div_nonneg; nra).
  lra.
Qed.

Lemma phi1_pos : 0 < phi1.
Proof. unfold phi1. pose proof tauNSinv_pos. nra. Qed.

Lemma denom_pos : 0 < phi1 + sigma.
Proof. pose proof phi1_pos. lra. Qed.

Lemma tau1_pos : 0 < tau1.
Proof. unfold tau1. apply Rinv_0_lt_compat, denom_pos. Qed.

Lemma tau2_pos : 0 < tau2.
Proof.
  unfold tau2. apply Rdiv_lt_0_compat.
  - pose proof tauNSinv_pos. pose proof h2_pos. nra.
  - nra.
Qed.

Lemma sigt_nonneg : 0 <= sigt.
Proof.
  unfold sigt. apply div_nonneg; [| exact denom_pos].
  pose proof phi1_pos. nra.
Qed.

(* ---------- Identities (eq:sigmatilde and the Step-6 workhorse) ----------- *)

Theorem sigt_id_tau : sigt = sigma * phi1 * tau1.
Proof. unfold sigt, tau1. pose proof denom_pos. field. lra. Qed.

Theorem sigt_id_sub : sigt = sigma - sigma^2 * tau1.
Proof. unfold sigt, tau1. pose proof denom_pos. field. lra. Qed.

(*  1 - sigma tau1 = phi1 tau1   (used in Steps 6b and 6c). *)
Theorem one_minus_sigma_tau1 : 1 - sigma * tau1 = phi1 * tau1.
Proof. unfold tau1. pose proof denom_pos. field. lra. Qed.

(* ---------- (P1) ----------------------------------------------------------- *)

Theorem P1_sigma_tau1 : sigma * tau1 <= 1.
Proof.
  pose proof denom_pos as HD. pose proof phi1_pos as HP.
  apply (Rmult_le_reg_r (phi1 + sigma)); [lra |].
  replace (sigma * tau1 * (phi1 + sigma)) with sigma
    by (unfold tau1; field; lra).
  lra.
Qed.

Theorem P1_phi1_tau1 : phi1 * tau1 <= 1.
Proof.
  pose proof denom_pos as HD.
  apply (Rmult_le_reg_r (phi1 + sigma)); [lra |].
  replace (phi1 * tau1 * (phi1 + sigma)) with phi1
    by (unfold tau1; field; lra).
  lra.
Qed.

Theorem P1_tau1_le_inv_phi1 : tau1 <= / phi1.
Proof.
  unfold tau1.
  apply Rinv_le_contravar; [exact phi1_pos | lra].
Qed.

(* ---------- (P2) ----------------------------------------------------------- *)

Theorem P2_visc : c1 * nu * alphaK <= phi1 * h^2.
Proof.
  replace (phi1 * h^2)
    with (alphaK * (c1 * nu) + alphaK * (c2 * amag * h))
    by (unfold phi1, tauNSinv; field; lra).
  assert (H1 : 0 <= c2 * amag) by nra.
  assert (H2 : 0 <= alphaK * (c2 * amag)) by nra.
  assert (H3 : 0 <= alphaK * (c2 * amag) * h) by nra.
  nra.
Qed.

Theorem P2_conv : c2 * alphaK * amag * h <= phi1 * h^2.
Proof.
  replace (phi1 * h^2)
    with (alphaK * (c1 * nu) + alphaK * (c2 * amag * h))
    by (unfold phi1, tauNSinv; field; lra).
  assert (H0 : 0 < c1 * nu) by nra.
  assert (H1 : 0 < alphaK * (c1 * nu)) by nra.
  nra.
Qed.

(*  The two divided forms used in Steps 4, 5 and 9. *)
Lemma visc_le_phi1 : c1 * nu * alphaK / h^2 <= phi1.
Proof.
  pose proof h2_pos as Hh.
  apply (Rmult_le_reg_r (h^2)); [exact Hh |].
  replace (c1 * nu * alphaK / h^2 * h^2) with (c1 * nu * alphaK)
    by (field; lra).
  exact P2_visc.
Qed.

Lemma conv_le_phi1 : c2 * alphaK * amag / h <= phi1.
Proof.
  pose proof h2_pos as Hh.
  apply (Rmult_le_reg_r (h^2)); [exact Hh |].
  replace (c2 * alphaK * amag / h * h^2) with (c2 * alphaK * amag * h)
    by (field; lra).
  exact P2_conv.
Qed.

(*  nu tau1 alpha_K <= h^2/c1   (the coefficient behind eq:keyvisc). *)
Theorem P2_key : nu * tau1 * alphaK <= h^2 / c1.
Proof.
  pose proof denom_pos as HD.
  apply (Rmult_le_reg_r (c1 * (phi1 + sigma))); [nra |].
  replace (nu * tau1 * alphaK * (c1 * (phi1 + sigma)))
    with (c1 * nu * alphaK) by (unfold tau1; field; lra).
  replace (h^2 / c1 * (c1 * (phi1 + sigma)))
    with (h^2 * (phi1 + sigma)) by (field; lra).
  pose proof P2_visc as HP.
  pose proof h2_pos as Hh.
  assert (H1 : 0 <= h^2 * sigma) by nra.
  nra.
Qed.

(* ---------- (P3) ----------------------------------------------------------- *)

Theorem P3_id : phi1 * h^2 = c1 * alphaK^2 * tau2.
Proof. unfold phi1, tau2. field. lra. Qed.

Theorem P3_sqrt : sqrt phi1 * h = sqrt c1 * alphaK * sqrt tau2.
Proof.
  pose proof phi1_pos as HP. pose proof tau2_pos as HT.
  apply nonneg_eq_of_sqr.
  - assert (0 <= sqrt phi1) by apply sqrt_pos. nra.
  - assert (H1 : 0 <= sqrt c1) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt tau2) by apply sqrt_pos.
    assert (H3 : 0 <= sqrt c1 * alphaK) by nra.
    nra.
  - replace ((sqrt phi1 * h) * (sqrt phi1 * h))
      with ((sqrt phi1 * sqrt phi1) * h^2) by ring.
    replace ((sqrt c1 * alphaK * sqrt tau2) * (sqrt c1 * alphaK * sqrt tau2))
      with ((sqrt c1 * sqrt c1) * alphaK^2 * (sqrt tau2 * sqrt tau2)) by ring.
    rewrite !sqrt_sqrt by lra.
    exact P3_id.
Qed.

Corollary P3_sqrt_le : sqrt phi1 * h <= sqrt c1 * sqrt tau2.
Proof.
  rewrite P3_sqrt.
  assert (H1 : 0 <= sqrt c1) by apply sqrt_pos.
  assert (H2 : 0 <= sqrt tau2) by apply sqrt_pos.
  assert (H3 : 0 <= sqrt c1 * sqrt tau2) by nra.
  nra.
Qed.

(* ---------- (P4) ----------------------------------------------------------- *)

Theorem P4_le_sigma : sigt <= sigma.
Proof.
  pose proof denom_pos as HD.
  apply (Rmult_le_reg_r (phi1 + sigma)); [lra |].
  replace (sigt * (phi1 + sigma)) with (sigma * phi1)
    by (unfold sigt; field; lra).
  nra.
Qed.

Theorem P4_le_phi1 : sigt <= phi1.
Proof.
  pose proof denom_pos as HD. pose proof phi1_pos as HP.
  apply (Rmult_le_reg_r (phi1 + sigma)); [lra |].
  replace (sigt * (phi1 + sigma)) with (sigma * phi1)
    by (unfold sigt; field; lra).
  nra.
Qed.

Theorem P4_sigt_tau1 : sigt * tau1 <= 1.
Proof.
  pose proof denom_pos as HD. pose proof phi1_pos as HP.
  apply (Rmult_le_reg_r ((phi1 + sigma) * (phi1 + sigma))); [nra |].
  replace (sigt * tau1 * ((phi1 + sigma) * (phi1 + sigma)))
    with (sigma * phi1) by (unfold sigt, tau1; field; lra).
  assert (H1 : 0 <= phi1 * phi1) by nra.
  assert (H2 : 0 <= sigma * sigma) by nra.
  assert (H3 : 0 <= phi1 * sigma) by nra.
  nra.
Qed.

(* ---------- (P5): the epsilon condition eq:epscond ------------------------ *)

Section EpsilonBound.

Variables (eps C2 : R).
Hypothesis eps_nonneg : 0 <= eps.
Hypothesis C2_lo      : 0 <= C2.
Hypothesis C2_hi      : C2 < 1.
Hypothesis eps_le     : eps <= C2 * c1 * alphaK^2 * tau1 / h^2.

Theorem P5_eps_tau2_phi : eps * tau2 <= C2 * (phi1 * tau1).
Proof.
  pose proof tau2_pos as HT.
  assert (Hm : eps * tau2 <= (C2 * c1 * alphaK^2 * tau1 / h^2) * tau2) by nra.
  replace ((C2 * c1 * alphaK^2 * tau1 / h^2) * tau2)
    with (C2 * (phi1 * tau1)) in Hm.
  - exact Hm.
  - unfold phi1, tau2, tauNSinv. field. lra.
Qed.

Corollary P5_eps_tau2_C2 : eps * tau2 <= C2.
Proof.
  pose proof P5_eps_tau2_phi as H.
  pose proof P1_phi1_tau1 as HP.
  nra.
Qed.

Corollary P5_eps_tau2_lt1 : eps * tau2 < 1.
Proof. pose proof P5_eps_tau2_C2. lra. Qed.

Theorem P5_eps_h2 : eps * h^2 <= C2 * c1 * alphaK^2 * tau1.
Proof.
  pose proof h2_pos as Hh.
  assert (Hm : eps * h^2 <= (C2 * c1 * alphaK^2 * tau1 / h^2) * h^2) by nra.
  replace ((C2 * c1 * alphaK^2 * tau1 / h^2) * h^2)
    with (C2 * c1 * alphaK^2 * tau1) in Hm by (field; lra).
  exact Hm.
Qed.

(*  eps^(1/2) h <= c1^(1/2) alpha_K tau1^(1/2)  (the eq:absorb3 coefficient). *)
Theorem P5_sqrt : sqrt eps * h <= sqrt c1 * alphaK * sqrt tau1.
Proof.
  pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (0 <= sqrt eps) by apply sqrt_pos. nra.
  - assert (H1 : 0 <= sqrt c1) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt tau1) by apply sqrt_pos.
    assert (H3 : 0 <= sqrt c1 * alphaK) by nra.
    nra.
  - replace ((sqrt eps * h)^2) with ((sqrt eps * sqrt eps) * h^2) by ring.
    replace ((sqrt c1 * alphaK * sqrt tau1)^2)
      with ((sqrt c1 * sqrt c1) * alphaK^2 * (sqrt tau1 * sqrt tau1)) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P5_eps_h2 as H5.
    assert (Ha2 : 0 < alphaK^2) by nra.
    assert (Hca : 0 < c1 * alphaK^2) by nra.
    assert (Hq : 0 <= c1 * alphaK^2 * tau1) by nra.
    nra.
Qed.

(*  Step 7 scalar:  eps tau2^(1/2) <= C2^(1/2) eps^(1/2). *)
Theorem step7_scalar : eps * sqrt tau2 <= sqrt C2 * sqrt eps.
Proof.
  pose proof tau2_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (0 <= sqrt tau2) by apply sqrt_pos. nra.
  - assert (H1 : 0 <= sqrt C2) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt eps) by apply sqrt_pos.
    nra.
  - replace ((eps * sqrt tau2)^2)
      with (eps^2 * (sqrt tau2 * sqrt tau2)) by ring.
    replace ((sqrt C2 * sqrt eps)^2)
      with ((sqrt C2 * sqrt C2) * (sqrt eps * sqrt eps)) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P5_eps_tau2_C2 as HC.
    nra.
Qed.

(* ---------- Analysis tau2 vs implemented tau2 (eq:Tau2Final vs eq:Tau2) --- *)
(*  The implemented parameter keeps the + eps h^2 term:                      *)
(*    tau2_impl = h^2 / (c1 alpha_K tau_NS + eps h^2)                        *)
(*             = h^2 tauNSinv / (c1 alpha_K + eps h^2 tauNSinv).             *)
(*  Under eq:epscond the two are equivalent up to the factor 1 + C2:         *)
(*    tau2_impl <= tau2 <= (1 + C2) tau2_impl,                               *)
(*  so every estimate of the analysis transfers to the implemented           *)
(*  parameter with constants inflated by at most 1 + C2 < 2.                 *)

Definition tau2impl : R :=
  h^2 * tauNSinv / (c1 * alphaK + eps * h^2 * tauNSinv).

Lemma tau2impl_denom_pos : 0 < c1 * alphaK + eps * h^2 * tauNSinv.
Proof.
  pose proof tauNSinv_pos as HT. pose proof h2_pos as Hh.
  assert (H1 : 0 < c1 * alphaK) by nra.
  assert (H2 : 0 <= eps * h^2) by nra.
  assert (H3 : 0 <= eps * h^2 * tauNSinv) by nra.
  lra.
Qed.

Theorem tau2impl_le_tau2 : tau2impl <= tau2.
Proof.
  pose proof tauNSinv_pos as HT. pose proof h2_pos as Hh.
  pose proof tau2impl_denom_pos as HD.
  assert (Hnum : 0 < h^2 * tauNSinv) by nra.
  assert (Hca : 0 < c1 * alphaK) by nra.
  unfold tau2impl, tau2.
  apply (Rmult_le_reg_r ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)));
    [nra |].
  replace (h^2 * tauNSinv / (c1 * alphaK + eps * h^2 * tauNSinv)
           * ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)))
    with (h^2 * tauNSinv * (c1 * alphaK)) by (field; lra).
  replace (h^2 * tauNSinv / (c1 * alphaK)
           * ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)))
    with (h^2 * tauNSinv * (c1 * alphaK + eps * h^2 * tauNSinv))
    by (field; lra).
  assert (H2 : 0 <= eps * h^2) by nra.
  assert (H3 : 0 <= eps * h^2 * tauNSinv) by nra.
  nra.
Qed.

Theorem tau2_le_scaled_impl : tau2 <= (1 + C2) * tau2impl.
Proof.
  pose proof tauNSinv_pos as HT. pose proof h2_pos as Hh.
  pose proof tau2impl_denom_pos as HD.
  pose proof tau1_pos as Ht1.
  assert (Hca : 0 < c1 * alphaK) by nra.
  (*  Key step:  eps h^2 tauNSinv <= C2 c1 alpha_K,  from eq:epscond and     *)
  (*  phi1 tau1 <= 1:                                                        *)
  (*    eps h^2 tauNSinv <= C2 c1 alpha_K^2 tau1 tauNSinv                    *)
  (*                      = C2 c1 alpha_K (phi1 tau1) <= C2 c1 alpha_K.      *)
  assert (Hkey : eps * h^2 * tauNSinv <= C2 * c1 * alphaK).
  { pose proof P5_eps_h2 as H5.
    assert (S1 : eps * h^2 * tauNSinv <= C2 * c1 * alphaK^2 * tau1 * tauNSinv)
      by nra.
    assert (Hid : C2 * c1 * alphaK^2 * tau1 * tauNSinv
                  = C2 * c1 * alphaK * (phi1 * tau1))
      by (unfold phi1; ring).
    pose proof P1_phi1_tau1 as HP.
    assert (Hcc : 0 <= C2 * c1 * alphaK) by nra.
    nra. }
  unfold tau2impl, tau2.
  apply (Rmult_le_reg_r ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)));
    [nra |].
  replace (h^2 * tauNSinv / (c1 * alphaK)
           * ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)))
    with (h^2 * tauNSinv * (c1 * alphaK + eps * h^2 * tauNSinv))
    by (field; lra).
  replace ((1 + C2) * (h^2 * tauNSinv / (c1 * alphaK + eps * h^2 * tauNSinv))
           * ((c1 * alphaK + eps * h^2 * tauNSinv) * (c1 * alphaK)))
    with ((1 + C2) * (h^2 * tauNSinv) * (c1 * alphaK)) by (field; lra).
  assert (Hnum : 0 < h^2 * tauNSinv) by nra.
  nra.
Qed.

End EpsilonBound.

(* ---------- Step 4: eq:keyvisc and the T8 chain --------------------------- *)

(*  Squared form of the eq:keyvisc coefficient. *)
Theorem keyvisc_sq : nu * tau1 * alphaK * c1 <= h^2.
Proof.
  pose proof c1_pos as Hc.
  apply (Rmult_le_reg_r (/ c1)); [apply Rinv_0_lt_compat; exact Hc |].
  replace (nu * tau1 * alphaK * c1 * / c1) with (nu * tau1 * alphaK)
    by (field; lra).
  replace (h^2 * / c1) with (h^2 / c1) by (unfold Rdiv; ring).
  exact P2_key.
Qed.

(*  nu^(1/2) tau1^(1/2) alpha_K^(1/2) c1^(1/2) <= h. *)
Theorem keyvisc_sqrt :
  sqrt nu * sqrt tau1 * sqrt alphaK * sqrt c1 <= h.
Proof.
  pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (H1 : 0 <= sqrt nu * sqrt tau1)
      by (apply Rmult_le_pos; apply sqrt_pos).
    assert (H2 : 0 <= sqrt alphaK * sqrt c1)
      by (apply Rmult_le_pos; apply sqrt_pos).
    nra.
  - lra.
  - replace ((sqrt nu * sqrt tau1 * sqrt alphaK * sqrt c1)^2)
      with ((sqrt nu * sqrt nu) * (sqrt tau1 * sqrt tau1)
            * (sqrt alphaK * sqrt alphaK) * (sqrt c1 * sqrt c1)) by ring.
    rewrite !sqrt_sqrt by lra.
    replace (h^2) with (h * h) by ring.
    pose proof keyvisc_sq as HK.
    nra.
Qed.

(*  T8 chain:  sigma (c1 nu alpha_K / h^2) tau1 <= sigmatilde. *)
Theorem T8_chain : sigma * (c1 * nu * alphaK / h^2) * tau1 <= sigt.
Proof.
  rewrite sigt_id_tau.
  pose proof visc_le_phi1 as HX.
  pose proof tau1_pos as HT.
  assert (Hst : 0 <= sigma * tau1) by nra.
  nra.
Qed.

(* ---------- Step 5: the T13 chain ------------------------------------------ *)

Theorem T13_chain : sigma * tau1 * (c2 * alphaK * amag / h) <= sigt.
Proof.
  rewrite sigt_id_tau.
  pose proof conv_le_phi1 as HX.
  pose proof tau1_pos as HT.
  assert (Hst : 0 <= sigma * tau1) by nra.
  nra.
Qed.

(* ---------- Step 6b: eq:volpart --------------------------------------------- *)

(*  phi1 tau1 <= phi1^(1/2) tau1^(1/2)   (from phi1 tau1 <= 1). *)
Theorem volpart_scalar : phi1 * tau1 <= sqrt phi1 * sqrt tau1.
Proof.
  pose proof phi1_pos as HP. pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - nra.
  - assert (H1 : 0 <= sqrt phi1) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt tau1) by apply sqrt_pos.
    nra.
  - replace ((sqrt phi1 * sqrt tau1)^2)
      with ((sqrt phi1 * sqrt phi1) * (sqrt tau1 * sqrt tau1)) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P1_phi1_tau1 as H1.
    assert (H0 : 0 <= phi1 * tau1) by nra.
    nra.
Qed.

(* ---------- Step 6c ---------------------------------------------------------- *)

(*  phi1^(1/2) tau1 <= tau1^(1/2). *)
Theorem sqrtphi_tau_le : sqrt phi1 * tau1 <= sqrt tau1.
Proof.
  pose proof phi1_pos as HP. pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (0 <= sqrt phi1) by apply sqrt_pos. nra.
  - apply sqrt_pos.
  - replace ((sqrt phi1 * tau1)^2)
      with ((sqrt phi1 * sqrt phi1) * tau1^2) by ring.
    replace ((sqrt tau1)^2) with (sqrt tau1 * sqrt tau1) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P1_phi1_tau1 as H1.
    nra.
Qed.

(*  phi1 tau1^(1/2) <= phi1^(1/2)   (the eq:absorb4 velocity chain, last step). *)
Theorem phi_sqrttau_le : phi1 * sqrt tau1 <= sqrt phi1.
Proof.
  pose proof phi1_pos as HP. pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (0 <= sqrt tau1) by apply sqrt_pos. nra.
  - apply sqrt_pos.
  - replace ((phi1 * sqrt tau1)^2)
      with (phi1^2 * (sqrt tau1 * sqrt tau1)) by ring.
    replace ((sqrt phi1)^2) with (sqrt phi1 * sqrt phi1) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P1_phi1_tau1 as H1.
    nra.
Qed.

(*  phi1^(1/2) tau1^(1/2) <= 1   (used in Step 6d). *)
Theorem sqrtphi_sqrttau_le_1 : sqrt phi1 * sqrt tau1 <= 1.
Proof.
  pose proof phi1_pos as HP. pose proof tau1_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (H1 : 0 <= sqrt phi1) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt tau1) by apply sqrt_pos.
    nra.
  - lra.
  - replace ((sqrt phi1 * sqrt tau1)^2)
      with ((sqrt phi1 * sqrt phi1) * (sqrt tau1 * sqrt tau1)) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof P1_phi1_tau1. lra.
Qed.

(* ---------- Step 9: the eq:absorb1--eq:absorb5 coefficient chains ------------ *)

(*  eq:absorb1 core:  nu <= alpha_K tau2   (equivalently                     *)
(*  nu^(1/2) alpha_K^(1/2) (Cinv/h) <= Cinv alpha_K tau2^(1/2) / h).         *)
Theorem absorb1_core : nu <= alphaK * tau2.
Proof.
  replace (alphaK * tau2) with (nu + c2 * amag * h / c1)
    by (unfold tau2, tauNSinv; field; lra).
  assert (H0 : 0 <= c2 * amag) by nra.
  assert (H1 : 0 <= c2 * amag * h) by nra.
  assert (H : 0 <= c2 * amag * h / c1)
    by (apply div_nonneg; [exact H1 | exact c1_pos]).
  lra.
Qed.

Theorem absorb1_sqrt : sqrt nu * sqrt alphaK <= alphaK * sqrt tau2.
Proof.
  pose proof tau2_pos as HT.
  apply nonneg_le_of_sqr.
  - assert (H1 : 0 <= sqrt nu) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt alphaK) by apply sqrt_pos.
    nra.
  - assert (0 <= sqrt tau2) by apply sqrt_pos. nra.
  - replace ((sqrt nu * sqrt alphaK)^2)
      with ((sqrt nu * sqrt nu) * (sqrt alphaK * sqrt alphaK)) by ring.
    replace ((alphaK * sqrt tau2)^2)
      with (alphaK^2 * (sqrt tau2 * sqrt tau2)) by ring.
    rewrite !sqrt_sqrt by lra.
    pose proof absorb1_core as HA.
    nra.
Qed.

(*  eq:absorb2:  sigmatilde^(1/2) <= phi1^(1/2), then with (P3). *)
Theorem absorb2_sqrt : sqrt sigt <= sqrt phi1.
Proof. apply sqrt_le_1_alt, P4_le_phi1. Qed.

Corollary absorb2_full : sqrt sigt * h <= sqrt c1 * alphaK * sqrt tau2.
Proof.
  rewrite <- P3_sqrt.
  pose proof absorb2_sqrt as HA.
  nra.
Qed.

(*  eq:absorb4, velocity part:  tau1^(1/2) (c2 alpha_K |a| / h) <= phi1^(1/2). *)
Theorem absorb4_vel : sqrt tau1 * (c2 * alphaK * amag / h) <= sqrt phi1.
Proof.
  pose proof conv_le_phi1 as HX.
  pose proof phi_sqrttau_le as HP.
  assert (Hs : 0 <= sqrt tau1) by apply sqrt_pos.
  assert (H1 : sqrt tau1 * (c2 * alphaK * amag / h) <= sqrt tau1 * phi1)
    by nra.
  nra.
Qed.

Corollary absorb4_full :
  sqrt tau1 * (c2 * alphaK * amag / h) * h <= sqrt c1 * alphaK * sqrt tau2.
Proof.
  rewrite <- P3_sqrt.
  pose proof absorb4_vel as HA.
  nra.
Qed.

End ContinuityParameters.

(* ========================================================================= *)
(*  [3]  lem:jump: the jump of tau1 across an interior face, with explicit   *)
(*       constants.  phiA, phiB are the values of phi1 on the two elements   *)
(*       sharing the face; H:jump reads  cJ phiA <= phiB <= cJ' phiA.        *)
(* ========================================================================= *)

Section JumpLemma.

Variables (phiA phiB sigma cJ cJ' : R).
Hypothesis phiA_pos     : 0 < phiA.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis cJ_pos       : 0 < cJ.
Hypothesis cJ_le_1      : cJ <= 1.
Hypothesis one_le_cJ'   : 1 <= cJ'.
Hypothesis jump_low     : cJ * phiA <= phiB.
Hypothesis jump_high    : phiB <= cJ' * phiA.

Definition tA  : R := / (phiA + sigma).
Definition tB  : R := / (phiB + sigma).
Definition sgA : R := sigma * phiA / (phiA + sigma).
Definition sgB : R := sigma * phiB / (phiB + sigma).
Definition CJ  : R := Rmax (cJ' - 1) (1 - cJ).

Lemma phiB_pos : 0 < phiB.
Proof. nra. Qed.

Lemma DA_pos : 0 < phiA + sigma.
Proof. lra. Qed.

Lemma DB_pos : 0 < phiB + sigma.
Proof. pose proof phiB_pos. lra. Qed.

Lemma tA_pos : 0 < tA.
Proof. unfold tA. apply Rinv_0_lt_compat, DA_pos. Qed.

Lemma tB_pos : 0 < tB.
Proof. unfold tB. apply Rinv_0_lt_compat, DB_pos. Qed.

Lemma sgA_nonneg : 0 <= sgA.
Proof.
  unfold sgA. apply div_nonneg; [nra | exact DA_pos].
Qed.

Lemma sgB_nonneg : 0 <= sgB.
Proof.
  unfold sgB. pose proof phiB_pos.
  apply div_nonneg; [nra | exact DB_pos].
Qed.

Lemma CJ_nonneg : 0 <= CJ.
Proof.
  pose proof (Rmax_l (cJ' - 1) (1 - cJ)) as H. fold CJ in H. lra.
Qed.

(* ---------- The jump formula and the increment bound ---------------------- *)

Theorem jump_formula :
  tA - tB = (phiB - phiA) / ((phiA + sigma) * (phiB + sigma)).
Proof.
  unfold tA, tB. pose proof DA_pos. pose proof DB_pos.
  field. split; lra.
Qed.

Theorem phi_diff_bound : Rabs (phiB - phiA) <= CJ * phiA.
Proof.
  apply Rabs_le. split.
  - pose proof (Rmax_r (cJ' - 1) (1 - cJ)) as HM. fold CJ in HM.
    nra.
  - pose proof (Rmax_l (cJ' - 1) (1 - cJ)) as HM. fold CJ in HM.
    nra.
Qed.

(* ---------- Denominator and elementwise comparabilities ------------------- *)

Lemma DB_lower : cJ * (phiA + sigma) <= phiB + sigma.
Proof. nra. Qed.

Lemma DB_upper : phiB + sigma <= cJ' * (phiA + sigma).
Proof. nra. Qed.

(*  tau1 comparability across the face:  tB <= (1/cJ) tA,  tA <= cJ' tB. *)
Lemma inv_DB_le : / (phiB + sigma) <= / cJ * / (phiA + sigma).
Proof.
  pose proof DA_pos as HA. pose proof DB_pos as HB.
  pose proof DB_lower as HL.
  assert (H : / (phiB + sigma) <= / (cJ * (phiA + sigma)))
    by (apply Rinv_le_contravar; [nra | exact HL]).
  rewrite Rinv_mult in H. exact H.
Qed.

Lemma inv_DA_le : / (phiA + sigma) <= cJ' * / (phiB + sigma).
Proof.
  pose proof DA_pos as HA. pose proof DB_pos as HB.
  pose proof DB_upper as HU.
  assert (H : / (cJ' * (phiA + sigma)) <= / (phiB + sigma))
    by (apply Rinv_le_contravar; [exact HB | exact HU]).
  rewrite Rinv_mult in H.
  assert (HicJ' : 0 < / cJ') by (apply Rinv_0_lt_compat; lra).
  assert (HiA : 0 < / (phiA + sigma)) by (apply Rinv_0_lt_compat; lra).
  (* From (1/cJ') iA <= iB, multiply by cJ' > 0. *)
  assert (Hm : cJ' * (/ cJ' * / (phiA + sigma)) <= cJ' * / (phiB + sigma))
    by nra.
  replace (cJ' * (/ cJ' * / (phiA + sigma))) with (/ (phiA + sigma)) in Hm
    by (field; lra).
  exact Hm.
Qed.

Theorem tau_comp_BA : tB <= / cJ * tA.
Proof. unfold tA, tB. apply inv_DB_le. Qed.

Theorem tau_comp_AB : tA <= cJ' * tB.
Proof. unfold tA, tB. apply inv_DA_le. Qed.

(*  phi tau <= 1 on each element (local (P1)). *)
Lemma phiA_tA_le_1 : phiA * tA <= 1.
Proof.
  pose proof DA_pos as HD.
  apply (Rmult_le_reg_r (phiA + sigma)); [lra |].
  replace (phiA * tA * (phiA + sigma)) with phiA by (unfold tA; field; lra).
  lra.
Qed.

Lemma sgA_tA_le_1 : sgA * tA <= 1.
Proof.
  pose proof DA_pos as HD.
  apply (Rmult_le_reg_r ((phiA + sigma) * (phiA + sigma))); [nra |].
  replace (sgA * tA * ((phiA + sigma) * (phiA + sigma)))
    with (sigma * phiA) by (unfold sgA, tA; field; lra).
  assert (H1 : 0 <= phiA * phiA) by nra.
  assert (H2 : 0 <= sigma * sigma) by nra.
  assert (H3 : 0 <= phiA * sigma) by nra.
  nra.
Qed.

(* ---------- eq:jumpest with the explicit constant CJ/cJ -------------------- *)

Theorem jump_bound_A :
  sigma * Rabs (tA - tB) <= (CJ / cJ) * (sgA * tA).
Proof.
  rewrite jump_formula.
  unfold Rdiv.
  rewrite Rabs_mult.
  rewrite (Rabs_right (/ ((phiA + sigma) * (phiB + sigma)))).
  2:{ apply Rle_ge. left. apply Rinv_0_lt_compat.
      pose proof DA_pos. pose proof DB_pos. nra. }
  rewrite Rinv_mult.
  pose proof phi_diff_bound as HD.
  pose proof (Rabs_pos (phiB - phiA)) as HD0.
  pose proof DA_pos as HDA. pose proof DB_pos as HDB.
  pose proof CJ_nonneg as HCJ.
  assert (HiA : 0 < / (phiA + sigma)) by (apply Rinv_0_lt_compat; lra).
  assert (HiB : 0 < / (phiB + sigma)) by (apply Rinv_0_lt_compat; lra).
  pose proof inv_DB_le as HinvB.
  replace (sgA * tA)
    with (sigma * phiA * / (phiA + sigma) * / (phiA + sigma)).
  2:{ unfold sgA, tA. field. lra. }
  assert (S1 : Rabs (phiB - phiA) * / (phiB + sigma)
               <= CJ * phiA * / (phiB + sigma)) by nra.
  assert (HCp : 0 <= CJ * phiA) by nra.
  assert (S2 : CJ * phiA * / (phiB + sigma)
               <= CJ * phiA * (/ cJ * / (phiA + sigma))) by nra.
  assert (S12 : Rabs (phiB - phiA) * / (phiB + sigma)
                <= CJ * phiA * / cJ * / (phiA + sigma)) by lra.
  assert (S3 : 0 <= sigma * / (phiA + sigma)) by nra.
  nra.
Qed.

(*  sigmatilde comparability across the face. *)
Theorem sg_comp_A_le_B : sgA <= (cJ' / cJ) * sgB.
Proof.
  pose proof DA_pos as HDA. pose proof DB_pos as HDB.
  pose proof phiB_pos as HPB.
  assert (HiB : 0 < / (phiB + sigma)) by (apply Rinv_0_lt_compat; lra).
  assert (HphiA : phiA <= / cJ * phiB).
  { apply (Rmult_le_reg_l cJ); [lra |].
    replace (cJ * (/ cJ * phiB)) with phiB by (field; lra).
    exact jump_low. }
  pose proof inv_DA_le as HIA.
  (* sgA = sigma phiA / DA <= sigma (phiB/cJ) (cJ'/DB) *)
  unfold sgA, sgB, Rdiv.
  assert (H1 : 0 <= sigma * phiA) by nra.
  assert (S1 : sigma * phiA * / (phiA + sigma)
               <= sigma * phiA * (cJ' * / (phiB + sigma))) by nra.
  assert (Hsg : 0 <= sigma * (cJ' * / (phiB + sigma))).
  { assert (0 < cJ' * / (phiB + sigma)) by nra. nra. }
  assert (S2 : sigma * phiA * (cJ' * / (phiB + sigma))
               <= sigma * (/ cJ * phiB) * (cJ' * / (phiB + sigma))).
  { assert (Hgap : 0 <= / cJ * phiB - phiA) by lra.
    nra. }
  nra.
Qed.

(*  The second form of eq:jumpsplit (indices i = A, j = B):                  *)
(*    sigma |tau_A - tau_B|                                                  *)
(*      <= (CJ cJ'/cJ) (1/cJ)^(1/2) sigmatilde_A^(1/2) tau_B^(1/2),          *)
(*  stated with the unsplit square root of the product.                      *)
Theorem jumpsplit_AB :
  sigma * Rabs (tA - tB)
  <= (CJ * cJ' / cJ) * sqrt (/ cJ) * sqrt (sgA * tB).
Proof.
  pose proof jump_bound_A as HJ.
  pose proof tau_comp_AB as HAB.
  pose proof tau_comp_BA as HBA.
  pose proof sgA_nonneg as HsA.
  pose proof sgA_tA_le_1 as H1.
  pose proof tB_pos as HtB. pose proof tA_pos as HtA.
  assert (HicJ : 0 < / cJ) by (apply Rinv_0_lt_compat; lra).
  (* sgA tA <= cJ' (sgA tB) *)
  assert (S1 : sgA * tA <= cJ' * (sgA * tB)) by nra.
  (* sgA tB <= (1/cJ) (sgA tA) <= 1/cJ *)
  assert (S2 : sgA * tB <= / cJ).
  { assert (Ht : sgA * tB <= sgA * (/ cJ * tA)) by nra.
    assert (Hm : sgA * (/ cJ * tA) <= / cJ) by nra.
    lra. }
  assert (S3 : 0 <= sgA * tB) by nra.
  (* sgA tB <= sqrt(1/cJ) sqrt(sgA tB) *)
  pose proof (prod_sqrt_bound (sgA * tB) (/ cJ) S3 S2) as S4.
  (* assemble: sigma|dt| <= (CJ/cJ)(sgA tA) <= (CJ cJ'/cJ)(sgA tB)
                        <= (CJ cJ'/cJ) sqrt(1/cJ) sqrt(sgA tB) *)
  pose proof CJ_nonneg as HCJ.
  assert (HK1 : 0 <= CJ / cJ) by (apply div_nonneg; lra).
  assert (T1 : (CJ / cJ) * (sgA * tA) <= (CJ / cJ) * (cJ' * (sgA * tB)))
    by nra.
  assert (HK2 : 0 <= CJ / cJ * cJ') by nra.
  assert (T2 : (CJ / cJ) * (cJ' * (sgA * tB))
               <= (CJ / cJ) * cJ' * (sqrt (/ cJ) * sqrt (sgA * tB))) by nra.
  assert (Hfin : sigma * Rabs (tA - tB)
                 <= CJ / cJ * cJ' * (sqrt (/ cJ) * sqrt (sgA * tB)))
    by lra.
  replace ((CJ * cJ' / cJ) * sqrt (/ cJ) * sqrt (sgA * tB))
    with (CJ / cJ * cJ' * (sqrt (/ cJ) * sqrt (sgA * tB)))
    by (unfold Rdiv; ring).
  exact Hfin.
Qed.

(* ---------- Step 6d absorption:  tau_i^(1/2) phi_j^(1/2) <= C -------------- *)

Theorem III_absorb_AB : sqrt tA * sqrt phiB <= sqrt cJ'.
Proof.
  pose proof tA_pos as HtA. pose proof phiB_pos as HPB.
  pose proof phiA_tA_le_1 as H1.
  apply nonneg_le_of_sqr.
  - assert (H2 : 0 <= sqrt tA) by apply sqrt_pos.
    assert (H3 : 0 <= sqrt phiB) by apply sqrt_pos.
    nra.
  - apply sqrt_pos.
  - replace ((sqrt tA * sqrt phiB)^2)
      with ((sqrt tA * sqrt tA) * (sqrt phiB * sqrt phiB)) by ring.
    replace ((sqrt cJ')^2) with (sqrt cJ' * sqrt cJ') by ring.
    rewrite !sqrt_sqrt by lra.
    (* tA phiB <= tA (cJ' phiA) = cJ'(phiA tA) <= cJ' *)
    assert (S1 : tA * phiB <= tA * (cJ' * phiA)) by nra.
    nra.
Qed.

Theorem III_absorb_BA : sqrt tB * sqrt phiA <= sqrt (/ cJ).
Proof.
  pose proof tB_pos as HtB. pose proof tA_pos as HtA.
  pose proof phiA_tA_le_1 as H1.
  pose proof tau_comp_BA as HBA.
  assert (HicJ : 0 < / cJ) by (apply Rinv_0_lt_compat; lra).
  apply nonneg_le_of_sqr.
  - assert (H2 : 0 <= sqrt tB) by apply sqrt_pos.
    assert (H3 : 0 <= sqrt phiA) by apply sqrt_pos.
    nra.
  - apply sqrt_pos.
  - replace ((sqrt tB * sqrt phiA)^2)
      with ((sqrt tB * sqrt tB) * (sqrt phiA * sqrt phiA)) by ring.
    replace ((sqrt (/ cJ))^2) with (sqrt (/ cJ) * sqrt (/ cJ)) by ring.
    rewrite !sqrt_sqrt by lra.
    (* tB phiA <= (tA/cJ) phiA = (1/cJ)(phiA tA) <= 1/cJ *)
    assert (S1 : tB * phiA <= / cJ * tA * phiA) by nra.
    nra.
Qed.

End JumpLemma.

(* ========================================================================= *)
(*  [4]  The scalar chain behind H:porosity and lem:winv.                    *)
(*       Given the resolved-porosity increment bound                         *)
(*       alpha_inf - alpha_0 <= C_alpha alpha_0 (from eq:resolved along a    *)
(*       segment within the convex element), the appendix derives            *)
(*       alpha_inf <= (1 + C_alpha) alpha_0 and, in the proof of             *)
(*       eq:winv-divvisc,                                                    *)
(*       alpha_inf / alpha_0^(1/2) <= delta^(1/2) alpha_inf^(1/2).           *)
(* ========================================================================= *)

Section PorosityResolution.

Variables (a0 aInf Ca : R).
Hypothesis a0_pos    : 0 < a0.
Hypothesis a_ord     : a0 <= aInf.
Hypothesis Ca_nonneg : 0 <= Ca.
Hypothesis increment : aInf - a0 <= Ca * a0.

Theorem delta_alpha_bound : aInf <= (1 + Ca) * a0.
Proof. nra. Qed.

Theorem winv_ratio : aInf / sqrt a0 <= sqrt (1 + Ca) * sqrt aInf.
Proof.
  assert (aInf_pos : 0 < aInf) by lra.
  assert (Hs0 : 0 < sqrt a0) by (apply sqrt_lt_R0; exact a0_pos).
  apply nonneg_le_of_sqr.
  - apply div_nonneg; lra.
  - assert (H1 : 0 <= sqrt (1 + Ca)) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt aInf) by apply sqrt_pos.
    nra.
  - replace ((aInf / sqrt a0)^2)
      with (aInf^2 * / (sqrt a0 * sqrt a0)) by (field; lra).
    rewrite (sqrt_sqrt a0) by lra.
    replace ((sqrt (1 + Ca) * sqrt aInf)^2)
      with ((sqrt (1 + Ca) * sqrt (1 + Ca)) * (sqrt aInf * sqrt aInf))
      by ring.
    rewrite !sqrt_sqrt by lra.
    apply (Rmult_le_reg_r a0); [exact a0_pos |].
    replace (aInf^2 * / a0 * a0) with (aInf^2) by (field; lra).
    pose proof delta_alpha_bound as HD.
    nra.
Qed.

End PorosityResolution.

(* ========================================================================= *)
(*  [5]  Finite Cauchy--Schwarz and the Step-9 aggregation.                  *)
(* ========================================================================= *)

(*  Two-term Lagrange identity and Cauchy--Schwarz. *)
Lemma lagrange2 :
  forall a b c d : R,
    (a^2 + c^2) * (b^2 + d^2) - (a*b + c*d)^2 = (a*d - c*b)^2.
Proof. intros. ring. Qed.

Theorem CS2 :
  forall a b c d : R,
    0 <= a -> 0 <= b -> 0 <= c -> 0 <= d ->
    a*b + c*d <= sqrt (a^2 + c^2) * sqrt (b^2 + d^2).
Proof.
  intros a b c d Ha Hb Hc Hd.
  apply nonneg_le_of_sqr.
  - nra.
  - assert (H1 : 0 <= sqrt (a^2 + c^2)) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt (b^2 + d^2)) by apply sqrt_pos.
    nra.
  - replace ((sqrt (a^2 + c^2) * sqrt (b^2 + d^2))^2)
      with ((sqrt (a^2 + c^2) * sqrt (a^2 + c^2))
            * (sqrt (b^2 + d^2) * sqrt (b^2 + d^2))) by ring.
    rewrite !sqrt_sqrt
      by (pose proof (pow2_ge_0 a); pose proof (pow2_ge_0 b);
          pose proof (pow2_ge_0 c); pose proof (pow2_ge_0 d); lra).
    pose proof (lagrange2 a b c d) as HL.
    pose proof (pow2_ge_0 (a*d - c*b)).
    lra.
Qed.

(*  Three-term Lagrange identity and Cauchy--Schwarz. *)
Lemma lagrange3 :
  forall a b c d e f : R,
    (a^2 + c^2 + e^2) * (b^2 + d^2 + f^2) - (a*b + c*d + e*f)^2
    = (a*d - c*b)^2 + (a*f - e*b)^2 + (c*f - e*d)^2.
Proof. intros. ring. Qed.

Theorem CS3 :
  forall a b c d e f : R,
    0 <= a -> 0 <= b -> 0 <= c -> 0 <= d -> 0 <= e -> 0 <= f ->
    a*b + c*d + e*f <= sqrt (a^2 + c^2 + e^2) * sqrt (b^2 + d^2 + f^2).
Proof.
  intros a b c d e f Ha Hb Hc Hd He Hf.
  apply nonneg_le_of_sqr.
  - nra.
  - assert (H1 : 0 <= sqrt (a^2 + c^2 + e^2)) by apply sqrt_pos.
    assert (H2 : 0 <= sqrt (b^2 + d^2 + f^2)) by apply sqrt_pos.
    nra.
  - replace ((sqrt (a^2 + c^2 + e^2) * sqrt (b^2 + d^2 + f^2))^2)
      with ((sqrt (a^2 + c^2 + e^2) * sqrt (a^2 + c^2 + e^2))
            * (sqrt (b^2 + d^2 + f^2) * sqrt (b^2 + d^2 + f^2))) by ring.
    rewrite !sqrt_sqrt
      by (pose proof (pow2_ge_0 a); pose proof (pow2_ge_0 b);
          pose proof (pow2_ge_0 c); pose proof (pow2_ge_0 d);
          pose proof (pow2_ge_0 e); pose proof (pow2_ge_0 f); lra).
    pose proof (lagrange3 a b c d e f) as HL.
    pose proof (pow2_ge_0 (a*d - c*b)).
    pose proof (pow2_ge_0 (a*f - e*b)).
    pose proof (pow2_ge_0 (c*f - e*d)).
    lra.
Qed.

(*  Step 1 (eq:easystep):  2ab + cd + ef <= 2 |||U||| |||V|||  in scalar     *)
(*  form, for the three norm pieces entering T1, T10 and T18. *)
Theorem step1_bound :
  forall a b c d e f : R,
    0 <= a -> 0 <= b -> 0 <= c -> 0 <= d -> 0 <= e -> 0 <= f ->
    2*(a*b) + c*d + e*f
    <= 2 * (sqrt (a^2 + c^2 + e^2) * sqrt (b^2 + d^2 + f^2)).
Proof.
  intros a b c d e f Ha Hb Hc Hd He Hf.
  pose proof (CS3 a b c d e f Ha Hb Hc Hd He Hf) as H.
  assert (H1 : 0 <= c*d) by nra.
  assert (H2 : 0 <= e*f) by nra.
  lra.
Qed.

(*  Step 9 (eq:absorb1--eq:normconv): five squared elemental pieces, each    *)
(*  dominated by a coefficient times A (velocity bracket norm), B (pressure  *)
(*  bracket norm), or a combination, aggregate to a triple-norm bound by     *)
(*  an explicit multiple of A + B.                                           *)
Theorem norm5_absorption :
  forall p1 p2 p3 p4 p5 A B k1 k2 k3 k4 k5 k6 : R,
    0 <= p1 -> 0 <= p2 -> 0 <= p3 -> 0 <= p4 -> 0 <= p5 ->
    0 <= A -> 0 <= B ->
    0 <= k1 -> 0 <= k2 -> 0 <= k3 -> 0 <= k4 -> 0 <= k5 -> 0 <= k6 ->
    p1 <= k1 * A -> p2 <= k2 * A -> p3 <= k3 * B ->
    p4 <= k4 * A + k5 * B -> p5 <= k6 * A ->
    sqrt (p1^2 + p2^2 + p3^2 + p4^2 + p5^2)
    <= sqrt (k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2) * (A + B).
Proof.
  intros p1 p2 p3 p4 p5 A B k1 k2 k3 k4 k5 k6
         Hp1 Hp2 Hp3 Hp4 Hp5 HA HB Hk1 Hk2 Hk3 Hk4 Hk5 Hk6
         H1 H2 H3 H4 H5.
  assert (HPs : 0 <= p1^2 + p2^2 + p3^2 + p4^2 + p5^2).
  { pose proof (pow2_ge_0 p1); pose proof (pow2_ge_0 p2);
    pose proof (pow2_ge_0 p3); pose proof (pow2_ge_0 p4);
    pose proof (pow2_ge_0 p5). lra. }
  assert (HKs : 0 <= k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2).
  { pose proof (pow2_ge_0 k1); pose proof (pow2_ge_0 k2);
    pose proof (pow2_ge_0 k3); pose proof (pow2_ge_0 (k4 + k5));
    pose proof (pow2_ge_0 k6). lra. }
  apply nonneg_le_of_sqr.
  - apply sqrt_pos.
  - assert (0 <= sqrt (k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2))
      by apply sqrt_pos.
    nra.
  - replace ((sqrt (p1^2 + p2^2 + p3^2 + p4^2 + p5^2))^2)
      with (sqrt (p1^2 + p2^2 + p3^2 + p4^2 + p5^2)
            * sqrt (p1^2 + p2^2 + p3^2 + p4^2 + p5^2)) by ring.
    replace ((sqrt (k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2) * (A + B))^2)
      with ((sqrt (k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2)
             * sqrt (k1^2 + k2^2 + k3^2 + (k4 + k5)^2 + k6^2))
            * (A + B)^2) by ring.
    rewrite !sqrt_sqrt by lra.
    (* per-piece squared bounds against (A + B) *)
    assert (Q1 : p1^2 <= k1^2 * (A + B)^2).
    { assert (Hm : p1 <= k1 * (A + B)) by nra.
      assert (Hs : 0 <= p1 + k1 * (A + B)) by nra.
      nra. }
    assert (Q2 : p2^2 <= k2^2 * (A + B)^2).
    { assert (Hm : p2 <= k2 * (A + B)) by nra.
      assert (Hs : 0 <= p2 + k2 * (A + B)) by nra.
      nra. }
    assert (Q3 : p3^2 <= k3^2 * (A + B)^2).
    { assert (Hm : p3 <= k3 * (A + B)) by nra.
      assert (Hs : 0 <= p3 + k3 * (A + B)) by nra.
      nra. }
    assert (Q4 : p4^2 <= (k4 + k5)^2 * (A + B)^2).
    { assert (Hm : p4 <= (k4 + k5) * (A + B)) by nra.
      assert (Hs : 0 <= p4 + (k4 + k5) * (A + B)) by nra.
      nra. }
    assert (Q5 : p5^2 <= k6^2 * (A + B)^2).
    { assert (Hm : p5 <= k6 * (A + B)) by nra.
      assert (Hs : 0 <= p5 + k6 * (A + B)) by nra.
      nra. }
    lra.
Qed.
