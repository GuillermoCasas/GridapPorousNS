(* ========================================================================= *)
(*  StabilityAlgebra.v                                                       *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the algebraic    *)
(*  identities underlying the coercivity / stability estimate of Section 5   *)
(*  of *A stabilized finite element method for incompressible, inertial      *)
(*  flows in inhomogeneous porous media* (Casas, González-Usúa, Codina,      *)
(*  de-Pouplana).                                                            *)
(*                                                                           *)
(*  Mirrors theory/verification scripts/stability_estimate_verification.py.  *)
(*  With                                                                     *)
(*     tau1NS^{-1} = c1 nu/h^2 + c2 |a|/h            (eq:TauNavierStokes)    *)
(*     tau1  = (alpha_K tau1NS^{-1} + sigma)^{-1}    (eq:Tau1Final)          *)
(*     tau2  = h^2 tau1NS^{-1}/(c1 alpha_K)          (eq:Tau2Final)          *)
(*     phi1  = alpha_K tau1NS^{-1}                                           *)
(*     sigt  = tau1NS^{-1} sigma/(tau1NS^{-1}+sigma/alpha_K)  (eq:SigmaAlpha)*)
(*  it proves:                                                               *)
(*   (1) the four equivalent forms of sigma-tilde (eq:SigmaAlpha);           *)
(*   (2) Young's inequality  -2xy >= -x^2/xi - xi y^2 (perfect square);      *)
(*   (3) the viscous-coefficient expansion (eq:847 form);                    *)
(*   (4) the velocity-coefficient expansion (eq:855) and its reduction       *)
(*       to  >= (1 - xi Cinv^2/c1) * sigma-tilde;                            *)
(*   (5) the epsilon smallness chain (eq:UpperBoundOnEpsilon, amendment A1): *)
(*       eps <= eps_max  ==>  eps*tau2 <= C2  ==>                            *)
(*       eps(1 - eps tau2) >= (1 - C2) eps.                                  *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

(* Helper: nonneg / pos division (name Rdiv_nonneg is absent in 8.18). *)
Lemma Rdiv_nonneg : forall a b : R, 0 <= a -> 0 < b -> 0 <= a / b.
Proof.
  intros a b Ha Hb. unfold Rdiv. apply Rmult_le_pos; [exact Ha |].
  left. apply Rinv_0_lt_compat. exact Hb.
Qed.

Section StabilityEstimateAlgebra.

(* ---------- Data of the estimate, all strictly positive ------------------ *)
Variables (nu h alphaK sigma amag : R).   (* viscosity, mesh size, porosity  *)
                                          (* value on K, reaction, |a|       *)
Variables (c1 c2 Cinv xi : R).            (* tau constants, inverse-estimate *)
                                          (* constant, Young parameter       *)
Hypothesis nu_pos     : 0 < nu.
Hypothesis h_pos      : 0 < h.
Hypothesis alphaK_pos : 0 < alphaK.
Hypothesis sigma_nonneg : 0 <= sigma.   (* H:data allows sigma = 0 *)
Hypothesis amag_nonneg: 0 <= amag.        (* |a| >= 0 : it is a norm         *)
Hypothesis c1_pos     : 0 < c1.
Hypothesis c2_pos     : 0 < c2.
Hypothesis Cinv_pos   : 0 < Cinv.
Hypothesis xi_pos     : 0 < xi.

(* ---------- The stabilization parameters (eq:TauNavierStokes,             *)
(*            eq:Tau1Final, eq:Tau2Final) and eq:SigmaAlpha ---------------- *)
Definition tau1NS_inv : R := c1 * nu / h^2 + c2 * amag / h.
Definition tau1       : R := / (alphaK * tau1NS_inv + sigma).
Definition tau2       : R := h^2 * tau1NS_inv / (c1 * alphaK).
Definition tau1NS     : R := / tau1NS_inv.
Definition phi1       : R := alphaK * tau1NS_inv.
Definition sigt       : R := tau1NS_inv * sigma / (tau1NS_inv + sigma / alphaK).

(* ---------- Positivity facts used to discharge field side conditions ----- *)
Lemma h2_pos : 0 < h^2.
Proof. apply pow_lt; exact h_pos. Qed.

Lemma tau1NS_inv_pos : 0 < tau1NS_inv.
Proof.
  unfold tau1NS_inv.
  assert (0 < c1 * nu / h^2)
    by (apply Rdiv_lt_0_compat; [nra | exact h2_pos]).
  assert (0 <= c2 * amag / h) by (apply Rdiv_nonneg; nra).
  lra.
Qed.

Lemma tau1_denom_pos : 0 < alphaK * tau1NS_inv + sigma.
Proof. pose proof tau1NS_inv_pos; nra. Qed.

Lemma tau1_pos : 0 < tau1.
Proof. unfold tau1. apply Rinv_0_lt_compat, tau1_denom_pos. Qed.

Lemma tau1NS_pos : 0 < tau1NS.
Proof. unfold tau1NS. apply Rinv_0_lt_compat, tau1NS_inv_pos. Qed.

Lemma sigt_denom_pos : 0 < tau1NS_inv + sigma / alphaK.
Proof.
  pose proof tau1NS_inv_pos.
  assert (0 <= sigma / alphaK) by (apply Rdiv_nonneg; lra).
  lra.
Qed.

Lemma tau2_pos : 0 < tau2.
Proof.
  unfold tau2.
  pose proof tau1NS_inv_pos. pose proof h2_pos.
  apply Rdiv_lt_0_compat; nra.
Qed.


(* Cleared-denominator positivity, in the exact shapes emitted by            *)
(* [field_simplify_eq] (staged for nra's pairwise product search).           *)
Lemma cleared_NS_pos : 0 < c1 * nu + c2 * amag * h.
Proof.
  assert (0 < c1 * nu) by nra.
  assert (0 <= c2 * amag) by nra.
  assert (0 <= c2 * amag * h) by nra.
  lra.
Qed.

Lemma cleared_tau1_pos :
  0 < alphaK * (c1 * nu + c2 * amag * h) + sigma * (h * h).
Proof.
  pose proof cleared_NS_pos.
  assert (0 < alphaK * (c1 * nu + c2 * amag * h)) by nra.
  assert (0 < h * h) by nra.
  assert (0 <= sigma * (h * h)) by nra.
  lra.
Qed.

Ltac clear_denoms :=
  field_simplify_eq;
  [ ring
  | pose proof cleared_NS_pos; pose proof cleared_tau1_pos;
    repeat split; lra ].

(* ========================================================================= *)
(*  (1)  The four equivalent forms of sigma-tilde (eq:SigmaAlpha).           *)
(*       sigt := tau1NS^{-1} sigma / (tau1NS^{-1} + sigma/alpha_K)           *)
(* ========================================================================= *)

(* Form 2:  sigt = sigma * phi1 / (phi1 + sigma). *)
Theorem sigt_form_phi : sigt = sigma * phi1 / (phi1 + sigma).
Proof.
  unfold sigt, phi1.
  pose proof tau1NS_inv_pos as Ht. pose proof sigt_denom_pos as Hd.
  field. split; nra.
Qed.

(* Form 3:  sigt = sigma - sigma^2 * tau1. *)
Theorem sigt_form_tau1 : sigt = sigma - sigma^2 * tau1.
Proof.
  unfold sigt, tau1.
  pose proof tau1NS_inv_pos as Ht. pose proof tau1_denom_pos as Hd.
  pose proof sigt_denom_pos as Hs.
  field. repeat split; nra.
Qed.

(* Form 4 (the key form used in the velocity coefficient):                   *)
(*        sigt = sigma * phi1 * tau1.                                        *)
Theorem sigt_form_key : sigt = sigma * phi1 * tau1.
Proof.
  unfold sigt, phi1, tau1.
  pose proof tau1NS_inv_pos as Ht. pose proof tau1_denom_pos as Hd.
  pose proof sigt_denom_pos as Hs.
  field. repeat split; nra.
Qed.

(*  With sigma >= 0 (H:data) this is nonnegativity, not strict positivity:   *)
(*  at sigma = 0 one has sigt = 0, which is exactly the reaction-free case.   *)
Lemma sigt_nonneg : 0 <= sigt.
Proof.
  rewrite sigt_form_key. unfold phi1.
  pose proof tau1NS_inv_pos as H1. pose proof tau1_pos as H2.
  assert (Hp : 0 < alphaK * tau1NS_inv) by nra.
  assert (0 <= sigma * (alphaK * tau1NS_inv)) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  (2)  Young's inequality as used throughout Section 5:                    *)
(*         x^2/xi - 2xy + xi y^2  =  (x/sqrt(xi) - sqrt(xi) y)^2  >=  0,     *)
(*       hence  -2xy >= -x^2/xi - xi y^2.                                    *)
(* ========================================================================= *)

Theorem young_perfect_square :
  forall x y : R,
    x^2 / xi - 2 * x * y + xi * y^2 = (x / sqrt xi - sqrt xi * y)^2.
Proof.
  intros x y.
  assert (Hs : sqrt xi * sqrt xi = xi) by (apply sqrt_sqrt; lra).
  assert (Hspos : 0 < sqrt xi) by (apply sqrt_lt_R0; lra).
  field_simplify; [| lra | lra].
  rewrite <- Hs at 1 2 3.  (* express xi through sqrt xi where needed *)
  field. lra.
Qed.

Theorem young_inequality :
  forall x y : R, - (2 * x * y) >= - x^2 / xi - xi * y^2.
Proof.
  intros x y.
  assert (H := young_perfect_square x y).
  assert (Hsq := pow2_ge_0 (x / sqrt xi - sqrt xi * y)).
  nra.
Qed.

(* ========================================================================= *)
(*  (3)  Viscous coefficient of eq:StabilityEstimateFinal equals the eq:847  *)
(*       expansion:                                                          *)
(*    nu tau1 (2/tau1 - 4 Cinv^2 alpha_K nu/h^2 - 4 sigma/xi)                *)
(*      = nu tau1 ( alpha_K (2 - 4Cinv^2/c1)(c1 nu/h^2)                      *)
(*                  + 2 alpha_K c2 |a|/h + 2(1 - 2/xi) sigma ).              *)
(* ========================================================================= *)

Definition visc_final : R :=
  nu * tau1 * (2 / tau1 - 4 * Cinv^2 * alphaK * nu / h^2 - 4 * sigma / xi).

Definition visc_847 : R :=
  nu * tau1 * (alphaK * (2 - 4 * Cinv^2 / c1) * (c1 * nu / h^2)
               + 2 * alphaK * c2 * amag / h + 2 * (1 - 2 / xi) * sigma).

Theorem viscous_coefficient_expansion : visc_final = visc_847.
Proof.
  unfold visc_final, visc_847, tau1, tau1NS_inv.
  pose proof tau1_denom_pos as Hd. unfold tau1NS_inv in Hd.
  clear_denoms.
Qed.

(* ========================================================================= *)
(*  (4)  Velocity coefficient: eq:StabilityEstimateFinal == eq:855, and its  *)
(*       reduction to  >= (1 - xi Cinv^2/c1) * sigma-tilde.                  *)
(* ========================================================================= *)

Definition u_final : R :=
  sigma * tau1 * (/ tau1 - sigma - xi * Cinv^2 * alphaK * nu / h^2).

Definition u_855 : R :=
  alphaK * tau1 * sigma * ((1 - xi * Cinv^2 / c1) * (c1 * nu / h^2)
                           + c2 * amag / h).

Definition C_u : R := 1 - xi * Cinv^2 / c1.

Theorem velocity_coefficient_expansion : u_final = u_855.
Proof.
  unfold u_final, u_855, tau1, tau1NS_inv.
  pose proof tau1_denom_pos as Hd. unfold tau1NS_inv in Hd.
  clear_denoms.
Qed.

(* The exact slack identity:                                                 *)
(*   eq:855 - C_u * sigt = alpha_K tau1 sigma (xi Cinv^2/c1)(c2 |a|/h) >= 0. *)
Theorem velocity_slack_identity :
  u_855 - C_u * sigt = alphaK * tau1 * sigma * (xi * Cinv^2 / c1) * (c2 * amag / h).
Proof.
  rewrite sigt_form_key.
  unfold u_855, C_u, phi1, tau1, tau1NS_inv.
  clear_denoms.
Qed.

Corollary velocity_coefficient_lower_bound : u_855 >= C_u * sigt.
Proof.
  pose proof velocity_slack_identity as Hs.
  pose proof tau1_pos as Ht.
  assert (HB : 0 <= c2 * amag / h) by (apply Rdiv_nonneg; nra).
  assert (HCinv2 : 0 < Cinv^2) by (apply pow_lt; lra).
  assert (HA : 0 < xi * Cinv^2 / c1) by (apply Rdiv_lt_0_compat; nra).
  assert (P1 : 0 < alphaK * tau1) by nra.
  assert (P2 : 0 <= alphaK * tau1 * sigma) by nra.
  assert (P3 : 0 <= alphaK * tau1 * sigma * (xi * Cinv^2 / c1)) by nra.
  assert (P4 : 0 <= alphaK * tau1 * sigma * (xi * Cinv^2 / c1) * (c2 * amag / h))
    by nra.
  lra.
Qed.

Corollary u_final_lower_bound : u_final >= C_u * sigt.
Proof. rewrite velocity_coefficient_expansion. apply velocity_coefficient_lower_bound. Qed.

(* ========================================================================= *)
(*  (4b) Viscous coefficient lower bound (main text, unlabelled display      *)
(*       after eq:UpperBoundOnEpsilon: bracket >= C nu with                  *)
(*       C = min{2 - 4 Cinv^2/c1, 2(1 - 2/xi)}), and positivity of the       *)
(*       stability constants under eq:conditions_on_num_param                *)
(*       (c1 > 2 xi Cinv^2, xi > 2).                                         *)
(* ========================================================================= *)

Definition C_visc : R := Rmin (2 - 4 * Cinv^2 / c1) (2 * (1 - 2 / xi)).

Theorem viscous_coefficient_lower_bound : visc_847 >= C_visc * nu.
Proof.
  pose proof (Rmin_l (2 - 4 * Cinv^2 / c1) (2 * (1 - 2 / xi))) as HL.
  pose proof (Rmin_r (2 - 4 * Cinv^2 / c1) (2 * (1 - 2 / xi))) as HR.
  fold C_visc in HL, HR.
  assert (Hq : 0 < 4 * Cinv^2 / c1) by (apply Rdiv_lt_0_compat; nra).
  assert (H2 : C_visc < 2) by lra.
  assert (Ht1 : 0 < c1 * nu / h^2).
  { apply Rdiv_lt_0_compat; [nra | apply pow_lt; exact h_pos]. }
  assert (Ht2 : 0 <= c2 * amag / h) by (apply Rdiv_nonneg; nra).
  assert (G1 : 0 <= alphaK * (c1 * nu / h^2) * (2 - 4 * Cinv^2 / c1 - C_visc)).
  { assert (0 < alphaK * (c1 * nu / h^2)) by nra. nra. }
  assert (G2 : 0 <= alphaK * (c2 * amag / h) * (2 - C_visc)).
  { assert (0 <= alphaK * (c2 * amag / h)) by nra. nra. }
  assert (G3 : 0 <= sigma * (2 * (1 - 2 / xi) - C_visc)) by nra.
  assert (HS : alphaK * (2 - 4 * Cinv^2 / c1) * (c1 * nu / h^2)
               + 2 * alphaK * c2 * amag / h + 2 * (1 - 2 / xi) * sigma
               >= C_visc * (alphaK * tau1NS_inv + sigma)).
  { unfold tau1NS_inv. lra. }
  assert (Hnt : 0 < nu * tau1) by (pose proof tau1_pos; nra).
  assert (Hprod :
      nu * tau1 * (alphaK * (2 - 4 * Cinv^2 / c1) * (c1 * nu / h^2)
                   + 2 * alphaK * c2 * amag / h + 2 * (1 - 2 / xi) * sigma)
      >= nu * tau1 * (C_visc * (alphaK * tau1NS_inv + sigma))) by nra.
  unfold visc_847.
  replace (C_visc * nu)
    with (nu * tau1 * (C_visc * (alphaK * tau1NS_inv + sigma))).
  2:{ unfold tau1. field. pose proof tau1_denom_pos. lra. }
  lra.
Qed.

Corollary visc_final_lower_bound : visc_final >= C_visc * nu.
Proof.
  rewrite viscous_coefficient_expansion.
  apply viscous_coefficient_lower_bound.
Qed.

(*  eq:conditions_on_num_param:  c1 > 2 xi Cinv^2 with xi > 2 makes both     *)
(*  stability constants strictly positive.                                   *)
(*  SHARP form of eq:conditions_on_num_param.  Positivity of the two          *)
(*  coefficients needs only  c1 > xi * Cinv^2  (together with xi > 2, which    *)
(*  already forces c1 > 2 Cinv^2, the condition C_visc needs).  This is a      *)
(*  factor of two weaker than the condition the manuscript states.            *)
Theorem stability_constants_positive_sharp :
  c1 > xi * Cinv^2 -> xi > 2 -> 0 < C_visc /\ 0 < C_u.
Proof.
  intros Hc Hxi.
  assert (HC2p : 0 < Cinv^2) by nra.
  assert (HA : 0 < 2 - 4 * Cinv^2 / c1).
  { assert (Hlt : 4 * Cinv^2 / c1 < 2).
    { apply (Rmult_lt_reg_r c1); [exact c1_pos |].
      replace (4 * Cinv^2 / c1 * c1) with (4 * Cinv^2) by (field; lra).
      nra. }
    lra. }
  assert (HB : 0 < 2 * (1 - 2 / xi)).
  { assert (H2x : 2 / xi < 1).
    { apply (Rmult_lt_reg_r xi); [lra |].
      replace (2 / xi * xi) with 2 by (field; lra). lra. }
    lra. }
  assert (HCu : 0 < C_u).
  { unfold C_u.
    assert (Hlt : xi * Cinv^2 / c1 < 1).
    { apply (Rmult_lt_reg_r c1); [exact c1_pos |].
      replace (xi * Cinv^2 / c1 * c1) with (xi * Cinv^2) by (field; lra).
      nra. }
    lra. }
  split; [unfold C_visc; apply Rmin_pos; assumption | exact HCu].
Qed.

(*  The manuscript's condition (eq:conditions_on_num_param, prop:stability).   *)
(*  It implies the sharp one, since xi > 2 > 1.                                *)
Corollary stability_constants_positive :
  c1 > 2 * xi * Cinv^2 -> xi > 2 -> 0 < C_visc /\ 0 < C_u.
Proof.
  intros Hc Hxi.
  apply stability_constants_positive_sharp; [| exact Hxi].
  assert (0 < Cinv^2) by nra. nra.
Qed.



(* ========================================================================= *)
(*  (5)  The epsilon smallness chain (amendment A1).                         *)
(*       eq:UpperBoundOnEpsilon gives eps <= eps_max with                    *)
(*         eps_max := C2 c1 alpha_K^2 tau1 / h^2 ;                           *)
(*       then eps * tau2 <= C2, hence eps(1 - eps tau2) >= (1 - C2) eps.     *)
(* ========================================================================= *)

Section EpsilonChain.
Variables (eps C2 : R).
Hypothesis eps_nonneg : 0 <= eps.   (* H:data allows eps = 0 *)
Hypothesis C2_nonneg  : 0 <= C2.    (* eq:epscond allows C2 = 0 *)

Definition eps_max : R := C2 * c1 * alphaK^2 * tau1 / h^2.

(* Step 1 (identity):  eps_max * tau2 = C2 * alpha_K * tau1 / tau1NS.        *)
Theorem eps_max_tau2_identity : eps_max * tau2 = C2 * alphaK * tau1 / tau1NS.
Proof.
  unfold eps_max, tau2, tau1NS, tau1, tau1NS_inv.
  pose proof tau1_denom_pos as Hd. unfold tau1NS_inv in Hd.
  pose proof tau1NS_inv_pos as Ht. unfold tau1NS_inv in Ht.
  clear_denoms.
Qed.

(* Step 2 (identity):  tau1NS - alpha_K tau1 = sigma * tau1 * tau1NS,        *)
(* which is > 0; hence alpha_K tau1 < tau1NS, i.e. alpha_K tau1/tau1NS < 1.  *)
Theorem tau1NS_minus_alphaK_tau1 : tau1NS - alphaK * tau1 = sigma * tau1 * tau1NS.
Proof.
  unfold tau1NS, tau1, tau1NS_inv.
  pose proof tau1_denom_pos as Hd. unfold tau1NS_inv in Hd.
  pose proof tau1NS_inv_pos as Ht. unfold tau1NS_inv in Ht.
  clear_denoms.
Qed.

(*  With sigma >= 0 this is  <=  rather than  < : at sigma = 0 the two sides *)
(*  coincide (tau1 = tau1NS/alpha_K).  The chain below only needs <=.        *)
Lemma alphaK_tau1_le_tau1NS : alphaK * tau1 <= tau1NS.
Proof.
  pose proof tau1NS_minus_alphaK_tau1 as Hid.
  pose proof tau1_pos as H1. pose proof tau1NS_pos as H2.
  assert (0 <= sigma * tau1) by nra.
  assert (0 <= sigma * tau1 * tau1NS) by nra.
  lra.
Qed.

(* Step 3 (the chain):  eps <= eps_max  ==>  eps * tau2 <= C2.               *)
Theorem eps_tau2_le_C2 : eps <= eps_max -> eps * tau2 <= C2.
Proof.
  intros Heps.
  pose proof eps_max_tau2_identity as Hid.
  pose proof alphaK_tau1_le_tau1NS as Hlt.
  pose proof tau1NS_pos as HNS. pose proof tau1_pos as Ht1.
  pose proof tau2_pos as Ht2.
  (* eps * tau2 <= eps_max * tau2 = C2 * (alphaK*tau1/tau1NS) <= C2 * 1 *)
  assert (Hstep1 : eps * tau2 <= eps_max * tau2) by nra.
  assert (Hratio : alphaK * tau1 / tau1NS <= 1).
  { apply (Rmult_le_reg_r tau1NS); [exact HNS |].
    unfold Rdiv. rewrite Rmult_assoc, Rinv_l; lra. }
  assert (Hstep2 : C2 * alphaK * tau1 / tau1NS <= C2).
  { replace (C2 * alphaK * tau1 / tau1NS) with (C2 * (alphaK * tau1 / tau1NS))
      by (unfold Rdiv; ring).
    nra. }
  lra.
Qed.

(* Step 4 (coercivity of the pressure term):                                 *)
(*   eps * tau2 <= C2  ==>  eps (1 - eps tau2) >= (1 - C2) eps.              *)
Theorem pressure_term_coercive :
  eps * tau2 <= C2 -> eps * (1 - eps * tau2) >= (1 - C2) * eps.
Proof. intros H. nra. Qed.

(* The full chain in one statement. *)
Corollary epsilon_smallness_chain :
  eps <= eps_max -> eps * (1 - eps * tau2) >= (1 - C2) * eps.
Proof. intros H. apply pressure_term_coercive, eps_tau2_le_C2, H. Qed.


(* ========================================================================= *)
(*  Elemental coercivity assembly (the algebraic skeleton of                 *)
(*  lemma:Stability): the five per-element coefficient bounds of             *)
(*  eq:StabilityEstimateFinal dominate  C_stab  times the corresponding      *)
(*  contributions of the working norm  triplenorm  (eq after                 *)
(*  lemma:Stability), with                                                   *)
(*    C_stab := min{ C_visc, C_u, 1 - C2, 1 }.                               *)
(*  Pn, Vn, Un, Xn, Dn stand for the elemental squared (semi)norms           *)
(*  ||p||^2, ||alpha^(1/2) P grad u||^2, ||u||^2,                            *)
(*  ||tau1^(1/2) alpha X(U)||^2, ||tau2^(1/2) div(alpha u)||^2.              *)
(* ========================================================================= *)

Definition C_stab : R := Rmin (Rmin C_visc C_u) (Rmin (1 - C2) 1).

(*  Why the manuscript's factor of two is not idle: it buys a quantitative     *)
(*  MARGIN, not merely positivity.  Under c1 > 2 xi Cinv^2 and xi > 2,         *)
(*                                                                             *)
(*      C_u > 1/2       and       2 - 4 Cinv^2 / c1 > 1,                       *)
(*                                                                             *)
(*  so that the coercivity constant enjoys a floor that is free of Cinv:       *)
(*                                                                             *)
(*      C_stab >= min{ 2(1 - 2/xi),  1/2,  1 - C2 }.                           *)
(*                                                                             *)
(*  Under the sharp condition c1 > xi Cinv^2 the constants are still positive  *)
(*  but C_u -> 0 as c1 decreases to xi Cinv^2, so the floor is lost.           *)
Theorem C_stab_margin :
  c1 > 2 * xi * Cinv^2 -> xi > 2 ->
  C_u > 1/2
  /\ 2 - 4 * Cinv^2 / c1 > 1
  /\ C_stab >= Rmin (Rmin (2 * (1 - 2 / xi)) (1/2)) (1 - C2).
Proof.
  intros Hc Hxi.
  assert (HCinv2 : 0 < Cinv^2) by nra.
  assert (HA : 2 - 4 * Cinv^2 / c1 > 1).
  { assert (Hlt : 4 * Cinv^2 / c1 < 1).
    { apply (Rmult_lt_reg_r c1); [exact c1_pos |].
      replace (4 * Cinv^2 / c1 * c1) with (4 * Cinv^2) by (field; lra).
      nra. }
    lra. }
  assert (HCu : C_u > 1/2).
  { unfold C_u.
    assert (Hlt : xi * Cinv^2 / c1 < 1/2).
    { apply (Rmult_lt_reg_r c1); [exact c1_pos |].
      replace (xi * Cinv^2 / c1 * c1) with (xi * Cinv^2) by (field; lra).
      lra. }
    lra. }
  repeat split; try assumption.
  pose proof C2_nonneg as HC2.
  unfold C_stab, C_visc, Rmin.
  repeat (destruct (Rle_dec _ _)); lra.
Qed.

Theorem elemental_coercivity :
  forall Pn Vn Un Xn Dn : R,
    eps <= eps_max ->
    0 <= Pn -> 0 <= Vn -> 0 <= Un -> 0 <= Xn -> 0 <= Dn ->
    eps * (1 - eps * tau2) * Pn + visc_final * Vn + u_final * Un + Xn + Dn
    >= C_stab * (eps * Pn + nu * Vn + sigt * Un + Xn + Dn).
Proof.
  intros Pn Vn Un Xn Dn Heps HP HV HU HX HD.
  pose proof (epsilon_smallness_chain Heps) as He.
  pose proof visc_final_lower_bound as Hv.
  pose proof u_final_lower_bound as Hu.
  assert (M1 : C_stab <= C_visc).
  { unfold C_stab.
    apply Rle_trans with (Rmin C_visc C_u); apply Rmin_l. }
  assert (M2 : C_stab <= C_u).
  { unfold C_stab.
    apply Rle_trans with (Rmin C_visc C_u); [apply Rmin_l | apply Rmin_r]. }
  assert (M3 : C_stab <= 1 - C2).
  { unfold C_stab.
    apply Rle_trans with (Rmin (1 - C2) 1); [apply Rmin_r | apply Rmin_l]. }
  assert (M4 : C_stab <= 1).
  { unfold C_stab.
    apply Rle_trans with (Rmin (1 - C2) 1); apply Rmin_r. }
  assert (B1 : eps * (1 - eps * tau2) * Pn >= C_stab * (eps * Pn)).
  { assert (S1 : eps * (1 - eps * tau2) * Pn >= (1 - C2) * eps * Pn) by nra.
    assert (HeP : 0 <= eps * Pn) by nra.
    assert (S2 : (1 - C2) * eps * Pn >= C_stab * (eps * Pn)) by nra.
    lra. }
  assert (B2 : visc_final * Vn >= C_stab * (nu * Vn)).
  { assert (S1 : visc_final * Vn >= C_visc * nu * Vn) by nra.
    assert (Hnv : 0 <= nu * Vn) by nra.
    assert (S2 : C_visc * nu * Vn >= C_stab * (nu * Vn)) by nra.
    lra. }
  assert (B3 : u_final * Un >= C_stab * (sigt * Un)).
  { assert (S1 : u_final * Un >= C_u * sigt * Un) by nra.
    assert (Hsu : 0 <= sigt * Un) by (pose proof sigt_nonneg; nra).
    assert (S2 : C_u * sigt * Un >= C_stab * (sigt * Un)) by nra.
    lra. }
  assert (B4 : Xn >= C_stab * Xn) by nra.
  assert (B5 : Dn >= C_stab * Dn) by nra.
  lra.
Qed.

End EpsilonChain.

End StabilityEstimateAlgebra.
