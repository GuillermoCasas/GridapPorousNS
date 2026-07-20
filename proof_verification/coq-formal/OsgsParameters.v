(* ========================================================================= *)
(*  OsgsParameters.v                                                         *)
(*                                                                           *)
(*  Machine-checked verification (Rocq/Coq, stdlib only) of the algebraic    *)
(*  core of the OSGS a priori analysis, theory/osgs_a_priori/                *)
(*  osgs_convergence.tex, lem:parameters (Lemma 3.1: the parameter           *)
(*  inequalities (P1)--(P6) and (K1)--(K3)).                                  *)
(*                                                                           *)
(*  The OSGS variant is analysed with EXACTLY the stabilization parameters    *)
(*  of the companion manuscript (eq:TauDefs of the note = eq:taus of the      *)
(*  companion): with tauNSinv = c1 nu/h^2 + c2 |a|/h, phi1 = alpha_K          *)
(*  tauNSinv,                                                                 *)
(*    tau1 = 1/(phi1 + sigma),   tau2 = h^2 tauNSinv/(c1 alpha_K),            *)
(*    sigt = sigma phi1/(phi1 + sigma),                                       *)
(*  which are the closed definitions of ContinuityAlgebra.v, reused verbatim  *)
(*  here (so that the OSGS files and the ASGS files share one parameter set,  *)
(*  reconciled inside the kernel).  Only two objects are genuinely new to the *)
(*  OSGS note: the elementwise mesh Damkohler number Dah = sigma h^2/(alpha_K *)
(*  nu), and the design constants c1 >= gamma^2 Cinv^2, c2 >= gamma Cinv of   *)
(*  hypothesis (A6), which drive (K1)--(K3).                                  *)
(*                                                                           *)
(*  The single genuinely new inequality is (P5),                             *)
(*    alpha_K^2 tau1 tau2 >= h^2/(c1 + Dah),                                  *)
(*  which is the SOLE source of the Damkohler factor (1 + Dah)^{1/2} in the   *)
(*  OSGS convergence estimate (note eq:MainResult); together with (P6),       *)
(*    sigma^{1/2} <= Dah^{1/2}(alpha_K/h) tau2^{1/2},                         *)
(*  it is what the orthogonal projection's deletion of the reactive subscale  *)
(*  costs (note rem:mechanism).                                              *)
(*                                                                           *)
(*  Everything is proved from the real-number axioms; no Admitted, no Axiom.  *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
From PNSFormal Require Import ContinuityAlgebra.
Local Open Scope R_scope.

Section OsgsParameters.

Variables (nu h alphaK sigma amag c1 c2 Cinv gam : R).
Hypothesis nu_pos       : 0 < nu.
Hypothesis h_pos        : 0 < h.
Hypothesis alphaK_pos   : 0 < alphaK.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis amag_nonneg  : 0 <= amag.
Hypothesis c1_pos       : 0 < c1.
Hypothesis c2_pos       : 0 < c2.
Hypothesis Cinv_pos     : 0 < Cinv.
Hypothesis gam_pos      : 0 < gam.
(*  (A6): c1 >= gamma^2 Cinv^2 and c2 >= gamma Cinv.  *)
Hypothesis c1_ge : gam^2 * Cinv^2 <= c1.
Hypothesis c2_ge : gam * Cinv <= c2.

(*  The companion parameters, reused verbatim (SAME as the note's eq:TauDefs). *)
Notation TNS := (ContinuityAlgebra.tauNSinv nu h amag c1 c2).
Notation PH  := (ContinuityAlgebra.phi1 nu h alphaK amag c1 c2).
Notation T1  := (ContinuityAlgebra.tau1 nu h alphaK sigma amag c1 c2).
Notation T2  := (ContinuityAlgebra.tau2 nu h alphaK amag c1 c2).
Notation SG  := (ContinuityAlgebra.sigt nu h alphaK sigma amag c1 c2).

(*  The elementwise mesh Damkohler number (note eq:TauDefs).  *)
Definition Dah : R := sigma * h^2 / (alphaK * nu).

(*  A robust positivity discharger: split a product into pairwise positivity   *)
(*  goals, each closed by nra against the section hypotheses.                   *)
Local Ltac pos := repeat apply Rmult_lt_0_compat; nra.

(*  Companion nonnegativity discharger: split products/quotients/square roots.  *)
Local Ltac nn :=
  repeat first [ apply div_nonneg | apply Rmult_le_pos | apply sqrt_pos ]; nra.

(* ---------- Positivity ----------------------------------------------------- *)

Lemma h2_pos : 0 < h^2.
Proof. apply pow_lt; exact h_pos. Qed.

Lemma TNS_pos : 0 < TNS.
Proof.
  unfold ContinuityAlgebra.tauNSinv.
  assert (H1 : 0 < c1 * nu / h^2)
    by (apply Rdiv_lt_0_compat; [nra | exact h2_pos]).
  assert (H2 : 0 <= c2 * amag / h) by (apply div_nonneg; nra).
  lra.
Qed.

Lemma PH_pos : 0 < PH.
Proof. unfold ContinuityAlgebra.phi1. pose proof TNS_pos. nra. Qed.

Lemma denom_pos : 0 < PH + sigma.
Proof. pose proof PH_pos. lra. Qed.

Lemma T1_pos : 0 < T1.
Proof. unfold ContinuityAlgebra.tau1. apply Rinv_0_lt_compat, denom_pos. Qed.

Lemma T2_pos : 0 < T2.
Proof.
  unfold ContinuityAlgebra.tau2. apply Rdiv_lt_0_compat.
  - pose proof TNS_pos. pose proof h2_pos. nra.
  - nra.
Qed.

Lemma SG_nonneg : 0 <= SG.
Proof.
  unfold ContinuityAlgebra.sigt. apply div_nonneg; [| exact denom_pos].
  pose proof PH_pos. nra.
Qed.

Lemma Dah_nonneg : 0 <= Dah.
Proof. unfold Dah. apply div_nonneg; nra. Qed.

Lemma c1_Dah_pos : 0 < c1 + Dah.
Proof. pose proof Dah_nonneg. lra. Qed.

(*  The key closed identities used throughout.  *)
Lemma T1_inv : / T1 = PH + sigma.
Proof.
  unfold ContinuityAlgebra.tau1. rewrite Rinv_inv. reflexivity.
Qed.

Lemma TNS_h2_id : h^2 * TNS = c1 * nu + c2 * amag * h.
Proof. unfold ContinuityAlgebra.tauNSinv. field. lra. Qed.

Lemma Dah_id : alphaK * nu * Dah = sigma * h^2.
Proof. unfold Dah. field. nra. Qed.

(* ========================================================================= *)
(*  (P1):  sigma tau1 <= 1,   alpha_K tau1 <= tau_NS (= /tauNSinv).          *)
(* ========================================================================= *)

Theorem P1_sigma_tau1 : sigma * T1 <= 1.
Proof.
  pose proof denom_pos as HD. pose proof PH_pos as HP.
  apply (Rmult_le_reg_r (PH + sigma)); [lra |].
  replace (sigma * T1 * (PH + sigma)) with sigma
    by (unfold ContinuityAlgebra.tau1; field; lra).
  lra.
Qed.

Theorem P1_alphaK_tau1 : alphaK * T1 <= / TNS.
Proof.
  pose proof denom_pos as HD. pose proof PH_pos as HP. pose proof TNS_pos as HT.
  apply (Rmult_le_reg_r ((PH + sigma) * TNS)); [nra |].
  replace (alphaK * T1 * ((PH + sigma) * TNS))
    with (alphaK * TNS)
    by (unfold ContinuityAlgebra.tau1; field; lra).
  replace (/ TNS * ((PH + sigma) * TNS))
    with (PH + sigma) by (field; lra).
  unfold ContinuityAlgebra.phi1. nra.
Qed.

(* ========================================================================= *)
(*  (P2):  tau1 tau2 <= h^2/(c1 alpha_K^2).                                  *)
(* ========================================================================= *)

Theorem P2 : T1 * T2 <= h^2 / (c1 * alphaK^2).
Proof.
  pose proof TNS_pos as HT. pose proof denom_pos as HD. pose proof h2_pos as Hh.
  unfold ContinuityAlgebra.tau1, ContinuityAlgebra.tau2, ContinuityAlgebra.phi1 in *.
  set (T := ContinuityAlgebra.tauNSinv nu h amag c1 c2) in *.
  assert (Ha2 : 0 < alphaK^2) by nra.
  apply (Rmult_le_reg_r (c1 * alphaK^2 * (alphaK * T + sigma))); [pos |].
  replace (/ (alphaK * T + sigma) * (h^2 * T / (c1 * alphaK))
           * (c1 * alphaK^2 * (alphaK * T + sigma)))
    with (h^2 * T * alphaK) by (field; repeat split; lra).
  replace (h^2 / (c1 * alphaK^2) * (c1 * alphaK^2 * (alphaK * T + sigma)))
    with (h^2 * (alphaK * T + sigma)) by (field; repeat split; lra).
  nra.
Qed.

(* ========================================================================= *)
(*  (P3):  h^2 tauNSinv = c1 alpha_K tau2, and the square-root form           *)
(*         sqrt phi1 = sqrt c1 (alpha_K/h) sqrt tau2.                         *)
(* ========================================================================= *)

Theorem P3_id : h^2 * TNS = c1 * alphaK * T2.
Proof. unfold ContinuityAlgebra.tau2. field. lra. Qed.

Theorem P3_sqrt : sqrt PH = sqrt c1 * alphaK * sqrt T2 / h.
Proof.
  pose proof PH_pos as HP. pose proof T2_pos as HT2. pose proof h_pos as Hh.
  apply nonneg_eq_of_sqr.
  - apply sqrt_pos.
  - nn.
  - rewrite (sqrt_sqrt PH) by lra.
    replace (sqrt c1 * alphaK * sqrt T2 / h * (sqrt c1 * alphaK * sqrt T2 / h))
      with ((sqrt c1 * sqrt c1) * alphaK^2 * (sqrt T2 * sqrt T2) / h^2)
      by (field; lra).
    rewrite (sqrt_sqrt c1) by lra.
    rewrite (sqrt_sqrt T2) by lra.
    (*  goal: PH = c1 * alphaK^2 * T2 / h^2  *)
    apply (Rmult_eq_reg_r (h^2)); [| pose proof h2_pos; lra].
    replace (c1 * alphaK^2 * T2 / h^2 * h^2) with (c1 * alphaK^2 * T2)
      by (field; pose proof h2_pos; lra).
    unfold ContinuityAlgebra.phi1.
    pose proof P3_id as HP3. unfold ContinuityAlgebra.phi1 in HP3.
    nra.
Qed.

(* ========================================================================= *)
(*  (P4):  nu <= alpha_K tau2, and sqrt tau1 |a| <= (sqrt c1 / c2) sqrt tau2. *)
(* ========================================================================= *)

Theorem P4_nu : nu <= alphaK * T2.
Proof.
  unfold ContinuityAlgebra.tau2, ContinuityAlgebra.tauNSinv.
  replace (alphaK * (h^2 * (c1 * nu / h^2 + c2 * amag / h) / (c1 * alphaK)))
    with (nu + c2 * amag * h / c1) by (field; repeat split; lra).
  assert (0 <= c2 * amag * h / c1) by nn.
  lra.
Qed.

(*  Squared form of (P4b).  *)
Lemma P4b_sq : c2^2 * T1 * amag^2 <= c1 * T2.
Proof.
  pose proof TNS_pos as HT. pose proof denom_pos as HD.
  pose proof h2_pos as Hh.
  unfold ContinuityAlgebra.tau1, ContinuityAlgebra.tau2, ContinuityAlgebra.phi1 in *.
  set (T := ContinuityAlgebra.tauNSinv nu h amag c1 c2) in *.
  (*  Multiply out: c2^2 amag^2 alphaK <= h^2 T (alphaK T + sigma).  *)
  apply (Rmult_le_reg_r ((alphaK * T + sigma) * alphaK)); [pos |].
  replace (c2^2 * / (alphaK * T + sigma) * amag^2 * ((alphaK * T + sigma) * alphaK))
    with (c2^2 * amag^2 * alphaK) by (field; lra).
  replace (c1 * (h^2 * T / (c1 * alphaK)) * ((alphaK * T + sigma) * alphaK))
    with (h^2 * T * (alphaK * T + sigma)) by (field; lra).
  (*  h^2 T = c1 nu + c2 amag h, so c2 amag <= h T.  *)
  assert (Hid : h^2 * T = c1 * nu + c2 * amag * h) by exact TNS_h2_id.
  assert (HhT : c2 * amag <= h * T).
  { apply (Rmult_le_reg_r h); [exact h_pos |].
    replace (h * T * h) with (h^2 * T) by ring.
    rewrite Hid. nra. }
  assert (Hcam : 0 <= c2 * amag) by nn.
  assert (HhTp : 0 <= h * T) by nn.
  assert (Hsq : (c2 * amag)^2 <= (h * T)^2) by nra.
  assert (HTsig : 0 <= h^2 * T * sigma) by nn.
  assert (HaK : 0 <= alphaK) by lra.
  assert (E1 : c2^2 * amag^2 * alphaK = (c2 * amag)^2 * alphaK) by ring.
  assert (E2 : h^2 * T * (alphaK * T + sigma) = (h * T)^2 * alphaK + h^2 * T * sigma)
    by ring.
  rewrite E1, E2.
  assert (Hmul : (c2 * amag)^2 * alphaK <= (h * T)^2 * alphaK) by nra.
  lra.
Qed.

Theorem P4b : sqrt T1 * amag <= sqrt c1 / c2 * sqrt T2.
Proof.
  pose proof T1_pos as HT1. pose proof T2_pos as HT2.
  apply nonneg_le_of_sqr.
  - nn.
  - nn.
  - replace ((sqrt T1 * amag)^2) with ((sqrt T1 * sqrt T1) * amag^2) by ring.
    replace ((sqrt c1 / c2 * sqrt T2)^2)
      with ((sqrt c1 * sqrt c1) / c2^2 * (sqrt T2 * sqrt T2)) by (field; lra).
    rewrite (sqrt_sqrt T1) by lra.
    rewrite (sqrt_sqrt c1) by lra.
    rewrite (sqrt_sqrt T2) by lra.
    (*  T1 amag^2 <= c1/c2^2 T2, i.e. c2^2 T1 amag^2 <= c1 T2  *)
    pose proof P4b_sq as HP.
    apply (Rmult_le_reg_r (c2^2)); [nra |].
    replace (T1 * amag^2 * c2^2) with (c2^2 * T1 * amag^2) by ring.
    replace (c1 / c2^2 * T2 * c2^2) with (c1 * T2) by (field; lra).
    exact HP.
Qed.

(* ========================================================================= *)
(*  (P5):  alpha_K^2 tau1 tau2 >= h^2/(c1 + Dah)  --- the Damkohler source.   *)
(*  and the consequence  /sqrt tau2 <= sqrt(c1+Dah)(alpha_K/h) sqrt tau1.     *)
(* ========================================================================= *)

Theorem P5_prod : h^2 / (c1 + Dah) <= alphaK^2 * T1 * T2.
Proof.
  pose proof TNS_pos as HT. pose proof denom_pos as HD. pose proof h2_pos as Hh.
  pose proof c1_Dah_pos as HcD. pose proof Dah_nonneg as HD0.
  pose proof Dah_id as HDid. pose proof TNS_h2_id as Hid.
  set (D := Dah) in *.
  unfold ContinuityAlgebra.tau1, ContinuityAlgebra.tau2, ContinuityAlgebra.phi1 in *.
  set (T := ContinuityAlgebra.tauNSinv nu h amag c1 c2) in *.
  (*  HDid : alphaK*nu*D = sigma*h^2 ;  Hid : h^2*T = c1*nu + c2*amag*h.  *)
  apply (Rmult_le_reg_r ((c1 + D) * (c1 * alphaK) * (alphaK * T + sigma)));
    [ pos |].
  replace (h^2 / (c1 + D) * ((c1 + D) * (c1 * alphaK) * (alphaK * T + sigma)))
    with (h^2 * (c1 * alphaK) * (alphaK * T + sigma)) by (field; lra).
  replace (alphaK^2 * / (alphaK * T + sigma) * (h^2 * T / (c1 * alphaK))
           * ((c1 + D) * (c1 * alphaK) * (alphaK * T + sigma)))
    with (alphaK^2 * h^2 * T * (c1 + D)) by (field; repeat split; lra).
  (*  Goal: h^2 (c1 alphaK)(alphaK T + sigma) <= alphaK^2 h^2 T (c1 + D).
      Cancel c1 alphaK^2 T h^2 from both sides; remains
      h^2 c1 alphaK sigma <= alphaK^2 h^2 T D.
      Sub sigma h^2 = alphaK nu D: LHS = c1 alphaK^2 nu D; RHS = alphaK^2 h^2 T D;
      difference alphaK^2 D (h^2 T - c1 nu) = alphaK^2 D (c2 amag h) >= 0. *)
  assert (Hkey : 0 <= alphaK^2 * D * (h^2 * T - c1 * nu)).
  { assert (Hp : 0 <= alphaK^2 * D) by nn.
    assert (Hq : 0 <= h^2 * T - c1 * nu).
    { assert (Eq : h^2 * T - c1 * nu = c2 * amag * h) by (rewrite Hid; ring).
      rewrite Eq. nn. }
    nra. }
  (*  B - A = alphaK^2 D (h^2 T) - c1 alphaK (sigma h^2); substitute HDid.  *)
  assert (Ediff : alphaK^2 * h^2 * T * (c1 + D)
                  - h^2 * (c1 * alphaK) * (alphaK * T + sigma)
                  = alphaK^2 * D * (h^2 * T)
                    - c1 * alphaK * (sigma * h^2)) by ring.
  rewrite <- HDid in Ediff.
  assert (Ediff2 : alphaK^2 * D * (h^2 * T) - c1 * alphaK * (alphaK * nu * D)
                   = alphaK^2 * D * (h^2 * T - c1 * nu)) by ring.
  rewrite Ediff2 in Ediff.
  lra.
Qed.

Theorem P5_sqrt : / sqrt T2 <= sqrt (c1 + Dah) * (alphaK / h) * sqrt T1.
Proof.
  pose proof T1_pos as HT1. pose proof T2_pos as HT2. pose proof h_pos as Hh.
  pose proof c1_Dah_pos as HcD. pose proof h2_pos as Hh2.
  assert (Hs2 : 0 < sqrt T2) by (apply sqrt_lt_R0; exact HT2).
  apply nonneg_le_of_sqr.
  - left. apply Rinv_0_lt_compat; exact Hs2.
  - nn.
  - replace ((/ sqrt T2)^2) with (/ (sqrt T2 * sqrt T2)) by (field; lra).
    rewrite (sqrt_sqrt T2) by lra.
    replace ((sqrt (c1 + Dah) * (alphaK / h) * sqrt T1)^2)
      with ((sqrt (c1 + Dah) * sqrt (c1 + Dah)) * (alphaK / h)^2 * (sqrt T1 * sqrt T1))
      by (field; lra).
    rewrite (sqrt_sqrt (c1 + Dah)) by lra.
    rewrite (sqrt_sqrt T1) by lra.
    (*  / T2 <= (c1+Dah)(alphaK/h)^2 T1, from h^2 <= (c1+Dah) alphaK^2 T1 T2 (P5_prod). *)
    pose proof P5_prod as HP.
    assert (Hlow : h^2 <= (c1 + Dah) * alphaK^2 * T1 * T2).
    { apply (Rmult_le_reg_r (/ (c1 + Dah))); [apply Rinv_0_lt_compat; lra |].
      replace (h^2 * / (c1 + Dah)) with (h^2 / (c1 + Dah)) by (unfold Rdiv; ring).
      replace ((c1 + Dah) * alphaK^2 * T1 * T2 * / (c1 + Dah))
        with (alphaK^2 * T1 * T2) by (field; lra).
      exact HP. }
    apply (Rmult_le_reg_r (T2 * h^2)); [nra |].
    replace (/ T2 * (T2 * h^2)) with (h^2) by (field; lra).
    replace ((c1 + Dah) * (alphaK / h)^2 * T1 * (T2 * h^2))
      with ((c1 + Dah) * alphaK^2 * T1 * T2) by (field; lra).
    exact Hlow.
Qed.

(* ========================================================================= *)
(*  (P6):  sqrt sigma <= sqrt Dah (alpha_K/h) sqrt tau2.                     *)
(* ========================================================================= *)

Theorem P6 : sqrt sigma <= sqrt Dah * (alphaK / h) * sqrt T2.
Proof.
  pose proof T2_pos as HT2. pose proof h_pos as Hh. pose proof h2_pos as Hh2.
  pose proof Dah_nonneg as HD0. pose proof Dah_id as HDid. pose proof P4_nu as HP4.
  apply nonneg_le_of_sqr.
  - apply sqrt_pos.
  - nn.
  - replace ((sqrt sigma)^2) with (sqrt sigma * sqrt sigma) by ring.
    rewrite (sqrt_sqrt sigma) by exact sigma_nonneg.
    replace ((sqrt Dah * (alphaK / h) * sqrt T2)^2)
      with ((sqrt Dah * sqrt Dah) * (alphaK / h)^2 * (sqrt T2 * sqrt T2))
      by (field; lra).
    rewrite (sqrt_sqrt Dah) by exact HD0.
    rewrite (sqrt_sqrt T2) by lra.
    (*  sigma <= Dah (alphaK/h)^2 T2.
        sigma = alphaK nu Dah / h^2 (from HDid), and nu <= alphaK T2, so
        Dah alphaK^2/h^2 T2 >= Dah alphaK^2/h^2 (nu/alphaK) = Dah alphaK nu/h^2 = sigma. *)
    assert (Hsig : sigma = alphaK * nu * Dah / h^2).
    { apply (Rmult_eq_reg_r (h^2)); [| lra].
      replace (alphaK * nu * Dah / h^2 * h^2) with (alphaK * nu * Dah)
        by (field; lra).
      rewrite HDid. reflexivity. }
    rewrite Hsig.
    apply (Rmult_le_reg_r (h^2)); [lra |].
    replace (alphaK * nu * Dah / h^2 * h^2) with (alphaK * nu * Dah)
      by (field; lra).
    replace (Dah * (alphaK / h)^2 * T2 * h^2) with (Dah * alphaK^2 * T2)
      by (field; lra).
    (*  alphaK nu Dah <= Dah alphaK^2 T2, i.e. Dah alphaK (nu - alphaK T2) <= 0 *)
    assert (Hp : 0 <= Dah * alphaK) by nra.
    nra.
Qed.

(* ========================================================================= *)
(*  (K1)--(K3):  the porosity-weighted inverse-estimate coefficients          *)
(*  (note lem:parameters), driven by (A6).                                   *)
(* ========================================================================= *)

(*  (K1):  Cinv (alpha_K nu)^{1/2}/h <= (1/gamma) tau1^{-1/2}.  *)
Theorem K1 : Cinv * sqrt (alphaK * nu) / h <= / gam * / sqrt T1.
Proof.
  pose proof T1_pos as HT1. pose proof h_pos as Hh. pose proof h2_pos as Hh2.
  assert (Hs1 : 0 < sqrt T1) by (apply sqrt_lt_R0; exact HT1).
  assert (Han : 0 <= alphaK * nu) by nra.
  assert (Hg2 : 0 < gam^2) by nra.
  assert (HsT1 : 0 < sqrt T1 * sqrt T1) by nra.
  apply nonneg_le_of_sqr.
  - nn.
  - assert (Hg : 0 <= / gam) by (left; apply Rinv_0_lt_compat; lra).
    assert (Hsq : 0 <= / sqrt T1) by (left; apply Rinv_0_lt_compat; exact Hs1).
    apply Rmult_le_pos; assumption.
  - replace ((Cinv * sqrt (alphaK * nu) / h)^2)
      with (Cinv^2 * (sqrt (alphaK * nu) * sqrt (alphaK * nu)) / h^2)
      by (field; repeat split; lra).
    rewrite (sqrt_sqrt (alphaK * nu)) by exact Han.
    replace ((/ gam * / sqrt T1)^2) with (/ gam^2 * / (sqrt T1 * sqrt T1))
      by (field; repeat split; lra).
    rewrite (sqrt_sqrt T1) by lra.
    (*  Cinv^2 alphaK nu/h^2 <= /(gam^2 T1),  i.e.  gam^2 T1 Cinv^2 alphaK nu <= h^2  *)
    apply (Rmult_le_reg_r (gam^2 * T1 * h^2)); [ pos |].
    replace (Cinv^2 * (alphaK * nu) / h^2 * (gam^2 * T1 * h^2))
      with (gam^2 * Cinv^2 * (alphaK * nu) * T1) by (field; repeat split; lra).
    replace (/ gam^2 * / T1 * (gam^2 * T1 * h^2)) with (h^2)
      by (field; repeat split; lra).
    (*  gam^2 Cinv^2 (alphaK nu) T1 <= h^2, using 1/T1 = alphaK TNS + sigma. *)
    pose proof T1_inv as Hinv.
    pose proof TNS_h2_id as Hid.
    pose proof TNS_pos as HTN. pose proof denom_pos as HDN.
    apply (Rmult_le_reg_r (/ T1)); [apply Rinv_0_lt_compat; exact HT1 |].
    replace (gam^2 * Cinv^2 * (alphaK * nu) * T1 * / T1)
      with (gam^2 * Cinv^2 * (alphaK * nu)) by (field; lra).
    rewrite Hinv.
    unfold ContinuityAlgebra.phi1.
    assert (Hstep1 : gam^2 * Cinv^2 * (alphaK * nu) <= c1 * (alphaK * nu)) by nra.
    assert (Hstep2 : c1 * (alphaK * nu) <= h^2 * (alphaK * TNS + sigma)).
    { assert (Hle : c1 * nu <= h^2 * TNS).
      { rewrite Hid. assert (0 <= c2 * amag * h) by nn. lra. }
      assert (H0 : 0 <= h^2 * sigma) by nn.
      nra. }
    lra.
Qed.

(*  (K2):  Cinv alpha_K |a|/h <= (1/gamma) tau1^{-1}.  *)
Theorem K2 : Cinv * alphaK * amag / h <= / gam * / T1.
Proof.
  pose proof T1_pos as HT1. pose proof h_pos as Hh.
  pose proof T1_inv as Hinv. pose proof TNS_h2_id as Hid.
  pose proof TNS_pos as HTN.
  rewrite Hinv. unfold ContinuityAlgebra.phi1.
  (*  Cinv alphaK amag/h <= (1/gam)(alphaK TNS + sigma).
      h*TNS >= c2 amag (from Hid), and gam Cinv <= c2, so
      alphaK TNS >= alphaK c2 amag/h >= alphaK gam Cinv amag/h. *)
  assert (HhT : c2 * amag <= h * TNS).
  { apply (Rmult_le_reg_r h); [exact h_pos |].
    replace (h * TNS * h) with (h^2 * TNS) by ring.
    rewrite Hid. assert (0 <= c1 * nu) by nn. lra. }
  apply (Rmult_le_reg_r (gam * h)); [ pos |].
  replace (Cinv * alphaK * amag / h * (gam * h))
    with (gam * Cinv * alphaK * amag) by (field; lra).
  replace (/ gam * (alphaK * TNS + sigma) * (gam * h))
    with (h * (alphaK * TNS + sigma)) by (field; lra).
  (*  gam Cinv alphaK amag <= h(alphaK TNS + sigma).  *)
  assert (Haa : 0 <= alphaK * amag) by nn.
  assert (Hstep1 : gam * Cinv * alphaK * amag <= c2 * alphaK * amag).
  { replace (gam * Cinv * alphaK * amag) with ((gam * Cinv) * (alphaK * amag)) by ring.
    replace (c2 * alphaK * amag) with (c2 * (alphaK * amag)) by ring.
    apply Rmult_le_compat_r; [exact Haa | exact c2_ge]. }
  assert (Hstep2 : c2 * alphaK * amag <= alphaK * (h * TNS)).
  { replace (c2 * alphaK * amag) with (alphaK * (c2 * amag)) by ring.
    apply Rmult_le_compat_l; [lra | exact HhT]. }
  assert (Hstep3 : alphaK * (h * TNS) <= h * (alphaK * TNS + sigma)).
  { assert (0 <= h * sigma) by nn. nra. }
  lra.
Qed.

(*  (K3):  Cinv alpha_K (tau1 tau2)^{1/2}/h <= 1/gamma.  *)
Theorem K3 : Cinv * alphaK * sqrt (T1 * T2) / h <= / gam.
Proof.
  pose proof T1_pos as HT1. pose proof T2_pos as HT2.
  pose proof h_pos as Hh. pose proof h2_pos as Hh2.
  assert (HT12 : 0 <= T1 * T2) by nra.
  assert (Hg2 : 0 < gam^2) by nra.
  assert (Ha2 : 0 < alphaK^2) by nra.
  apply nonneg_le_of_sqr.
  - nn.
  - left; apply Rinv_0_lt_compat; lra.
  - replace ((Cinv * alphaK * sqrt (T1 * T2) / h)^2)
      with (Cinv^2 * alphaK^2 * (sqrt (T1 * T2) * sqrt (T1 * T2)) / h^2)
      by (field; repeat split; lra).
    rewrite (sqrt_sqrt (T1 * T2)) by exact HT12.
    replace ((/ gam)^2) with (/ (gam^2)) by (field; repeat split; lra).
    (*  Cinv^2 alphaK^2 (T1 T2)/h^2 <= 1/gam^2, from (P2) and c1 >= gam^2 Cinv^2. *)
    pose proof P2 as HP2.
    apply (Rmult_le_reg_r (gam^2 * h^2)); [ pos |].
    replace (Cinv^2 * alphaK^2 * (T1 * T2) / h^2 * (gam^2 * h^2))
      with (gam^2 * Cinv^2 * alphaK^2 * (T1 * T2)) by (field; repeat split; lra).
    replace (/ gam^2 * (gam^2 * h^2)) with (h^2) by (field; repeat split; lra).
    (*  gam^2 Cinv^2 alphaK^2 (T1 T2) <= h^2.  *)
    assert (Hcoef : 0 <= gam^2 * Cinv^2 * alphaK^2) by nn.
    assert (Hstep : gam^2 * Cinv^2 * alphaK^2 * (T1 * T2)
                    <= gam^2 * Cinv^2 * alphaK^2 * (h^2 / (c1 * alphaK^2))).
    { apply Rmult_le_compat_l; [exact Hcoef | exact HP2]. }
    assert (Eid : gam^2 * Cinv^2 * alphaK^2 * (h^2 / (c1 * alphaK^2))
                  = gam^2 * Cinv^2 * h^2 / c1) by (field; repeat split; lra).
    rewrite Eid in Hstep.
    assert (Hfrac : gam^2 * Cinv^2 * h^2 / c1 <= h^2).
    { apply (Rmult_le_reg_r c1); [lra |].
      replace (gam^2 * Cinv^2 * h^2 / c1 * c1) with (gam^2 * Cinv^2 * h^2)
        by (field; lra).
      (*  gam^2 Cinv^2 h^2 <= h^2 c1, from gam^2 Cinv^2 <= c1 and h^2 >= 0.  *)
      assert (Hh0 : 0 <= h^2) by nn.
      replace (gam^2 * Cinv^2 * h^2) with ((gam^2 * Cinv^2) * h^2) by ring.
      replace (h^2 * c1) with (c1 * h^2) by ring.
      apply Rmult_le_compat_r; [exact Hh0 | exact c1_ge]. }
    lra.
Qed.

End OsgsParameters.
