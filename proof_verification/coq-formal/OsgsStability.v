(* ========================================================================= *)
(*  OsgsStability.v                                                          *)
(*                                                                           *)
(*  Theorem 4.1 (th:stability) of theory/osgs_a_priori/osgs_convergence.tex: *)
(*  the inf--sup stability of the OSGS bilinear form, proved via the special  *)
(*  test function W = U_h + V0 of the note.  The theorem exhibits W with      *)
(*                                                                           *)
(*     B_osgs(U_h, W) >= beta_stab * |||U_h|||^2,                            *)
(*     |||W|||^2      <= C_W^2 * |||U_h|||^2,                                 *)
(*                                                                           *)
(*  which is exactly what the convergence composition (Theorem 5.4) consumes: *)
(*  applied to the discrete error it yields  beta_stab |||eta|||             *)
(*  <= sup_V B_osgs(eta,V)/|||V|||  (eq:InfSup).                             *)
(*                                                                           *)
(*  ENCODING.  Steps 1--3 of th:stability -- the diagonal identity            *)
(*  eq:Diagonal (= eq:StepDiag) and the seven term estimates T1--T7 of the    *)
(*  special test function, each packaging the smoothing lemma lem:smoothing,  *)
(*  the porosity-weighted inverse-estimate coefficients (K1)--(K3) of         *)
(*  OsgsParameters.v, and elemental Cauchy--Schwarz / Young -- are the NAMED  *)
(*  ANALYTIC TRUSTED BASE (Hdiag_expand, HT1--HT7), together with the         *)
(*  projection-stability condition (A7) HA7, the exact mass decomposition     *)
(*  Hmass, projection nonexpansiveness HP1nex, and the test-function          *)
(*  boundedness HV0/HW.  Steps 4--5 -- the coefficient collection (every      *)
(*  collected coefficient is >= 1/8 under gamma >= 10(1+M2) and               *)
(*  psi <= min(1/8, beta0^2/16)) and the recovery of the triple norm -- are   *)
(*  proved IN FULL here from the real-number axioms.                          *)
(*                                                                           *)
(*  Scalar quantities:                                                       *)
(*    Dnu = 2 nu ||a^{1/2} P grad u_h||^2,  Dsig = ||sigma^{1/2} u_h||^2,    *)
(*    Deps = eps ||p_h||^2,  Xtau = ||alpha X(U_h)||_tau1^2,                 *)
(*    Dtau = ||div(alpha u_h)||_tau2^2,  the OSGS triple norm squared being  *)
(*    NormSq = Dnu + Dsig + Deps + Xtau + Dtau;                             *)
(*    O1 = ||Pi1perp xi||_tau1^2,  O2 = ||Pi2perp delta||_tau2^2,           *)
(*    P1 = ||Pi10 xi||_tau1^2,     P2 = ||Pi2 delta||_tau2^2.               *)
(*                                                                           *)
(*  No Admitted, no Axiom.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
Local Open Scope R_scope.

Section AbstractOsgsStability.

(* ---------- The collection constants -------------------------------------- *)

Variables (beta0 gam psi M2 C2 : R).
Hypothesis beta0_pos : 0 < beta0.       (*  (A7) projection-stability const  *)
Hypothesis M2_ge_1   : 1 <= M2.         (*  M2 = 1 + C_nabla/C_inv >= 1       *)
Hypothesis gam_ge    : 10 * (1 + M2) <= gam.   (*  (A6): gamma >= gamma0        *)
Hypothesis psi_nonneg : 0 <= psi.
Hypothesis psi_le_18  : 8 * psi <= 1.          (*  psi <= 1/8                   *)
Hypothesis psi_le_b   : 16 * psi <= beta0^2.   (*  psi <= beta0^2/16            *)
Hypothesis C2_nonneg  : 0 <= C2.
Hypothesis C2_le_1    : C2 <= 1.

Lemma gam_ge_20 : 20 <= gam.
Proof. lra. Qed.

Lemma gam_pos : 0 < gam.
Proof. lra. Qed.

(* ---------- The scalar energies ------------------------------------------- *)

Variables (Dnu Dsig Deps O1 O2 P1 P2 Xtau Dtau NormV0Sq NormWSq Cnorm : R).
Hypothesis Dnu_nn  : 0 <= Dnu.
Hypothesis Dsig_nn : 0 <= Dsig.
Hypothesis Deps_nn : 0 <= Deps.
Hypothesis O1_nn   : 0 <= O1.
Hypothesis O2_nn   : 0 <= O2.
Hypothesis P1_nn   : 0 <= P1.
Hypothesis P2_nn   : 0 <= P2.
Hypothesis Xtau_nn : 0 <= Xtau.
Hypothesis Dtau_nn : 0 <= Dtau.
Hypothesis NormV0Sq_nn : 0 <= NormV0Sq.
Hypothesis Cnorm_nn : 0 <= Cnorm.

Definition NormSq : R := Dnu + Dsig + Deps + Xtau + Dtau.

(* ---------- The value B_osgs(U_h, W) and the seven term estimates ---------- *)

Variables (Bosgs_UW T1 T2 T3 T4 T5 T6 T7 : R).

(*  eq:StepDiag + the T1--T7 expansion of B_osgs(U_h, V0):                    *)
(*     B_osgs(U_h, W) = (Dnu+Dsig+Deps+O1+O2) + (T1+...+T7).                  *)
Hypothesis Hdiag_expand :
  Bosgs_UW = (Dnu + Dsig + Deps + O1 + O2) + (T1 + T2 + T3 + T4 + T5 + T6 + T7).

(*  Step 3 term estimates (each packaging lem:smoothing + (K1)--(K3) + CS).   *)
Hypothesis HT1 : T1 >= - Dnu / 4 - 2 * ((1 + psi) / gam)^2 * P1.
Hypothesis HT2 : T2 >= P1 - (2 * psi / beta0^2) * (P1 + O1).
Hypothesis HT3 : T3 >= - Dsig / 2 - ((1 + psi)^2 / 2) * P1.
Hypothesis HT4 : T4 >= - Deps / 2 - ((1 + psi)^2 * C2 / 2) * P2.
Hypothesis HT5 : T5 >= P2 - psi * (P2 + O2).
Hypothesis HT6 : T6 >= - ((1 + psi) / gam) * (O1 + (P1 + P2) / 2).
Hypothesis HT7 : T7 >= - ((M2 * (1 + psi) / gam) / 2) * (O2 + P1).

(*  (A7), squared consequence eq:Hstab:  ||xi||_tau1^2 <= (2/beta0^2)(P1+O1). *)
Hypothesis HA7 : Xtau <= 2 / beta0^2 * (P1 + O1).
(*  Exact tau2 decomposition of the mass slot:  ||delta||_tau2^2 = P2 + O2.   *)
Hypothesis Hmass : Dtau = P2 + O2.
(*  Projection nonexpansiveness:  P1 = ||Pi10 xi||^2 <= ||xi||^2 = Xtau.      *)
Hypothesis HP1nex : P1 <= Xtau.
(*  Step 5 boundedness of the test function:  |||V0|||^2 <= Cnorm (P1 + P2).  *)
Hypothesis HV0 : NormV0Sq <= Cnorm * (P1 + P2).
(*  Triangle inequality (squared): |||W|||^2 <= 2 |||U_h|||^2 + 2 |||V0|||^2. *)
Hypothesis HW : NormWSq <= 2 * NormSq + 2 * NormV0Sq.

(* ========================================================================= *)
(*  Elementary reciprocal bounds driven by (A6) and psi <= psi0.             *)
(* ========================================================================= *)

Lemma inv_gam_pos : 0 < / gam.
Proof. apply Rinv_0_lt_compat, gam_pos. Qed.

Lemma inv_gam_le : / gam <= / 20.
Proof. apply Rinv_le_contravar; [lra | exact gam_ge_20]. Qed.

Lemma pg_nonneg : 0 <= (1 + psi) / gam.
Proof.
  unfold Rdiv. apply Rmult_le_pos; [lra | left; apply inv_gam_pos].
Qed.

(*  (1+psi)/gam <= 9/160.  *)
Lemma pg_bound : (1 + psi) / gam <= 9 / 160.
Proof.
  pose proof inv_gam_le as Hg. pose proof inv_gam_pos as Hgp.
  assert (E : (1 + psi) / gam = (1 + psi) * / gam) by (unfold Rdiv; ring).
  rewrite E.
  assert (Hb : (1 + psi) * / gam <= (9 / 8) * (/ 20)).
  { apply Rmult_le_compat; lra. }
  lra.
Qed.

(*  ((1+psi)/gam)^2 <= (9/160)^2.  *)
Lemma pg_sq_bound : ((1 + psi) / gam)^2 <= (9 / 160)^2.
Proof.
  pose proof pg_bound as Hpg. pose proof pg_nonneg as Hpg0. nra.
Qed.

(*  M2 * /gam <= 1/10.  *)
Lemma M2g_bound : M2 * / gam <= / 10.
Proof.
  pose proof gam_pos as Hg.
  apply (Rmult_le_reg_r gam); [exact Hg |].
  replace (M2 * / gam * gam) with M2 by (field; lra).
  replace (/ 10 * gam) with (gam / 10) by field.
  lra.
Qed.

(*  M2 (1+psi)/gam <= 9/80.  *)
Lemma M2pg_bound : M2 * (1 + psi) / gam <= 9 / 80.
Proof.
  pose proof M2g_bound as Hm. pose proof gam_pos as Hg. pose proof inv_gam_pos as Hgp.
  assert (HM0 : 0 <= M2 * / gam) by nra.
  assert (E : M2 * (1 + psi) / gam = (M2 * / gam) * (1 + psi)) by (field; lra).
  rewrite E.
  assert (Hb : (M2 * / gam) * (1 + psi) <= (/ 10) * (9 / 8)).
  { apply Rmult_le_compat; lra. }
  lra.
Qed.

(*  2 psi / beta0^2 <= 1/8.  *)
Lemma two_psi_bsq : 2 * psi / beta0^2 <= / 8.
Proof.
  assert (Hb2 : 0 < beta0^2) by nra.
  apply (Rmult_le_reg_r (beta0^2)); [exact Hb2 |].
  replace (2 * psi / beta0^2 * beta0^2) with (2 * psi) by (field; nra).
  replace (/ 8 * beta0^2) with (beta0^2 / 8) by field.
  lra.
Qed.

Lemma two_psi_bsq_nonneg : 0 <= 2 * psi / beta0^2.
Proof.
  unfold Rdiv. apply Rmult_le_pos; [nra | left; apply Rinv_0_lt_compat; nra].
Qed.

(* ========================================================================= *)
(*  Step 4:  every collected coefficient is >= 1/8.                          *)
(* ========================================================================= *)

Lemma coeff_O1 : 1 - 2 * psi / beta0^2 - (1 + psi) / gam >= / 8.
Proof.
  pose proof two_psi_bsq as H1. pose proof pg_bound as H2. lra.
Qed.

Lemma coeff_O2 : 1 - psi - (M2 * (1 + psi) / gam) / 2 >= / 8.
Proof.
  pose proof M2pg_bound as H2. lra.
Qed.

Lemma coeff_P1 :
  1 - 2 * psi / beta0^2 - 2 * ((1 + psi) / gam)^2 - (1 + psi)^2 / 2
    - ((1 + psi) / gam) / 2 - (M2 * (1 + psi) / gam) / 2 >= / 8.
Proof.
  pose proof two_psi_bsq as H1.
  pose proof pg_sq_bound as H2.
  pose proof pg_bound as H3.
  pose proof M2pg_bound as H4.
  assert (Hp2 : (1 + psi)^2 <= (9 / 8)^2) by nra.
  lra.
Qed.

Lemma coeff_P2 :
  1 - psi - (1 + psi)^2 * C2 / 2 - ((1 + psi) / gam) / 2 >= / 8.
Proof.
  pose proof pg_bound as H3.
  assert (Hp2 : (1 + psi)^2 * C2 <= (9 / 8)^2).
  { assert (Hp : (1 + psi)^2 <= (9 / 8)^2) by nra.
    assert (Hc : (1 + psi)^2 * C2 <= (1 + psi)^2 * 1).
    { apply Rmult_le_compat_l; [nra | lra]. }
    nra. }
  lra.
Qed.

(*  The collected lower bound (eq:Collected):  B_osgs(U_h,W) >= C0 * (all).   *)
Lemma step4 :
  Bosgs_UW >= / 8 * (Dnu + Dsig + Deps + O1 + O2 + P1 + P2).
Proof.
  (*  lower-bound B_osgs(U_h,W) via the diagonal identity and HT1--HT7.       *)
  assert (Hlb :
    Bosgs_UW >=
      (Dnu + Dsig + Deps + O1 + O2)
      + ((- Dnu / 4 - 2 * ((1 + psi) / gam)^2 * P1)
         + (P1 - (2 * psi / beta0^2) * (P1 + O1))
         + (- Dsig / 2 - ((1 + psi)^2 / 2) * P1)
         + (- Deps / 2 - ((1 + psi)^2 * C2 / 2) * P2)
         + (P2 - psi * (P2 + O2))
         + (- ((1 + psi) / gam) * (O1 + (P1 + P2) / 2))
         + (- ((M2 * (1 + psi) / gam) / 2) * (O2 + P1)))).
  { rewrite Hdiag_expand.
    pose proof HT1. pose proof HT2. pose proof HT3. pose proof HT4.
    pose proof HT5. pose proof HT6. pose proof HT7. lra. }
  (*  regroup into per-energy coefficients (ring identity in the atoms).      *)
  set (cO1 := 1 - 2 * psi / beta0^2 - (1 + psi) / gam) in *.
  set (cO2 := 1 - psi - (M2 * (1 + psi) / gam) / 2) in *.
  set (cP1 := 1 - 2 * psi / beta0^2 - 2 * ((1 + psi) / gam)^2 - (1 + psi)^2 / 2
              - ((1 + psi) / gam) / 2 - (M2 * (1 + psi) / gam) / 2) in *.
  set (cP2 := 1 - psi - (1 + psi)^2 * C2 / 2 - ((1 + psi) / gam) / 2) in *.
  assert (Hid :
    (Dnu + Dsig + Deps + O1 + O2)
      + ((- Dnu / 4 - 2 * ((1 + psi) / gam)^2 * P1)
         + (P1 - (2 * psi / beta0^2) * (P1 + O1))
         + (- Dsig / 2 - ((1 + psi)^2 / 2) * P1)
         + (- Deps / 2 - ((1 + psi)^2 * C2 / 2) * P2)
         + (P2 - psi * (P2 + O2))
         + (- ((1 + psi) / gam) * (O1 + (P1 + P2) / 2))
         + (- ((M2 * (1 + psi) / gam) / 2) * (O2 + P1)))
    = (3 / 4) * Dnu + (1 / 2) * Dsig + (1 / 2) * Deps
      + cO1 * O1 + cO2 * O2 + cP1 * P1 + cP2 * P2).
  { unfold cO1, cO2, cP1, cP2.
    pose proof gam_pos as Hgam. assert (Hb2 : 0 < beta0^2) by nra.
    field. repeat split; lra. }
  rewrite Hid in Hlb.
  (*  each coefficient dominates 1/8; energies are nonnegative.               *)
  pose proof coeff_O1 as HcO1. fold cO1 in HcO1.
  pose proof coeff_O2 as HcO2. fold cO2 in HcO2.
  pose proof coeff_P1 as HcP1. fold cP1 in HcP1.
  pose proof coeff_P2 as HcP2. fold cP2 in HcP2.
  assert (B1 : (3 / 4) * Dnu >= / 8 * Dnu) by nra.
  assert (B2 : (1 / 2) * Dsig >= / 8 * Dsig) by nra.
  assert (B3 : (1 / 2) * Deps >= / 8 * Deps) by nra.
  assert (B4 : cO1 * O1 >= / 8 * O1).
  { apply Rle_ge. apply Rmult_le_compat_r; [exact O1_nn | lra]. }
  assert (B5 : cO2 * O2 >= / 8 * O2).
  { apply Rle_ge. apply Rmult_le_compat_r; [exact O2_nn | lra]. }
  assert (B6 : cP1 * P1 >= / 8 * P1).
  { apply Rle_ge. apply Rmult_le_compat_r; [exact P1_nn | lra]. }
  assert (B7 : cP2 * P2 >= / 8 * P2).
  { apply Rle_ge. apply Rmult_le_compat_r; [exact P2_nn | lra]. }
  lra.
Qed.

(* ========================================================================= *)
(*  Step 5:  recovery of the triple norm and boundedness of W.               *)
(* ========================================================================= *)

Definition beta_stab : R := Rmin (/ 8) (beta0^2 / 16).

Lemma beta_stab_pos : 0 < beta_stab.
Proof.
  unfold beta_stab. apply Rmin_pos; [lra | nra].
Qed.

Lemma beta_stab_le_18 : beta_stab <= / 8.
Proof. unfold beta_stab. apply Rmin_l. Qed.

Lemma beta_stab_le_b : beta_stab <= beta0^2 / 16.
Proof. unfold beta_stab. apply Rmin_r. Qed.

(*  beta_stab * (2/beta0^2) <= 1/8.  *)
Lemma beta_stab_scaled : beta_stab * (2 / beta0^2) <= / 8.
Proof.
  assert (Hb2 : 0 < beta0^2) by nra.
  pose proof beta_stab_le_b as Hbb.
  apply (Rmult_le_reg_r (beta0^2)); [exact Hb2 |].
  replace (beta_stab * (2 / beta0^2) * beta0^2) with (2 * beta_stab)
    by (field; nra).
  replace (/ 8 * beta0^2) with (beta0^2 / 8) by field.
  lra.
Qed.

(*  The recovery:  B_osgs(U_h, W) >= beta_stab * NormSq  (eq before eq:InfSup). *)
Lemma recovery : Bosgs_UW >= beta_stab * NormSq.
Proof.
  pose proof step4 as Hs4.
  pose proof beta_stab_pos as Hbp.
  pose proof beta_stab_le_18 as Hb18.
  pose proof beta_stab_scaled as Hbsc.
  (*  NormSq = Dnu+Dsig+Deps+Xtau+Dtau; bound Xtau by (2/b^2)(P1+O1) and       *)
  (*  Dtau by P2+O2, then compare coefficient-wise against (1/8)(...).         *)
  assert (HXtau : Xtau <= 2 / beta0^2 * (P1 + O1)) by exact HA7.
  assert (HDtau : Dtau = P2 + O2) by exact Hmass.
  unfold NormSq.
  (*  It suffices:  (1/8)(Dnu+Dsig+Deps+O1+O2+P1+P2)
                    >= beta_stab (Dnu+Dsig+Deps+Xtau+Dtau).                    *)
  assert (Hbound :
    / 8 * (Dnu + Dsig + Deps + O1 + O2 + P1 + P2)
    >= beta_stab * (Dnu + Dsig + Deps + Xtau + Dtau)).
  { (*  compare term by term  *)
    assert (D1 : / 8 * Dnu >= beta_stab * Dnu).
    { apply Rle_ge. apply Rmult_le_compat_r; [exact Dnu_nn | exact Hb18]. }
    assert (D2 : / 8 * Dsig >= beta_stab * Dsig).
    { apply Rle_ge. apply Rmult_le_compat_r; [exact Dsig_nn | exact Hb18]. }
    assert (D3 : / 8 * Deps >= beta_stab * Deps).
    { apply Rle_ge. apply Rmult_le_compat_r; [exact Deps_nn | exact Hb18]. }
    (*  Xtau <= (2/b^2)(P1+O1);  beta_stab (2/b^2) <= 1/8;  P1,O1 >= 0.        *)
    assert (DX : / 8 * (P1 + O1) >= beta_stab * Xtau).
    { assert (HX2 : beta_stab * Xtau <= beta_stab * (2 / beta0^2 * (P1 + O1))).
      { apply Rmult_le_compat_l; [lra | exact HXtau]. }
      assert (HX3 : beta_stab * (2 / beta0^2 * (P1 + O1)) <= / 8 * (P1 + O1)).
      { assert (HPO : 0 <= P1 + O1) by lra.
        assert (E : beta_stab * (2 / beta0^2 * (P1 + O1))
                    = (beta_stab * (2 / beta0^2)) * (P1 + O1)) by ring.
        rewrite E.
        apply Rmult_le_compat_r; [exact HPO | exact Hbsc]. }
      lra. }
    (*  Dtau = P2 + O2;  beta_stab <= 1/8;  P2,O2 >= 0.                        *)
    assert (DD : / 8 * (P2 + O2) >= beta_stab * Dtau).
    { rewrite HDtau.
      apply Rle_ge. apply Rmult_le_compat_r; [lra | exact Hb18]. }
    lra. }
  lra.
Qed.

(*  Boundedness of the test function:  |||W|||^2 <= C_W^2 |||U_h|||^2.        *)
Lemma norm_W_bound : NormWSq <= 2 * (1 + Cnorm) * NormSq.
Proof.
  pose proof HW as HW'. pose proof HV0 as HV0'.
  (*  P1 + P2 <= Xtau + Dtau <= NormSq.                                       *)
  assert (HP2D : P2 <= Dtau) by (rewrite Hmass; lra).
  assert (HPP : P1 + P2 <= NormSq).
  { unfold NormSq. pose proof HP1nex. lra. }
  assert (HV0N : NormV0Sq <= Cnorm * NormSq).
  { assert (HcPP : Cnorm * (P1 + P2) <= Cnorm * NormSq).
    { apply Rmult_le_compat_l; [exact Cnorm_nn | exact HPP]. }
    lra. }
  lra.
Qed.

(* ========================================================================= *)
(*  Theorem 4.1 (th:stability), abstract form.                               *)
(* ========================================================================= *)

Theorem abstract_osgs_stability :
  0 < beta_stab
  /\ Bosgs_UW >= beta_stab * NormSq
  /\ NormWSq <= 2 * (1 + Cnorm) * NormSq.
Proof.
  split; [exact beta_stab_pos |].
  split; [exact recovery | exact norm_W_bound].
Qed.

End AbstractOsgsStability.
