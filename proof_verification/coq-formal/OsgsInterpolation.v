(* ========================================================================= *)
(*  OsgsInterpolation.v                                                      *)
(*                                                                           *)
(*  Lemma 5.1 (lem:consistency, the OSGS consistency error) and Lemma 5.2    *)
(*  (lem:interpolation, interpolation continuity) of                         *)
(*  theory/osgs_a_priori/osgs_convergence.tex.                              *)
(*                                                                           *)
(*  The OSGS error function E(h) (eq:ErrorFunction) is defined here          *)
(*  CONCRETELY as the l1 sum over the mesh                                    *)
(*                                                                           *)
(*    E(h) = sum_K (alpha_K/h_K)(1 + Da_{h,K})^{1/2}                         *)
(*             ( tau_{2,K}^{1/2} E_int(u) + tau_{1,K}^{1/2} E_int(p) ),      *)
(*                                                                           *)
(*  the ASGS error function of the companion multiplied elementwise by the   *)
(*  square root of one plus the local mesh Damkohler number Da_{h,K} =       *)
(*  sigma h_K^2/(alpha_K nu) -- the SOLE new factor (OsgsParameters.(P5)).   *)
(*                                                                           *)
(*  ENCODING.  The seven per-term interpolation bounds (I1)--(I7) of         *)
(*  lem:interpolation and the two consistency-slot bounds (S1),(S2) of       *)
(*  lem:consistency -- each of which the note derives from the nodal /       *)
(*  Scott--Zhang interpolation estimates (eq:NodalInterp), the best-         *)
(*  approximation lemma lem:bestapprox, and the parameter conversions        *)
(*  (P3)--(P6) of OsgsParameters.v (the two Damkohler factors entering in    *)
(*  (I2)/(I5) and the compressibility (I4)) together with elemental          *)
(*  Cauchy--Schwarz -- are the NAMED ANALYTIC TRUSTED BASE.  The ASSEMBLY    *)
(*  (the triangle inequality over the terms, collecting them into            *)
(*  C E(h) |||V|||) is proved IN FULL here from the real-number axioms.      *)
(*                                                                           *)
(*  No Admitted, no Axiom.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

Section AbstractOsgsInterp.

(* ---------- Ambient space and mesh ---------------------------------------- *)

Variable Hs : PreHilbert.
Variable K  : Type.
Variable Th : list K.

Variables (nu sigma eps c1 c2 : R).
Hypothesis nu_pos       : 0 < nu.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis eps_nonneg   : 0 <= eps.
Hypothesis c1_pos       : 0 < c1.
Hypothesis c2_pos       : 0 < c2.

Variables (hK aK am : K -> R).
Hypothesis hK_pos    : forall k, 0 < hK k.
Hypothesis aK_pos    : forall k, 0 < aK k.
Hypothesis am_nonneg : forall k, 0 <= am k.

Variables (IU IP : K -> R).
Hypothesis IU_nonneg : forall k, 0 <= IU k.
Hypothesis IP_nonneg : forall k, 0 <= IP k.

(*  Parameters and the elementwise mesh Damkohler number.  *)
Definition t1 (k : K) : R := ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2.
Definition t2 (k : K) : R := ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2.
Definition Dah (k : K) : R := sigma * (hK k)^2 / (aK k * nu).

Lemma t1_pos : forall k, 0 < t1 k.
Proof. intro k. unfold t1. apply tau1_pos; auto. Qed.
Lemma t2_pos : forall k, 0 < t2 k.
Proof. intro k. unfold t2. apply tau2_pos; auto. Qed.
Lemma Dah_nonneg : forall k, 0 <= Dah k.
Proof.
  intro k. unfold Dah. apply div_nonneg.
  - pose proof (hK_pos k). nra.
  - pose proof (aK_pos k). nra.
Qed.

(* ---------- The OSGS error function E(h) (eq:ErrorFunction) ---------------- *)

Definition ErrTerm (k : K) : R :=
  aK k / hK k * sqrt (1 + Dah k)
    * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k).

Definition Eh : R := Rsum Th ErrTerm.

Lemma ErrTerm_nonneg : forall k, 0 <= ErrTerm k.
Proof.
  intro k. unfold ErrTerm.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  pose proof (IU_nonneg k) as Hu. pose proof (IP_nonneg k) as Hp.
  pose proof (Dah_nonneg k) as Hd.
  assert (Ha0 : 0 <= aK k / hK k) by (apply div_nonneg; lra).
  assert (Hs1 : 0 <= sqrt (1 + Dah k)) by apply sqrt_pos.
  assert (Hs2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
  assert (Hs3 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
  assert (Hbr : 0 <= sqrt (t2 k) * IU k + sqrt (t1 k) * IP k) by nra.
  apply Rmult_le_pos; [apply Rmult_le_pos; [exact Ha0 | exact Hs1] | exact Hbr].
Qed.

Lemma Eh_nonneg : 0 <= Eh.
Proof. unfold Eh. apply Rsum_nonneg. exact ErrTerm_nonneg. Qed.

(* ========================================================================= *)
(*  Lemma 5.2 (lem:interpolation): interpolation continuity.                 *)
(*                                                                           *)
(*  The value B_osgs(Ehat, V_h) decomposes into the seven groups (I1)--(I7)  *)
(*  of the proof (viscous, convective+mass-divergence, reactive,             *)
(*  compressibility, pressure-gradient, momentum-stabilization,              *)
(*  mass-stabilization); each is bounded by its interpolation contribution   *)
(*  times |||V_h|||.                                                         *)
(* ========================================================================= *)

Variable NV : R.                 (*  |||V_h|||  *)
Hypothesis NV_nonneg : 0 <= NV.

Variables (BEV I1 I2 I3 I4 I5 I6 I7 : R).
Variables (kI1 kI2 kI3 kI4 kI5 kI6 kI7 : R).
Hypothesis kI1_nn : 0 <= kI1.  Hypothesis kI2_nn : 0 <= kI2.
Hypothesis kI3_nn : 0 <= kI3.  Hypothesis kI4_nn : 0 <= kI4.
Hypothesis kI5_nn : 0 <= kI5.  Hypothesis kI6_nn : 0 <= kI6.
Hypothesis kI7_nn : 0 <= kI7.

Hypothesis HdecompI : BEV = I1 + I2 + I3 + I4 + I5 + I6 + I7.
Hypothesis HI1 : Rabs I1 <= kI1 * (Eh * NV).
Hypothesis HI2 : Rabs I2 <= kI2 * (Eh * NV).
Hypothesis HI3 : Rabs I3 <= kI3 * (Eh * NV).
Hypothesis HI4 : Rabs I4 <= kI4 * (Eh * NV).
Hypothesis HI5 : Rabs I5 <= kI5 * (Eh * NV).
Hypothesis HI6 : Rabs I6 <= kI6 * (Eh * NV).
Hypothesis HI7 : Rabs I7 <= kI7 * (Eh * NV).

Definition CtotI : R := kI1 + kI2 + kI3 + kI4 + kI5 + kI6 + kI7.

Lemma CtotI_nonneg : 0 <= CtotI.
Proof. unfold CtotI. lra. Qed.

Theorem abstract_osgs_continterp : Rabs BEV <= CtotI * (Eh * NV).
Proof.
  rewrite HdecompI.
  (*  triangle inequality over the seven groups  *)
  assert (Htri :
    Rabs (I1 + (I2 + (I3 + (I4 + (I5 + (I6 + I7))))))
    <= Rabs I1 + Rabs I2 + Rabs I3 + Rabs I4 + Rabs I5 + Rabs I6 + Rabs I7).
  { pose proof (Rabs_triang I1 (I2 + (I3 + (I4 + (I5 + (I6 + I7)))))) as T1.
    pose proof (Rabs_triang I2 (I3 + (I4 + (I5 + (I6 + I7))))) as T2.
    pose proof (Rabs_triang I3 (I4 + (I5 + (I6 + I7)))) as T3.
    pose proof (Rabs_triang I4 (I5 + (I6 + I7))) as T4.
    pose proof (Rabs_triang I5 (I6 + I7)) as T5.
    pose proof (Rabs_triang I6 I7) as T6.
    lra. }
  (*  reassociate the sum to match Htri, then apply the term bounds  *)
  assert (Erw : I1 + I2 + I3 + I4 + I5 + I6 + I7
                = I1 + (I2 + (I3 + (I4 + (I5 + (I6 + I7)))))) by ring.
  rewrite Erw.
  eapply Rle_trans; [ exact Htri |].
  pose proof HI1. pose proof HI2. pose proof HI3. pose proof HI4.
  pose proof HI5. pose proof HI6. pose proof HI7.
  unfold CtotI. nra.
Qed.

(* ========================================================================= *)
(*  Lemma 5.1 (lem:consistency): the OSGS consistency error.                 *)
(*                                                                           *)
(*  B_osgs(U - U_h, V_h) = S1(U,V_h) + S2(U,V_h)  (eq:ConsistencyId), and     *)
(*  both slots are bounded by C E(h) |||V_h|||  (eq:ConsistencyBound); the S2 *)
(*  slot carries the compressibility Damkohler factor through (P5).          *)
(* ========================================================================= *)

Variables (BcU S1v S2v : R).
Variables (kS1 kS2 : R).
Hypothesis kS1_nn : 0 <= kS1.  Hypothesis kS2_nn : 0 <= kS2.
Hypothesis HdecompC : BcU = S1v + S2v.
Hypothesis HS1 : Rabs S1v <= kS1 * (Eh * NV).
Hypothesis HS2 : Rabs S2v <= kS2 * (Eh * NV).

Definition CconsI : R := kS1 + kS2.

Lemma CconsI_nonneg : 0 <= CconsI.
Proof. unfold CconsI. lra. Qed.

Theorem abstract_osgs_consistency : Rabs BcU <= CconsI * (Eh * NV).
Proof.
  rewrite HdecompC.
  eapply Rle_trans; [ apply Rabs_triang |].
  pose proof HS1. pose proof HS2. unfold CconsI. nra.
Qed.

End AbstractOsgsInterp.
