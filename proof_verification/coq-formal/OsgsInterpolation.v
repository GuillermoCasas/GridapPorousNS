(* ========================================================================= *)
(*  OsgsInterpolation.v                                                      *)
(*                                                                           *)
(*  Lemma 5.1 (lem:consistency, the OSGS consistency error) and Lemma 5.2    *)
(*  (lem:interpolation, interpolation continuity) of                         *)
(*  theory/osgs_a_priori/osgs_convergence.tex.                              *)
(*                                                                           *)
(*  BOTH OSGS error functionals of the appendix are defined here             *)
(*  CONCRETELY, and the theorems are stated in the SHARP one:                 *)
(*                                                                           *)
(*    Psi_O(h) = ( sum_K (c1 + Da_{h,K}) (alpha_K/h_K)^2                     *)
(*                  ( tau_{2,K} E_int(u)^2 + tau_{1,K} E_int(p)^2 ) )^{1/2}  *)
(*                                            -- oa:eq:ErrorFunctionL2,      *)
(*    E(h)     = sum_K (alpha_K/h_K)(1 + Da_{h,K})^{1/2}                     *)
(*             ( tau_{2,K}^{1/2} E_int(u) + tau_{1,K}^{1/2} E_int(p) )       *)
(*                                            -- oa:eq:ErrorFunction,        *)
(*                                                                           *)
(*  the latter being the ASGS error function of the companion multiplied      *)
(*  elementwise by the square root of one plus the local mesh Damkohler       *)
(*  number Da_{h,K} = sigma h_K^2/(alpha_K nu) (OsgsParameters.(P5)).         *)
(*  Psi_O is the functional in which the appendix proves both lemmas (its     *)
(*  per-group bounds are assembled by elemental Cauchy--Schwarz and an l2     *)
(*  sum) and in which the paper states oa:th:convergence; the l1 passage to   *)
(*  E(h) can lose a mesh-cardinality factor ~h^{-d/2}, so the optimal-order   *)
(*  corollaries must be read off Psi_O.  The relation                         *)
(*  Psi_O(h) <= c1^{1/2} E(h) printed in the appendix is proved here as       *)
(*  PsiO_le_Eh.                                                              *)
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
(*  C Psi_O(h) |||V|||) is proved IN FULL here from the real-number axioms.  *)
(*                                                                           *)
(*  No Admitted, no Axiom.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

(*  One elementary monotonicity step, factored out so that the nonlinear
    arithmetic stays first-degree at each use (nra does not see through the
    triple product on its own).                                              *)
Lemma prod_mono3 : forall c d W S Q : R,
  0 <= W -> 0 <= S -> S <= Q -> 0 <= c -> c <= d -> c * W * S <= d * (W * Q).
Proof.
  intros c d W S Q HW HS HSQ Hc Hcd.
  assert (HQ : 0 <= Q) by lra.
  assert (HcW : 0 <= c * W) by nra.
  assert (H1 : c * W * S <= c * W * Q) by nra.
  assert (HWQ : 0 <= W * Q) by nra.
  assert (H2 : c * (W * Q) <= d * (W * Q)) by nra.
  nra.
Qed.

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

(* ---------- The OSGS error functionals ------------------------------------- *)
(*  The theorems below are stated in the porosity-weighted broken l2 functional
    Psi_O(h) (oa:eq:ErrorFunctionL2), which is the functional the appendix's
    interpolation and consistency lemmas deliver directly (their per-group
    bounds are assembled by elemental Cauchy--Schwarz and an l2 sum, with no
    l2-in-l1 passage) and the functional in which the paper states
    oa:th:convergence.  Its coarser l1 majorant E(h) (oa:eq:ErrorFunction) is
    also defined, and the relation Psi_O(h) <= c1^(1/2) E(h) printed in the
    appendix is proved below as PsiO_le_Eh.  The distinction is not cosmetic:
    the l1 passage can lose a mesh-cardinality factor ~h^(-d/2) on a
    quasi-uniform family, so the optimal-order corollaries must be read off
    Psi_O.                                                                    *)

Definition PsiTerm (k : K) : R :=
  (c1 + Dah k) * (aK k / hK k)^2
    * (t2 k * (IU k)^2 + t1 k * (IP k)^2).

Definition PsiO : R := sqrt (Rsum Th PsiTerm).

Definition ErrTerm (k : K) : R :=
  aK k / hK k * sqrt (1 + Dah k)
    * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k).

Definition Eh : R := Rsum Th ErrTerm.

Lemma PsiTerm_nonneg : forall k, 0 <= PsiTerm k.
Proof.
  intro k. unfold PsiTerm.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  pose proof (IU_nonneg k) as Hu. pose proof (IP_nonneg k) as Hp.
  pose proof (Dah_nonneg k) as Hd.
  pose proof (pow2_ge_0 (aK k / hK k)) as Hw.
  pose proof (pow2_ge_0 (IU k)) as Hu2. pose proof (pow2_ge_0 (IP k)) as Hp2.
  assert (Hsq : 0 <= t2 k * (IU k)^2 + t1 k * (IP k)^2) by nra.
  assert (Hc : 0 <= c1 + Dah k) by lra.
  apply Rmult_le_pos; [ apply Rmult_le_pos; [ exact Hc | exact Hw ] | exact Hsq ].
Qed.

Lemma PsiO_nonneg : 0 <= PsiO.
Proof. unfold PsiO. apply sqrt_pos. Qed.

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

(*  The relation printed in the appendix just below oa:eq:ErrorFunction:
        Psi_O(h) <= c1^(1/2) E(h),
    valid because c1 + Da_{h,K} <= c1 (1 + Da_{h,K}) when c1 >= 1 (which
    H:design supplies), the elementwise a^2 + b^2 <= (a + b)^2 for nonnegative
    a, b, and the discrete l2-in-l1 inequality (AbstractSums).  The c1 >= 1
    hypothesis is local to this lemma, so the theorems below keep the weaker
    section hypothesis 0 < c1.                                                *)
Lemma PsiO_le_Eh : 1 <= c1 -> PsiO <= sqrt c1 * Eh.
Proof.
  intro Hc1.
  (*  Step 1: elementwise, PsiTerm k <= c1 * (ErrTerm k)^2.  *)
  assert (Hterm : forall k, PsiTerm k <= c1 * (ErrTerm k)^2).
  { intro k.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
    pose proof (IU_nonneg k) as Hu. pose proof (IP_nonneg k) as Hp.
    pose proof (Dah_nonneg k) as Hd.
    unfold PsiTerm, ErrTerm.
    assert (Hdas : sqrt (1 + Dah k) * sqrt (1 + Dah k) = 1 + Dah k)
      by (apply sqrt_sqrt; lra).
    assert (Ht2s : sqrt (t2 k) * sqrt (t2 k) = t2 k) by (apply sqrt_sqrt; lra).
    assert (Ht1s : sqrt (t1 k) * sqrt (t1 k) = t1 k) by (apply sqrt_sqrt; lra).
    assert (Hq2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (Hq1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    (*  expand the square of the l1 element factor (pure ring)  *)
    assert (Es : (aK k / hK k * sqrt (1 + Dah k)
                  * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k))^2
                 = (aK k / hK k)^2 * (sqrt (1 + Dah k) * sqrt (1 + Dah k))
                   * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k)^2) by ring.
    rewrite Es, Hdas.
    assert (Erhs : c1 * ((aK k / hK k)^2 * (1 + Dah k)
                         * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k)^2)
                   = c1 * (1 + Dah k) * ((aK k / hK k)^2
                         * (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k)^2)) by ring.
    rewrite Erhs.
    (*  the two elementwise facts, then the factored monotonicity step  *)
    assert (Ha2 : t2 k * (IU k)^2 = (sqrt (t2 k) * IU k)^2).
    { replace ((sqrt (t2 k) * IU k)^2)
        with ((sqrt (t2 k) * sqrt (t2 k)) * (IU k)^2) by ring.
      rewrite Ht2s. ring. }
    assert (Hb2 : t1 k * (IP k)^2 = (sqrt (t1 k) * IP k)^2).
    { replace ((sqrt (t1 k) * IP k)^2)
        with ((sqrt (t1 k) * sqrt (t1 k)) * (IP k)^2) by ring.
      rewrite Ht1s. ring. }
    assert (Hsq : t2 k * (IU k)^2 + t1 k * (IP k)^2
                  <= (sqrt (t2 k) * IU k + sqrt (t1 k) * IP k)^2).
    { rewrite Ha2, Hb2.
      assert (0 <= sqrt (t2 k) * IU k) by nra.
      assert (0 <= sqrt (t1 k) * IP k) by nra.
      nra. }
    apply prod_mono3.
    - apply pow2_ge_0.
    - rewrite Ha2, Hb2.
      pose proof (pow2_ge_0 (sqrt (t2 k) * IU k)).
      pose proof (pow2_ge_0 (sqrt (t1 k) * IP k)). lra.
    - exact Hsq.
    - lra.
    - nra. }
  (*  Step 2: sum over the mesh, then l2-in-l1.  *)
  unfold PsiO, Eh.
  assert (HS : Rsum Th PsiTerm <= Rsum Th (fun k => c1 * (ErrTerm k)^2))
    by (apply Rsum_le; exact Hterm).
  assert (HE : Rsum Th (fun k => c1 * (ErrTerm k)^2)
               = c1 * Rsum Th (fun k => (ErrTerm k)^2))
    by (apply Rsum_scal).
  assert (Hmono : sqrt (Rsum Th PsiTerm)
                  <= sqrt (c1 * Rsum Th (fun k => (ErrTerm k)^2)))
    by (apply sqrt_mono; lra).
  assert (Hsplit : sqrt (c1 * Rsum Th (fun k => (ErrTerm k)^2))
                   = sqrt c1 * sqrt (Rsum Th (fun k => (ErrTerm k)^2)))
    by (apply sqrt_mult; [ lra | apply Rsum_nonneg; intro k; apply pow2_ge_0 ]).
  assert (Hl1 : sqrt (Rsum Th (fun k => (ErrTerm k)^2)) <= Rsum Th ErrTerm)
    by (apply sqrt_sum_sq_le_sum; exact ErrTerm_nonneg).
  assert (Hsc : 0 <= sqrt c1) by apply sqrt_pos.
  assert (Hchain : sqrt c1 * sqrt (Rsum Th (fun k => (ErrTerm k)^2))
                   <= sqrt c1 * Rsum Th ErrTerm)
    by (apply Rmult_le_compat_l; [ exact Hsc | exact Hl1 ]).
  lra.
Qed.

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
Hypothesis HI1 : Rabs I1 <= kI1 * (PsiO * NV).
Hypothesis HI2 : Rabs I2 <= kI2 * (PsiO * NV).
Hypothesis HI3 : Rabs I3 <= kI3 * (PsiO * NV).
Hypothesis HI4 : Rabs I4 <= kI4 * (PsiO * NV).
Hypothesis HI5 : Rabs I5 <= kI5 * (PsiO * NV).
Hypothesis HI6 : Rabs I6 <= kI6 * (PsiO * NV).
Hypothesis HI7 : Rabs I7 <= kI7 * (PsiO * NV).

Definition CtotI : R := kI1 + kI2 + kI3 + kI4 + kI5 + kI6 + kI7.

Lemma CtotI_nonneg : 0 <= CtotI.
Proof. unfold CtotI. lra. Qed.

Theorem abstract_osgs_continterp : Rabs BEV <= CtotI * (PsiO * NV).
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
Hypothesis HS1 : Rabs S1v <= kS1 * (PsiO * NV).
Hypothesis HS2 : Rabs S2v <= kS2 * (PsiO * NV).

Definition CconsI : R := kS1 + kS2.

Lemma CconsI_nonneg : 0 <= CconsI.
Proof. unfold CconsI. lra. Qed.

Theorem abstract_osgs_consistency : Rabs BcU <= CconsI * (PsiO * NV).
Proof.
  rewrite HdecompC.
  eapply Rle_trans; [ apply Rabs_triang |].
  pose proof HS1. pose proof HS2. unfold CconsI. nra.
Qed.

End AbstractOsgsInterp.
