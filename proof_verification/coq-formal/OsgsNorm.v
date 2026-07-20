(* ========================================================================= *)
(*  OsgsNorm.v                                                               *)
(*                                                                           *)
(*  The OSGS mesh-dependent triple norm (theory/osgs_a_priori/               *)
(*  osgs_convergence.tex, eq:TripleNorm) over the abstract inner-product /   *)
(*  mesh interface of InnerSpace.v, together with the norm-comparison lemma   *)
(*  (Lemma 3.5, lem:normcomparison): the OSGS triple norm DOMINATES, term by  *)
(*  term, the ASGS ("S") triple norm of the companion manuscript.  The       *)
(*  single structural difference between the two norms is the fate of the     *)
(*  reactive term: the OSGS norm controls the FULL reactive contribution      *)
(*  ||sigma^{1/2} v||^2, whereas the ASGS norm controls only the damped       *)
(*  ||sigmatilde^{1/2} v||_h^2 (and its viscous coefficient is nu rather than *)
(*  2 nu).  Machine-checked here from sigmatilde <= sigma                     *)
(*  (ContinuityAlgebra.P4_le_sigma) and nu <= 2 nu.                          *)
(*                                                                           *)
(*  The per-element OSGS energy perN defined here is the SHARED energy that   *)
(*  the OSGS stability, interpolation-continuity and convergence files all    *)
(*  build their triple norm from, so their norms coincide by construction     *)
(*  (no cross-module bridge is needed, unlike the ASGS files).               *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

Section OsgsNorm.

Variable Hs : PreHilbert.
Notation V := (carrier Hs).
Notation "'<<' x , y '>>'" := (ip Hs x y) (at level 0).
Notation "x '+v' y" := (vadd Hs x y) (at level 50, left associativity).

Variable K : Type.
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

(*  The companion parameters (SAME as the OSGS note's eq:TauDefs).  *)
Definition t1 (k : K) : R := ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2.
Definition t2 (k : K) : R := ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2.
Definition sg (k : K) : R := ContinuityAlgebra.sigt nu (hK k) (aK k) sigma (am k) c1 c2.

Lemma t1_pos : forall k, 0 < t1 k.
Proof. intro k. unfold t1. apply tau1_pos; auto. Qed.
Lemma t2_pos : forall k, 0 < t2 k.
Proof. intro k. unfold t2. apply tau2_pos; auto. Qed.
Lemma sg_nonneg : forall k, 0 <= sg k.
Proof. intro k. unfold sg. apply sigt_nonneg; auto. Qed.

(* ---------- The per-element OSGS energy (eq:TripleNorm) --------------------- *)

(*  For a field V = [v; q] with elemental atoms                              *)
(*    g   ~ (alpha^{1/2} P grad v)|_K,   v ~ v|_K,   q ~ q|_K,               *)
(*    cx  ~ (alpha a . grad v)|_K,       gp ~ (alpha grad q)|_K,             *)
(*    dv  ~ (div(alpha v))|_K,                                              *)
(*  the composite  x = cx + gp = (alpha X(V))|_K, and the energy            *)
(*    2 nu ||g||^2 + sigma ||v||^2 + eps ||q||^2                            *)
(*      + tau1 ||x||^2 + tau2 ||dv||^2.                                     *)
Definition perN
    (g vv qq cx gp dv : K -> V) (k : K) : R :=
  2 * nu * << (g k) , (g k) >>
  + sigma * << (vv k) , (vv k) >>
  + eps * << (qq k) , (qq k) >>
  + t1 k * << ((cx k) +v (gp k)) , ((cx k) +v (gp k)) >>
  + t2 k * << (dv k) , (dv k) >>.

(*  The ASGS ("S") per-element energy of the companion (lem:normcomparison): *)
(*  viscous coefficient nu (not 2 nu) and the DAMPED reaction sigmatilde.     *)
Definition perS
    (g vv qq cx gp dv : K -> V) (k : K) : R :=
  nu * << (g k) , (g k) >>
  + sg k * << (vv k) , (vv k) >>
  + eps * << (qq k) , (qq k) >>
  + t1 k * << ((cx k) +v (gp k)) , ((cx k) +v (gp k)) >>
  + t2 k * << (dv k) , (dv k) >>.

Definition NN2 (g vv qq cx gp dv : K -> V) : R := Rsum Th (perN g vv qq cx gp dv).
Definition NN  (g vv qq cx gp dv : K -> V) : R := sqrt (NN2 g vv qq cx gp dv).

(* ---------- Nonnegativity of the energies ---------------------------------- *)

Lemma perN_nonneg :
  forall g vv qq cx gp dv k, 0 <= perN g vv qq cx gp dv k.
Proof.
  intros g vv qq cx gp dv k. unfold perN.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  pose proof (ip_pos Hs (g k)) as P1.
  pose proof (ip_pos Hs (vv k)) as P2.
  pose proof (ip_pos Hs (qq k)) as P3.
  pose proof (ip_pos Hs ((cx k) +v (gp k))) as P4.
  pose proof (ip_pos Hs (dv k)) as P5.
  assert (0 <= 2 * nu * << (g k) , (g k) >>) by nra.
  assert (0 <= sigma * << (vv k) , (vv k) >>) by nra.
  assert (0 <= eps * << (qq k) , (qq k) >>) by nra.
  assert (0 <= t1 k * << ((cx k) +v (gp k)) , ((cx k) +v (gp k)) >>) by nra.
  assert (0 <= t2 k * << (dv k) , (dv k) >>) by nra.
  lra.
Qed.

Lemma perS_nonneg :
  forall g vv qq cx gp dv k, 0 <= perS g vv qq cx gp dv k.
Proof.
  intros g vv qq cx gp dv k. unfold perS.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  pose proof (sg_nonneg k) as Hsg.
  pose proof (ip_pos Hs (g k)) as P1.
  pose proof (ip_pos Hs (vv k)) as P2.
  pose proof (ip_pos Hs (qq k)) as P3.
  pose proof (ip_pos Hs ((cx k) +v (gp k))) as P4.
  pose proof (ip_pos Hs (dv k)) as P5.
  assert (0 <= nu * << (g k) , (g k) >>) by nra.
  assert (0 <= sg k * << (vv k) , (vv k) >>) by nra.
  assert (0 <= eps * << (qq k) , (qq k) >>) by nra.
  assert (0 <= t1 k * << ((cx k) +v (gp k)) , ((cx k) +v (gp k)) >>) by nra.
  assert (0 <= t2 k * << (dv k) , (dv k) >>) by nra.
  lra.
Qed.

Lemma NN2_nonneg : forall g vv qq cx gp dv, 0 <= NN2 g vv qq cx gp dv.
Proof.
  intros. unfold NN2. apply Rsum_nonneg. intro k. apply perN_nonneg.
Qed.

Lemma NN_nonneg : forall g vv qq cx gp dv, 0 <= NN g vv qq cx gp dv.
Proof. intros. unfold NN. apply sqrt_pos. Qed.

(* ========================================================================= *)
(*  Lemma 3.5 (lem:normcomparison):  perS k <= perN k, term by term.          *)
(*  Uses nu <= 2 nu (on ||g||^2 >= 0) and sigmatilde <= sigma                 *)
(*  (ContinuityAlgebra.P4_le_sigma, on ||v||^2 >= 0); the remaining three     *)
(*  terms coincide.                                                          *)
(* ========================================================================= *)

Lemma perS_le_perN :
  forall g vv qq cx gp dv k, perS g vv qq cx gp dv k <= perN g vv qq cx gp dv k.
Proof.
  intros g vv qq cx gp dv k. unfold perS, perN.
  pose proof (ip_pos Hs (g k)) as Pg.
  pose proof (ip_pos Hs (vv k)) as Pv.
  assert (Hnu : nu * << (g k) , (g k) >> <= 2 * nu * << (g k) , (g k) >>) by nra.
  assert (Hsg : sg k <= sigma).
  { unfold sg. apply P4_le_sigma; auto. }
  assert (Hreac : sg k * << (vv k) , (vv k) >> <= sigma * << (vv k) , (vv k) >>).
  { apply Rmult_le_compat_r; [exact Pv | exact Hsg]. }
  lra.
Qed.

(*  Global comparison of the two triple norms (squared).  *)
Lemma NormS2_le_NN2 :
  forall g vv qq cx gp dv,
    Rsum Th (perS g vv qq cx gp dv) <= NN2 g vv qq cx gp dv.
Proof.
  intros. unfold NN2. apply Rsum_le. intro k. apply perS_le_perN.
Qed.

End OsgsNorm.
