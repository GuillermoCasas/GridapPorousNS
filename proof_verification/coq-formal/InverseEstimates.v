(* ========================================================================= *)
(*  InverseEstimates.v                                                       *)
(*                                                                           *)
(*  A NOTATIONAL schema for the Class-I weighted inverse estimates of        *)
(*  lem:winv (the "winv" family).  Each of the nine hypotheses               *)
(*    Hw_gu, Hw_gv, Hw_du, Hw_dv, Hw_cxu, Hw_cxv, Hw_gpu, Hw_divu, (S3)      *)
(*  shares the shape                                                         *)
(*        forall k, ||A k|| <= C / h_K k * W k * ||B k||                     *)
(*  with A a bounded discrete atom, B a source atom, C a constant and        *)
(*  W k in { sqrt(alpha_K), alpha_K, alpha_K |a| }.  `winv_est C W A B'      *)
(*  packages exactly that proposition.                                       *)
(*                                                                           *)
(*  [must-test / known-fragility]  winv_est is a PREDICATE the nine          *)
(*  hypotheses INSTANTIATE, not a single hypothesis that replaces them.      *)
(*  Each `Hypothesis Hw_x : winv_est ... ' unfolds DEFINITIONALLY to the     *)
(*  very estimate it used to state (the "C / h * W * ||.||" association is    *)
(*  chosen so six of the nine match up to nothing more than beta; the        *)
(*  cxu/cxv/divu weights re-associate and so match only PROPOSITIONALLY,      *)
(*  which every consumer -- all `pose proof (Hw_x k); ... nra' -- absorbs).   *)
(*  So the trusted base is logically UNCHANGED: nine named inverse           *)
(*  estimates remain nine named inverse estimates.  This does NOT reduce     *)
(*  the trusted base, and MUST NOT be turned into one.                       *)
(*                                                                           *)
(*  The atoms A, B are DISTINCT per hypothesis and mutually independent; a   *)
(*  single estimate quantified over an ARBITRARY vector                      *)
(*        forall x : V, ||D x|| <= C / h * W * ||x||                         *)
(*  would be UNSOUND -- the discrete atoms and the interpolation-error       *)
(*  atoms inhabit the SAME carrier, so a forall-x inverse estimate would     *)
(*  license one for a non-polynomial interpolation error, which is false.    *)
(*  The schema is notation over the nine, never a tenth hypothesis.          *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
From PNSFormal Require Import InnerSpace.
Local Open Scope R_scope.

Section InverseEstimates.

Variable Hs : PreHilbert.
Variable K  : Type.
Variable hK : K -> R.
Hypothesis hK_pos : forall k, 0 < hK k.

(*  The winv shape, in the "C / h * W * ||.||" association (paper "Shape A"). *)
Definition winv_est (C : R) (W : K -> R) (A B : K -> carrier Hs) : Prop :=
  forall k, nrm (A k) <= C / hK k * W k * nrm (B k).

(*  Composition of two winv estimates (eq:doubleinv):                        *)
(*      ||A|| <= C1 / h * W1 * ||M||   and   ||M|| <= C2 / h * W2 * ||B||     *)
(*  chain into                                                               *)
(*      ||A|| <= C1 C2 (W1 W2) / h^2 * ||B||.                                 *)
(*  This is the single lemma that retires the byte-for-byte                  *)
(*  double_inv_u / double_inv_v triplication.  It yields the weight PRODUCT  *)
(*  W1 k * W2 k literally (e.g. sqrt(aK) * sqrt(aK)); the caller folds it     *)
(*  (sqrt_sqrt) into aK.  hK k > 0 is needed for the field step and is       *)
(*  supplied by hK_pos in all four abstract consumers.                       *)
Lemma winv_compose :
  forall (C1 C2 : R) (W1 W2 : K -> R) (A M B : K -> carrier Hs),
    0 <= C1 -> (forall k, 0 <= W1 k) ->
    winv_est C1 W1 A M -> winv_est C2 W2 M B ->
    forall k, nrm (A k) <= C1 * C2 * (W1 k * W2 k) / (hK k)^2 * nrm (B k).
Proof.
  intros C1 C2 W1 W2 A M B HC1 HW1 H1 H2 k.
  pose proof (H1 k) as E1.
  pose proof (H2 k) as E2.
  pose proof (hK_pos k) as Hh.
  pose proof (nrm_nonneg Hs (B k)) as HB.
  pose proof (nrm_nonneg Hs (M k)) as HM.
  assert (Hinv : 0 <= / hK k) by (apply Rlt_le, Rinv_0_lt_compat; exact Hh).
  assert (Hcoef1 : 0 <= C1 / hK k * W1 k).
  { apply Rmult_le_pos;
      [ apply Rmult_le_pos; [ exact HC1 | exact Hinv ] | exact (HW1 k) ]. }
  assert (S1 : nrm (A k)
               <= (C1 / hK k * W1 k) * (C2 / hK k * W2 k * nrm (B k))) by nra.
  assert (E : (C1 / hK k * W1 k) * (C2 / hK k * W2 k * nrm (B k))
              = C1 * C2 * (W1 k * W2 k) / (hK k)^2 * nrm (B k))
    by (field; lra).
  rewrite E in S1. exact S1.
Qed.

End InverseEstimates.
