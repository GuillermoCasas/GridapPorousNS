(* ========================================================================= *)
(*  Limits.v                                                                 *)
(*                                                                           *)
(*  A minimal, self-contained limit theory (Coq 8.18, stdlib only):          *)
(*  limits at +oo, one-sided limits at 0+ and 1-, extensionality on the      *)
(*  positive half-line, and the generic rational limits used by the          *)
(*  robustness analysis (Asymptotics.v) and the plateau-bump analysis        *)
(*  (ManufacturedSolution.v).                                                *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  A minimal, self-contained limit theory (one-sided and at infinity).      *)
(* ========================================================================= *)

Definition tendsto_at_top (f : R -> R) (l : R) : Prop :=
  forall eps : R, 0 < eps -> exists M : R, forall x : R, M < x -> Rabs (f x - l) < eps.

Definition tendsto_at_0plus (f : R -> R) (l : R) : Prop :=
  forall eps : R, 0 < eps ->
    exists delta : R, 0 < delta /\
      forall x : R, 0 < x < delta -> Rabs (f x - l) < eps.

(* Limits only see eventual/positive values, so pointwise equality on        *)
(* (0, +oo) transfers them. *)
Lemma tendsto_at_top_ext_pos :
  forall f g l, (forall x, 0 < x -> f x = g x) ->
    tendsto_at_top g l -> tendsto_at_top f l.
Proof.
  intros f g l Hext Hg eps Heps.
  destruct (Hg eps Heps) as [M HM].
  exists (Rmax M 0). intros x Hx.
  assert (HxM : M < x) by (eapply Rle_lt_trans; [apply Rmax_l | exact Hx]).
  assert (Hx0 : 0 < x) by (eapply Rle_lt_trans; [apply Rmax_r | exact Hx]).
  rewrite Hext by exact Hx0. apply HM, HxM.
Qed.

Lemma tendsto_at_0plus_ext_pos :
  forall f g l, (forall x, 0 < x -> f x = g x) ->
    tendsto_at_0plus g l -> tendsto_at_0plus f l.
Proof.
  intros f g l Hext Hg eps Heps.
  destruct (Hg eps Heps) as [delta [Hd HD]].
  exists delta. split; [exact Hd |].
  intros x Hx. rewrite Hext by lra. apply HD, Hx.
Qed.

(*  The three generic rational limits every regime statement reduces to.     *)

(*  x / (C + x)  ->  1  as x -> +oo   (C > 0). *)
Lemma lim_x_over_Cpx : forall C : R, 0 < C ->
  tendsto_at_top (fun x => x / (C + x)) 1.
Proof.
  intros C HC eps Heps.
  exists (Rmax (C / eps) 0). intros x Hx.
  assert (Hx0 : 0 < x) by (eapply Rle_lt_trans; [apply Rmax_r | exact Hx]).
  assert (HxC : C / eps < x) by (eapply Rle_lt_trans; [apply Rmax_l | exact Hx]).
  assert (Hden : 0 < C + x) by lra.
  replace (x / (C + x) - 1) with (- (C / (C + x))) by (field; lra).
  rewrite Rabs_Ropp, Rabs_right
    by (apply Rle_ge; apply Rlt_le, Rdiv_lt_0_compat; lra).
  apply (Rmult_lt_reg_r (C + x)); [lra |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra.
  (* goal:  C * 1 < eps * (C + x);  from x > C/eps:  eps x > C. *)
  assert (Heps_x : C < eps * x).
  { apply (Rmult_lt_reg_l (/ eps)).
    - apply Rinv_0_lt_compat; lra.
    - rewrite <- Rmult_assoc, Rinv_l by lra.
      unfold Rdiv in HxC. lra. }
  nra.
Qed.

(*  (C + x) / x  ->  1  as x -> +oo   (C > 0). *)
Lemma lim_Cpx_over_x : forall C : R, 0 < C ->
  tendsto_at_top (fun x => (C + x) / x) 1.
Proof.
  intros C HC eps Heps.
  exists (Rmax (C / eps) 0). intros x Hx.
  assert (Hx0 : 0 < x) by (eapply Rle_lt_trans; [apply Rmax_r | exact Hx]).
  assert (HxC : C / eps < x) by (eapply Rle_lt_trans; [apply Rmax_l | exact Hx]).
  replace ((C + x) / x - 1) with (C / x) by (field; lra).
  rewrite Rabs_right
    by (apply Rle_ge; apply Rlt_le, Rdiv_lt_0_compat; lra).
  apply (Rmult_lt_reg_r x); [lra |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra.
  assert (Heps_x : C < eps * x).
  { apply (Rmult_lt_reg_l (/ eps)).
    - apply Rinv_0_lt_compat; lra.
    - rewrite <- Rmult_assoc, Rinv_l by lra.
      unfold Rdiv in HxC. lra. }
  nra.
Qed.

(*  A / (A + x)  ->  1  as x -> 0+   (A > 0). *)
Lemma lim_A_over_Apx_at0 : forall A : R, 0 < A ->
  tendsto_at_0plus (fun x => A / (A + x)) 1.
Proof.
  intros A HA eps Heps.
  exists (A * eps). split; [nra |].
  intros x [Hx0 Hxd].
  assert (Hden : 0 < A + x) by lra.
  replace (A / (A + x) - 1) with (- (x / (A + x))) by (field; lra).
  rewrite Rabs_Ropp, Rabs_right
    by (apply Rle_ge; apply Rlt_le, Rdiv_lt_0_compat; lra).
  apply (Rmult_lt_reg_r (A + x)); [lra |].
  unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra.
  (* goal: x * 1 < eps * (A + x);  from x < A eps. *)
  nra.
Qed.


(*  (A + x) / (B + x)  ->  1  as x -> +oo   (any A, B). *)
Lemma lim_Apx_over_Bpx : forall A B : R,
  tendsto_at_top (fun x => (A + x) / (B + x)) 1.
Proof.
  intros A B eps Heps.
  exists (Rmax (Rabs (A - B) / eps - B) (- B)).
  intros x Hx.
  assert (HxB : - B < x) by (eapply Rle_lt_trans; [apply Rmax_r | exact Hx]).
  assert (HxE : Rabs (A - B) / eps - B < x)
    by (eapply Rle_lt_trans; [apply Rmax_l | exact Hx]).
  assert (Hden : 0 < B + x) by lra.
  replace ((A + x) / (B + x) - 1) with ((A - B) / (B + x)) by (field; lra).
  unfold Rdiv. rewrite Rabs_mult.
  rewrite (Rabs_right (/ (B + x)))
    by (apply Rle_ge; left; apply Rinv_0_lt_compat; lra).
  apply (Rmult_lt_reg_r (B + x)); [lra |].
  rewrite Rmult_assoc, Rinv_l by lra. rewrite Rmult_1_r.
  assert (Rabs (A - B) < eps * (B + x)).
  { apply (Rmult_lt_reg_l (/ eps)); [apply Rinv_0_lt_compat; lra |].
    rewrite <- Rmult_assoc, Rinv_l, Rmult_1_l by lra.
    unfold Rdiv in HxE. lra. }
  lra.
Qed.


(*  One-sided limit from the left at 1 (for the plateau-bump join). *)
Definition tendsto_at_1minus (f : R -> R) (l : R) : Prop :=
  forall eps : R, 0 < eps ->
    exists delta : R, 0 < delta /\
      forall x : R, 1 - delta < x < 1 -> Rabs (f x - l) < eps.
