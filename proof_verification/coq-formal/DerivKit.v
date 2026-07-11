(* ========================================================================= *)
(*  DerivKit.v                                                               *)
(*                                                                           *)
(*  A small derivative toolkit on top of the standard library's Ranalysis    *)
(*  (Coq 8.18, stdlib only), shared by ManufacturedSolution.v and            *)
(*  PlateauBump.v: value rewriting and extensionality for                    *)
(*  derivable_pt_lim, plus derivatives of the concrete building blocks       *)
(*  a t, sin(a t), cos(a t), c f(t) and e^{t-1}.                             *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  Small derivative toolkit on top of Ranalysis.                            *)
(* ========================================================================= *)

(* Replace the derivative value by an equal one. *)
Lemma dpl_val_eq :
  forall (f : R -> R) (x l l' : R),
    derivable_pt_lim f x l -> l = l' -> derivable_pt_lim f x l'.
Proof. intros f x l l' H E. rewrite <- E. exact H. Qed.

(* Pointwise-equal functions have the same derivatives. *)
Lemma dpl_ext :
  forall (f g : R -> R) (x l : R),
    (forall t, g t = f t) ->
    derivable_pt_lim f x l -> derivable_pt_lim g x l.
Proof.
  intros f g x l Hgf Hf.
  unfold derivable_pt_lim in *.
  intros eps Heps.
  destruct (Hf eps Heps) as [delta Hd].
  exists delta. intros hh Hne Hlt.
  rewrite !Hgf. apply Hd; assumption.
Qed.

(* d/dt (a t) = a. *)
Lemma dpl_linear : forall a x : R, derivable_pt_lim (fun t => a * t) x a.
Proof.
  intros a x.
  eapply dpl_val_eq.
  - apply derivable_pt_lim_scal, derivable_pt_lim_id.
  - ring.
Qed.

(* d/dt sin(a t) = a cos(a x). *)
Lemma dpl_sin_lin :
  forall a x : R, derivable_pt_lim (fun t => sin (a * t)) x (a * cos (a * x)).
Proof.
  intros a x.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_comp (fun t => a * t) sin).
    + apply dpl_linear.
    + apply derivable_pt_lim_sin.
  - ring.
Qed.

(* d/dt cos(a t) = - a sin(a x). *)
Lemma dpl_cos_lin :
  forall a x : R,
    derivable_pt_lim (fun t => cos (a * t)) x (- (a * sin (a * x))).
Proof.
  intros a x.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_comp (fun t => a * t) cos).
    + apply dpl_linear.
    + apply derivable_pt_lim_cos.
  - ring.
Qed.

(* d/dt (c f t) = c f'(x). *)
Lemma dpl_scale :
  forall (c : R) (f : R -> R) (x l : R),
    derivable_pt_lim f x l -> derivable_pt_lim (fun t => c * f t) x (c * l).
Proof. intros c f x l H. apply (derivable_pt_lim_scal f c x l H). Qed.

Lemma dpl_exp_shift :
  forall y : R, derivable_pt_lim (fun t => exp (t - 1)) y (exp (y - 1)).
Proof.
  intro y.
  eapply dpl_val_eq.
  - apply (derivable_pt_lim_comp (fun t => t - 1) exp).
    + apply (derivable_pt_lim_minus id (fct_cte 1)).
      * apply derivable_pt_lim_id.
      * apply derivable_pt_lim_const.
    + apply derivable_pt_lim_exp.
  - cbv beta. ring.
Qed.

