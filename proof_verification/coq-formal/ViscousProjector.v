(* ========================================================================= *)
(*  ViscousProjector.v                                                       *)
(*                                                                           *)
(*  The abstract algebra behind the standing assumption (A3) of the paper     *)
(*  ("Viscous projector", display eq:PiKorn) and behind the main-text        *)
(*  subsection "The viscous projector" (sec:ViscousProjector, lem:projfamily);*)
(*  that material sat in an Appendix C subsubsection until 2026-07-29.        *)
(*                                                                           *)
(*  WHAT IS MODELLED.  The carrier is a PreHilbert space; the intended model  *)
(*  is L^2(Omega)^{d x d}, the square-integrable second-order tensor fields   *)
(*  with the integrated Frobenius inner product.  A viscous projector acting  *)
(*  pointwise with constant coefficients, idempotent and self-adjoint for the *)
(*  Frobenius product, IS an orthogonal projection of that space -- so the    *)
(*  pointwise statements of lem:projfamily and their integrated forms are one *)
(*  and the same abstract statement here.                                     *)
(*                                                                           *)
(*  WHAT IS PROVED (from the four PreHilbert axioms; no Admitted, no Axiom):  *)
(*    proj_nonexpansive   |P x| <= |x|                (the contraction of     *)
(*                                                     (A3): eq:winv-grad,    *)
(*                                                     eq:interpgrad, OSGS)   *)
(*    nested_pythagoras   |P x|^2 = |Q x|^2 + |P x - Q x|^2   (eq:projmonotone)*)
(*    proj_monotone_sq    |Q x|^2 <= |P x|^2  when ran(P) contains ran(Q)     *)
(*    korn_chain_sq       the Korn constant transfers from Q to P             *)
(*    korn_norm           its norm form, |x| <= C |P x|                       *)
(*    korn_sqrt2          |x| <= sqrt 2 |P x| from |Q x|^2 >= |x|^2 / 2       *)
(*                                          (eq:devsymidentity => eq:PiKorn)  *)
(*    korn_definite       |P x| = 0 -> |x| = 0        (lem:definiteness, its  *)
(*                                                     velocity step)         *)
(*                                                                           *)
(*  WHAT IS NOT MODELLED, and is deliberately left to Stratum 1 / SymPy: the  *)
(*  CONCRETE identity eq:devsymidentity, i.e. that the deviatoric-symmetric   *)
(*  projector satisfies |Q grad v|^2 >= |grad v|^2 / 2 on H_0^1.  That is an  *)
(*  integration-by-parts fact about a particular differential operator on a   *)
(*  particular function space, outside this algebraic kernel; it is checked   *)
(*  in proof_verification/sympy/projector_algebra_verification.py.  Here it   *)
(*  enters as the hypothesis Hkorn, restricted to the subclass G of admissible*)
(*  arguments (the gradients of H_0^1 velocity fields).                       *)
(*                                                                           *)
(*  The section closes with a non-vacuity witness (Section Witness): a        *)
(*  concrete PreHilbert with P <> Q, P not the identity and G a proper        *)
(*  subclass, on which every hypothesis of the section holds with kappa = 1/2.*)
(*                                                                           *)
(*  Rocq/Coq, stdlib only.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
From PNSFormal Require Import InnerSpace.
Local Open Scope R_scope.

Section ViscousProjector.

Variable H : PreHilbert.

Notation V := (carrier H).
Notation "'<<' x , y '>>'" := (ip H x y) (at level 0).

(* ------------------------------------------------------------------------- *)
(*  An orthogonal projection: idempotent and self-adjoint for the inner       *)
(*  product.  Linearity is not needed by anything below and is therefore not  *)
(*  assumed, which keeps the hypothesis bundle minimal.                       *)
(* ------------------------------------------------------------------------- *)

Variable P Q : V -> V.

Hypothesis P_idem : forall x, P (P x) = P x.
Hypothesis P_sa   : forall x y, << P x , y >> = << x , P y >>.
Hypothesis Q_idem : forall x, Q (Q x) = Q x.
Hypothesis Q_sa   : forall x y, << Q x , y >> = << x , Q y >>.

(* ---------- Non-expansiveness --------------------------------------------- *)

(*  The pointwise contraction |Pi T| <= |T| of (A3), used throughout          *)
(*  Appendix C (eq:winv-grad, eq:interpgrad) and in Step 3 of the OSGS        *)
(*  stability proof.                                                          *)
Lemma proj_fixes_own_image : forall x, << P x , P x >> = << x , P x >>.
Proof. intro x. rewrite P_sa, P_idem. reflexivity. Qed.

Lemma proj_nonexpansive_sq : forall x, << P x , P x >> <= << x , x >>.
Proof.
  intro x.
  pose proof (proj_fixes_own_image x) as Hfix.
  pose proof (ip_pos H (vsub x (P x))) as Hp.
  rewrite (ip_expand_sub H) in Hp.
  lra.
Qed.

Theorem proj_nonexpansive : forall x, nrm (P x) <= nrm x.
Proof.
  intro x.
  apply nonneg_le_of_sqr'; try apply nrm_nonneg.
  replace ((nrm (P x))^2) with (nrm (P x) * nrm (P x)) by ring.
  replace ((nrm x)^2) with (nrm x * nrm x) by ring.
  rewrite !(nrm_sq H).
  apply proj_nonexpansive_sq.
Qed.

(* ---------- Nested ranges: Pythagoras and monotonicity --------------------- *)

(*  "The range of P contains the range of Q", i.e. P fixes every element of   *)
(*  ran(Q).  This is exactly the criterion of lem:projfamily: a projector      *)
(*  whose range contains the deviatoric-symmetric tensors fixes them.          *)
Hypothesis PQ : forall x, P (Q x) = Q x.

(*  The cross term of the splitting.  Only self-adjointness of P and the      *)
(*  idempotence plus self-adjointness of Q are used: in particular the        *)
(*  companion identity Q P = Q is never needed in OPERATOR form, which is     *)
(*  what lets the argument avoid assuming the inner product definite.         *)
Lemma nested_cross : forall x, << P x , Q x >> = << Q x , Q x >>.
Proof.
  intro x.
  rewrite (Q_sa x (Q x)), Q_idem.
  rewrite P_sa, PQ.
  reflexivity.
Qed.

(*  eq:projmonotone: |P x|^2 = |Q x|^2 + |P x - Q x|^2.                       *)
Theorem nested_pythagoras :
  forall x,
    << P x , P x >>
    = << Q x , Q x >> + << vsub (P x) (Q x) , vsub (P x) (Q x) >>.
Proof.
  intro x.
  rewrite (ip_expand_sub H).
  pose proof (nested_cross x) as Hc.
  lra.
Qed.

Corollary proj_monotone_sq : forall x, << Q x , Q x >> <= << P x , P x >>.
Proof.
  intro x.
  pose proof (nested_pythagoras x) as Hp.
  pose proof (ip_pos H (vsub (P x) (Q x))) as Hnn.
  lra.
Qed.

Corollary proj_monotone : forall x, nrm (Q x) <= nrm (P x).
Proof.
  intro x.
  apply nonneg_le_of_sqr'; try apply nrm_nonneg.
  replace ((nrm (Q x))^2) with (nrm (Q x) * nrm (Q x)) by ring.
  replace ((nrm (P x))^2) with (nrm (P x) * nrm (P x)) by ring.
  rewrite !(nrm_sq H).
  apply proj_monotone_sq.
Qed.

(* ---------- Korn compatibility transfers along the nesting ----------------- *)

(*  G is the class of admissible arguments -- in the intended model, the      *)
(*  gradients of H_0^1(Omega)^d velocity fields, on which alone the Korn      *)
(*  inequality eq:PiKorn is asserted.  It is left opaque here.                *)
Variable G : V -> Prop.
Variable kappa : R.

Hypothesis Hkorn : forall x, G x -> kappa * << x , x >> <= << Q x , Q x >>.

(*  This is the chaining step of lem:projfamily: whatever Korn constant the   *)
(*  deviatoric-symmetric projector Q enjoys, every projector P whose range    *)
(*  contains that of Q enjoys the same one.                                   *)
Theorem korn_chain_sq :
  forall x, G x -> kappa * << x , x >> <= << P x , P x >>.
Proof.
  intros x Hx.
  pose proof (Hkorn x Hx) as H1.
  pose proof (proj_monotone_sq x) as H2.
  lra.
Qed.

(*  Norm form: |x| <= C |P x| whenever |x|^2 <= C^2 |P x|^2.                  *)
Lemma korn_norm_of_sq :
  forall (C : R) (x : V),
    0 <= C ->
    << x , x >> <= C * C * << P x , P x >> ->
    nrm x <= C * nrm (P x).
Proof.
  intros C x HC Hsq.
  apply nonneg_le_of_sqr'.
  - apply nrm_nonneg.
  - apply Rmult_le_pos; [exact HC | apply nrm_nonneg].
  - replace ((nrm x)^2) with (nrm x * nrm x) by ring.
    replace ((C * nrm (P x))^2) with (C * C * (nrm (P x) * nrm (P x))) by ring.
    rewrite !(nrm_sq H).
    exact Hsq.
Qed.

(*  The headline instance: kappa = 1/2 -- the value eq:devsymidentity supplies *)
(*  for the deviatoric-symmetric projector -- yields the uniform constant      *)
(*  C_K = sqrt 2 of eq:PiKorn for the whole family.                           *)
Hypothesis kappa_half : kappa = 1 / 2.

Theorem korn_sqrt2 : forall x, G x -> nrm x <= sqrt 2 * nrm (P x).
Proof.
  intros x Hx.
  apply korn_norm_of_sq.
  - apply sqrt_pos.
  - pose proof (korn_chain_sq x Hx) as Hk.
    rewrite kappa_half in Hk.
    assert (Hs : sqrt 2 * sqrt 2 = 2) by (apply sqrt_sqrt; lra).
    rewrite Hs. lra.
Qed.

(*  The velocity step of lem:definiteness: a field whose projected gradient    *)
(*  vanishes has vanishing gradient.                                          *)
Theorem korn_definite :
  forall x, G x -> nrm (P x) = 0 -> nrm x = 0.
Proof.
  intros x Hx H0.
  pose proof (korn_sqrt2 x Hx) as Hk.
  rewrite H0, Rmult_0_r in Hk.
  pose proof (nrm_nonneg H x).
  lra.
Qed.

End ViscousProjector.

(* ========================================================================= *)
(*  Non-vacuity witness.                                                     *)
(*                                                                           *)
(*  Every hypothesis of the section above is satisfied simultaneously by a    *)
(*  concrete triple (H, P, Q) with P <> Q, P NOT the identity, and G a proper *)
(*  nontrivial subclass, with kappa = 1/2.  The carrier R^3 stands in for the *)
(*  tensor space: Q keeps the first coordinate (the analogue of the           *)
(*  deviatoric-symmetric part), P keeps the first two (a strictly larger      *)
(*  range), and G is the cone on which the Korn bound holds.  So the theorems *)
(*  above are not vacuously true.                                            *)
(* ========================================================================= *)

Section Witness.

Local Open Scope R_scope.

Definition T3 : Type := (R * R * R)%type.
Definition e1 (x : T3) : R := fst (fst x).
Definition e2 (x : T3) : R := snd (fst x).
Definition e3 (x : T3) : R := snd x.

Definition t3add (x y : T3) : T3 := (e1 x + e1 y, e2 x + e2 y, e3 x + e3 y).
Definition t3scal (a : R) (x : T3) : T3 := (a * e1 x, a * e2 x, a * e3 x).
Definition t3ip (x y : T3) : R := e1 x * e1 y + e2 x * e2 y + e3 x * e3 y.

Lemma t3ip_sym : forall x y, t3ip x y = t3ip y x.
Proof. intros x y. unfold t3ip. ring. Qed.

Lemma t3ip_add_l : forall x y z, t3ip (t3add x y) z = t3ip x z + t3ip y z.
Proof.
  intros x y z. unfold t3ip, t3add, e1, e2, e3. simpl. ring.
Qed.

Lemma t3ip_scal_l : forall a x y, t3ip (t3scal a x) y = a * t3ip x y.
Proof.
  intros a x y. unfold t3ip, t3scal, e1, e2, e3. simpl. ring.
Qed.

Lemma t3ip_pos : forall x, 0 <= t3ip x x.
Proof. intro x. unfold t3ip. nra. Qed.

Definition PH3 : PreHilbert :=
  mkPreHilbert T3 t3add t3scal t3ip t3ip_sym t3ip_add_l t3ip_scal_l t3ip_pos.

Definition Pw (x : T3) : T3 := (e1 x, e2 x, 0).
Definition Qw (x : T3) : T3 := (e1 x, 0, 0).
Definition Gw (x : T3) : Prop := (e2 x)^2 + (e3 x)^2 <= (e1 x)^2.

Lemma w_P_idem : forall x, Pw (Pw x) = Pw x.
Proof. intro x. unfold Pw, e1, e2, e3. simpl. reflexivity. Qed.

Lemma w_P_sa : forall x y, ip PH3 (Pw x) y = ip PH3 x (Pw y).
Proof. intros x y. unfold ip, PH3, t3ip, Pw, e1, e2, e3. simpl. ring. Qed.

Lemma w_Q_idem : forall x, Qw (Qw x) = Qw x.
Proof. intro x. unfold Qw, e1, e2, e3. simpl. reflexivity. Qed.

Lemma w_Q_sa : forall x y, ip PH3 (Qw x) y = ip PH3 x (Qw y).
Proof. intros x y. unfold ip, PH3, t3ip, Qw, e1, e2, e3. simpl. ring. Qed.

Lemma w_PQ : forall x, Pw (Qw x) = Qw x.
Proof. intro x. unfold Pw, Qw, e1, e2, e3. simpl. reflexivity. Qed.

Lemma w_korn :
  forall x, Gw x -> (1 / 2) * ip PH3 x x <= ip PH3 (Qw x) (Qw x).
Proof.
  intros x Hx. unfold Gw in Hx.
  unfold ip, PH3, t3ip, Qw, e1, e2, e3 in *. simpl in *.
  nra.
Qed.

(*  P and Q really differ, and P is not the identity, so the nesting is       *)
(*  strict at both ends: the witness is not a disguised triviality.           *)
Lemma w_P_ne_Q : Pw (0, 1, 0) <> Qw (0, 1, 0).
Proof.
  unfold Pw, Qw, e1, e2, e3. simpl. intro Hc. inversion Hc. lra.
Qed.

Lemma w_P_ne_id : Pw (0, 0, 1) <> (0, 0, 1).
Proof.
  unfold Pw, e1, e2, e3. simpl. intro Hc. inversion Hc. lra.
Qed.

(*  G is nonempty and proper.                                                 *)
Lemma w_G_inhabited : Gw (1, 0, 0).
Proof. unfold Gw, e1, e2, e3. simpl. nra. Qed.

Lemma w_G_proper : ~ Gw (0, 1, 0).
Proof. unfold Gw, e1, e2, e3. simpl. nra. Qed.

(*  The instantiated conclusion, obtained from the general theorem.           *)
Theorem witness_korn_sqrt2 :
  forall x : T3, Gw x -> nrm (H:=PH3) x <= sqrt 2 * nrm (H:=PH3) (Pw x).
Proof.
  apply (korn_sqrt2 PH3 Pw Qw w_P_sa w_Q_idem w_Q_sa w_PQ Gw (1 / 2) w_korn).
  reflexivity.
Qed.

End Witness.
