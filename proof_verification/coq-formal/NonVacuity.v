(* ========================================================================= *)
(*  NonVacuity.v                                                             *)
(*                                                                           *)
(*  A machine-checked WITNESS that the hypothesis bundle of                  *)
(*  AbstractStability.abstract_stability is CONSISTENT -- i.e. that the      *)
(*  theorem is not vacuously true.                                           *)
(*                                                                           *)
(*  A theorem with contradictory hypotheses is true and worthless.  The      *)
(*  abstract theorems of this development carry a long list of quantitative  *)
(*  assumptions (positivity, the tested identity HBS, the weighted inverse   *)
(*  estimate S3, the epsilon-condition Heps), so a reader is entitled to     *)
(*  ask whether that list is satisfiable at all.  This file answers YES, by  *)
(*  exhibiting an explicit instance and letting the kernel check it:         *)
(*                                                                           *)
(*    * the carrier is R with < x , y > := x * y   (a PreHilbert space);     *)
(*    * the mesh is the one-element list [tt];                               *)
(*    * nu = sigma = 1,  h_K = alpha_K = 1,  |a| = 0,                        *)
(*      xi = 3,  Cbar = 1,  c1 = 7 (> 2*xi*Cbar^2 = 6),  c2 = 1,  C2 = 1/2,  *)
(*      eps = 7/16 (= eps_max, so Heps holds with equality);                 *)
(*    * the atoms are  uu = 1  and  gu = pp = xu = divu = du = 0.            *)
(*                                                                           *)
(*  The parameter formulas then give  tau1 = 1/8,  tau2 = 1,  sigt = 7/8,    *)
(*  and the conclusion of abstract_stability instantiates to the             *)
(*  NON-DEGENERATE inequality                                                *)
(*                                                                           *)
(*      B_S = 7/8  >=  C_stab * ||| U_h |||^2  =  (1/2) * (7/8)  =  7/16 ,   *)
(*                                                                           *)
(*  with BOTH sides strictly positive -- so the theorem is not being         *)
(*  satisfied by a vacuous 0 >= 0.                                           *)
(*                                                                           *)
(*  What this does and does not show.  It shows the hypothesis set is        *)
(*  CONSISTENT (no theorem here is vacuously true).  It does NOT show that   *)
(*  the finite element objects satisfy the hypotheses -- that is the         *)
(*  soundness direction, and it is what the hand audit establishes.          *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import StabilityAlgebra InnerSpace AbstractSums AbstractStability.
Local Open Scope R_scope.

(* ------------------------------------------------------------------------- *)
(*  1.  R itself is a PreHilbert space under  < x , y > := x * y.             *)
(* ------------------------------------------------------------------------- *)

Lemma R_ip_pos : forall x : R, 0 <= x * x.
Proof. intro x. nra. Qed.

Definition RPH : PreHilbert :=
  mkPreHilbert R Rplus Rmult Rmult
    Rmult_comm            (* ip_sym    *)
    Rmult_plus_distr_r    (* ip_add_l  *)
    Rmult_assoc           (* ip_scal_l *)
    R_ip_pos.             (* ip_pos    *)

(* ------------------------------------------------------------------------- *)
(*  2.  The witness data.                                                     *)
(* ------------------------------------------------------------------------- *)

Definition wnu   : R := 1.
Definition wsig  : R := 1.
Definition weps  : R := 7/16.
Definition wc1   : R := 7.
Definition wc2   : R := 1.
Definition wCb   : R := 1.
Definition wxi   : R := 3.
Definition wC2   : R := 1/2.

Definition whK (_ : unit) : R := 1.
Definition waK (_ : unit) : R := 1.
Definition wam (_ : unit) : R := 0.

Definition wgu   (_ : unit) : carrier RPH := 0.
Definition wuu   (_ : unit) : carrier RPH := 1.
Definition wpp   (_ : unit) : carrier RPH := 0.
Definition wxu   (_ : unit) : carrier RPH := 0.
Definition wdivu (_ : unit) : carrier RPH := 0.
Definition wdu   (_ : unit) : carrier RPH := 0.

Definition wTh : list unit := [tt].
Definition wBS : R := 7/8.

(* ------------------------------------------------------------------------- *)
(*  3.  The elementary positivity hypotheses.                                 *)
(* ------------------------------------------------------------------------- *)

Lemma w_nu_pos       : 0 < wnu.    Proof. unfold wnu; lra.  Qed.
Lemma w_sigma_nonneg : 0 <= wsig.  Proof. unfold wsig; lra. Qed.
Lemma w_eps_nonneg   : 0 <= weps.  Proof. unfold weps; lra. Qed.
Lemma w_c1_pos       : 0 < wc1.    Proof. unfold wc1; lra.  Qed.
Lemma w_c2_pos       : 0 < wc2.    Proof. unfold wc2; lra.  Qed.
Lemma w_Cb_pos       : 0 < wCb.    Proof. unfold wCb; lra.  Qed.
Lemma w_C2_nonneg    : 0 <= wC2.   Proof. unfold wC2; lra.  Qed.
Lemma w_C2_lt_1      : wC2 < 1.    Proof. unfold wC2; lra.  Qed.

(*  eq:conditions_on_num_param, sharp form:  c1 = 7 > 3 = xi * Cbar^2.
    (The manuscript's stronger c1 > 2 xi Cbar^2 = 6 also holds here.)  *)
Lemma w_c1_large : wc1 > wxi * wCb ^ 2.
Proof. unfold wc1, wxi, wCb; lra. Qed.

Lemma w_xi_large : wxi > 2.
Proof. unfold wxi; lra. Qed.

Lemma w_hK_pos    : forall k, 0 < whK k.   Proof. intro; unfold whK; lra. Qed.
Lemma w_aK_pos    : forall k, 0 < waK k.   Proof. intro; unfold waK; lra. Qed.
Lemma w_am_nonneg : forall k, 0 <= wam k.  Proof. intro; unfold wam; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  4.  The stabilization parameters, computed.                               *)
(* ------------------------------------------------------------------------- *)

Lemma w_t1 : forall k, AbstractStability.t1 unit wnu wsig wc1 wc2 whK waK wam k = 1/8.
Proof.
  intro k.
  unfold AbstractStability.t1, tau1, tau1NS_inv,
         wnu, wsig, wc1, wc2, whK, waK, wam.
  field.
Qed.

Lemma w_t2 : forall k, AbstractStability.t2 unit wnu wc1 wc2 whK waK wam k = 1.
Proof.
  intro k.
  unfold AbstractStability.t2, tau2, tau1NS_inv,
         wnu, wc1, wc2, whK, waK, wam.
  field.
Qed.

Lemma w_sg : forall k, AbstractStability.sg unit wnu wsig wc1 wc2 whK waK wam k = 7/8.
Proof.
  intro k.
  unfold AbstractStability.sg, sigt, tau1NS_inv,
         wnu, wsig, wc1, wc2, whK, waK, wam.
  field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  5.  The three non-trivial hypotheses of abstract_stability.               *)
(* ------------------------------------------------------------------------- *)

(*  (HBS) the tested identity.  With uu = 1 and every other atom 0:
      G(u) = du - sigma*uu = -1,  L_m = L*_m = 1,  L_c = L*_c = 0,
      so the right-hand side is  0 + 1 + 0 - (1/8)*1 - 1*0 = 7/8 = B_S.       *)
Lemma w_HBS :
  wBS =
  2 * wnu * Rsum wTh (fun k => ip RPH (wgu k) (wgu k))
  + wsig * Rsum wTh (fun k => ip RPH (wuu k) (wuu k))
  + weps * Rsum wTh (fun k => ip RPH (wpp k) (wpp k))
  - Rsum wTh (fun k =>
      AbstractStability.t1 unit wnu wsig wc1 wc2 whK waK wam k
      * ip RPH (AbstractStability.Lstar_m RPH unit wsig wuu wxu wdu k)
               (AbstractStability.L_m RPH unit wsig wuu wxu wdu k))
  - Rsum wTh (fun k =>
      AbstractStability.t2 unit wnu wc1 wc2 whK waK wam k
      * ip RPH (AbstractStability.Lstar_c RPH unit weps wpp wdivu k)
               (AbstractStability.L_c RPH unit weps wpp wdivu k)).
Proof.
  unfold wTh. simpl Rsum.
  rewrite !w_t1, !w_t2.
  unfold AbstractStability.Lstar_m, AbstractStability.L_m,
         AbstractStability.Lstar_c, AbstractStability.L_c,
         AbstractStability.Bv,
         vsub, vopp,
         wgu, wuu, wpp, wxu, wdivu, wdu, wBS, wnu, wsig, weps.
  simpl. field.
Qed.

(*  (S3) the weighted inverse estimate.  Both sides are 0.                    *)
Lemma w_S3 :
  forall k, nrm (wdu k) <= (2 * wnu * wCb / whK k) * sqrt (waK k) * nrm (wgu k).
Proof.
  intro k.
  unfold nrm, wdu, wgu, wnu, wCb, whK, waK.
  simpl.
  replace (0 * 0) with 0 by ring.
  rewrite sqrt_0.
  lra.
Qed.

(*  (Heps) eq:UpperBoundOnEpsilon:  eps = 7/16 = eps_max, so it holds with
    equality -- the extreme admissible choice.                               *)
Lemma w_Heps :
  forall k, weps <= eps_max wnu (whK k) (waK k) wsig (wam k) wc1 wc2 wC2.
Proof.
  intro k.
  unfold eps_max, tau1, tau1NS_inv,
         weps, wnu, wsig, wc1, wc2, wC2, whK, waK, wam.
  field_simplify; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  6.  The instance:  abstract_stability, applied.                           *)
(* ------------------------------------------------------------------------- *)

Definition wCstab : R := C_stab wc1 wCb wxi wC2.

Definition wNormSq : R :=
  AbstractStability.NormSq RPH unit wTh wnu wsig weps wc1 wc2
    whK waK wam wgu wuu wpp wxu wdivu.

Theorem witness_stability :
  0 < wCstab /\ wBS >= wCstab * wNormSq.
Proof.
  exact (AbstractStability.abstract_stability
           RPH unit wTh wnu wsig weps wc1 wc2 wCb wxi wC2
           w_nu_pos w_sigma_nonneg w_eps_nonneg w_c1_pos w_c2_pos w_Cb_pos
           w_C2_nonneg w_C2_lt_1 w_c1_large w_xi_large
           whK waK wam w_hK_pos w_aK_pos w_am_nonneg
           wgu wuu wpp wxu wdivu wdu wBS
           w_HBS w_S3 w_Heps).
Qed.

(* ------------------------------------------------------------------------- *)
(*  7.  The constants, computed:  C_stab = 1/2  and  |||U_h|||^2 = 7/8.       *)
(* ------------------------------------------------------------------------- *)

(*  C_u = 1 - xi*Cbar^2/c1 = 4/7 ;
    C_visc = min{2 - 4*Cbar^2/c1, 2(1 - 2/xi)} = min{10/7, 2/3} = 2/3 ;
    C_stab = min{min{2/3, 4/7}, min{1/2, 1}} = min{4/7, 1/2} = 1/2.          *)
Lemma w_Cstab_val : wCstab = 1/2.
Proof.
  unfold wCstab, C_stab, C_visc, C_u, wc1, wCb, wxi, wC2, Rmin.
  repeat (destruct (Rle_dec _ _); try lra).
Qed.

(*  |||U_h|||^2 = eps*|p|^2 + nu*|gu|^2 + sigt*|u|^2 + tau1*|X|^2 + tau2*|div|^2
                = 0 + 0 + (7/8)*1 + 0 + 0 = 7/8.                             *)
Lemma w_NormSq_val : wNormSq = 7/8.
Proof.
  unfold wNormSq, AbstractStability.NormSq, wTh. simpl Rsum.
  unfold AbstractStability.perNorm,
         AbstractStability.Vn, AbstractStability.Un, AbstractStability.Pn,
         AbstractStability.Xn, AbstractStability.Dn.
  rewrite w_sg, w_t1, w_t2.
  unfold wgu, wuu, wpp, wxu, wdivu, wnu, wsig, weps.
  simpl. field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  8.  The headline: the hypotheses are satisfiable, and the conclusion      *)
(*      they yield is a NON-DEGENERATE inequality between positive numbers.   *)
(* ------------------------------------------------------------------------- *)

Theorem abstract_stability_is_not_vacuous :
  (*  the conclusion of abstract_stability, at this instance ...  *)
  wBS >= wCstab * wNormSq
  (*  ... with every quantity in it strictly positive ...  *)
  /\ 0 < wCstab /\ 0 < wBS /\ 0 < wNormSq
  (*  ... and equal to these explicit rationals.  *)
  /\ wCstab = 1/2 /\ wNormSq = 7/8 /\ wBS = 7/8
  (*  So the bound reads  7/8 >= 7/16 : true, and not a vacuous 0 >= 0.  *)
  /\ wCstab * wNormSq = 7/16.
Proof.
  destruct witness_stability as [Hpos Hbound].
  repeat split;
    rewrite ?w_Cstab_val, ?w_NormSq_val in *;
    unfold wBS in *; lra.
Qed.

(*  The two theorems above are proved from the SAME abstract_stability that
    the paper's lemma:Stability is identified with.  Hence:

      (i)  the hypothesis bundle of abstract_stability is CONSISTENT
           (else no instance could exist), and

      (ii) the conclusion is not degenerate: there is an instance in which
           both sides of  B_S >= C_stab |||U_h|||^2  are strictly positive.

    Run  Print Assumptions abstract_stability_is_not_vacuous.  to confirm that
    this rests on nothing beyond the standard library's real-number axioms.   *)
