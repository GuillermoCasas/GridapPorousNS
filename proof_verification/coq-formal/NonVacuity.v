(* ========================================================================= *)
(*  NonVacuity.v                                                             *)
(*                                                                           *)
(*  Machine-checked WITNESSES that the hypothesis bundle of                  *)
(*  AbstractStability.abstract_stability is CONSISTENT -- i.e. that the      *)
(*  theorem is not vacuously true.                                           *)
(*                                                                           *)
(*  A theorem with contradictory hypotheses is true and worthless.  The      *)
(*  abstract theorems of this development carry a long list of quantitative  *)
(*  assumptions (positivity, the two diagonal Green identities H_skew_diag   *)
(*  and H_ibp_diag, the weighted inverse estimate S3, the epsilon-condition  *)
(*  Heps), so a reader is entitled to ask whether that list is satisfiable   *)
(*  at all.  This file answers YES, by exhibiting explicit instances and     *)
(*  letting the kernel check them.                                           *)
(*                                                                           *)
(*  There are TWO instances, because no single one exercises the bundle.     *)
(*  Both use the carrier R with < x , y > := x * y (a PreHilbert space) and  *)
(*  the constants  nu = sigma = 1,  h_K = alpha_K = 1,  xi = 3,  Cbar = 1,   *)
(*  c1 = 7 (> 2*xi*Cbar^2 = 6),  c2 = 1,  C2 = 1/2.  They differ in |a|, and *)
(*  that difference is the whole point: |a| enters tau_1 through             *)
(*  tau1NS_inv = c1*nu/h^2 + c2*|a|/h, so at |a| = 0 the c2 branch is        *)
(*  multiplied by zero and a whole face of the parameter space collapses.    *)
(*                                                                           *)
(*  WITNESS 1 -- the Darcy corner |a| = 0  (prefix w, Sections 2-8).         *)
(*    Mesh [tt];  eps = 7/16 (= eps_max, so Heps holds with equality);       *)
(*    atoms  uu = 1  and  gu = pp = cxu = gpu = divu = du = 0.  The          *)
(*    parameter formulas give  tau1 = 1/8,  tau2 = 1,  sigt = 7/8,  and the  *)
(*    conclusion of abstract_stability instantiates to the NON-DEGENERATE    *)
(*    inequality                                                             *)
(*                                                                           *)
(*      B_S = 7/8  >=  C_stab * ||| U_h |||^2  =  (1/2) * (7/8)  =  7/16 ,   *)
(*                                                                           *)
(*    with BOTH sides strictly positive -- so the theorem is not being       *)
(*    satisfied by a vacuous 0 >= 0.  It certifies the sigma > 0, |a| = 0    *)
(*    Darcy limit, which is a regime of the paper in its own right.          *)
(*                                                                           *)
(*  WITNESS 2 -- strict advection |a| = 1  (prefix w2, Sections 9-16).       *)
(*    Mesh [true; false] (TWO elements);  eps = 7/18 (= eps_max again);      *)
(*    atoms  gu = uu = pp = divu = 1,  gpu = du = -1,  cxu = (2, -2).  Now   *)
(*    tau1 = 1/9,  tau2 = 8/7,  sigt = 8/9,  and the conclusion reads        *)
(*                                                                           *)
(*      B_S = 5069/567  >=  C_stab * ||| U_h |||^2  =  (1/2) * (167/21)      *)
(*                                                  =  167/42 ,              *)
(*                                                                           *)
(*    again with both sides strictly positive.  Witness 2 exists because     *)
(*    witness 1 leaves five things unexercised, each now a THEOREM of        *)
(*    Section 9 rather than a claim in a comment:                            *)
(*                                                                           *)
(*      (a) am_nonneg is met only at its BOUNDARY |a| = 0, so no strictly    *)
(*          positive advection field is shown to satisfy the bundle;         *)
(*      (b) c2 is numerically DEAD at |a| = 0: tau1, tau2, sigt and eps_max  *)
(*          are all constant in c2 there, so c2_pos is discharged but never  *)
(*          exercised -- witness 1 would yield the same 7/8 >= 7/16 for      *)
(*          c2 = 1 or c2 = 10^6  (w1_c2_is_dead_at_am0);                     *)
(*      (c) the velocity coefficient underlying Step 4 is SATURATED at       *)
(*          |a| = 0 -- u_final = C_u * sigt exactly, because the slack of    *)
(*          StabilityAlgebra.velocity_slack_identity carries a factor        *)
(*          c2*|a|/h -- so the key estimate holds only with equality         *)
(*          (w1_velocity_bound_saturated_at_am0);                            *)
(*      (d) S3 reads 0 <= 0 and H_ibp_diag reads 0 = -0, both sides zero;    *)
(*      (e) on a ONE-element mesh H_skew_diag literally says                 *)
(*          uu tt * cxu tt = 0, which with uu tt = 1 FORCES cxu tt = 0 --    *)
(*          in R orthogonality means a factor vanishes.  This is proved, not *)
(*          assumed (w1_cxu_forced_zero_on_singleton), and it is why witness *)
(*          2 needs a second element: on [true; false] the identity is met   *)
(*          by the genuine cancellation  1*2 + 1*(-2) = 0  between two       *)
(*          NON-ZERO convective contributions.                               *)
(*                                                                           *)
(*    Witness 2 repairs (a)-(e): |a| = 1 > 0 strictly, c2 is live, the       *)
(*    velocity bound is strict (slack 1/21), S3 is strict between strictly   *)
(*    positive sides (1 <= 2), both Green identities are met by non-zero     *)
(*    data (-2 = -2 and 2 + (-2) = 0), and all five norm slots Pn, Vn, Un,   *)
(*    Xn, Dn are strictly positive on BOTH elements.                         *)
(*                                                                           *)
(*    Note that |a| = 1 FORCES eps down: it raises tau1NS_inv from 7 to 8,   *)
(*    shrinking tau1 to 1/9 and hence eps_max to 7/18 < 7/16.  Witness 1's   *)
(*    eps is therefore INADMISSIBLE at |a| = 1 (w1_eps_inadmissible_at_am1), *)
(*    so no witness differing from witness 1 only in |a| exists.  eps = 7/18 *)
(*    keeps witness 1's extreme-admissible-choice property: Heps again holds *)
(*    with equality, at the sharp boundary rather than with slack.           *)
(*                                                                           *)
(*    What witness 2 still does NOT exercise, stated so that no reader has   *)
(*    to discover it:                                                        *)
(*      - the mesh is UNIFORM (h_K = alpha_K = |a| = 1 on both elements),    *)
(*        so hK_pos/aK_pos/am_nonneg are met at one value each rather than   *)
(*        across a varying mesh.  The assembly step is still genuinely       *)
(*        exercised -- the two elements carry DIFFERENT data (Xn = 1 and 9,  *)
(*        < L*_m , L_m > = 3 and -5, perNorm = 445/126 and 557/126), so the  *)
(*        Rsum adds two distinct numbers rather than doubling one.           *)
(*      - Heps is met with EQUALITY, not strictly.  For an upper bound that  *)
(*        is the extremal, hardest admissible choice, not a degenerate one   *)
(*        (and eps = 7/18 > 0, so eps_nonneg is met strictly); a slacker eps *)
(*        would test less.  It is recorded here only because "boundary" and  *)
(*        "degenerate" are easy to confuse.                                  *)
(*      - the carrier is R, i.e. ONE-dimensional, which is why H_skew_diag   *)
(*        needs the two-element cancellation: on R a vanishing inner product *)
(*        forces a factor to vanish (9(d)).  A >= 2-dimensional carrier      *)
(*        would admit cxu genuinely orthogonal to uu on a single element.    *)
(*      - sigma = 1 > 0 throughout, so the sigma = 0 corner the theorem      *)
(*        allows (sigma_nonneg, eps_nonneg and C2_nonneg are all >= 0) is    *)
(*        witnessed by neither instance.                                     *)
(*                                                                           *)
(*  What the 7/8 and the 5069/567 now mean.  B_S(U_h,U_h) used to be a free  *)
(*  real of the stability lemma: the witness set it by fiat (wBS := 7/8) and *)
(*  then justified that choice against the ASSUMED tested identity, via the  *)
(*  obligation w_HBS.  It is now no free real -- it is DEFINED as the        *)
(*  eighteen-term form of eq:Bstab evaluated at the diagonal -- so both      *)
(*  headline numbers are COMPUTED, not posited, and w_BS_val / w2_BS_val     *)
(*  merely evaluate them.  The witnesses have strictly less freedom than     *)
(*  they used to, and the headline bounds are the stronger for it.  In place *)
(*  of w_HBS the two Green identities must now be witnessed.                 *)
(*                                                                           *)
(*  What this does and does not show.  It shows the hypothesis set is        *)
(*  CONSISTENT (no theorem here is vacuously true) and that it is consistent *)
(*  with a strictly positive advection field on a multi-element mesh.  It    *)
(*  does NOT show that the finite element objects satisfy the hypotheses --  *)
(*  that is the soundness direction, and it is what the hand audit           *)
(*  establishes.  Neither witness speaks to abstract_continuity,             *)
(*  abstract_continterp or abstract_convergence: all three carry a face      *)
(*  apparatus (F, Fl, FBp, FBc), and the latter two additionally CI_pos.     *)
(*  None of those occurs anywhere in the discharged signature of             *)
(*  abstract_stability, which quantifies over no faces at all, so nothing    *)
(*  here witnesses them.  By the same token the two classic vacuity escapes  *)
(*  for face hypotheses -- F := Empty_set, making every  forall f : F        *)
(*  vacuous, and Fl := [], making every face sum read 0 = 0 -- are not       *)
(*  dodged here so much as unavailable: they cannot arise for a theorem      *)
(*  that has no faces in it.  They are live risks only for the other three.  *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import StabilityAlgebra InnerSpace AbstractSums AbstractStability.
Local Open Scope R_scope.

(* ------------------------------------------------------------------------- *)
(*  1.  R itself is a PreHilbert space under  < x , y > := x * y.             *)
(*      Shared by both witnesses.                                             *)
(* ------------------------------------------------------------------------- *)

Lemma R_ip_pos : forall x : R, 0 <= x * x.
Proof. intro x. nra. Qed.

Definition RPH : PreHilbert :=
  mkPreHilbert R Rplus Rmult Rmult
    Rmult_comm            (* ip_sym    *)
    Rmult_plus_distr_r    (* ip_add_l  *)
    Rmult_assoc           (* ip_scal_l *)
    R_ip_pos.             (* ip_pos    *)

(* ========================================================================= *)
(*  WITNESS 1 -- THE DARCY CORNER  |a| = 0.   (Sections 2-8, prefix w.)      *)
(*                                                                           *)
(*  One element, sigma > 0, no advection.  Everything but uu vanishes, so    *)
(*  the two Green identities and S3 are met trivially; what this witness     *)
(*  certifies is that the bundle is consistent AT ALL and that its           *)
(*  conclusion is not 0 >= 0.  Section 9 records, as theorems, exactly what  *)
(*  it leaves unexercised.                                                   *)
(* ========================================================================= *)

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

(*  The seven atom families.  The x-component of the stability lemma is now   *)
(*  split into its convective and pressure-gradient halves (cxu, gpu) -- the  *)
(*  two diagonal Green identities speak about them separately -- so the       *)
(*  witness supplies both; here both vanish, as the fused wxu used to.        *)
Definition wgu   (_ : unit) : carrier RPH := 0.
Definition wuu   (_ : unit) : carrier RPH := 1.
Definition wpp   (_ : unit) : carrier RPH := 0.
Definition wcxu  (_ : unit) : carrier RPH := 0.   (*  (alpha a . grad u_h)|_K *)
Definition wgpu  (_ : unit) : carrier RPH := 0.   (*  (alpha grad p_h)|_K     *)
Definition wdivu (_ : unit) : carrier RPH := 0.
Definition wdu   (_ : unit) : carrier RPH := 0.

Definition wTh : list unit := [tt].

(*  B_S(U_h,U_h) is NOT a free choice of this witness any more: it is the     *)
(*  eighteen-term form of eq:Bstab (AbstractStability.BS) at the atoms above. *)
(*  Its value -- 7/8, the number the old witness had to posit -- is computed  *)
(*  in w_BS_val below.                                                        *)
Definition wBS : R :=
  AbstractStability.BS RPH unit wTh wnu wsig weps wc1 wc2 whK waK wam
    wgu wuu wpp wcxu wgpu wdivu wdu.

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
(*  5.  The four non-trivial hypotheses of abstract_stability.                *)
(* ------------------------------------------------------------------------- *)

(*  (H_skew_diag) eq:skew on the diagonal.  Every atom but uu vanishes, so
    the convective sum reads  1 * 0 = 0.                                     *)
Lemma w_skew_diag :
  Rsum wTh (fun k => ip RPH (wuu k) (wcxu k)) = 0.
Proof.
  unfold wTh. simpl Rsum. unfold wuu, wcxu. simpl. ring.
Qed.

(*  (H_ibp_diag) eq:globalibp on the diagonal.  Both sides vanish: the
    identity reads  0 = -0.                                                  *)
Lemma w_ibp_diag :
  Rsum wTh (fun k => ip RPH (wuu k) (wgpu k))
  = - Rsum wTh (fun k => ip RPH (wpp k) (wdivu k)).
Proof.
  unfold wTh. simpl Rsum. unfold wuu, wgpu, wpp, wdivu. simpl. ring.
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
    whK waK wam wgu wuu wpp wcxu wgpu wdivu.

Theorem witness_stability :
  0 < wCstab /\ wBS >= wCstab * wNormSq.
Proof.
  exact (AbstractStability.abstract_stability
           RPH unit wTh wnu wsig weps wc1 wc2 wCb wxi wC2
           w_nu_pos w_sigma_nonneg w_eps_nonneg w_c1_pos w_c2_pos w_Cb_pos
           w_C2_nonneg w_C2_lt_1 w_c1_large w_xi_large
           whK waK wam w_hK_pos w_aK_pos w_am_nonneg
           wgu wuu wpp wcxu wgpu wdivu wdu
           w_skew_diag w_ibp_diag w_S3 w_Heps).
Qed.

(* ------------------------------------------------------------------------- *)
(*  7.  The three numbers, computed:  B_S = 7/8,  C_stab = 1/2               *)
(*      and  |||U_h|||^2 = 7/8.                                              *)
(* ------------------------------------------------------------------------- *)

(*  B_S(U_h,U_h), read off the eighteen-term form through the stability
    file's tested identity -- which is a THEOREM there (AbstractStability.HBS,
    proved from the two Green identities above), not an assumption.  With
    uu = 1 and every other atom 0:
      alpha X(U_h) = cxu + gpu = 0,  G(u) = du - sigma*uu = -1,
      L_m = L*_m = 1,  L_c = L*_c = 0,
    so  B_S = 0 + 1 + 0 - (1/8)*1 - 1*0 = 7/8.                               *)
Lemma w_BS_val : wBS = 7/8.
Proof.
  unfold wBS.
  rewrite (AbstractStability.HBS RPH unit wTh wnu wsig weps wc1 wc2
             whK waK wam wgu wuu wpp wcxu wgpu wdivu wdu
             w_skew_diag w_ibp_diag).
  unfold wTh. simpl Rsum.
  rewrite !w_t1, !w_t2.
  unfold AbstractStability.Lstar_m, AbstractStability.L_m,
         AbstractStability.Lstar_c, AbstractStability.L_c,
         AbstractStability.Bv, AbstractStability.xu,
         vsub, vopp,
         wgu, wuu, wpp, wcxu, wgpu, wdivu, wdu, wnu, wsig, weps.
  simpl. field.
Qed.

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
         AbstractStability.Xn, AbstractStability.Dn, AbstractStability.xu.
  rewrite w_sg, w_t1, w_t2.
  unfold wgu, wuu, wpp, wcxu, wgpu, wdivu, wnu, wsig, weps.
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
    rewrite ?w_Cstab_val, ?w_NormSq_val, ?w_BS_val in *; lra.
Qed.

(* ========================================================================= *)
(*  WITNESS 2 -- STRICT ADVECTION  |a| = 1.   (Sections 9-16, prefix w2.)    *)
(*                                                                           *)
(*  Two elements, sigma > 0, |a| = 1 > 0 strictly, and every atom non-zero.  *)
(*  This is the witness that takes am_nonneg off its boundary, brings c2 to  *)
(*  life, and meets S3 and the two Green identities with non-zero data.      *)
(* ========================================================================= *)

(* ------------------------------------------------------------------------- *)
(*  9.  Why a second witness, and why it cannot be witness 1 with |a|         *)
(*      retuned.  Four obstructions, each a theorem rather than a claim.      *)
(* ------------------------------------------------------------------------- *)

(*  (a)  c2 is numerically DEAD at |a| = 0.  It enters only through the
    c2*|a|/h branch of tau1NS_inv, which vanishes there, so ALL FOUR derived
    parameters are constant in c2 -- and C_stab, C_visc, C_u never mention c2
    at all.  Witness 1 therefore discharges c2_pos without exercising it: it
    would produce the identical 7/8 >= 7/16 for c2 = 1 or c2 = 10^6.         *)
Lemma w1_c2_is_dead_at_am0 :
  forall c2 : R,
    tau1    1 1 1 1 0 7 c2       = 1/8
    /\ tau2 1 1 1 0 7 c2         = 1
    /\ sigt 1 1 1 1 0 7 c2       = 7/8
    /\ eps_max 1 1 1 1 0 7 c2 (1/2) = 7/16.
Proof.
  intro c2.
  unfold eps_max, tau1, tau2, sigt, tau1NS_inv.
  repeat split; field.
Qed.

(*  ... whereas at |a| = 1 the value of c2 genuinely moves tau2.             *)
Lemma w2_c2_is_live_at_am1 :
  tau2 1 1 1 1 7 1 = 8/7 /\ tau2 1 1 1 1 7 4 = 11/7.
Proof. unfold tau2, tau1NS_inv. split; field. Qed.

(*  (b)  The velocity coefficient of Step 4 is SATURATED at |a| = 0.  By
    StabilityAlgebra.velocity_slack_identity the slack in
    u_final >= C_u * sigt is  alpha_K tau1 sigma (xi Cbar^2/c1)(c2 |a|/h),
    which carries a factor |a| -- so at |a| = 0 the estimate holds with
    EQUALITY, and witness 1 exercises it only at its boundary.              *)
Lemma w1_velocity_bound_saturated_at_am0 :
  u_final 1 1 1 1 0 7 1 1 3 = 1/2 /\ C_u 7 1 3 * sigt 1 1 1 1 0 7 1 = 1/2.
Proof. unfold u_final, C_u, sigt, tau1, tau1NS_inv. split; field. Qed.

(*  ... at |a| = 1 it is strict, with slack exactly 1/21.                    *)
Lemma w2_velocity_bound_strict_at_am1 :
  u_final 1 1 1 1 1 7 1 1 3 = 5/9
  /\ C_u 7 1 3 * sigt 1 1 1 1 1 7 1 = 32/63
  /\ u_final 1 1 1 1 1 7 1 1 3 - C_u 7 1 3 * sigt 1 1 1 1 1 7 1 = 1/21.
Proof. unfold u_final, C_u, sigt, tau1, tau1NS_inv. repeat split; field. Qed.

(*  (c)  Witness 1's eps is INADMISSIBLE at |a| = 1.  Advection raises
    tau1NS_inv from 7 to 8, so tau1 falls to 1/9 and eps_max to
    (1/2)*7*(1/9) = 7/18 < 7/16.  Hence NO witness differing from witness 1
    only in |a| exists: eps must drop with it, and 7/18 is the extreme
    admissible choice that preserves equality in Heps.                      *)
Lemma w1_eps_inadmissible_at_am1 :
  ~ (7/16 <= eps_max 1 1 1 1 1 7 1 (1/2)).
Proof.
  assert (E : eps_max 1 1 1 1 1 7 1 (1/2) = 7/18).
  { unfold eps_max, tau1, tau1NS_inv. field. }
  rewrite E. lra.
Qed.

(*  (d)  On a ONE-element mesh, H_skew_diag FORCES the convective atom to
    vanish.  On carrier R the identity is literally  uu tt * cxu tt = 0, and
    a product of reals is zero only if a factor is: with uu tt = 1 this
    leaves cxu tt = 0 with no choice in the matter.  So witness 1's cxu = 0
    is not a degenerate CHOICE -- it is forced -- and escaping it requires
    either a >= 2-dimensional carrier or, as here, a second element, on which
    the sum can cancel between two NON-ZERO contributions.                  *)
Theorem w1_cxu_forced_zero_on_singleton :
  forall uu cxu : unit -> carrier RPH,
    uu tt <> 0 ->
    Rsum [tt] (fun k => ip RPH (uu k) (cxu k)) = 0 ->
    cxu tt = 0.
Proof.
  intros uu cxu Hu H. simpl in H.
  replace (ip RPH (uu tt) (cxu tt)) with (uu tt * cxu tt) in H by reflexivity.
  nra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  10.  The witness data.                                                    *)
(* ------------------------------------------------------------------------- *)

Definition w2nu   : R := 1.
Definition w2sig  : R := 1.
Definition w2eps  : R := 7/18.   (*  = eps_max at |a| = 1;  cf. 9(c)         *)
Definition w2c1   : R := 7.
Definition w2c2   : R := 1.
Definition w2Cb   : R := 1.
Definition w2xi   : R := 3.
Definition w2C2   : R := 1/2.

(*  The mesh is bool, i.e. the TWO-element list [true; false].  The second
    element is not decoration: by w1_cxu_forced_zero_on_singleton it is the
    cheapest way to satisfy H_skew_diag with a non-zero convective atom.    *)
Definition w2hK (_ : bool) : R := 1.
Definition w2aK (_ : bool) : R := 1.
Definition w2am (_ : bool) : R := 1.   (*  |a| = 1 > 0 : THE POINT           *)

(*  The seven atom families -- every one non-zero.  du is NEGATIVE on
    purpose: du := +1 would make G(u) = du - sigma*uu = 0, annihilating
    < L*_m , L_m > and costing B_S its tau_1 dependence, i.e. quietly
    undoing the very advection sensitivity this witness exists to show.     *)
Definition w2gu   (_ : bool) : carrier RPH := 1.
Definition w2uu   (_ : bool) : carrier RPH := 1.
Definition w2pp   (_ : bool) : carrier RPH := 1.
Definition w2cxu  (b : bool) : carrier RPH := if b then 2 else -2.
                                          (*  (alpha a . grad u_h)|_K        *)
Definition w2gpu  (_ : bool) : carrier RPH := -1.
                                          (*  (alpha grad p_h)|_K, forced by
                                              H_ibp_diag given uu, pp, divu  *)
Definition w2divu (_ : bool) : carrier RPH := 1.
Definition w2du   (_ : bool) : carrier RPH := -1.

Definition w2Th : list bool := [true; false].

Definition w2BS : R :=
  AbstractStability.BS RPH bool w2Th w2nu w2sig w2eps w2c1 w2c2 w2hK w2aK w2am
    w2gu w2uu w2pp w2cxu w2gpu w2divu w2du.

(* ------------------------------------------------------------------------- *)
(*  11.  The elementary positivity hypotheses.                                *)
(* ------------------------------------------------------------------------- *)

Lemma w2_nu_pos       : 0 < w2nu.    Proof. unfold w2nu; lra.  Qed.
Lemma w2_sigma_nonneg : 0 <= w2sig.  Proof. unfold w2sig; lra. Qed.
Lemma w2_eps_nonneg   : 0 <= w2eps.  Proof. unfold w2eps; lra. Qed.
Lemma w2_c1_pos       : 0 < w2c1.    Proof. unfold w2c1; lra.  Qed.
Lemma w2_c2_pos       : 0 < w2c2.    Proof. unfold w2c2; lra.  Qed.
Lemma w2_Cb_pos       : 0 < w2Cb.    Proof. unfold w2Cb; lra.  Qed.
Lemma w2_C2_nonneg    : 0 <= w2C2.   Proof. unfold w2C2; lra.  Qed.
Lemma w2_C2_lt_1      : w2C2 < 1.    Proof. unfold w2C2; lra.  Qed.

Lemma w2_c1_large : w2c1 > w2xi * w2Cb ^ 2.
Proof. unfold w2c1, w2xi, w2Cb; lra. Qed.

Lemma w2_xi_large : w2xi > 2.
Proof. unfold w2xi; lra. Qed.

Lemma w2_hK_pos    : forall k, 0 < w2hK k.   Proof. intro; unfold w2hK; lra. Qed.
Lemma w2_aK_pos    : forall k, 0 < w2aK k.   Proof. intro; unfold w2aK; lra. Qed.

(*  (am_nonneg) item 18 of the trusted base -- and here, unlike in witness 1,
    it is met with room to spare rather than at its boundary.               *)
Lemma w2_am_nonneg : forall k, 0 <= w2am k.  Proof. intro; unfold w2am; lra. Qed.

(*  ... which is exactly the gap this witness closes:  |a| > 0 STRICTLY.      *)
Lemma w2_am_pos : forall k, 0 < w2am k.  Proof. intro; unfold w2am; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  12.  The stabilization parameters, computed.                              *)
(*       tau1NS_inv = c1*nu/h^2 + c2*|a|/h = 7 + 1 = 8  (the c2 branch is    *)
(*       now LIVE), whence tau1 = 1/9, tau2 = 8/7, sigt = 8/9.               *)
(* ------------------------------------------------------------------------- *)

Lemma w2_t1 :
  forall k, AbstractStability.t1 bool w2nu w2sig w2c1 w2c2 w2hK w2aK w2am k = 1/9.
Proof.
  intro k.
  unfold AbstractStability.t1, tau1, tau1NS_inv,
         w2nu, w2sig, w2c1, w2c2, w2hK, w2aK, w2am.
  field.
Qed.

Lemma w2_t2 :
  forall k, AbstractStability.t2 bool w2nu w2c1 w2c2 w2hK w2aK w2am k = 8/7.
Proof.
  intro k.
  unfold AbstractStability.t2, tau2, tau1NS_inv,
         w2nu, w2c1, w2c2, w2hK, w2aK, w2am.
  field.
Qed.

Lemma w2_sg :
  forall k, AbstractStability.sg bool w2nu w2sig w2c1 w2c2 w2hK w2aK w2am k = 8/9.
Proof.
  intro k.
  unfold AbstractStability.sg, sigt, tau1NS_inv,
         w2nu, w2sig, w2c1, w2c2, w2hK, w2aK, w2am.
  field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  13.  The four non-trivial hypotheses, each with a certificate that it     *)
(*       is met by NON-ZERO data.                                             *)
(* ------------------------------------------------------------------------- *)

(*  (H_skew_diag) eq:skew on the diagonal.  Not "cxu = 0" as on a singleton
    (9(d)), but a genuine cancellation:  1*2 + 1*(-2) = 0.                  *)
Lemma w2_skew_diag :
  Rsum w2Th (fun k => ip RPH (w2uu k) (w2cxu k)) = 0.
Proof.
  unfold w2Th. simpl Rsum. unfold w2uu, w2cxu. simpl. ring.
Qed.

(*  ... and the two summands it cancels are 2 and -2, neither of them 0.      *)
Lemma w2_skew_nondegenerate :
  ip RPH (w2uu true) (w2cxu true) = 2
  /\ ip RPH (w2uu false) (w2cxu false) = -2.
Proof. unfold w2uu, w2cxu. simpl. split; ring. Qed.

(*  (H_ibp_diag) eq:globalibp on the diagonal.  It reads -2 = -2, where
    witness 1's read 0 = -0.                                                *)
Lemma w2_ibp_diag :
  Rsum w2Th (fun k => ip RPH (w2uu k) (w2gpu k))
  = - Rsum w2Th (fun k => ip RPH (w2pp k) (w2divu k)).
Proof.
  unfold w2Th. simpl Rsum. unfold w2uu, w2gpu, w2pp, w2divu. simpl. ring.
Qed.

Lemma w2_ibp_nondegenerate :
  Rsum w2Th (fun k => ip RPH (w2uu k) (w2gpu k)) = -2
  /\ Rsum w2Th (fun k => ip RPH (w2pp k) (w2divu k)) = 2.
Proof.
  unfold w2Th. simpl Rsum. unfold w2uu, w2gpu, w2pp, w2divu. simpl.
  split; ring.
Qed.

(*  (S3) the weighted inverse estimate:  ||du|| = 1 <= 2 = (2 nu Cbar/h)
    alpha_K^(1/2) ||gu||.  Witness 1's read 0 <= 0.                         *)
Lemma w2_S3 :
  forall k, nrm (w2du k) <= (2 * w2nu * w2Cb / w2hK k) * sqrt (w2aK k) * nrm (w2gu k).
Proof.
  intro k.
  assert (E1 : ip RPH (w2du k) (w2du k) = 1) by (unfold w2du; simpl; ring).
  assert (E2 : ip RPH (w2gu k) (w2gu k) = 1) by (unfold w2gu; simpl; ring).
  unfold nrm. rewrite E1, E2, sqrt_1.
  unfold w2nu, w2Cb, w2hK, w2aK. rewrite sqrt_1. lra.
Qed.

(*  ... and it is met STRICTLY, between strictly positive sides.              *)
Lemma w2_S3_nondegenerate :
  forall k,
    0 < nrm (w2du k)
    /\ nrm (w2du k) < (2 * w2nu * w2Cb / w2hK k) * sqrt (w2aK k) * nrm (w2gu k).
Proof.
  intro k.
  assert (E1 : ip RPH (w2du k) (w2du k) = 1) by (unfold w2du; simpl; ring).
  assert (E2 : ip RPH (w2gu k) (w2gu k) = 1) by (unfold w2gu; simpl; ring).
  unfold nrm. rewrite E1, E2, sqrt_1.
  unfold w2nu, w2Cb, w2hK, w2aK. rewrite sqrt_1. split; lra.
Qed.

(*  (Heps) eq:UpperBoundOnEpsilon:  eps = 7/18 = eps_max at |a| = 1.          *)
Lemma w2_Heps :
  forall k, w2eps <= eps_max w2nu (w2hK k) (w2aK k) w2sig (w2am k) w2c1 w2c2 w2C2.
Proof.
  intro k.
  unfold eps_max, tau1, tau1NS_inv,
         w2eps, w2nu, w2sig, w2c1, w2c2, w2C2, w2hK, w2aK, w2am.
  field_simplify; lra.
Qed.

(*  ... with EQUALITY, i.e. at the sharp boundary rather than with slack --
    witness 1's extreme-admissible-choice property, preserved.             *)
Lemma w2_Heps_sharp :
  forall k, w2eps = eps_max w2nu (w2hK k) (w2aK k) w2sig (w2am k) w2c1 w2c2 w2C2.
Proof.
  intro k.
  unfold eps_max, tau1, tau1NS_inv,
         w2eps, w2nu, w2sig, w2c1, w2c2, w2C2, w2hK, w2aK, w2am.
  field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  14.  The instance:  abstract_stability, applied.                          *)
(* ------------------------------------------------------------------------- *)

Definition w2Cstab : R := C_stab w2c1 w2Cb w2xi w2C2.

Definition w2NormSq : R :=
  AbstractStability.NormSq RPH bool w2Th w2nu w2sig w2eps w2c1 w2c2
    w2hK w2aK w2am w2gu w2uu w2pp w2cxu w2gpu w2divu.

Theorem witness2_stability :
  0 < w2Cstab /\ w2BS >= w2Cstab * w2NormSq.
Proof.
  exact (AbstractStability.abstract_stability
           RPH bool w2Th w2nu w2sig w2eps w2c1 w2c2 w2Cb w2xi w2C2
           w2_nu_pos w2_sigma_nonneg w2_eps_nonneg w2_c1_pos w2_c2_pos w2_Cb_pos
           w2_C2_nonneg w2_C2_lt_1 w2_c1_large w2_xi_large
           w2hK w2aK w2am w2_hK_pos w2_aK_pos w2_am_nonneg
           w2gu w2uu w2pp w2cxu w2gpu w2divu w2du
           w2_skew_diag w2_ibp_diag w2_S3 w2_Heps).
Qed.

(* ------------------------------------------------------------------------- *)
(*  15.  The three numbers, computed:  B_S = 5069/567,  C_stab = 1/2         *)
(*       and  |||U_h|||^2 = 167/21.                                          *)
(* ------------------------------------------------------------------------- *)

(*  Through AbstractStability.HBS again.  Per element G(u) = du - sigma*uu
    = -2, so < G , G > = 4; the x-components differ between the elements:
      alpha X = cxu + gpu = 2 - 1 = 1  on true,   Xn = 1 ;
      alpha X = cxu + gpu = -2 - 1 = -3 on false, Xn = 9 ,
    so < L*_m , L_m > = 4 - Xn is 3 and -5, and tau_1 sums them to
    (1/9)(3 - 5) = -2/9.  On the mass side L_c = 1 + 7/18 = 25/18 and
    L*_c = -1 + 7/18 = -11/18, so < L*_c , L_c > = -275/324 per element and
    tau_2 sums to 2*(8/7)*(-275/324) = -1100/567.  Hence
      B_S = 2*1*2 + 1*2 + (7/18)*2 + 2/9 + 1100/567 = 7 + 1100/567
          = 5069/567 .                                                      *)
Lemma w2_BS_val : w2BS = 5069/567.
Proof.
  unfold w2BS.
  rewrite (AbstractStability.HBS RPH bool w2Th w2nu w2sig w2eps w2c1 w2c2
             w2hK w2aK w2am w2gu w2uu w2pp w2cxu w2gpu w2divu w2du
             w2_skew_diag w2_ibp_diag).
  unfold w2Th. simpl Rsum.
  rewrite !w2_t1, !w2_t2.
  unfold AbstractStability.Lstar_m, AbstractStability.L_m,
         AbstractStability.Lstar_c, AbstractStability.L_c,
         AbstractStability.Bv, AbstractStability.xu,
         vsub, vopp,
         w2gu, w2uu, w2pp, w2cxu, w2gpu, w2divu, w2du, w2nu, w2sig, w2eps.
  simpl. field.
Qed.

(*  C_stab has NO |a| dependence -- C_visc, C_u, C2 do not mention am -- so
    it is 1/2 here exactly as in witness 1.  That is a feature: the two
    witnesses bound the same constant from instances at opposite ends of the
    advection range.                                                        *)
Lemma w2_Cstab_val : w2Cstab = 1/2.
Proof.
  unfold w2Cstab, C_stab, C_visc, C_u, w2c1, w2Cb, w2xi, w2C2, Rmin.
  repeat (destruct (Rle_dec _ _); try lra).
Qed.

(*  |||U_h|||^2 = sum over the two elements of
      eps*Pn + nu*Vn + sigt*Un + tau1*Xn + tau2*Dn
    = (7/18 + 1 + 8/9 + 1/9 + 8/7) + (7/18 + 1 + 8/9 + 9/9 + 8/7)
    = 445/126 + 557/126 = 1002/126 = 167/21 .                              *)
Lemma w2_NormSq_val : w2NormSq = 167/21.
Proof.
  unfold w2NormSq, AbstractStability.NormSq, w2Th. simpl Rsum.
  unfold AbstractStability.perNorm,
         AbstractStability.Vn, AbstractStability.Un, AbstractStability.Pn,
         AbstractStability.Xn, AbstractStability.Dn, AbstractStability.xu.
  rewrite !w2_sg, !w2_t1, !w2_t2.
  unfold w2gu, w2uu, w2pp, w2cxu, w2gpu, w2divu, w2nu, w2sig, w2eps.
  simpl. field.
Qed.

(*  Every one of the five norm slots contributes, on BOTH elements: none of
    the triple norm's terms is switched off.                                *)
Lemma w2_slots_positive :
  forall k,
    0 < AbstractStability.Pn RPH bool w2pp k
    /\ 0 < AbstractStability.Vn RPH bool w2gu k
    /\ 0 < AbstractStability.Un RPH bool w2uu k
    /\ 0 < AbstractStability.Xn RPH bool w2cxu w2gpu k
    /\ 0 < AbstractStability.Dn RPH bool w2divu k.
Proof.
  intro k.
  unfold AbstractStability.Pn, AbstractStability.Vn, AbstractStability.Un,
         AbstractStability.Xn, AbstractStability.Dn, AbstractStability.xu,
         w2pp, w2gu, w2uu, w2cxu, w2gpu, w2divu.
  destruct k; simpl; repeat split; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  16.  The headline: the bundle is satisfiable with a STRICTLY POSITIVE     *)
(*       advection field, and the conclusion it yields is again a             *)
(*       NON-DEGENERATE inequality between positive numbers.                  *)
(* ------------------------------------------------------------------------- *)

Theorem abstract_stability_is_not_vacuous_with_advection :
  (*  the conclusion of abstract_stability, at this instance ...  *)
  w2BS >= w2Cstab * w2NormSq
  (*  ... with every quantity in it strictly positive ...  *)
  /\ 0 < w2Cstab /\ 0 < w2BS /\ 0 < w2NormSq
  (*  ... the advection strictly positive on every element (the gap this
      witness closes: am_nonneg is no longer met only at its boundary) ...  *)
  /\ (forall k, 0 < w2am k)
  (*  ... and equal to these explicit rationals.  *)
  /\ w2Cstab = 1/2 /\ w2NormSq = 167/21 /\ w2BS = 5069/567
  (*  So the bound reads  5069/567 >= 167/42 : true, and not a vacuous
      0 >= 0.  Over the common denominator 1134 it is 10138 >= 4509.       *)
  /\ w2Cstab * w2NormSq = 167/42.
Proof.
  destruct witness2_stability as [Hpos Hbound].
  repeat split;
    rewrite ?w2_Cstab_val, ?w2_NormSq_val, ?w2_BS_val in *;
    try (intro k; unfold w2am); lra.
Qed.

(*  The theorems above are proved from the SAME abstract_stability that the
    paper's lemma:Stability is identified with.  Hence:

      (i)   the hypothesis bundle of abstract_stability is CONSISTENT
            (else no instance could exist);

      (ii)  the conclusion is not degenerate: in both instances both sides
            of  B_S >= C_stab |||U_h|||^2  are strictly positive; and

      (iii) the bundle is consistent with a strictly positive advection
            field on a multi-element mesh, with every atom non-zero, every
            norm slot live, S3 strict, and both Green identities carrying
            non-zero data -- so the |a|-dependent half of the hypothesis
            set (am_nonneg, and the c2 branch of tau_1) is exercised, not
            merely discharged.

    Note that (i) is now a claim about a SMALLER bundle: what used to be the
    assumed tested identity is a theorem of AbstractStability.v, so the only
    Green-type inputs left to witness are the two diagonal identities, and
    B_S itself is computed rather than posited.

    Run  Print Assumptions abstract_stability_is_not_vacuous.  and
         Print Assumptions abstract_stability_is_not_vacuous_with_advection.
    to confirm that these rest on nothing beyond the standard library's
    real-number axioms.                                                      *)
