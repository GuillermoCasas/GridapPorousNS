(* ========================================================================= *)
(*  NonVacuityInterp.v                                                       *)
(*                                                                           *)
(*  A machine-checked WITNESS that the hypothesis bundle of                  *)
(*  AbstractInterpolation.abstract_continterp is CONSISTENT -- i.e. that     *)
(*  the theorem is not vacuously true.                                       *)
(*                                                                           *)
(*  This is the interpolation sibling of NonVacuity.v (which witnesses       *)
(*  AbstractStability.abstract_stability).  A theorem with contradictory     *)
(*  hypotheses is true and worthless; abstract_continterp carries THIRTY-    *)
(*  NINE named assumptions -- fifteen scalar, three mesh, twenty-one         *)
(*  structural -- so a reader is entitled to ask whether that list is        *)
(*  satisfiable at all.  This file answers YES, by exhibiting an explicit    *)
(*  instance and letting the kernel check it.                                *)
(*                                                                           *)
(*  *** WHY THIS FILE EXISTS: CI_pos. ***                                    *)
(*                                                                           *)
(*  The recent refactor made ONE hypothesis of the trusted base strictly     *)
(*  STRONGER.  IU >= 0 and IP >= 0 used to be assumed; they are now THEOREMS *)
(*  (AbstractInterpolation.IU_nonneg / IP_nonneg), derived from HI_uu / HI_pp *)
(*  together with  CI_pos : 0 < CI  -- which replaced the former CI_nonneg.   *)
(*  That strengthened hypothesis lives ONLY in AbstractInterpolation.v and    *)
(*  AbstractConvergence.v, neither of which had ANY non-vacuity witness.  So  *)
(*  the strengthening was, until this file, entirely unwitnessed: nothing     *)
(*  ruled out that  0 < CI  had quietly made the bundle contradictory.        *)
(*  Here CI = 1, so  0 < CI  is satisfied STRICTLY, jointly with the other    *)
(*  thirty-eight hypotheses.  That joint satisfaction -- not the trivial      *)
(*  observation that some positive real exists -- is what this file adds.     *)
(*                                                                           *)
(*  The instance:                                                            *)
(*                                                                           *)
(*    * the carrier is R with < x , y > := x * y   (a PreHilbert space);     *)
(*    * the mesh is the TWO-element list [K1; K2] with ONE genuine interior  *)
(*      face f, e1 f = K1 =/= K2 = e2 f  (two triangles sharing an edge);    *)
(*    * nu = sigma = 1,  h_K = alpha_K = 1,  |a|_inf = 1 on K1 and 2 on K2,  *)
(*      c1 = 7,  c2 = 1,  C2 = 1/2,  Cinv = Cb = 1,  cJ = 1,  cJ' = 9/8,     *)
(*      Cface = 1/2,  Nf = 1,  CI = 1,  eps = 7/20 (= eps_max, attained);    *)
(*    * ALL FOURTEEN vector atom families are NON-ZERO on BOTH elements,     *)
(*      and IU = IP = 1 are STRICTLY POSITIVE.                               *)
(*                                                                           *)
(*  The parameter formulas then give  phi1 = (8, 9),  tau1 = (1/9, 1/10),    *)
(*  tau2 = (8/7, 9/7),  sigt = (8/9, 9/10),  and the conclusion of           *)
(*  abstract_continterp instantiates to the NON-DEGENERATE inequality        *)
(*                                                                           *)
(*      |B_S(E,V)| = 72181/8400 = 8.5929...                                  *)
(*                 <=  C_totI * ((Psi_U + Psi_P) * |||V|||)  >=  131 ,       *)
(*                                                                           *)
(*  with BOTH sides strictly positive and both bounded by explicit           *)
(*  rationals -- so the theorem is not being satisfied by a vacuous 0 <= 0.  *)
(*                                                                           *)
(* ------------------------------------------------------------------------- *)
(*  WHY A TWO-ELEMENT MESH IS MANDATORY (the central design point).          *)
(*                                                                           *)
(*  FBp_e and FBc_e are DEFINITIONS, not free variables:                     *)
(*      FBp_e k = <vv k, gpu k> + <divv k, pp k>                             *)
(*      FBc_e k = <cxu k, vv k> + <cxv k, uu k>                              *)
(*  so H_ibp_vp is literally  Sum_k FBp_e k = 0  and H_skew is literally     *)
(*  Sum_k FBc_e k = 0.  On a ONE-element mesh (e1 = e2, as in NonVacuity.v)  *)
(*  THREE degeneracies fire at once:                                         *)
(*    (i)   Dt1 f = t1(e1 f) - t1(e2 f) = 0 identically;                     *)
(*    (ii)  H_assemble_p and H_assemble_c both read 0 = 0, so FBp and FBc    *)
(*          become TOTALLY UNCONSTRAINED by assembly;                        *)
(*    (iii) hence JmpP = 0 and IIIterm = 0 -- the ENTIRE Step-6b/6d jump     *)
(*          machinery drops out of B_S identically.                          *)
(*  Additionally H_jump would degenerate to cJ*ph <= ph <= cJ'*ph, which is  *)
(*  implied by cJ <= 1 <= cJ' alone and carries zero information.  That is a *)
(*  genuine degenerate escape, and it is REJECTED here.                      *)
(*                                                                           *)
(*  With Th = [K1;K2] and Fl = [f] instead, H_ibp_vp forces                  *)
(*  FBp_e K2 = -FBp_e K1, so the assembly identity reads                     *)
(*      (t1 K1 - t1 K2) * FBp_e K1  =  (t1 K1 - t1 K2) * FBp                 *)
(*  and since t1 K1 - t1 K2 = 1/9 - 1/10 = 1/90 =/= 0 it PINS FBp = 2        *)
(*  exactly (likewise FBc = 2).  FBp and FBc are therefore determined by the *)
(*  data, and H_face_p / H_face_c become genuine TESTS rather than free      *)
(*  choices.  Section 21 below machine-checks that liveness: every quantity  *)
(*  that a one-element mesh would have zeroed is a non-zero rational here.   *)
(* ------------------------------------------------------------------------- *)
(*  WHAT THIS FILE DOES **NOT** SHOW -- stated with equal prominence.        *)
(*                                                                           *)
(*  (a) It does NOT show that the finite element objects satisfy the         *)
(*      hypotheses.  That is the SOUNDNESS direction, and it is what the     *)
(*      hand audit establishes; consistency and soundness are different      *)
(*      claims and this file makes only the first.                           *)
(*                                                                           *)
(*  (b) It does NOT show the bound is SHARP.  At this instance the left side *)
(*      is 8.59... and the right side is at least 131 -- roughly fifteenfold *)
(*      slack, because C_totI tracks constants crudely (it is a sum of       *)
(*      products of Cauchy-Schwarz constants).  Non-vacuity is not tightness, *)
(*      and no claim of tightness is made or implied.                        *)
(*                                                                           *)
(*  (c) It does NOT witness abstract_convergence.  THIS file discharges only *)
(*      the abstract_continterp bundle; the convergence half of gap G2 is    *)
(*      closed by the sibling file NonVacuityConv.v, which witnesses the     *)
(*      ~44-hypothesis abstract_convergence bundle (CI_pos among them).      *)
(*                                                                           *)
(*  (d) The LOWER half of H_jump  (cJ * ph(e1 f) <= ph(e2 f))  is satisfied  *)
(*      with SLACK here (8 <= 9), not at equality.  This is UNAVOIDABLE, not *)
(*      an oversight: liveness of the assembly identities REQUIRES           *)
(*      ph(e1 f) =/= ph(e2 f)  (else Dt1 = 0 and we are back in case (i)     *)
(*      above), and given cJ <= 1 <= cJ' at most ONE of the two halves of    *)
(*      H_jump can then be at equality.  This witness puts the UPPER half at *)
(*      equality (9 <= (9/8)*8 = 9), where it genuinely forces cJ' >= 9/8,   *)
(*      strictly beyond the hypothesis one_le_cJ' -- see w_cJ'_forced.  The  *)
(*      mirror choice (making the lower half sharp instead) is equally       *)
(*      defensible; both halves cannot be sharp at once.                     *)
(*                                                                           *)
(*  (e) Two of the twenty pointwise-estimate instances -- Hw_cxv and HI_cxu  *)
(*      at K2 -- hold with slack (1 <= 2) rather than equality, because the  *)
(*      mesh heterogeneity is carried by am = (1, 2).  The other EIGHTEEN    *)
(*      hold with EQUALITY, so the data saturates its bounds rather than     *)
(*      fitting inside them with room to spare.                             *)
(*                                                                           *)
(*  (f) c1_large / xi_large are NOT hypotheses of abstract_continterp (they  *)
(*      belong to abstract_stability), so nothing here exercises them.       *)
(*      ContinuityAlgebra's section carries alphaK_le_1, but it is likewise  *)
(*      absent from the discharged signature -- verified with About.         *)
(* ------------------------------------------------------------------------- *)
(*  [known-fragility]  This file deliberately RE-DEFINES the PreHilbert      *)
(*  instance RPH rather than importing NonVacuity's.  StabilityAlgebra and   *)
(*  ContinuityAlgebra BOTH define tau1, tau2, phi1 and sigt; NonVacuity.v    *)
(*  imports StabilityAlgebra, while this file needs ContinuityAlgebra (which *)
(*  is what AbstractInterpolation's parameters are built from).  Requiring   *)
(*  NonVacuity here would pull StabilityAlgebra in transitively and make the *)
(*  unqualified names tau1/tau2/phi1/sigt resolve to the WRONG file's        *)
(*  definitions.  Six lines of duplication buy immunity from that; do not    *)
(*  "de-duplicate" RPH by importing NonVacuity.                              *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums
                              AbstractInterpolation InverseEstimates.
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

(*  On this carrier the induced norm is the absolute value: nrm x =          *)
(*  sqrt <x,x> = sqrt (x*x) = |x|.  Every pointwise estimate below is        *)
(*  discharged through this bridge.                                          *)
Lemma nrm_val : forall x : carrier RPH, nrm x = Rabs x.
Proof.
  intro x. unfold nrm. simpl.
  rewrite <- (sqrt_Rsqr_abs x). unfold Rsqr. reflexivity.
Qed.

(* ------------------------------------------------------------------------- *)
(*  2.  The mesh: TWO elements, ONE genuine interior face.                    *)
(*                                                                           *)
(*  K := bool, with K1 := true and K2 := false.  The single face f := tt has *)
(*  e1 f = K1 and e2 f = K2, so e1 =/= e2 and the face is interior in the    *)
(*  only sense the abstract theorem knows: it has two distinct neighbours.   *)
(*  Neither F := Empty_set nor Fl := [] -- both of which would make the      *)
(*  face hypotheses vacuous -- is used.                                      *)
(* ------------------------------------------------------------------------- *)

Definition wTh : list bool := [true; false].
Definition wFl : list unit := [tt].
Definition we1 (_ : unit) : bool := true.    (*  K1  *)
Definition we2 (_ : unit) : bool := false.   (*  K2  *)

(* ------------------------------------------------------------------------- *)
(*  3.  The thirteen scalars.                                                 *)
(* ------------------------------------------------------------------------- *)

Definition wnu    : R := 1.
Definition wsig   : R := 1.       (*  sigma > 0 is REQUIRED: sg = sigma*ph*t1, *)
                                  (*  and at sigma = 0 every jump bound and    *)
                                  (*  JmpP/IIIterm would collapse to 0.        *)
Definition weps   : R := 7/20.    (*  = min_k C2*c1*aK^2*t1/hK^2, attained at K2 *)
Definition wc1    : R := 7.
Definition wc2    : R := 1.
Definition wC2    : R := 1/2.
Definition wCinv  : R := 1.
Definition wCb    : R := 1.
Definition wcJ    : R := 1.       (*  largest admissible: cJ_le_1 at equality  *)
Definition wcJ'   : R := 9/8.     (*  sharp: forced by ph(K2)/ph(K1) = 9/8     *)
Definition wCface : R := 1/2.     (*  sharp for H_face_p: 2 <= 2               *)
Definition wNf    : R := 1.       (*  forced minimum -- see w_Nf_forced        *)
Definition wCI    : R := 1.       (*  STRICTLY POSITIVE -- the point of G2     *)

(* ------------------------------------------------------------------------- *)
(*  4.  Mesh data.  All the heterogeneity is carried by am.                   *)
(*                                                                           *)
(*  am > 0 on BOTH elements is essential, and is exactly what NonVacuity.v   *)
(*  cannot show (it sets am := 0, gap G1).  The right-hand sides of H_face_c *)
(*  and HI_cxu carry a factor am, so at am = 0 they read  |FBc| <= 0  and    *)
(*  |cxu| <= 0 -- forcing cxu = 0, hence FBc_e = 0, hence no constraint on   *)
(*  FBc at all.  A strictly positive advection field is what gives those two *)
(*  hypotheses content, and this witness supplies one.                       *)
(* ------------------------------------------------------------------------- *)

Definition whK (_ : bool) : R := 1.
Definition waK (_ : bool) : R := 1.
Definition wam (k : bool) : R := if k then 1 else 2.

(* ------------------------------------------------------------------------- *)
(*  5.  The atoms.  Fourteen vector families, all NON-ZERO on BOTH elements,  *)
(*      plus the two strictly positive interpolation sizes IU, IP and the     *)
(*      two face data FBp, FBc (which the assembly identities PIN to 2).      *)
(*                                                                           *)
(*  The sign flips are not decoration: cxu/cxv balance H_skew, gpu/divv      *)
(*  balance H_ibp_vp, and gpv (which carries no upper bound of its own)      *)
(*  solves H_ibp_qu.  gpv K2 = -3 rather than -1 so that xv = cxv +v gpv is  *)
(*  non-zero on BOTH elements (-1 would have given xv K1 = 0).               *)
(* ------------------------------------------------------------------------- *)

Definition wgu   (_ : bool) : carrier RPH := 1.
Definition wgv   (_ : bool) : carrier RPH := 1.
Definition wdu   (_ : bool) : carrier RPH := 2.
Definition wdv   (_ : bool) : carrier RPH := 2.
Definition wcxu  (k : bool) : carrier RPH := if k then 1 else -1.
Definition wcxv  (k : bool) : carrier RPH := if k then 1 else -1.
Definition wgpu  (k : bool) : carrier RPH := if k then 1 else -1.
Definition wgpv  (k : bool) : carrier RPH := if k then 1 else -3.
Definition wuu   (_ : bool) : carrier RPH := 1.
Definition wvv   (_ : bool) : carrier RPH := 1.
Definition wpp   (_ : bool) : carrier RPH := 1.
Definition wqq   (_ : bool) : carrier RPH := 1.
Definition wdivu (_ : bool) : carrier RPH := 1.
Definition wdivv (k : bool) : carrier RPH := if k then 1 else -1.

Definition wIU (_ : bool) : R := 1.   (*  STRICTLY POSITIVE  *)
Definition wIP (_ : bool) : R := 1.   (*  STRICTLY POSITIVE  *)

Definition wFBp (_ : unit) : R := 2.  (*  pinned by H_assemble_p  *)
Definition wFBc (_ : unit) : R := 2.  (*  pinned by H_assemble_c  *)

(* ------------------------------------------------------------------------- *)
(*  6.  The stabilization parameters, computed.                              *)
(*                                                                           *)
(*  [debugging-lore]  Dependency order matters in the unfolding tactic:      *)
(*  tau1/sigt REINTRODUCE phi1, and phi1 reintroduces tauNSinv, so the       *)
(*  derived names must be unfolded BEFORE the base ones or the goal is left  *)
(*  with folded occurrences that `field` cannot see through.                 *)
(* ------------------------------------------------------------------------- *)

Ltac unf := unfold AbstractInterpolation.ph, AbstractInterpolation.t1,
  AbstractInterpolation.t2, AbstractInterpolation.sg,
  tau1, tau2, sigt, phi1, tauNSinv,
  wnu, wsig, weps, wc1, wc2, wC2, wCinv, wCb, wcJ, wcJ', wCface, wNf, wCI,
  whK, waK, wam in *.

(*  ph = phi1 = aK*(c1*nu/h^2 + c2*am/h) : 1*(7+1) = 8 and 1*(7+2) = 9.     *)
Lemma w_ph : forall k, AbstractInterpolation.ph bool wnu wc1 wc2 whK waK wam k
                       = if k then 8 else 9.
Proof. intro k. destruct k; unf; field. Qed.

(*  t1 = 1/(ph+sigma) : 1/9 and 1/10.  The jump  t1 K1 - t1 K2 = 1/90 =/= 0  *)
(*  is what makes the assembly identities live.                              *)
Lemma w_t1 : forall k, AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                       = if k then 1/9 else 1/10.
Proof. intro k. destruct k; unf; field. Qed.

(*  t2 = h^2*tauNSinv/(c1*aK) : 8/7 and 9/7.                                 *)
Lemma w_t2 : forall k, AbstractInterpolation.t2 bool wnu wc1 wc2 whK waK wam k
                       = if k then 8/7 else 9/7.
Proof. intro k. destruct k; unf; field. Qed.

(*  sg = sigma*ph/(ph+sigma) : 8/9 and 9/10.                                 *)
Lemma w_sg : forall k, AbstractInterpolation.sg bool wnu wsig wc1 wc2 whK waK wam k
                       = if k then 8/9 else 9/10.
Proof. intro k. destruct k; unf; field. Qed.

(* ------------------------------------------------------------------------- *)
(*  7.  The fifteen scalar hypotheses.                                        *)
(* ------------------------------------------------------------------------- *)

Lemma w_nu_pos       : 0 < wnu.     Proof. unf; lra. Qed.
Lemma w_sigma_nonneg : 0 <= wsig.   Proof. unf; lra. Qed.
Lemma w_eps_nonneg   : 0 <= weps.   Proof. unf; lra. Qed.
Lemma w_c1_pos       : 0 < wc1.     Proof. unf; lra. Qed.
Lemma w_c2_pos       : 0 < wc2.     Proof. unf; lra. Qed.
Lemma w_C2_nonneg    : 0 <= wC2.    Proof. unf; lra. Qed.
Lemma w_C2_lt_1      : wC2 < 1.     Proof. unf; lra. Qed.
Lemma w_Cinv_pos     : 0 < wCinv.   Proof. unf; lra. Qed.
Lemma w_Cb_pos       : 0 < wCb.     Proof. unf; lra. Qed.
Lemma w_cJ_pos       : 0 < wcJ.     Proof. unf; lra. Qed.
Lemma w_cJ_le_1      : wcJ <= 1.    Proof. unf; lra. Qed.
Lemma w_one_le_cJ'   : 1 <= wcJ'.   Proof. unf; lra. Qed.
Lemma w_Cface_nonneg : 0 <= wCface. Proof. unf; lra. Qed.
Lemma w_Nf_nonneg    : 0 <= wNf.    Proof. unf; lra. Qed.

(*  *** THE G2 TARGET ***  The hypothesis the refactor strengthened from     *)
(*  CI_nonneg (0 <= CI) to CI_pos (0 < CI).  Satisfied here STRICTLY, and    *)
(*  -- crucially -- jointly with the other thirty-eight.                     *)
Lemma w_CI_pos       : 0 < wCI.     Proof. unf; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  8.  The three mesh hypotheses.                                            *)
(* ------------------------------------------------------------------------- *)

Lemma w_hK_pos    : forall k, 0 < whK k.  Proof. intro k; destruct k; unf; lra. Qed.
Lemma w_aK_pos    : forall k, 0 < waK k.  Proof. intro k; destruct k; unf; lra. Qed.

(*  am_nonneg is met at am = (1,2): STRICTLY POSITIVE on both elements, not  *)
(*  at the am = 0 boundary that NonVacuity.v is confined to.                 *)
Lemma w_am_nonneg : forall k, 0 <= wam k. Proof. intro k; destruct k; unf; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  9.  The three Green identities (eq:skew, eq:globalibp).                   *)
(*                                                                           *)
(*  H_skew and H_ibp_vp read 0 = -0 here, but they are NOT idle: since       *)
(*  FBc_e and FBp_e are definitions, these two identities are literally the  *)
(*  statements  Sum_k FBc_e k = 0  and  Sum_k FBp_e k = 0, and it is exactly *)
(*  that which pins FBc and FBp through the assembly identities below.       *)
(*  H_ibp_qu has BOTH sides non-zero (= 2).                                  *)
(* ------------------------------------------------------------------------- *)

Lemma w_skew :
  Rsum wTh (fun k => ip RPH (wvv k) (wcxu k))
  = - Rsum wTh (fun k => ip RPH (wuu k) (wcxv k)).
Proof. unfold wTh. simpl. unfold wvv, wcxu, wuu, wcxv. simpl. ring. Qed.

Lemma w_ibp_vp :
  Rsum wTh (fun k => ip RPH (wvv k) (wgpu k))
  = - Rsum wTh (fun k => ip RPH (wpp k) (wdivv k)).
Proof. unfold wTh. simpl. unfold wvv, wgpu, wpp, wdivv. simpl. ring. Qed.

Lemma w_ibp_qu :
  Rsum wTh (fun k => ip RPH (wqq k) (wdivu k))
  = - Rsum wTh (fun k => ip RPH (wuu k) (wgpv k)).
Proof. unfold wTh. simpl. unfold wqq, wdivu, wuu, wgpv. simpl. ring. Qed.

(* ------------------------------------------------------------------------- *)
(*  10.  Facewise assembly (Steps 6b, 6d).  BOTH sides of BOTH identities    *)
(*       equal the non-zero rational 1/45 -- see w_assemble_p_live below.    *)
(* ------------------------------------------------------------------------- *)

Lemma w_assemble_p :
  Rsum wTh (fun k => AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractInterpolation.FBp_e RPH bool wgpu wvv wpp wdivv k)
  = Rsum wFl (fun f => (AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
                        - AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
                       * wFBp f).
Proof.
  unfold wTh, wFl. simpl. rewrite !w_t1.
  unfold we1, we2, AbstractInterpolation.FBp_e, wgpu, wvv, wpp, wdivv, wFBp.
  simpl. field.
Qed.

Lemma w_assemble_c :
  Rsum wTh (fun k => AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractInterpolation.FBc_e RPH bool wcxu wcxv wuu wvv k)
  = Rsum wFl (fun f => (AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
                        - AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
                       * wFBc f).
Proof.
  unfold wTh, wFl. simpl. rewrite !w_t1.
  unfold we1, we2, AbstractInterpolation.FBc_e, wcxu, wcxv, wuu, wvv, wFBc.
  simpl. field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  11.  The two face estimates.                                             *)
(*                                                                           *)
(*  H_face_p reads  2 <= (1/2)*(1+1+1+1) = 2 : EQUALITY, so Cface = 1/2 is   *)
(*  sharp for it.  H_face_c reads  2 <= (1/2)*(1+2+1+2) = 3, and its RHS is  *)
(*  strictly positive ONLY because am > 0 -- at am = 0 it would read 2 <= 0, *)
(*  which is FALSE.  So this witness could not exist at the am = 0 boundary  *)
(*  with FBc pinned to 2; that is gap G1 made concrete.                      *)
(* ------------------------------------------------------------------------- *)

Lemma w_face_p :
  forall f, Rabs (wFBp f)
  <= wCface * (  waK (we1 f) / whK (we1 f) * (nrm (wvv (we1 f)) * wIP (we1 f))
               + waK (we2 f) / whK (we2 f) * (nrm (wvv (we1 f)) * wIP (we2 f))
               + waK (we1 f) / whK (we1 f) * (nrm (wvv (we2 f)) * wIP (we1 f))
               + waK (we2 f) / whK (we2 f) * (nrm (wvv (we2 f)) * wIP (we2 f))).
Proof.
  intro f. rewrite !nrm_val.
  unfold we1, we2, wFBp, wvv, wIP. unf. simpl.
  rewrite Rabs_R1. rewrite (Rabs_right 2) by lra. lra.
Qed.

Lemma w_face_c :
  forall f, Rabs (wFBc f)
  <= wCface * (  waK (we1 f) * wam (we1 f) / whK (we1 f)
                   * (wIU (we1 f) * nrm (wvv (we1 f)))
               + waK (we2 f) * wam (we2 f) / whK (we2 f)
                   * (wIU (we1 f) * nrm (wvv (we2 f)))
               + waK (we1 f) * wam (we1 f) / whK (we1 f)
                   * (wIU (we2 f) * nrm (wvv (we1 f)))
               + waK (we2 f) * wam (we2 f) / whK (we2 f)
                   * (wIU (we2 f) * nrm (wvv (we2 f)))).
Proof.
  intro f. rewrite !nrm_val.
  unfold we1, we2, wFBc, wvv, wIU. unf. simpl.
  rewrite Rabs_R1. rewrite (Rabs_right 2) by lra. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  12.  H:jump.  ph(K1) = 8 =/= 9 = ph(K2), so the upper half genuinely     *)
(*       forces cJ' >= 9/8 -- strictly beyond one_le_cJ' (see w_cJ'_forced). *)
(*       Met at EQUALITY.  On the lower half see caveat (d) in the banner.   *)
(* ------------------------------------------------------------------------- *)

Lemma w_jump :
  forall f, wcJ * AbstractInterpolation.ph bool wnu wc1 wc2 whK waK wam (we1 f)
            <= AbstractInterpolation.ph bool wnu wc1 wc2 whK waK wam (we2 f)
            /\ AbstractInterpolation.ph bool wnu wc1 wc2 whK waK wam (we2 f)
               <= wcJ' * AbstractInterpolation.ph bool wnu wc1 wc2 whK waK wam (we1 f).
Proof.
  intro f. rewrite !w_ph. unfold we1, we2, wcJ, wcJ'. split; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  13.  Bounded face multiplicity (H:mesh).  Nf = 1 is the FORCED minimum   *)
(*       at this mesh -- see w_Nf_forced.                                    *)
(* ------------------------------------------------------------------------- *)

Lemma w_mult1 :
  forall g : bool -> R, (forall k, 0 <= g k) ->
    Rsum wFl (fun f => g (we1 f)) <= wNf * Rsum wTh g.
Proof.
  intros g Hg. unfold wFl, wTh, wNf, we1. simpl. pose proof (Hg false). lra.
Qed.

Lemma w_mult2 :
  forall g : bool -> R, (forall k, 0 <= g k) ->
    Rsum wFl (fun f => g (we2 f)) <= wNf * Rsum wTh g.
Proof.
  intros g Hg. unfold wFl, wTh, wNf, we2. simpl. pose proof (Hg true). lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  14.  V-side weighted inverse estimates (lem:winv).                       *)
(*       Hw_gv and Hw_dv hold with EQUALITY on both elements; Hw_cxv with    *)
(*       equality on K1 and slack on K2 (where am = 2).                      *)
(* ------------------------------------------------------------------------- *)

Lemma w_gv : forall k, nrm (wgv k) <= wCinv / whK k * sqrt (waK k) * nrm (wvv k).
Proof.
  intro k. rewrite !nrm_val. unfold wgv, wvv. unf.
  rewrite sqrt_1, Rabs_R1. lra.
Qed.

Lemma w_dv : forall k, nrm (wdv k) <= 2 * wnu * wCb / whK k * sqrt (waK k) * nrm (wgv k).
Proof.
  intro k. rewrite !nrm_val. unfold wdv, wgv. unf.
  rewrite sqrt_1, Rabs_R1, (Rabs_right 2) by lra. lra.
Qed.

(*  Restated through the winv_est schema so it discharges abstract_continterp's
    (now winv_est) Hw_cxv slot.  The (aK am) weight matches only
    propositionally, so -- unlike the Shape-A w_gv/w_dv, which pass by
    conversion untouched -- this one must wear winv_est explicitly; `unfold
    winv_est; intro k; cbv beta' recovers the raw goal (up to the harmless
    (waK*wam) grouping, which the closing lra absorbs).  *)
Lemma w_cxv : winv_est RPH bool whK wCinv (fun k => waK k * wam k) wcxv wvv.
Proof.
  unfold winv_est. intro k. cbv beta.
  rewrite !nrm_val. unfold wcxv, wvv. destruct k; unf; simpl.
  - rewrite Rabs_R1. lra.
  - rewrite Rabs_R1, (Rabs_left (-1)) by lra. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  15.  E-side interpolation estimates (eq:interpdivvisc--eq:interpzero),   *)
(*       one constant CI for all seven.  All hold with EQUALITY except       *)
(*       HI_cxu at K2 (am = 2).  HI_uu and HI_pp, together with CI_pos, are  *)
(*       what DERIVE IU_nonneg and IP_nonneg inside AbstractInterpolation.v. *)
(* ------------------------------------------------------------------------- *)

Lemma w_HI_gu : forall k, nrm (wgu k) <= wCI * sqrt (waK k) / whK k * wIU k.
Proof.
  intro k. rewrite nrm_val. unfold wgu, wIU. unf. rewrite sqrt_1, Rabs_R1. lra.
Qed.

Lemma w_HI_du : forall k, nrm (wdu k) <= 2 * wnu * wCI * waK k / (whK k)^2 * wIU k.
Proof.
  intro k. rewrite nrm_val. unfold wdu, wIU. unf. rewrite (Rabs_right 2) by lra. lra.
Qed.

(*  The G1-sensitive one: the RHS carries a factor am, so it is 0 iff am = 0. *)
Lemma w_HI_cxu : forall k, nrm (wcxu k) <= wCI * waK k * wam k / whK k * wIU k.
Proof.
  intro k. rewrite nrm_val. unfold wcxu, wIU. destruct k; unf; simpl.
  - rewrite Rabs_R1. lra.
  - rewrite (Rabs_left (-1)) by lra. lra.
Qed.

Lemma w_HI_gpu : forall k, nrm (wgpu k) <= wCI * waK k / whK k * wIP k.
Proof.
  intro k. rewrite nrm_val. unfold wgpu, wIP. destruct k; unf; simpl.
  - rewrite Rabs_R1. lra.
  - rewrite (Rabs_left (-1)) by lra. lra.
Qed.

Lemma w_HI_divu : forall k, nrm (wdivu k) <= wCI * waK k / whK k * wIU k.
Proof.
  intro k. rewrite nrm_val. unfold wdivu, wIU. unf. rewrite Rabs_R1. lra.
Qed.

Lemma w_HI_uu : forall k, nrm (wuu k) <= wCI * wIU k.
Proof.
  intro k. rewrite nrm_val. unfold wuu, wIU. unf. rewrite Rabs_R1. lra.
Qed.

Lemma w_HI_pp : forall k, nrm (wpp k) <= wCI * wIP k.
Proof.
  intro k. rewrite nrm_val. unfold wpp, wIP. unf. rewrite Rabs_R1. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  16.  eq:epscond.  eps = 7/20 <= 7/18 at K1 and = 7/20 at K2: the extreme *)
(*       admissible value, attained -- mirroring the stability witness.      *)
(* ------------------------------------------------------------------------- *)

Lemma w_Heps :
  forall k, weps <= wC2 * wc1 * (waK k)^2
                    * AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                    / (whK k)^2.
Proof.
  intro k. rewrite w_t1. destruct k; unf; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  17.  The instance:  abstract_continterp, applied to all 39 hypotheses.   *)
(* ------------------------------------------------------------------------- *)

Definition wBS : R :=
  AbstractInterpolation.BS RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgu wgv wdu wdv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv.

Definition wCtotI : R :=
  AbstractInterpolation.CtotI wc1 wc2 wC2 wCinv wCb wcJ wcJ' wCface wNf wCI.

Definition wPsU : R := AbstractInterpolation.PsU bool wTh wnu wc1 wc2 whK waK wam wIU.
Definition wPsP : R := AbstractInterpolation.PsP bool wTh wnu wsig wc1 wc2 whK waK wam wIP.
Definition wNV  : R :=
  AbstractInterpolation.NV RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgv wcxv wgpv wvv wqq wdivv.

Theorem witness_continterp : Rabs wBS <= wCtotI * ((wPsU + wPsP) * wNV).
Proof.
  exact (AbstractInterpolation.abstract_continterp
           RPH bool wTh unit wFl we1 we2
           wnu wsig weps wc1 wc2 wC2 wCinv wCb wcJ wcJ' wCface wNf wCI
           w_nu_pos w_sigma_nonneg w_eps_nonneg w_c1_pos w_c2_pos
           w_C2_nonneg w_C2_lt_1 w_Cinv_pos w_Cb_pos w_cJ_pos w_cJ_le_1
           w_one_le_cJ' w_Cface_nonneg w_Nf_nonneg w_CI_pos
           whK waK wam w_hK_pos w_aK_pos w_am_nonneg
           wgu wgv wdu wdv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv
           wIU wIP wFBp wFBc
           w_skew w_ibp_vp w_ibp_qu w_assemble_p w_assemble_c
           w_face_p w_face_c w_jump w_mult1 w_mult2
           w_gv w_dv w_cxv
           w_HI_gu w_HI_du w_HI_cxu w_HI_gpu w_HI_divu w_HI_uu w_HI_pp
           w_Heps).
Qed.

(* ------------------------------------------------------------------------- *)
(*  18.  The numbers, computed.                                              *)
(*                                                                           *)
(*  B_S is the eighteen-term form of the appendix evaluated at the atoms     *)
(*  above -- a number this witness COMPUTES, not one it posits.              *)
(* ------------------------------------------------------------------------- *)

Lemma w_BS_val : wBS = 72181/8400.
Proof.
  unfold wBS, AbstractInterpolation.BS,
    AbstractInterpolation.T1, AbstractInterpolation.T2, AbstractInterpolation.T3,
    AbstractInterpolation.T4, AbstractInterpolation.T5, AbstractInterpolation.T6,
    AbstractInterpolation.T7, AbstractInterpolation.T8, AbstractInterpolation.T9,
    AbstractInterpolation.T10, AbstractInterpolation.T11, AbstractInterpolation.T12,
    AbstractInterpolation.T13, AbstractInterpolation.T14, AbstractInterpolation.T15,
    AbstractInterpolation.T16, AbstractInterpolation.T17, AbstractInterpolation.T18,
    wTh.
  simpl Rsum.
  rewrite !w_t1, !w_t2.
  unfold AbstractInterpolation.xu, AbstractInterpolation.xv,
    wgu, wgv, wdu, wdv, wcxu, wcxv, wgpu, wgpv, wuu, wvv, wpp, wqq, wdivu, wdivv,
    wnu, wsig, weps.
  simpl. field.
Qed.

(*  Psi_U^2 = Sum aK^2*t2/hK^2*IU^2 = 8/7 + 9/7 = 17/7.                      *)
Lemma w_PsU2_val : AbstractInterpolation.PsU2 bool wTh wnu wc1 wc2 whK waK wam wIU = 17/7.
Proof.
  unfold AbstractInterpolation.PsU2, wTh. simpl Rsum.
  rewrite !w_t2. unfold wIU. unf. field.
Qed.

(*  Psi_P^2 = Sum aK^2*t1/hK^2*IP^2 = 1/9 + 1/10 = 19/90.                    *)
Lemma w_PsP2_val : AbstractInterpolation.PsP2 bool wTh wnu wsig wc1 wc2 whK waK wam wIP = 19/90.
Proof.
  unfold AbstractInterpolation.PsP2, wTh. simpl Rsum.
  rewrite !w_t1. unfold wIP. unf. field.
Qed.

(*  |||V|||^2 = Sum perV = 3.826... + 5.135... = 941/105.                    *)
Lemma w_NV2_val :
  AbstractInterpolation.NV2 RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgv wcxv wgpv wvv wqq wdivv = 941/105.
Proof.
  unfold AbstractInterpolation.NV2, AbstractInterpolation.perV, wTh. simpl Rsum.
  rewrite !w_sg, !w_t1, !w_t2.
  unfold AbstractInterpolation.xv, wgv, wcxv, wgpv, wvv, wqq, wdivv, wnu, wsig, weps.
  simpl. field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  19.  Non-degeneracy: every factor of the conclusion is strictly positive, *)
(*       and the right-hand side admits an explicit RATIONAL lower bound.     *)
(*                                                                           *)
(*  C_totI, Psi_U, Psi_P and |||V||| are all irrational at this instance      *)
(*  (they are square roots), so instead of exact values we bracket them by    *)
(*  rationals.  That is what turns "the RHS is positive" -- which a reader     *)
(*  could satisfy with 10^-30 -- into "the RHS is at least 131".              *)
(* ------------------------------------------------------------------------- *)

Lemma sqrt_lower : forall a b : R, 0 <= a -> a * a <= b -> a <= sqrt b.
Proof.
  intros a b Ha Hab.
  rewrite <- (sqrt_Rsqr a Ha). apply sqrt_le_1_alt. unfold Rsqr. lra.
Qed.

Lemma w_BS_pos : 0 < Rabs wBS.
Proof. rewrite w_BS_val, (Rabs_right (72181/8400)) by lra. lra. Qed.

Lemma w_LHS_val : Rabs wBS = 72181/8400.
Proof. rewrite w_BS_val. apply Rabs_right. lra. Qed.

(*  Psi_U = sqrt(17/7) >= 3/2, since (3/2)^2 = 9/4 <= 17/7.                  *)
Lemma w_PsU_lower : 3/2 <= wPsU.
Proof.
  unfold wPsU, AbstractInterpolation.PsU. rewrite w_PsU2_val.
  apply sqrt_lower; lra.
Qed.

Lemma w_PsU_pos : 0 < wPsU.
Proof. pose proof w_PsU_lower. lra. Qed.

(*  Psi_P = sqrt(19/90) > 0.                                                  *)
Lemma w_PsP_pos : 0 < wPsP.
Proof.
  unfold wPsP, AbstractInterpolation.PsP. rewrite w_PsP2_val.
  apply sqrt_lt_R0. lra.
Qed.

Lemma w_PsP_nonneg : 0 <= wPsP.
Proof. pose proof w_PsP_pos. lra. Qed.

(*  |||V||| = sqrt(941/105) >= 5/2, since (5/2)^2 = 25/4 <= 941/105.         *)
Lemma w_NV_lower : 5/2 <= wNV.
Proof.
  unfold wNV, AbstractInterpolation.NV. rewrite w_NV2_val.
  apply sqrt_lower; lra.
Qed.

Lemma w_NV_pos : 0 < wNV.
Proof. pose proof w_NV_lower. lra. Qed.

(*  sqrt 7 >= 2, and (sqrt 7)^2 = 7 -- the two facts every constant below     *)
(*  needs.                                                                    *)
Lemma w_sqrt7 : 2 <= sqrt 7 /\ sqrt 7 ^ 2 = 7.
Proof. split; [apply sqrt_lower; lra | apply pow2_sqrt; lra]. Qed.

(*  KaggI = sqrt(1 + 7 + 7 + (sqrt 7 + 1)^2 + 1) = sqrt(24 + 2*sqrt 7) >= 4. *)
Lemma w_KaggI_lower : 4 <= AbstractInterpolation.KaggI wc1 wc2 wCI.
Proof.
  unfold AbstractInterpolation.KaggI, wc1, wc2, wCI.
  destruct w_sqrt7 as [Hs7 Hs7sq].
  apply sqrt_lower; [lra | nra].
Qed.

(*  KabsI = sqrt 2 * KaggI >= 1.4 * 4 = 5.6 >= 5.                             *)
Lemma w_KabsI_lower : 5 <= AbstractInterpolation.KabsI wc1 wc2 wCI.
Proof.
  unfold AbstractInterpolation.KabsI.
  pose proof w_KaggI_lower as HK.
  assert (Hs2 : 14/10 <= sqrt 2) by (apply sqrt_lower; lra).
  nra.
Qed.

(*  KUV_I = 2+1+1+1/2 + 2/sqrt 7 + 1 + 2/7 + 1 + 2*sqrt(1/2) + 1 >= 7,       *)
(*  dropping the three nonnegative square-root terms.                        *)
Lemma w_KUV_I_lower : 7 <= AbstractInterpolation.KUV_I wc1 wc2 wC2 wCinv wCb.
Proof.
  unfold AbstractInterpolation.KUV_I.
  assert (H1 : 0 < sqrt wc1) by (apply sqrt_lt_R0; unfold wc1; lra).
  assert (H2 : 0 <= 2 * wCb / sqrt wc1)
    by (apply div_nonneg; [unfold wCb; lra | lra]).
  assert (H3 : 0 <= sqrt wC2) by apply sqrt_pos.
  unfold wc1, wc2, wC2, wCinv, wCb in *. lra.
Qed.

(*  C_totI = KUV_I*KabsI + KPU_I + KPP_I >= 7*5 + 0 + 0 = 35.                *)
Lemma w_CtotI_lower : 35 <= wCtotI.
Proof.
  unfold wCtotI, AbstractInterpolation.CtotI.
  pose proof w_KUV_I_lower. pose proof w_KabsI_lower.
  assert (HP : 0 <= AbstractInterpolation.KPU_I wc1 wc2 wCb wcJ wcJ' wCface wNf wCI).
  { apply AbstractInterpolation.KPU_I_nonneg;
      unfold wc1, wc2, wCb, wcJ, wcJ', wCface, wNf, wCI; lra. }
  assert (HQ : 0 <= AbstractInterpolation.KPP_I wc1 wcJ wcJ' wCface wNf wCI).
  { apply AbstractInterpolation.KPP_I_nonneg;
      unfold wc1, wcJ, wcJ', wCface, wNf, wCI; lra. }
  nra.
Qed.

Lemma w_CtotI_pos : 0 < wCtotI.
Proof. pose proof w_CtotI_lower. lra. Qed.

(*  RHS >= 35 * ((3/2 + 0) * 5/2) = 35 * 15/4 = 131.25 >= 131.               *)
Lemma w_RHS_lower : 131 <= wCtotI * ((wPsU + wPsP) * wNV).
Proof.
  pose proof w_CtotI_lower as HC.
  pose proof w_PsU_lower as HU.
  pose proof w_PsP_nonneg as HP.
  pose proof w_NV_lower as HN.
  assert (H1 : 15/4 <= (wPsU + wPsP) * wNV) by nra.
  nra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  20.  The headline: the hypotheses are satisfiable, and the conclusion     *)
(*       they yield is a NON-DEGENERATE inequality between positive numbers.  *)
(* ------------------------------------------------------------------------- *)

Theorem abstract_continterp_is_not_vacuous :
  (*  the conclusion of abstract_continterp, at this instance ...  *)
  Rabs wBS <= wCtotI * ((wPsU + wPsP) * wNV)
  (*  ... with every quantity in it strictly positive ...  *)
  /\ 0 < Rabs wBS /\ 0 < wCtotI /\ 0 < wPsU /\ 0 < wPsP /\ 0 < wNV
  (*  ... the left side an explicit positive rational ...  *)
  /\ Rabs wBS = 72181/8400
  (*  ... the right side bounded below by an explicit positive rational,
      so the inequality reads  8.5929... <= (something >= 131) : true,
      and not a vacuous  0 <= 0.  *)
  /\ 131 <= wCtotI * ((wPsU + wPsP) * wNV)
  (*  ... and the three squared moduli explicit positive rationals.  *)
  /\ AbstractInterpolation.PsU2 bool wTh wnu wc1 wc2 whK waK wam wIU = 17/7
  /\ AbstractInterpolation.PsP2 bool wTh wnu wsig wc1 wc2 whK waK wam wIP = 19/90
  /\ AbstractInterpolation.NV2 RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
       wgv wcxv wgpv wvv wqq wdivv = 941/105.
Proof.
  pose proof witness_continterp. pose proof w_BS_pos. pose proof w_CtotI_pos.
  pose proof w_PsU_pos. pose proof w_PsP_pos. pose proof w_NV_pos.
  repeat split; try assumption.
  - exact w_LHS_val.
  - exact w_RHS_lower.
  - exact w_PsU2_val.
  - exact w_PsP2_val.
  - exact w_NV2_val.
Qed.

(* ========================================================================= *)
(*  21.  ADVERSARIAL SELF-CHECK.                                             *)
(*                                                                           *)
(*  A witness that trivializes the hypotheses it claims to witness certifies *)
(*  nothing.  This section machine-checks that the face machinery is LIVE:   *)
(*  every quantity below would be identically 0 on a one-element mesh, and   *)
(*  every one of them is a NON-ZERO rational here.  It also checks that two  *)
(*  of the scalar constants are at values this instance FORCES, rather than  *)
(*  values chosen for convenience.                                           *)
(* ========================================================================= *)

(*  The jump of tau1 across the interior face is NON-ZERO.  Everything in    *)
(*  Step 6b/6d is proportional to it.                                        *)
Lemma w_Dt1_val :
  forall f, AbstractInterpolation.Dt1 bool unit we1 we2 wnu wsig wc1 wc2 whK waK wam f
            = 1/90.
Proof.
  intro f. unfold AbstractInterpolation.Dt1. rewrite !w_t1. unfold we1, we2. lra.
Qed.

(*  Both sides of H_assemble_p are the SAME NON-ZERO number 1/45: FBp is     *)
(*  PINNED by the assembly identity, not free.                              *)
Lemma w_assemble_p_live :
  Rsum wTh (fun k => AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractInterpolation.FBp_e RPH bool wgpu wvv wpp wdivv k) = 1/45
  /\ Rsum wFl (fun f => (AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
                         - AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
                        * wFBp f) = 1/45.
Proof.
  split; [unfold wTh | unfold wFl]; simpl; rewrite !w_t1;
    unfold we1, we2, AbstractInterpolation.FBp_e,
           wgpu, wvv, wpp, wdivv, wFBp; simpl; field.
Qed.

Lemma w_assemble_c_live :
  Rsum wTh (fun k => AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractInterpolation.FBc_e RPH bool wcxu wcxv wuu wvv k) = 1/45
  /\ Rsum wFl (fun f => (AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
                         - AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
                        * wFBc f) = 1/45.
Proof.
  split; [unfold wTh | unfold wFl]; simpl; rewrite !w_t1;
    unfold we1, we2, AbstractInterpolation.FBc_e,
           wcxu, wcxv, wuu, wvv, wFBc; simpl; field.
Qed.

(*  The Step-6b jump term and the Step-6d inter-element term are NON-ZERO.   *)
Lemma w_JmpP_val :
  AbstractInterpolation.JmpP bool unit wFl we1 we2 wnu wsig wc1 wc2 whK waK wam wFBp
  = 1/45.
Proof.
  unfold AbstractInterpolation.JmpP, wFl. simpl Rsum.
  rewrite !w_Dt1_val. unfold wFBp, wsig. field.
Qed.

Lemma w_IIIterm_val :
  AbstractInterpolation.IIIterm RPH bool wTh wnu wsig wc1 wc2 whK waK wam
    wcxu wcxv wuu wvv = -(1/45).
Proof.
  unfold AbstractInterpolation.IIIterm, wTh. simpl Rsum.
  rewrite !w_t1.
  unfold AbstractInterpolation.FBc_e, wcxu, wcxv, wuu, wvv, wsig. simpl. field.
Qed.

(*  kJ = CJ/cJ = max(cJ'-1, 1-cJ)/cJ = max(1/8, 0)/1 = 1/8.                  *)
Lemma w_kJ_val : AbstractInterpolation.kJ wcJ wcJ' = 1/8.
Proof.
  unfold AbstractInterpolation.kJ, CJ, wcJ, wcJ', Rmax.
  destruct (Rle_dec (9/8 - 1) (1 - 1)); lra.
Qed.

(*  The jump lemma F_base is exercised TIGHTLY: sigma*|[tau1]| = 1/90 against *)
(*  the bound kJ*(sg*tau1) = (1/8)*(8/9)*(1/9) = 1/81 -- both sides strictly *)
(*  positive, with only ~10% of slack.                                       *)
Lemma w_F_base_live :
  wsig * Rabs (AbstractInterpolation.Dt1 bool unit we1 we2 wnu wsig wc1 wc2 whK waK wam tt)
  = 1/90
  /\ AbstractInterpolation.kJ wcJ wcJ'
     * (AbstractInterpolation.sg bool wnu wsig wc1 wc2 whK waK wam (we1 tt)
        * AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 tt))
     = 1/81
  /\ 1/90 < 1/81.
Proof.
  repeat split.
  - rewrite w_Dt1_val, (Rabs_right (1/90)) by lra. unfold wsig. lra.
  - rewrite w_kJ_val, w_sg, w_t1. unfold we1. lra.
  - lra.
Qed.

(*  Every face hypothesis instance is a real inequality between non-zero     *)
(*  numbers -- H_face_p at EQUALITY (Cface = 1/2 is sharp for it), H_face_c  *)
(*  with an RHS that is positive ONLY because am > 0.                        *)
Lemma w_face_live :
  Rabs (wFBp tt) = 2 /\ Rabs (wFBc tt) = 2
  /\ wCface * (1 + 1 + 1 + 1) = 2          (* H_face_p RHS: 2 <= 2, SHARP *)
  /\ wCface * (1 + 2 + 1 + 2) = 3.         (* H_face_c RHS: 2 <= 3        *)
Proof.
  unfold wFBp, wFBc, wCface. repeat split;
    try (rewrite (Rabs_right 2) by lra); lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  Two constants are at values this instance FORCES, not values chosen for  *)
(*  convenience.  Both are proved by INSTANTIATING the witness's own         *)
(*  hypotheses -- so they are statements about what the bundle demands.      *)
(* ------------------------------------------------------------------------- *)

(*  H_mult1, at g = the indicator of K1, reads  1 <= Nf * 1.  So Nf >= 1 is  *)
(*  FORCED by this mesh, and Nf = 1 is the sharp admissible value.           *)
Lemma w_Nf_forced : 1 <= wNf.
Proof.
  assert (Hg : forall k : bool, 0 <= (if k then 1 else 0)) by (intro k; destruct k; lra).
  pose proof (w_mult1 (fun k : bool => if k then 1 else 0) Hg) as H.
  unfold wFl, wTh, we1 in H. simpl in H. lra.
Qed.

(*  The upper half of H_jump reads  9 <= cJ' * 8.  So cJ' >= 9/8 is FORCED,  *)
(*  STRICTLY beyond the hypothesis one_le_cJ' -- i.e. H_jump carries real    *)
(*  information at this instance, which it would not on a one-element mesh.  *)
Lemma w_cJ'_forced : 9/8 <= wcJ'.
Proof.
  destruct (w_jump tt) as [_ Hup].
  rewrite !w_ph in Hup. unfold we1, we2 in Hup. simpl in Hup. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  22.  The axiom check, run at build time rather than left to the reader.  *)
(* ------------------------------------------------------------------------- *)

Print Assumptions abstract_continterp_is_not_vacuous.

(*  The theorems above are proved from the SAME abstract_continterp that the
    paper's lem:continterp (Appendix B) is identified with.  Hence:

      (i)   the hypothesis bundle of abstract_continterp -- all THIRTY-NINE
            named hypotheses -- is CONSISTENT (else no instance could exist);

      (ii)  in particular  CI_pos : 0 < CI,  the one hypothesis the recent
            refactor made strictly stronger, is satisfiable JOINTLY with the
            other thirty-eight.  Before this file that was unwitnessed, since
            CI_pos occurs only in AbstractInterpolation.v and
            AbstractConvergence.v and neither had a witness;

      (iii) the conclusion is not degenerate: there is an instance in which
            |B_S(E,V)| = 72181/8400 > 0 and the bounding side is >= 131, with
            every factor strictly positive; and

      (iv)  the content-bearing hypotheses are not trivially met: the mesh has
            two elements and a genuine interior face (e1 =/= e2), |a|_inf > 0
            on both elements, IU = IP = 1 > 0, all fourteen vector atom
            families are non-zero on both elements, FBp and FBc are PINNED by
            assembly rather than chosen, and eighteen of the twenty pointwise
            estimate instances hold with EQUALITY.

    What is NOT shown is set out in the header banner -- soundness, sharpness,
    abstract_convergence, and the two hypotheses (the lower half of H_jump,
    and Hw_cxv / HI_cxu at K2) that are met with slack rather than equality.

    The Print Assumptions above confirms this rests on nothing beyond the
    standard library's real-number axioms -- the same three that
    NonVacuity.abstract_stability_is_not_vacuous rests on.                    *)
