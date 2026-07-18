(* ========================================================================= *)
(*  NonVacuityConv.v                                                         *)
(*                                                                           *)
(*  A machine-checked WITNESS that the hypothesis bundle of                  *)
(*  AbstractConvergence.abstract_convergence -- thm:convergence, the         *)
(*  deepest of the four abstract theorems, with forty-four named             *)
(*  hypotheses -- is CONSISTENT, i.e. that the theorem is not vacuously      *)
(*  true.                                                                    *)
(*                                                                           *)
(*  This is the convergence sibling of NonVacuity.v (which witnesses         *)
(*  abstract_stability) and NonVacuityInterp.v (abstract_continterp).  It    *)
(*  closes the last gap in the non-vacuity story: abstract_convergence is    *)
(*  the only theorem whose trusted base contains Horth, the eighteen-term    *)
(*  Galerkin-orthogonality identity, and it is one of the only two that      *)
(*  carry CI_pos -- the one hypothesis the B_S refactor made STRICTLY        *)
(*  STRONGER (0 <= CI became 0 < CI, which is what lets HI_uu / HI_pp        *)
(*  transfer the sign of the pre-Hilbert norm onto IU / IP).                 *)
(*                                                                           *)
(*  ------------------------------------------------------------------------ *)
(*  THE INSTANCE                                                             *)
(*                                                                           *)
(*    * the carrier is R with < x , y > := x * y   (a PreHilbert space),     *)
(*      so nrm x = |x| ;                                                     *)
(*                                                                           *)
(*    * the mesh has TWO elements, K := bool, Th := [A; B] (A := true,       *)
(*      B := false).  One element is NOT enough: the assembly hypotheses     *)
(*      H_assemble_p/c read                                                  *)
(*                                                                           *)
(*        Rsum Th (t1 k * FB*_e k) = Rsum Fl ((t1(e1 f) - t1(e2 f)) * FB* f) *)
(*                                                                           *)
(*      and on a mesh with t1 CONSTANT the right-hand side is identically    *)
(*      0 whatever FB* is, so the face data would be left wholly             *)
(*      unconstrained.  A one-element mesh forces e1 f = e2 f and is         *)
(*      exactly that degenerate case.  Here alpha_K = (1, 1/4) -- an         *)
(*      inhomogeneous porosity, the physically apt way to break uniformity   *)
(*      for a paper on inhomogeneous porous media -- gives t1 = (1/9, 1/3),  *)
(*      so t1(e1 f) - t1(e2 f) = -2/9 at fT and +2/9 at fF, nonzero at both  *)
(*      faces, and both assembly identities have both sides NONZERO          *)
(*      (16/9 and -4/9).                                                     *)
(*                                                                           *)
(*    * the interior faces are F := bool, Fl := [fT; fF], with               *)
(*      e1 := id and e2 := negb, i.e. TWO genuine faces shared by the two    *)
(*      distinct elements with opposite orientation -- the 1-D periodic      *)
(*      two-element mesh.  F is inhabited and Fl is non-empty, so no face    *)
(*      hypothesis is vacuous.                                               *)
(*                                                                           *)
(*    * nu = sigma = 1,  h_K = 1,  alpha_K = (1, 1/4),  |a| = 2,             *)
(*      c1 = 4, c2 = 2  (the paper's c1 = 4k^4, c2 = 2k^2 at k = 1),         *)
(*      xi = 3, Cbar = 1 (c1 = 4 > xi*Cbar^2 = 3), C2 = 1/2,                 *)
(*      eps = 1/24, C_inv = 2, c_J = 1/4, c_J' = 4, C_face = 1/3,            *)
(*      N_f = 1, C_I = 4.                                                    *)
(*                                                                           *)
(*      The parameter formulas then give                                     *)
(*        phi1 = (8, 2),  t1 = (1/9, 1/3),  t2 = (2, 8),  sg = (8/9, 2/3).   *)
(*                                                                           *)
(*    * |a| = 2 > 0 STRICTLY.  This is the point of the exercise for the     *)
(*      three hypotheses whose right-hand side carries |a| as a factor of    *)
(*      the WHOLE bracket -- H_face_c, HI_cxu and Hw_cxv.  (The brief names  *)
(*      the first two; Hw_cxv is a third casualty of |a| = 0, and it is one  *)
(*      the brief does not name.)  At |a| = 0 all three read "<= 0", which   *)
(*      forces FBc = 0 identically and makes the norms of cxu and cxv        *)
(*      vanish -- so H_skew_diag would then read 0 = 0 with no content       *)
(*      whatever.  Here all three are live, and Hw_cxv holds with EQUALITY   *)
(*      at B.  (Note that nrm x = 0 does NOT force x = 0 in a PreHilbert:    *)
(*      there is no definiteness axiom.  On THIS carrier it does, since      *)
(*      ip x x = x*x, but the collapse of the three hypotheses at |a| = 0    *)
(*      is a statement about their right-hand sides, not about definiteness.)*)
(*                                                                           *)
(*  ------------------------------------------------------------------------ *)
(*  WHAT IT CERTIFIES                                                        *)
(*                                                                           *)
(*  All forty-four hypotheses hold -- witness_convergence discharges the     *)
(*  real theorem, so the kernel has type-checked all forty-four against      *)
(*  their actual statements -- at data in which every one of the fourteen    *)
(*  atom families is NONZERO at BOTH elements (w_atoms_nonzero), and         *)
(*                                                                           *)
(*    * Horth  --  BSWW = -BSEW  --  holds with BOTH sides NONZERO:          *)
(*         B_S(W,W) = 3931/32   and   B_S(E,W) = -3931/32,                   *)
(*      and every one of the EIGHTEEN terms of B_S is individually nonzero   *)
(*      on both sides -- checked, not asserted, in w_BSWW_terms /            *)
(*      w_BSEW_terms, over the reflexivity-certified decompositions          *)
(*      w_BSWW_split / w_BSEW_split.  This is the crux: Horth is one linear  *)
(*      equation on the total error Z := E + W (B_S is bilinear in its       *)
(*      first slot, so Horth <=> B_S(Z,W) = 0), and the trap is to satisfy   *)
(*      it with E := -W, which makes Horth read 0 = -0 AND zeroes Z, hence   *)
(*      the whole conclusion.  Here Z <> 0.                                  *)
(*                                                                           *)
(*    * the conclusion is the NON-DEGENERATE inequality                      *)
(*         NErr = |||U - U_h|||  <=  Cconv * Psi                             *)
(*      with  NErr^2 = 91 > 0,  PsU^2 = 89/8 > 0,  PsP^2 = 7/36 > 0, hence   *)
(*      Psi > 0 -- so the bound is NOT the                                   *)
(*      vacuous 0 <= something -- and 0 < Cconv follows FOR FREE, since      *)
(*      Cconv * Psi >= NErr > 0 with Psi > 0.                                *)
(*                                                                           *)
(*  SIXTEEN of the hypothesis instances hold with EQUALITY, so the           *)
(*  constants are PINNED rather than merely admissible -- a witness whose     *)
(*  every estimate had a factor of 50 to spare would be one whose constants   *)
(*  do no work.  This too is checked rather than asserted, in                 *)
(*  w_constants_are_sharp, w_jump_is_sharp, w_face_p_is_sharp and             *)
(*  w_mult_is_sharp (each is the hypothesis with `<=' replaced by `='):       *)
(*    C_inv = 2   by Hw_cxv at B (and Hw_gv at both A and B);                *)
(*    Cbar  = 1   by Hw_dv at both A and B;                                  *)
(*    C_I   = 4   by HI_du at A, HI_pp at A, HI_gpu at B and HI_divu at B;   *)
(*    C_face= 1/3  by H_face_p at BOTH faces;                                *)
(*    N_f   = 1   -- H_mult1/H_mult2 hold with EQUALITY for EVERY nonneg g,  *)
(*                  the true maximal face multiplicity of this mesh;         *)
(*    c_J   = 1/4 tight at face fT,  c_J' = 4 tight at face fF -- BOTH       *)
(*                  interior to their admissible ranges (0 < c_J <= 1 and    *)
(*                  1 <= c_J'), and H_jump is an equality on one side at     *)
(*                  each face;                                               *)
(*    eps   = 1/24 = the maximum admitted by H_eps, tight at B.              *)
(*  The remaining instances are slack, but NONE of them reads 0 <= 0: every  *)
(*  left-hand side here is strictly positive, so every one is a bound        *)
(*  between positive numbers.  For the record the slackest are HI_pp at B    *)
(*  (1 <= 40 -- IP(B) = 10 is pinned by HI_gpu at B, and pp(B) is much        *)
(*  smaller than gpu(B)), the two non-pinning halves of H_jump (2 <= 32 and   *)
(*  1/2 <= 8, each pinned at the OTHER face), and H_eps at A (1/24 <= 2/9,    *)
(*  since eps is a single constant and B is the binding element).  The        *)
(*  tightest of the slack ones is HI_cxu at B (4 <= 49/12, 1.02x).            *)
(*                                                                           *)
(*  ------------------------------------------------------------------------ *)
(*  WHAT IT DOES NOT CERTIFY -- read this before quoting the file            *)
(*                                                                           *)
(*  (1) It shows the hypothesis set is CONSISTENT (no theorem here is        *)
(*      vacuously true).  It does NOT show that the finite element objects   *)
(*      satisfy the hypotheses -- that is the soundness direction, and it    *)
(*      is what the hand audit establishes.                                  *)
(*                                                                           *)
(*  (2) H_skew_diag -- Rsum Th (vv . cxv) = 0 -- CANNOT be exercised with    *)
(*      both sides nonzero, and no arrangement of the data would change      *)
(*      that: its right-hand side is the literal 0, because the diagonal of  *)
(*      a skew-symmetric form vanishes as a matter of mathematics.  The best *)
(*      attainable standard is NONZERO per-element terms that CANCEL, and    *)
(*      that is what is achieved here:  vv(A)*cxv(A) = +1,                   *)
(*      vv(B)*cxv(B) = -1,  sum 0.  This is strictly better than the         *)
(*      existing NonVacuity.v witness, where the sum is 0 only because cxv   *)
(*      is identically zero, but it is not a two-sided exercise and is not   *)
(*      claimed to be one.                                                   *)
(*                                                                           *)
(*  (3) HOW Horth is satisfied, stated plainly so that nobody has to reverse *)
(*      engineer it.  Everything in the instance is chosen freely EXCEPT one  *)
(*      number: du(A) = 3/16.  B_S is bilinear in its first slot, so          *)
(*      B_S(Z,W) is affine in the du-slot of Z; du(A) is the unique value     *)
(*      that makes it vanish, and it was solved for.  du is the right knob    *)
(*      because it occurs in exactly one hypothesis (HI_du) and in NO term    *)
(*      of the conclusion -- perErr pairs gu+gv, uu+vv, pp+qq, xe+xw and      *)
(*      divu+divv, never du+dv -- so using it to close Horth cannot distort   *)
(*      what the theorem is being asked to bound.                             *)
(*                                                                            *)
(*      The price, stated so it cannot be discovered as a surprise: du(A) is  *)
(*      small, so HI_du at A is the SLACKEST estimate of the witness          *)
(*      (3/16 <= 10, a factor of 53).  Its two sides are still strictly       *)
(*      positive, so it is not vacuous, but it is not sharp either.           *)
(*                                                                            *)
(*      What this buys is that NOTHING about Horth here is structural.  No    *)
(*      group of the eighteen terms was zeroed to make it come out:           *)
(*        Rsum Th (gv . Zg)  = -4 <> 0 ,   Rsum Th (qq . Zp)   =  4 <> 0 ,    *)
(*        Rsum Th (gpv . Zu) = -3 <> 0 ,   Rsum Th (qq . Zdiv) =  3 <> 0 ,    *)
(*        Zx = (-1,14) <> -Zu = (2,1) ,    Zdiv = (2,1) <> -eps*Zp .          *)
(*      In consequence -- and this is the fact worth checking, in             *)
(*      w_BSWW_terms and w_BSEW_terms -- NOT ONE of the eighteen terms        *)
(*      cancels pairwise between B_S(W,W) and B_S(E,W), and not one of the    *)
(*      thirty-six is zero.  The orthogonality is a genuine eighteen-term     *)
(*      cancellation between 3931/32 and -3931/32.                            *)
(*                                                                            *)
(*  (4) Cconv is not computed.  It is built from sqrt 2, sqrt c1, sqrt C2    *)
(*      and KaggI (itself a square root of a five-term sum), so its exact    *)
(*      value is an unilluminating surd; positivity is what the              *)
(*      non-vacuity claim needs, and that comes for free from the theorem    *)
(*      itself as noted above.                                               *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import StabilityAlgebra ContinuityAlgebra InnerSpace
                              AbstractSums AbstractStability
                              AbstractInterpolation AbstractConvergence.
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

(*  On this space the pre-Hilbert norm is the absolute value.                *)
Lemma w_nrm : forall x : carrier RPH, nrm x = Rabs x.
Proof.
  intro x. unfold nrm.
  replace (ip RPH x x) with (Rsqr x) by (unfold Rsqr; reflexivity).
  apply sqrt_Rsqr_abs.
Qed.

(*  Discharge a goal all of whose absolute values are at concrete numerals.  *)
Ltac rabs :=
  repeat match goal with
  | |- context [Rabs ?x] =>
      first [ rewrite (Rabs_right x) by lra | rewrite (Rabs_left x) by lra ]
  end.

(* ------------------------------------------------------------------------- *)
(*  2.  The mesh: two elements, two interior faces.                           *)
(* ------------------------------------------------------------------------- *)

(*  K := bool.  Element A := true, element B := false.                       *)
Definition wTh : list bool := [true; false].

(*  F := bool.  The 1-D periodic two-element mesh: A and B share TWO faces,  *)
(*  and the two faces label the neighbours in opposite order.  This is what  *)
(*  lets BOTH c_J and c_J' be pinned at interior values (see w_jump), and it *)
(*  makes N_f = 1 the exact face multiplicity (see w_mult1 / w_mult2).       *)
Definition wFl : list bool := [true; false].
Definition we1 (f : bool) : bool := f.
Definition we2 (f : bool) : bool := negb f.

(*  Sums over a two-element list, reduced without touching anything else.    *)
Lemma Rsum_bool : forall f : bool -> R, Rsum wTh f = f true + f false.
Proof. intro f. unfold wTh. simpl. ring. Qed.

Lemma Rsum_boolF : forall f : bool -> R, Rsum wFl f = f true + f false.
Proof. intro f. unfold wFl. simpl. ring. Qed.

(* ------------------------------------------------------------------------- *)
(*  3.  The witness data.                                                     *)
(* ------------------------------------------------------------------------- *)

Definition wnu    : R := 1.
Definition wsig   : R := 1.
Definition weps   : R := 1/24.
Definition wc1    : R := 4.        (*  paper: c1 = 4 k^4, k = 1  *)
Definition wc2    : R := 2.        (*  paper: c2 = 2 k^2, k = 1  *)
Definition wC2    : R := 1/2.
Definition wCinv  : R := 2.
Definition wCb    : R := 1.
Definition wcJ    : R := 1/4.
Definition wcJ'   : R := 4.
Definition wCface : R := 1/3.
Definition wNf    : R := 1.
Definition wCI    : R := 4.
Definition wxi    : R := 3.

Definition whK (_ : bool) : R := 1.
Definition waK (k : bool) : R := if k then 1 else 1/4.   (*  inhomogeneous  *)
Definition wam (_ : bool) : R := 2.                      (*  |a| > 0 : gap G1  *)

(*  The fourteen atom families.  E = (gu, du, cxu, gpu, uu, pp, divu) is the *)
(*  interpolation error U - Uhat_h;  W = (gv, dv, cxv, gpv, vv, qq, divv) is *)
(*  the discrete error Uhat_h - U_h.  Every one is nonzero at both elements. *)
Definition wgu   (_ : bool) : carrier RPH := -3.
Definition wgv   (k : bool) : carrier RPH := if k then 2 else 1.
(*  du(A) = 3/16 is the ONE atom whose value is not chosen but SOLVED FOR:
    it is what makes Horth come out exactly.  See section 16.               *)
Definition wdu   (k : bool) : carrier RPH := if k then 3/16 else -1.
Definition wdv   (k : bool) : carrier RPH := if k then 4 else 1.
Definition wcxu  (k : bool) : carrier RPH := if k then -4 else 5.
Definition wcxv  (k : bool) : carrier RPH := if k then 1 else -1.
Definition wgpu  (k : bool) : carrier RPH := if k then 5 else 1.
Definition wgpv  (k : bool) : carrier RPH := if k then -3 else 9.
Definition wuu   (k : bool) : carrier RPH := if k then -3 else -2.
Definition wvv   (_ : bool) : carrier RPH := 1.
Definition wpp   (_ : bool) : carrier RPH := 1.
Definition wqq   (_ : bool) : carrier RPH := 1.
Definition wdivu (k : bool) : carrier RPH := if k then 5 else 4.
Definition wdivv (_ : bool) : carrier RPH := -3.

(*  The elemental interpolation-error sizes.  Strictly positive, so Psi > 0. *)
Definition wIU (k : bool) : R := if k then 5/4 else 4.
Definition wIP (k : bool) : R := if k then 5/4 else 1.

(*  The face data.  NOT free: pinned by the two assembly identities, which   *)
(*  here determine FB*(fT) - FB*(fF); the antisymmetric split is the         *)
(*  orientation-consistent one.  Both values are nonzero at both faces.      *)
Definition wFBp (f : bool) : R := if f then 1 else -1.
Definition wFBc (f : bool) : R := if f then -7/2 else 7/2.

(* ------------------------------------------------------------------------- *)
(*  4.  The stabilization parameters, computed.                              *)
(* ------------------------------------------------------------------------- *)

Lemma w_sqrt_quarter : sqrt (1/4) = 1/2.
Proof.
  replace (1/4) with (Rsqr (1/2)) by (unfold Rsqr; field).
  apply sqrt_Rsqr; lra.
Qed.

(*  tauNSinv = c1 nu / h^2 + c2 |a| / h = 4 + 4 = 8, constant;               *)
(*  phi1 = alpha_K * 8 = (8, 2).                                             *)
Lemma w_ph_t : AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam true = 8.
Proof.
  unfold AbstractConvergence.ph, phi1, tauNSinv, wnu, wc1, wc2, whK, waK, wam.
  field.
Qed.
Lemma w_ph_f : AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam false = 2.
Proof.
  unfold AbstractConvergence.ph, phi1, tauNSinv, wnu, wc1, wc2, whK, waK, wam.
  field.
Qed.

(*  t1 = 1/(phi1 + sigma) = (1/9, 1/3).  Non-constant: this is what makes    *)
(*  the assembly hypotheses bite.                                            *)
Lemma w_t1_t : AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam true = 1/9.
Proof.
  unfold AbstractConvergence.t1, tau1, phi1, tauNSinv,
         wnu, wsig, wc1, wc2, whK, waK, wam. field.
Qed.
Lemma w_t1_f : AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam false = 1/3.
Proof.
  unfold AbstractConvergence.t1, tau1, phi1, tauNSinv,
         wnu, wsig, wc1, wc2, whK, waK, wam. field.
Qed.

(*  The interpolation file spells the same parameters; the two spellings are *)
(*  definitionally equal, so the same values serve there.                    *)
Lemma w_t1I_t : AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam true = 1/9.
Proof. exact w_t1_t. Qed.
Lemma w_t1I_f : AbstractInterpolation.t1 bool wnu wsig wc1 wc2 whK waK wam false = 1/3.
Proof. exact w_t1_f. Qed.

Lemma w_t2_t : AbstractConvergence.t2 bool wnu wc1 wc2 whK waK wam true = 2.
Proof.
  unfold AbstractConvergence.t2, tau2, tauNSinv, wnu, wc1, wc2, whK, waK, wam.
  field.
Qed.
Lemma w_t2_f : AbstractConvergence.t2 bool wnu wc1 wc2 whK waK wam false = 8.
Proof.
  unfold AbstractConvergence.t2, tau2, tauNSinv, wnu, wc1, wc2, whK, waK, wam.
  field.
Qed.
Lemma w_t2I_t : AbstractInterpolation.t2 bool wnu wc1 wc2 whK waK wam true = 2.
Proof. exact w_t2_t. Qed.
Lemma w_t2I_f : AbstractInterpolation.t2 bool wnu wc1 wc2 whK waK wam false = 8.
Proof. exact w_t2_f. Qed.

Lemma w_sg_t : AbstractConvergence.sg bool wnu wsig wc1 wc2 whK waK wam true = 8/9.
Proof.
  unfold AbstractConvergence.sg, sigt, phi1, tauNSinv,
         wnu, wsig, wc1, wc2, whK, waK, wam. field.
Qed.
Lemma w_sg_f : AbstractConvergence.sg bool wnu wsig wc1 wc2 whK waK wam false = 2/3.
Proof.
  unfold AbstractConvergence.sg, sigt, phi1, tauNSinv,
         wnu, wsig, wc1, wc2, whK, waK, wam. field.
Qed.

(* ------------------------------------------------------------------------- *)
(*  5.  Hypotheses 1-18:  the scalar positivity block and the mesh.           *)
(* ------------------------------------------------------------------------- *)

Lemma w_nu_pos       : 0 < wnu.     Proof. unfold wnu; lra.   Qed.
Lemma w_sigma_nonneg : 0 <= wsig.   Proof. unfold wsig; lra.  Qed.
Lemma w_eps_nonneg   : 0 <= weps.   Proof. unfold weps; lra.  Qed.
Lemma w_c1_pos       : 0 < wc1.     Proof. unfold wc1; lra.   Qed.
Lemma w_c2_pos       : 0 < wc2.     Proof. unfold wc2; lra.   Qed.
Lemma w_C2_nonneg    : 0 <= wC2.    Proof. unfold wC2; lra.   Qed.
Lemma w_C2_lt_1      : wC2 < 1.     Proof. unfold wC2; lra.   Qed.
Lemma w_Cinv_pos     : 0 < wCinv.   Proof. unfold wCinv; lra. Qed.
Lemma w_Cb_pos       : 0 < wCb.     Proof. unfold wCb; lra.   Qed.
Lemma w_cJ_pos       : 0 < wcJ.     Proof. unfold wcJ; lra.   Qed.
Lemma w_cJ_le_1      : wcJ <= 1.    Proof. unfold wcJ; lra.   Qed.
Lemma w_one_le_cJ'   : 1 <= wcJ'.   Proof. unfold wcJ'; lra.  Qed.
Lemma w_Cface_nonneg : 0 <= wCface. Proof. unfold wCface; lra. Qed.
Lemma w_Nf_nonneg    : 0 <= wNf.    Proof. unfold wNf; lra.   Qed.

(*  THE STRENGTHENED ONE.  CI_pos (0 < CI, not merely 0 <= CI) is what lets  *)
(*  HI_uu / HI_pp transfer the sign of the pre-Hilbert norm onto IU / IP,    *)
(*  and it lives ONLY in AbstractInterpolation and AbstractConvergence -- so *)
(*  before this file and its interpolation sibling, the strengthening was    *)
(*  entirely unwitnessed.                                                    *)
Lemma w_CI_pos : 0 < wCI. Proof. unfold wCI; lra. Qed.

Lemma w_hK_pos    : forall k, 0 < whK k.
Proof. intro k; unfold whK; lra. Qed.
Lemma w_aK_pos    : forall k, 0 < waK k.
Proof. intro k; destruct k; unfold waK; lra. Qed.

(*  GAP G1 CLOSED.  |a| = 2 > 0 strictly, not at the boundary am = 0 where   *)
(*  H_face_c, HI_cxu and Hw_cxv all collapse to "<= 0".                      *)
Lemma w_am_nonneg : forall k, 0 <= wam k.
Proof. intro k; unfold wam; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  6.  Hypotheses 19-21:  the three CROSS Green identities.                  *)
(*      All three carry NONZERO values (-3, 8, -6): none degenerates to 0=0.  *)
(* ------------------------------------------------------------------------- *)

(*  eq:skew.   1*1 + 1*(-4) = -3  =  -( 1*1 + (-2)*(-1) ) = -3.              *)
Lemma w_skew :
  Rsum wTh (fun k => ip RPH (wvv k) (wcxu k))
  = - Rsum wTh (fun k => ip RPH (wuu k) (wcxv k)).
Proof.
  rewrite !Rsum_bool. cbv beta.
  unfold wvv, wcxu, wuu, wcxv. simpl. lra.
Qed.

(*  eq:globalibp (v,p).   1*(-2) + 1*10 = 8  =  -( (-3)*2 + (-1)*2 ) = 8.    *)
Lemma w_ibp_vp :
  Rsum wTh (fun k => ip RPH (wvv k) (wgpu k))
  = - Rsum wTh (fun k => ip RPH (wpp k) (wdivv k)).
Proof.
  rewrite !Rsum_bool. cbv beta.
  unfold wvv, wgpu, wpp, wdivv. simpl. lra.
Qed.

(*  eq:globalibp (q,u).   1*5 + 1*4 = 9  =  -( (-3)*(-3) + (-2)*9 ) = 9.     *)
Lemma w_ibp_qu :
  Rsum wTh (fun k => ip RPH (wqq k) (wdivu k))
  = - Rsum wTh (fun k => ip RPH (wuu k) (wgpv k)).
Proof.
  rewrite !Rsum_bool. cbv beta.
  unfold wqq, wdivu, wuu, wgpv. simpl. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  7.  Hypotheses 22-23:  facewise assembly.                                 *)
(*                                                                            *)
(*  These are the hypotheses that a one-element (or any t1-uniform) mesh      *)
(*  silently trivializes: their right-hand side would be identically 0 and    *)
(*  FBp / FBc would be unconstrained.  Here t1(A) - t1(B) = -2/9 <> 0 and     *)
(*  BOTH sides of BOTH identities are nonzero:                                *)
(*      Rsum Th (t1 * FBp_e) =  16/9 = (-2/9)*(-4) + (2/9)*4 ;               *)
(*      Rsum Th (t1 * FBc_e) = -4/9 = (-2/9)*1    + (2/9)*(-1).              *)
(* ------------------------------------------------------------------------- *)

Lemma w_assemble_p :
  Rsum wTh (fun k => AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractConvergence.FBp_e RPH bool wgpu wvv wpp wdivv k)
  = Rsum wFl (fun f =>
      (AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
       - AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
      * wFBp f).
Proof.
  rewrite Rsum_bool, Rsum_boolF. cbv beta.
  unfold we1, we2. simpl negb.
  rewrite !w_t1_t, !w_t1_f.
  unfold AbstractConvergence.FBp_e, wgpu, wvv, wpp, wdivv, wFBp. simpl. lra.
Qed.

Lemma w_assemble_c :
  Rsum wTh (fun k => AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam k
                     * AbstractConvergence.FBc_e RPH bool wcxu wcxv wuu wvv k)
  = Rsum wFl (fun f =>
      (AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam (we1 f)
       - AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam (we2 f))
      * wFBc f).
Proof.
  rewrite Rsum_bool, Rsum_boolF. cbv beta.
  unfold we1, we2. simpl negb.
  rewrite !w_t1_t, !w_t1_f.
  unfold AbstractConvergence.FBc_e, wcxu, wcxv, wuu, wvv, wFBc. simpl. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  8.  Hypotheses 24-25:  the two face estimates.                            *)
(*                                                                            *)
(*  H_face_p is TIGHT at BOTH faces (1 = (1/3)*3), which is what pins         *)
(*  C_face = 1/3.  H_face_c is the one whose bracket carries |a| as a factor  *)
(*  of every term: at |a| = 0 it would read |FBc| <= 0 and force FBc = 0.     *)
(*  Here it reads 7/2 <= 35/8 -- both sides strictly positive.               *)
(* ------------------------------------------------------------------------- *)

Lemma w_face_p :
  forall f, Rabs (wFBp f)
  <= wCface * (  waK (we1 f) / whK (we1 f) * (nrm (wvv (we1 f)) * wIP (we1 f))
               + waK (we2 f) / whK (we2 f) * (nrm (wvv (we1 f)) * wIP (we2 f))
               + waK (we1 f) / whK (we1 f) * (nrm (wvv (we2 f)) * wIP (we1 f))
               + waK (we2 f) / whK (we2 f) * (nrm (wvv (we2 f)) * wIP (we2 f))).
Proof.
  intro f. rewrite !w_nrm.
  unfold we1, we2. destruct f; simpl negb;
    unfold wFBp, wCface, waK, whK, wvv, wIP; rabs; lra.
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
  intro f. rewrite !w_nrm.
  unfold we1, we2. destruct f; simpl negb;
    unfold wFBc, wCface, waK, wam, whK, wvv, wIU; rabs; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  9.  Hypothesis 26:  H:jump, with BOTH constants pinned.                   *)
(*                                                                            *)
(*  phi1 = (8, 2).  At face fT (e1 = A, e2 = B) the LEFT inequality is an     *)
(*  EQUALITY: c_J * 8 = 2 = phi1(B), so c_J = 1/4 is sharp.  At face fF       *)
(*  (e1 = B, e2 = A) the RIGHT one is: phi1(A) = 8 = c_J' * 2, so c_J' = 4    *)
(*  is sharp.  Both constants sit strictly inside their admissible ranges     *)
(*  (0 < 1/4 <= 1 and 1 <= 4); a mesh with only one face could not do this.   *)
(* ------------------------------------------------------------------------- *)

Lemma w_jump :
  forall f,
    wcJ * AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we1 f)
      <= AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we2 f)
    /\ AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we2 f)
      <= wcJ' * AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we1 f).
Proof.
  intro f. unfold we1, we2. destruct f; simpl negb;
    rewrite ?w_ph_t, ?w_ph_f; unfold wcJ, wcJ'; split; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  10.  Hypotheses 27-28:  bounded face multiplicity.                        *)
(*                                                                            *)
(*  N_f = 1 is the TRUE maximal multiplicity of this mesh: each element is    *)
(*  e1 of exactly one face and e2 of exactly one face.  Both hypotheses are   *)
(*  therefore proved for EVERY nonnegative g -- not checked at one g -- and   *)
(*  hold with EQUALITY.                                                       *)
(* ------------------------------------------------------------------------- *)

Lemma w_mult1 :
  forall g : bool -> R, (forall k, 0 <= g k) ->
    Rsum wFl (fun f => g (we1 f)) <= wNf * Rsum wTh g.
Proof.
  intros g Hg. rewrite Rsum_boolF, Rsum_bool. cbv beta.
  unfold we1, wNf. lra.
Qed.

Lemma w_mult2 :
  forall g : bool -> R, (forall k, 0 <= g k) ->
    Rsum wFl (fun f => g (we2 f)) <= wNf * Rsum wTh g.
Proof.
  intros g Hg. rewrite Rsum_boolF, Rsum_bool. cbv beta.
  unfold we2. simpl negb. unfold wNf. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  11.  Hypotheses 29-31:  the W-side weighted inverse estimates.            *)
(*                                                                            *)
(*  Hw_gv and Hw_dv are TIGHT at BOTH elements, and Hw_cxv -- the third of    *)
(*  the three |a|-carrying hypotheses -- is TIGHT at B.  Between them they    *)
(*  pin C_inv = 2 from below; Cbar = 1 is pinned by Hw_dv.                    *)
(* ------------------------------------------------------------------------- *)

(*  |gv| = (2, 1)  <=  C_inv * sqrt(alpha_K) * |vv| = (2*1*1, 2*(1/2)*1).     *)
Lemma w_Hw_gv :
  forall k, nrm (wgv k) <= wCinv / whK k * sqrt (waK k) * nrm (wvv k).
Proof.
  intro k. rewrite !w_nrm. destruct k; unfold wgv, wvv, wCinv, whK, waK.
  - rewrite sqrt_1. rabs. lra.
  - rewrite w_sqrt_quarter. rabs. lra.
Qed.

(*  (S3) -- literally the stability lemma's inverse-estimate input, at W.     *)
(*  |dv| = (4, 1)  <=  2*nu*Cbar*sqrt(alpha_K)*|gv| = (2*1*2, 2*(1/2)*1).     *)
Lemma w_Hw_dv :
  forall k, nrm (wdv k) <= 2 * wnu * wCb / whK k * sqrt (waK k) * nrm (wgv k).
Proof.
  intro k. rewrite !w_nrm. destruct k; unfold wdv, wgv, wnu, wCb, whK, waK.
  - rewrite sqrt_1. rabs. lra.
  - rewrite w_sqrt_quarter. rabs. lra.
Qed.

(*  GAP G1, third casualty.  The bound carries alpha_K * |a| as a factor of   *)
(*  the whole right-hand side: at |a| = 0 it reads |cxv| <= 0 and forces      *)
(*  cxv = 0, which in turn makes H_skew_diag read 0 = 0 with no content.      *)
(*  Here  |cxv(B)| = 1 = 2 * (1/4) * 2 * 1 : EQUALITY.                        *)
Lemma w_Hw_cxv :
  forall k, nrm (wcxv k) <= wCinv / whK k * waK k * wam k * nrm (wvv k).
Proof.
  intro k. rewrite !w_nrm.
  destruct k; unfold wcxv, wvv, wCinv, whK, waK, wam; rabs; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  12.  Hypotheses 32-38:  the E-side interpolation estimates.               *)
(*                                                                            *)
(*  C_I = 4 is pinned by four of them at EQUALITY: HI_du at A, HI_pp at A,    *)
(*  HI_gpu at B and HI_divu at B.  Every left-hand side is strictly positive, *)
(*  so none of these reads 0 <= 0 -- the escape that setting IU = IP := 0     *)
(*  (or every atom := 0) would produce.                                       *)
(* ------------------------------------------------------------------------- *)

Lemma w_HI_gu :
  forall k, nrm (wgu k) <= wCI * sqrt (waK k) / whK k * wIU k.
Proof.
  intro k. rewrite !w_nrm. destruct k; unfold wgu, wCI, whK, waK, wIU.
  - rewrite sqrt_1. rabs. lra.
  - rewrite w_sqrt_quarter. rabs. lra.
Qed.

(*  du(A) = 3/16 is the atom that SOLVES Horth, hence small, hence this is    *)
(*  the SLACKEST estimate of the witness (3/16 <= 10).  Both sides are still  *)
(*  strictly positive; see banner item (3).                                   *)
Lemma w_HI_du :
  forall k, nrm (wdu k) <= 2 * wnu * wCI * waK k / (whK k)^2 * wIU k.
Proof.
  intro k. rewrite !w_nrm.
  destruct k; unfold wdu, wnu, wCI, whK, waK, wIU; rabs; lra.
Qed.

(*  GAP G1, second casualty:  the bound carries alpha_K * |a|.                *)
Lemma w_HI_cxu :
  forall k, nrm (wcxu k) <= wCI * waK k * wam k / whK k * wIU k.
Proof.
  intro k. rewrite !w_nrm.
  destruct k; unfold wcxu, wCI, whK, waK, wam, wIU; rabs; lra.
Qed.

(*  TIGHT at BOTH:  |gpu| = (5,1) = (4*1/1 * 5/4, 4*(1/4)/1 * 1).             *)
Lemma w_HI_gpu :
  forall k, nrm (wgpu k) <= wCI * waK k / whK k * wIP k.
Proof.
  intro k. rewrite !w_nrm.
  destruct k; unfold wgpu, wCI, whK, waK, wIP; rabs; lra.
Qed.

(*  TIGHT at BOTH:  |divu| = (5,4) = (4*1/1 * 5/4, 4*(1/4)/1 * 4).            *)
Lemma w_HI_divu :
  forall k, nrm (wdivu k) <= wCI * waK k / whK k * wIU k.
Proof.
  intro k. rewrite !w_nrm.
  destruct k; unfold wdivu, wCI, whK, waK, wIU; rabs; lra.
Qed.

(*  HI_uu / HI_pp are the two that CI_pos turns into IU >= 0 / IP >= 0.       *)
Lemma w_HI_uu : forall k, nrm (wuu k) <= wCI * wIU k.
Proof.
  intro k. rewrite !w_nrm. destruct k; unfold wuu, wCI, wIU; rabs; lra.
Qed.

Lemma w_HI_pp : forall k, nrm (wpp k) <= wCI * wIP k.
Proof.
  intro k. rewrite !w_nrm. destruct k; unfold wpp, wCI, wIP; rabs; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  13.  Hypothesis 39:  eq:epscond.  eps = 1/24 is the LARGEST admissible    *)
(*       value -- the estimate is an EQUALITY at B -- so the extreme          *)
(*       admissible choice is exercised, as in NonVacuity.v.                  *)
(* ------------------------------------------------------------------------- *)

Lemma w_Heps :
  forall k, weps <= wC2 * wc1 * (waK k)^2
                    * AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam k
                    / (whK k)^2.
Proof.
  intro k. destruct k.
  - rewrite w_t1_t. unfold weps, wC2, wc1, waK, whK. lra.
  - rewrite w_t1_f. unfold weps, wC2, wc1, waK, whK. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  14.  Hypotheses 40-41:  the numerical-parameter conditions.               *)
(* ------------------------------------------------------------------------- *)

Lemma w_xi_large : wxi > 2.
Proof. unfold wxi; lra. Qed.

(*  eq:conditions_on_num_param:  c1 = 4 > 3 = xi * Cbar^2.                    *)
Lemma w_c1_large : wc1 > wxi * wCb ^ 2.
Proof. unfold wc1, wxi, wCb; lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  15.  Hypotheses 42-43:  the two DIAGONAL Green identities.                *)
(*                                                                            *)
(*  These are shared verbatim with AbstractStability's trusted base (the      *)
(*  stability lemma's tested identity is a THEOREM proved from them), so      *)
(*  witnessing them here re-witnesses that part of the stability bundle at    *)
(*  data far less degenerate than NonVacuity.v's.                             *)
(* ------------------------------------------------------------------------- *)

(*  eq:skew on the diagonal.                                                  *)
(*                                                                            *)
(*  HONEST CAVEAT (see the banner, item (2)).  This is the ONE hypothesis     *)
(*  that cannot be exercised with both sides nonzero: its right-hand side is  *)
(*  the literal 0, because the diagonal of a skew-symmetric form vanishes as  *)
(*  a matter of mathematics.  The best attainable standard is nonzero         *)
(*  per-element terms that cancel, and that is what happens here:             *)
(*      vv(A)*cxv(A) = 1*1 = +1,   vv(B)*cxv(B) = 1*(-1) = -1,   sum = 0.     *)
(*  Contrast NonVacuity.v, where the sum is 0 only because cxv = 0.           *)
Lemma w_skew_diag :
  Rsum wTh (fun k => ip RPH (wvv k) (wcxv k)) = 0.
Proof.
  rewrite !Rsum_bool. cbv beta. unfold wvv, wcxv. simpl. lra.
Qed.

(*  eq:globalibp on the diagonal.  BOTH sides nonzero:                        *)
(*      1*(-2) + 1*(-4) = -6  =  -( 1*2 + 2*2 ) = -6.                         *)
Lemma w_ibp_diag :
  Rsum wTh (fun k => ip RPH (wvv k) (wgpv k))
  = - Rsum wTh (fun k => ip RPH (wqq k) (wdivv k)).
Proof.
  rewrite !Rsum_bool. cbv beta. unfold wvv, wgpv, wqq, wdivv. simpl. lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  16.  Hypothesis 44:  Horth -- Galerkin orthogonality.  THE CRUX.          *)
(*                                                                            *)
(*  B_S is bilinear in its first slot (each of the eighteen terms pairs       *)
(*  exactly one u-slot atom with one v-slot atom), so                         *)
(*                                                                            *)
(*      Horth  <=>  B_S(W,W) + B_S(E,W) = 0  <=>  B_S(Z,W) = 0,               *)
(*                                                                            *)
(*  ONE linear equation on the total error Z := E + W.  The trap is to        *)
(*  satisfy it with E := -W: that makes Horth read 0 = -0 AND zeroes Z, so    *)
(*  NErr = 0 and the conclusion degenerates to 0 <= something.  Here Z is     *)
(*  emphatically nonzero -- Zu = (-2,-1), Zp = (2,2), Zx = (-1,14),           *)
(*  Zg = (-1,-2), Zdiv = (2,1) -- and B_S(W,W) = 3931/32 <> 0.                *)
(* ------------------------------------------------------------------------- *)

Definition wBSWW : R :=
  AbstractConvergence.BSWW RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgv wdv wcxv wgpv wvv wqq wdivv.

Definition wBSEW : R :=
  AbstractConvergence.BSEW RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgu wgv wdu wdv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv.

(*  Reduce either eighteen-term form to closed arithmetic: unfold B_S into    *)
(*  eq:T1--eq:T18, collapse the two-element sums, replace the stabilization   *)
(*  parameters by their computed values, and evaluate the inner products.     *)
Ltac wBSred :=
  unfold AbstractInterpolation.BS,
         AbstractInterpolation.T1,  AbstractInterpolation.T2,
         AbstractInterpolation.T3,  AbstractInterpolation.T4,
         AbstractInterpolation.T5,  AbstractInterpolation.T6,
         AbstractInterpolation.T7,  AbstractInterpolation.T8,
         AbstractInterpolation.T9,  AbstractInterpolation.T10,
         AbstractInterpolation.T11, AbstractInterpolation.T12,
         AbstractInterpolation.T13, AbstractInterpolation.T14,
         AbstractInterpolation.T15, AbstractInterpolation.T16,
         AbstractInterpolation.T17, AbstractInterpolation.T18;
  rewrite !Rsum_bool; cbv beta;
  rewrite !w_t1I_t, !w_t1I_f, !w_t2I_t, !w_t2I_f;
  unfold AbstractInterpolation.xu, AbstractInterpolation.xv,
         wgu, wgv, wdu, wdv, wcxu, wcxv, wgpu, wgpv,
         wuu, wvv, wpp, wqq, wdivu, wdivv, wnu, wsig, weps;
  simpl.

(*  B_S(W,W), the eighteen-term form of eq:Bstab at the DISCRETE error.       *)
Lemma w_BSWW_val : wBSWW = 3931/32.
Proof. unfold wBSWW, AbstractConvergence.BSWW. wBSred. field. Qed.

(*  B_S(E,W), the same form at (interpolation error, discrete error).         *)
Lemma w_BSEW_val : wBSEW = - (3931/32).
Proof. unfold wBSEW, AbstractConvergence.BSEW. wBSred. field. Qed.

(*  ... and hence Horth, with BOTH sides nonzero.                             *)
Lemma w_Horth : wBSWW = - wBSEW.
Proof. rewrite w_BSWW_val, w_BSEW_val. lra. Qed.

(* ------------------------------------------------------------------------- *)
(*  17.  The instance:  abstract_convergence, applied.                        *)
(* ------------------------------------------------------------------------- *)

Definition wNErr : R :=
  AbstractConvergence.NErr RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgu wgv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv.

Definition wPsi : R :=
  AbstractConvergence.Psi bool wTh wnu wsig wc1 wc2 whK waK wam wIU wIP.

Definition wCconv : R :=
  AbstractConvergence.Cconv wc1 wc2 wC2 wCinv wCb wcJ wcJ' wCface wNf wCI wxi.

Theorem witness_convergence : wNErr <= wCconv * wPsi.
Proof.
  exact (AbstractConvergence.abstract_convergence
           RPH bool wTh bool wFl we1 we2
           wnu wsig weps wc1 wc2 wC2 wCinv wCb wcJ wcJ' wCface wNf wCI
           w_nu_pos w_sigma_nonneg w_eps_nonneg w_c1_pos w_c2_pos
           w_C2_nonneg w_C2_lt_1 w_Cinv_pos w_Cb_pos w_cJ_pos w_cJ_le_1
           w_one_le_cJ' w_Cface_nonneg w_Nf_nonneg w_CI_pos
           whK waK wam w_hK_pos w_aK_pos w_am_nonneg
           wgu wgv wdu wdv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv
           wIU wIP wFBp wFBc
           w_skew w_ibp_vp w_ibp_qu w_assemble_p w_assemble_c
           w_face_p w_face_c w_jump w_mult1 w_mult2
           w_Hw_gv w_Hw_dv w_Hw_cxv
           w_HI_gu w_HI_du w_HI_cxu w_HI_gpu w_HI_divu w_HI_uu w_HI_pp
           w_Heps
           wxi w_xi_large w_c1_large
           w_skew_diag w_ibp_diag w_Horth).
Qed.

(* ------------------------------------------------------------------------- *)
(*  18.  The two sides of the conclusion, computed.                           *)
(* ------------------------------------------------------------------------- *)

Definition wNErr2 : R :=
  AbstractConvergence.NErr2 RPH bool wTh wnu wsig weps wc1 wc2 whK waK wam
    wgu wgv wcxu wcxv wgpu wgpv wuu wvv wpp wqq wdivu wdivv.

Definition wPsU2 : R := AbstractInterpolation.PsU2 bool wTh wnu wc1 wc2 whK waK wam wIU.
Definition wPsP2 : R :=
  AbstractInterpolation.PsP2 bool wTh wnu wsig wc1 wc2 whK waK wam wIP.

(*  |||U - U_h|||^2 = Rsum Th perErr, with per-element                        *)
(*    perErr = nu|Zg|^2 + sg|Zu|^2 + eps|Zp|^2 + t1|Zx|^2 + t2|Zdiv|^2 ,      *)
(*  Z := E + W the TOTAL error.  Every one of the five contributions is       *)
(*  nonzero at BOTH elements -- there is no slot of the error that this       *)
(*  witness quietly leaves at zero:                                           *)
(*    at A:  1 + 32/9 + 1/6 + 1/9 + 8   = 77/6  ;                             *)
(*    at B:  4 + 2/3  + 1/6 + 196/3 + 8 = 469/6 ;   total 91 > 0.             *)
Lemma w_NErr2_val : wNErr2 = 91.
Proof.
  unfold wNErr2, AbstractConvergence.NErr2, AbstractConvergence.perErr.
  rewrite !Rsum_bool. cbv beta.
  rewrite !w_sg_t, !w_sg_f, !w_t1_t, !w_t1_f, !w_t2_t, !w_t2_f.
  unfold AbstractConvergence.xe, AbstractConvergence.xw,
         AbstractInterpolation.xu, AbstractInterpolation.xv,
         wgu, wgv, wcxu, wcxv, wgpu, wgpv, wuu, wvv, wpp, wqq, wdivu, wdivv,
         wnu, weps.
  simpl. field.
Qed.

Lemma w_PsU2_val : wPsU2 = 89/8.
Proof.
  unfold wPsU2, AbstractInterpolation.PsU2.
  rewrite !Rsum_bool. cbv beta.
  rewrite !w_t2I_t, !w_t2I_f.
  unfold waK, whK, wIU. simpl. field.
Qed.

Lemma w_PsP2_val : wPsP2 = 7/36.
Proof.
  unfold wPsP2, AbstractInterpolation.PsP2.
  rewrite !Rsum_bool. cbv beta.
  rewrite !w_t1I_t, !w_t1I_f.
  unfold waK, whK, wIP. simpl. field.
Qed.

(* ---- strict positivity of both sides ------------------------------------- *)

Lemma w_NErr_eq : wNErr = sqrt wNErr2.
Proof. reflexivity. Qed.

Lemma w_NErr_pos : 0 < wNErr.
Proof.
  rewrite w_NErr_eq, w_NErr2_val. apply sqrt_lt_R0. lra.
Qed.

Lemma w_Psi_pos : 0 < wPsi.
Proof.
  unfold wPsi, AbstractConvergence.Psi.
  assert (HU : AbstractInterpolation.PsU bool wTh wnu wc1 wc2 whK waK wam wIU
               = sqrt wPsU2) by reflexivity.
  assert (HP : AbstractInterpolation.PsP bool wTh wnu wsig wc1 wc2 whK waK wam wIP
               = sqrt wPsP2) by reflexivity.
  rewrite HU, HP.
  assert (H1 : 0 < sqrt wPsU2)
    by (rewrite w_PsU2_val; apply sqrt_lt_R0; lra).
  assert (H2 : 0 <= sqrt wPsP2) by apply sqrt_pos.
  lra.
Qed.

(*  0 < Cconv comes FOR FREE from the theorem itself: Cconv * Psi >= NErr > 0 *)
(*  and Psi > 0.  Cconv is a surd (sqrt 2 * sqrt(...) + .../C_stab), so this  *)
(*  is both the cheapest and the most informative statement about it.         *)
Lemma w_Cconv_pos : 0 < wCconv.
Proof.
  pose proof witness_convergence as H.
  pose proof w_NErr_pos as H1. pose proof w_Psi_pos as H2. nra.
Qed.


(* ========================================================================= *)
(*  19.  ANTI-TRIVIALIZATION: Horth, term by term.                           *)
(*                                                                           *)
(*  The banner claims that every one of the eighteen terms of B_S is         *)
(*  individually nonzero on both sides of Horth.  A claim of that kind is    *)
(*  worthless unless it is checked, so it is checked here.                   *)
(*                                                                           *)
(*  The two `split' lemmas are proved by REFLEXIVITY: they certify that the  *)
(*  eighteen partial applications below really are the eighteen terms of the *)
(*  B_S that Horth speaks about, with the atoms in the right slots.  Had any *)
(*  argument been misassigned, reflexivity would fail -- so the term values   *)
(*  that follow cannot be about some other expression.                       *)
(* ========================================================================= *)

Ltac wTred :=
  cbv beta delta [AbstractInterpolation.T1  AbstractInterpolation.T2
                  AbstractInterpolation.T3  AbstractInterpolation.T4
                  AbstractInterpolation.T5  AbstractInterpolation.T6
                  AbstractInterpolation.T7  AbstractInterpolation.T8
                  AbstractInterpolation.T9  AbstractInterpolation.T10
                  AbstractInterpolation.T11 AbstractInterpolation.T12
                  AbstractInterpolation.T13 AbstractInterpolation.T14
                  AbstractInterpolation.T15 AbstractInterpolation.T16
                  AbstractInterpolation.T17 AbstractInterpolation.T18];
  rewrite !Rsum_bool; cbv beta;
  rewrite ?w_t1I_t, ?w_t1I_f, ?w_t2I_t, ?w_t2I_f;
  cbv beta iota delta [AbstractInterpolation.xu AbstractInterpolation.xv
                       wgu wgv wdu wdv wcxu wcxv wgpu wgpv
                       wuu wvv wpp wqq wdivu wdivv wnu wsig weps];
  simpl; field.

(* ---- B_S(W,W): the eighteen terms of eq:T1--eq:T18 at the W family ------ *)

Lemma w_BSWW_split :
  wBSWW =
    AbstractInterpolation.T1  RPH bool wTh wnu wgv wgv
  + AbstractInterpolation.T2  RPH bool wTh wcxv wgpv wvv
  + AbstractInterpolation.T3  RPH bool wTh wsig wvv wvv
  + AbstractInterpolation.T4  RPH bool wTh wqq wdivv
  + AbstractInterpolation.T5  RPH bool wTh weps wqq wqq
  + AbstractInterpolation.T6  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wdv
  + AbstractInterpolation.T7  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxv wgpv
  + AbstractInterpolation.T8  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wvv
  + AbstractInterpolation.T9  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxv wgpv
  + AbstractInterpolation.T10 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wcxv wgpv wgpv
  + AbstractInterpolation.T11 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wvv
  + AbstractInterpolation.T12 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wvv
  + AbstractInterpolation.T13 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wvv
  + AbstractInterpolation.T14 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wvv wvv
  + AbstractInterpolation.T15 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wqq
  + AbstractInterpolation.T16 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivv
  + AbstractInterpolation.T17 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivv
  + AbstractInterpolation.T18 RPH bool wTh wnu wc1 wc2 whK waK wam wdivv wdivv.
Proof. reflexivity. Qed.

Lemma w_BSWW_terms :
     AbstractInterpolation.T1  RPH bool wTh wnu wgv wgv = 10
  /\ AbstractInterpolation.T2  RPH bool wTh wcxv wgpv wvv = 6
  /\ AbstractInterpolation.T3  RPH bool wTh wsig wvv wvv = 2
  /\ AbstractInterpolation.T4  RPH bool wTh wqq wdivv = -6
  /\ AbstractInterpolation.T5  RPH bool wTh weps wqq wqq = 1/12
  /\ AbstractInterpolation.T6  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wdv = -19/9
  /\ AbstractInterpolation.T7  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxv wgpv = -16/9
  /\ AbstractInterpolation.T8  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wvv = 7/9
  /\ AbstractInterpolation.T9  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxv wgpv = 16/9
  /\ AbstractInterpolation.T10 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wcxv wgpv wgpv = 196/9
  /\ AbstractInterpolation.T11 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wvv = -22/9
  /\ AbstractInterpolation.T12 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wvv = 7/9
  /\ AbstractInterpolation.T13 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wvv = 22/9
  /\ AbstractInterpolation.T14 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wvv wvv = -4/9
  /\ AbstractInterpolation.T15 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wqq = -5/288
  /\ AbstractInterpolation.T16 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivv = -5/4
  /\ AbstractInterpolation.T17 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivv = 5/4
  /\ AbstractInterpolation.T18 RPH bool wTh wnu wc1 wc2 whK waK wam wdivv wdivv = 90.
Proof. repeat split; wTred. Qed.

(* ---- B_S(E,W): the same eighteen terms at (E, W) ------------------------ *)

Lemma w_BSEW_split :
  wBSEW =
    AbstractInterpolation.T1  RPH bool wTh wnu wgu wgv
  + AbstractInterpolation.T2  RPH bool wTh wcxu wgpu wvv
  + AbstractInterpolation.T3  RPH bool wTh wsig wuu wvv
  + AbstractInterpolation.T4  RPH bool wTh wqq wdivu
  + AbstractInterpolation.T5  RPH bool wTh weps wpp wqq
  + AbstractInterpolation.T6  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wdv
  + AbstractInterpolation.T7  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wcxv wgpv
  + AbstractInterpolation.T8  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wvv
  + AbstractInterpolation.T9  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxu wgpu
  + AbstractInterpolation.T10 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxu wcxv wgpu wgpv
  + AbstractInterpolation.T11 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxu wgpu wvv
  + AbstractInterpolation.T12 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wuu
  + AbstractInterpolation.T13 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wuu
  + AbstractInterpolation.T14 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wuu wvv
  + AbstractInterpolation.T15 RPH bool wTh wnu weps wc1 wc2 whK waK wam wpp wqq
  + AbstractInterpolation.T16 RPH bool wTh wnu weps wc1 wc2 whK waK wam wpp wdivv
  + AbstractInterpolation.T17 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivu
  + AbstractInterpolation.T18 RPH bool wTh wnu wc1 wc2 whK waK wam wdivu wdivv.
Proof. reflexivity. Qed.

Lemma w_BSEW_terms :
     AbstractInterpolation.T1  RPH bool wTh wnu wgu wgv = -18
  /\ AbstractInterpolation.T2  RPH bool wTh wcxu wgpu wvv = 7
  /\ AbstractInterpolation.T3  RPH bool wTh wsig wuu wvv = -5
  /\ AbstractInterpolation.T4  RPH bool wTh wqq wdivu = 9
  /\ AbstractInterpolation.T5  RPH bool wTh weps wpp wqq = 1/12
  /\ AbstractInterpolation.T6  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wdv = 1/4
  /\ AbstractInterpolation.T7  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wcxv wgpv = 65/24
  /\ AbstractInterpolation.T8  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdu wvv = -5/16
  /\ AbstractInterpolation.T9  RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wcxu wgpu = 22/9
  /\ AbstractInterpolation.T10 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxu wcxv wgpu wgpv = 142/9
  /\ AbstractInterpolation.T11 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxu wgpu wvv = -19/9
  /\ AbstractInterpolation.T12 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wdv wuu = -2
  /\ AbstractInterpolation.T13 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wcxv wgpv wuu = -14/3
  /\ AbstractInterpolation.T14 RPH bool wTh wnu wsig wc1 wc2 whK waK wam wuu wvv = 1
  /\ AbstractInterpolation.T15 RPH bool wTh wnu weps wc1 wc2 whK waK wam wpp wqq = -5/288
  /\ AbstractInterpolation.T16 RPH bool wTh wnu weps wc1 wc2 whK waK wam wpp wdivv = -5/4
  /\ AbstractInterpolation.T17 RPH bool wTh wnu weps wc1 wc2 whK waK wam wqq wdivu = -7/4
  /\ AbstractInterpolation.T18 RPH bool wTh wnu wc1 wc2 whK waK wam wdivu wdivv = -126.
Proof. repeat split; wTred. Qed.

(* ========================================================================= *)
(*  20.  ANTI-TRIVIALIZATION: no atom is zero, no interpolation size is zero.*)
(*                                                                           *)
(*  The cheapest way to "satisfy" this bundle would be to set every atom to  *)
(*  0 (every bound then reads 0 <= 0 and every identity 0 = 0) or to set     *)
(*  IU = IP := 0 (every interpolation estimate then reads 0 <= 0).  Neither  *)
(*  happens here, and that is checked rather than asserted.                  *)
(* ========================================================================= *)

Lemma w_atoms_nonzero :
  forall k : bool,
       wgu k <> 0  /\ wgv k <> 0  /\ wdu k <> 0   /\ wdv k <> 0
    /\ wcxu k <> 0 /\ wcxv k <> 0 /\ wgpu k <> 0  /\ wgpv k <> 0
    /\ wuu k <> 0  /\ wvv k <> 0  /\ wpp k <> 0   /\ wqq k <> 0
    /\ wdivu k <> 0 /\ wdivv k <> 0
    (*  and the two interpolation-error sizes are STRICTLY positive, which is
        what makes Psi > 0 and every HI_* estimate a bound between positive
        numbers rather than the vacuous 0 <= 0.  *)
    /\ 0 < wIU k /\ 0 < wIP k.
Proof.
  intro k. destruct k;
    unfold wgu, wgv, wdu, wdv, wcxu, wcxv, wgpu, wgpv,
           wuu, wvv, wpp, wqq, wdivu, wdivv, wIU, wIP;
    repeat split; simpl; lra.
Qed.

(*  The face data is nonzero too: a witness in which FBp = FBc = 0 would let
    C_face := 0 and turn both face estimates into 0 <= 0.  *)
Lemma w_face_data_nonzero : forall f : bool, wFBp f <> 0 /\ wFBc f <> 0.
Proof. intro f. destruct f; unfold wFBp, wFBc; split; lra. Qed.

(* ========================================================================= *)
(*  21.  ANTI-TRIVIALIZATION: the constants are SHARP, not merely admissible. *)
(*                                                                           *)
(*  The other cheap way to "satisfy" a bundle of estimates is to take every  *)
(*  constant enormous, so that each bound holds with a factor of 50 to       *)
(*  spare and none of them constrains anything.  The banner claims instead   *)
(*  that SIXTEEN of the hypothesis instances hold with EQUALITY, pinning     *)
(*  C_inv, Cbar, C_I, C_face, N_f, c_J, c_J' and eps from below.  A claim of *)
(*  that kind is worthless unless checked, so it is checked here: each       *)
(*  conjunct below is the corresponding hypothesis with `<=' replaced by     *)
(*  `=' (and, for H_mult1/H_mult2, quantified over EVERY nonnegative g).     *)
(* ========================================================================= *)

Lemma w_constants_are_sharp :
  (*  C_inv = 2 is pinned: Hw_gv is an equality at BOTH elements, and Hw_cxv
      -- the |a|-carrying one -- is an equality at B.  *)
     nrm (wgv true)  = wCinv / whK true  * sqrt (waK true)  * nrm (wvv true)
  /\ nrm (wgv false) = wCinv / whK false * sqrt (waK false) * nrm (wvv false)
  /\ nrm (wcxv false)
       = wCinv / whK false * waK false * wam false * nrm (wvv false)
  (*  Cbar = 1 is pinned: (S3) is an equality at BOTH elements.  *)
  /\ nrm (wdv true)
       = 2 * wnu * wCb / whK true  * sqrt (waK true)  * nrm (wgv true)
  /\ nrm (wdv false)
       = 2 * wnu * wCb / whK false * sqrt (waK false) * nrm (wgv false)
  (*  C_I = 4 is pinned by four of the fourteen interpolation-estimate
      instances -- HI_gpu and HI_divu, at BOTH elements.  Equivalently:
      IP(A), IU(A), IP(B) and IU(B) are each pinned from below by one of
      them, so none of the four interpolation sizes is free to be inflated. *)
  /\ nrm (wgpu true)  = wCI * waK true  / whK true  * wIP true
  /\ nrm (wdivu true) = wCI * waK true  / whK true  * wIU true
  /\ nrm (wgpu false) = wCI * waK false / whK false * wIP false
  /\ nrm (wdivu false)= wCI * waK false / whK false * wIU false
  (*  eps = 1/24 is exactly the maximum eq:epscond admits (at B).  *)
  /\ weps = wC2 * wc1 * (waK false)^2
            * AbstractConvergence.t1 bool wnu wsig wc1 wc2 whK waK wam false
            / (whK false)^2.
Proof.
  repeat split; rewrite ?w_nrm, ?w_t1_f;
    unfold wgv, wvv, wcxv, wdv, wdu, wpp, wgpu, wdivu,
           wCinv, wCb, wCI, wCface, wnu, wsig, weps, wC2, wc1,
           whK, waK, wam, wIU, wIP;
    rewrite ?sqrt_1, ?w_sqrt_quarter; rabs; lra.
Qed.

(*  c_J = 1/4 and c_J' = 4 are BOTH pinned, each at its own face -- the       *)
(*  payoff of giving the mesh two faces instead of one.                      *)
Lemma w_jump_is_sharp :
     wcJ * AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we1 true)
       = AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we2 true)
  /\ AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we2 false)
       = wcJ' * AbstractConvergence.ph bool wnu wc1 wc2 whK waK wam (we1 false).
Proof.
  unfold we1, we2. simpl negb.
  rewrite ?w_ph_t, ?w_ph_f. unfold wcJ, wcJ'. split; lra.
Qed.

(*  C_face = 1/3 is pinned: H_face_p is an EQUALITY at BOTH faces.           *)
Lemma w_face_p_is_sharp :
  forall f, Rabs (wFBp f)
  = wCface * (  waK (we1 f) / whK (we1 f) * (nrm (wvv (we1 f)) * wIP (we1 f))
              + waK (we2 f) / whK (we2 f) * (nrm (wvv (we1 f)) * wIP (we2 f))
              + waK (we1 f) / whK (we1 f) * (nrm (wvv (we2 f)) * wIP (we1 f))
              + waK (we2 f) / whK (we2 f) * (nrm (wvv (we2 f)) * wIP (we2 f))).
Proof.
  intro f. rewrite !w_nrm.
  unfold we1, we2. destruct f; simpl negb;
    unfold wFBp, wCface, waK, whK, wvv, wIP; rabs; lra.
Qed.

(*  N_f = 1 is the exact face multiplicity: both multiplicity hypotheses are *)
(*  EQUALITIES, for EVERY nonnegative g -- not merely satisfiable bounds.    *)
Lemma w_mult_is_sharp :
  forall g : bool -> R, (forall k, 0 <= g k) ->
       Rsum wFl (fun f => g (we1 f)) = wNf * Rsum wTh g
    /\ Rsum wFl (fun f => g (we2 f)) = wNf * Rsum wTh g.
Proof.
  (*  wFl and wTh are convertible, so one collapse lemma reaches both sums.  *)
  intros g Hg. split; rewrite !Rsum_boolF; cbv beta;
    unfold we1, we2; simpl negb; unfold wNf; lra.
Qed.

(* ------------------------------------------------------------------------- *)
(*  22.  The headline.                                                       *)
(* ------------------------------------------------------------------------- *)

Theorem abstract_convergence_is_not_vacuous :
  (*  the conclusion of abstract_convergence, at this instance ...  *)
  wNErr <= wCconv * wPsi
  (*  ... with every quantity in it strictly positive, so it is NOT the
      vacuous  0 <= something  ...  *)
  /\ 0 < wNErr /\ 0 < wPsi /\ 0 < wCconv
  (*  ... the two sides being these explicit rationals, squared ...  *)
  /\ wNErr2 = 91 /\ wPsU2 = 89/8 /\ wPsP2 = 7/36
  (*  ... the Galerkin orthogonality that glues the proof together holding
      with BOTH sides nonzero, not as a vacuous 0 = -0 ...  *)
  /\ wBSWW = 3931/32 /\ wBSEW = - (3931/32) /\ wBSWW = - wBSEW
  (*  ... and none of it obtained by zeroing the data: every atom is
      nonzero at both elements and both interpolation sizes are positive.  *)
  /\ (forall k : bool, 0 < wIU k /\ 0 < wIP k)
  /\ (forall k : bool, wcxu k <> 0 /\ wcxv k <> 0)
  /\ (forall f : bool, wFBp f <> 0 /\ wFBc f <> 0)
  (*  |a| = 2 > 0 strictly -- gap G1: the three |a|-carrying hypotheses
      H_face_c, HI_cxu and Hw_cxv are live, not "<= 0".  *)
  /\ (forall k : bool, 0 < wam k).
Proof.
  pose proof witness_convergence.
  pose proof w_NErr_pos. pose proof w_Psi_pos. pose proof w_Cconv_pos.
  pose proof w_NErr2_val. pose proof w_PsU2_val. pose proof w_PsP2_val.
  pose proof w_BSWW_val. pose proof w_BSEW_val. pose proof w_Horth.
  (*  `split' intros first, so the four universally quantified conjuncts are
      already broken into their seven leaves: 0 < wIU k, 0 < wIP k,
      wcxu k <> 0, wcxv k <> 0, wFBp f <> 0, wFBc f <> 0, and the last
      quantifier, which split leaves alone.  *)
  repeat split; try assumption.
  - pose proof (w_atoms_nonzero k). tauto.
  - pose proof (w_atoms_nonzero k). tauto.
  - pose proof (w_atoms_nonzero k). tauto.
  - pose proof (w_atoms_nonzero k). tauto.
  - pose proof (w_face_data_nonzero f). tauto.
  - pose proof (w_face_data_nonzero f). tauto.
  - intro k. unfold wam. lra.
Qed.
