(* ========================================================================= *)
(*  AbstractInterpolation.v                                                  *)
(*                                                                           *)
(*  lem:continterp of the paper (Appendix B): continuity of B_S with the     *)
(*  FIRST argument equal to the interpolation error E = U - Uhat_h, proved   *)
(*  IN FULL by rerunning the machinery of AbstractContinuity.v with the      *)
(*  discrete inverse estimates on the first argument replaced -- exactly as  *)
(*  the appendix's proof prescribes -- by the interpolation estimates        *)
(*  eq:interpdivvisc--eq:interpinftyE.  The first-slot vector family         *)
(*  (gu, du, cxu, gpu, uu, pp, divu) now models the elemental restrictions   *)
(*  of E; two nonnegative elemental size functionals                         *)
(*      IU k ~ E_int,K(u) = h_K^(k_u+1) |u|_(H^(k_u+1)(K)),                  *)
(*      IP k ~ E_int,K(p) = h_K^(k_p+1) |p|_(H^(k_p+1)(K)),                  *)
(*  enter through the seven interpolation hypotheses HI_* (the appendix's    *)
(*  replacement table, constants folded into one CI) and through the two     *)
(*  face-integral hypotheses, whose first-argument L-infinity content is     *)
(*  now the L-infinity interpolation estimate eq:interpinftyE.               *)
(*                                                                           *)
(*  The second argument V_h stays discrete: its inverse estimates            *)
(*  (Hw_gv, Hw_dv, Hw_cxv), the jump condition, the mesh multiplicity, the   *)
(*  integration-by-parts identities and the facewise assembly are the SAME   *)
(*  hypotheses as in AbstractContinuity.v (the appendix's observation (1):   *)
(*  the exact identities remain valid for E in H^1_0 with elementwise H^2    *)
(*  regularity -- the F1-amended H:advection covers eq:elemibp).             *)
(*                                                                           *)
(*  Conclusions (fully explicit constants):                                  *)
(*    interp_triple_norm      :  |||E||| <= sqrt2 KaggI (PsiU + PsiP)        *)
(*                               -- eq:Enorm, in its sharp l2 form;          *)
(*    abstract_continterp_sharp : |B_S(E,V)| bounded by the three groups;    *)
(*    abstract_continterp     :  |B_S(E,V)| <= CtotI (PsiU + PsiP) |||V|||   *)
(*                               -- lem:continterp, l2 form of psi(h);       *)
(*    abstract_continterp_l1  :  the same with the appendix's l1 psi(h),     *)
(*                               via the discrete l2-in-l1 inequality.       *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

Section AbstractInterpolation.

(* ---------- Ambient space, elements, interior faces ------------------------ *)

Variable Hs : PreHilbert.
Notation V := (carrier Hs).
Notation "'<<' x , y '>>'" := (ip Hs x y) (at level 0).
Notation "x '+v' y" := (vadd Hs x y) (at level 50, left associativity).
Notation "a '*v' x" := (vscal Hs a x) (at level 40).

Variable K : Type.
Variable Th : list K.               (*  the triangulation                    *)
Variable F : Type.
Variable Fl : list F.               (*  the interior faces                   *)
Variables (e1 e2 : F -> K).         (*  the two elements sharing a face      *)

(* ---------- Constants ------------------------------------------------------- *)

Variables (nu sigma eps c1 c2 C2 Cinv Cb cJ cJ' Cface Nf CI : R).
Hypothesis nu_pos       : 0 < nu.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis eps_nonneg   : 0 <= eps.
Hypothesis c1_pos       : 0 < c1.
Hypothesis c2_pos       : 0 < c2.
Hypothesis C2_nonneg    : 0 <= C2.
Hypothesis C2_lt_1      : C2 < 1.
Hypothesis Cinv_pos     : 0 < Cinv.       (*  C_inv of eq:inverse            *)
Hypothesis Cb_pos       : 0 < Cb.         (*  Cbar of lem:winv               *)
Hypothesis cJ_pos       : 0 < cJ.
Hypothesis cJ_le_1      : cJ <= 1.
Hypothesis one_le_cJ'   : 1 <= cJ'.
Hypothesis Cface_nonneg : 0 <= Cface.     (*  the face-estimate constant     *)
Hypothesis Nf_nonneg    : 0 <= Nf.        (*  max faces per element (H:mesh) *)
Hypothesis CI_pos       : 0 < CI.         (*  interpolation constant (eq:interp) *)

(*  CI > 0 (rather than merely CI >= 0) is what lets the interpolation        *)
(*  estimates HI_uu / HI_pp transfer the sign of the pre-Hilbert norm onto    *)
(*  IU / IP -- see the IU_nonneg / IP_nonneg lemmas below.  The weaker        *)
(*  0 <= CI is all the rest of the file ever needs, so we re-derive it here   *)
(*  under its old name and every downstream consumer stays untouched.         *)
Lemma CI_nonneg : 0 <= CI.
Proof. pose proof CI_pos. lra. Qed.

(* ---------- Mesh data -------------------------------------------------------- *)

Variables (hK aK am : K -> R).
Hypothesis hK_pos    : forall k, 0 < hK k.
Hypothesis aK_pos    : forall k, 0 < aK k.
Hypothesis am_nonneg : forall k, 0 <= am k.

(*  The parameters, per element, via the paper's formulas (eq:taus,          *)
(*  eq:phi1, eq:sigmatilde), reusing the closed definitions of               *)
(*  ContinuityAlgebra.v verbatim.                                            *)
Definition ph (k : K) : R := phi1 nu (hK k) (aK k) (am k) c1 c2.
Definition t1 (k : K) : R := tau1 nu (hK k) (aK k) sigma (am k) c1 c2.
Definition t2 (k : K) : R := tau2 nu (hK k) (aK k) (am k) c1 c2.
Definition sg (k : K) : R := sigt nu (hK k) (aK k) sigma (am k) c1 c2.

(* ---------- Discrete fields, elementwise ------------------------------------ *)

(*  For the interpolation error E = [e_u; e_p] and the discrete V = [v_h; q_h]:                                    *)
(*    gu/gv     ~  (alpha^(1/2) P grad .)|_K                                  *)
(*    du/dv     ~  (2 nu div(alpha P grad .))|_K                              *)
(*    cxu/cxv   ~  (alpha a . grad .)|_K                                      *)
(*    gpu/gpv   ~  (alpha grad p)|_K, (alpha grad q)|_K                       *)
(*    uu/vv     ~  u_h|_K, v_h|_K ;  pp/qq ~ p_h|_K, q_h|_K                   *)
(*    divu/divv ~  (div(alpha .))|_K                                          *)
Variables (gu gv du dv cxu cxv gpu gpv uu vv pp qq divu divv : K -> V).

Definition xu (k : K) : V := (cxu k) +v (gpu k).   (*  (alpha X(E))|_K  *)
Definition xv (k : K) : V := (cxv k) +v (gpv k).   (*  (alpha X(V))|_K  *)

(*  The elemental interpolation error sizes (eq:Eint).  Their nonnegativity   *)
(*  is NOT assumed: it is derived from HI_uu / HI_pp below.                   *)
Variables (IU IP : K -> R).

(* ---------- Face data --------------------------------------------------------- *)

Variable FBp : F -> R.   (*  int_{face} alpha (n . v) p                        *)
Variable FBc : F -> R.   (*  int_{face} alpha (n . a) (u . v)                  *)
(*  The elemental boundary terms of the two elementwise integration-by-parts *)
(*  identities.  These are NOT free variables: by the divergence theorem on K *)
(*  (eq:elemibp and its pressure sibling) the boundary integral EQUALS the    *)
(*  integration-by-parts defect, so we DEFINE it as that defect.  The two     *)
(*  identities H_elem_conv_ibp / H_elem_p_ibp then become theorems rather     *)
(*  than hypotheses -- as they must, since with FB*_e free they constrained   *)
(*  nothing.  All the real content of the boundary terms lives in the         *)
(*  assembly hypotheses H_assemble_p/c and the face estimates H_face_p/c.     *)
Definition FBp_e (k : K) : R := << (vv k) , (gpu k) >> + << (divv k) , (pp k) >>.
Definition FBc_e (k : K) : R := << (cxu k) , (vv k) >> + << (cxv k) , (uu k) >>.

(* ========================================================================= *)
(*  The trusted base.                                                        *)
(* ========================================================================= *)

(*  eq:skew (identity (i) of Step 6a).  *)
Hypothesis H_skew :
  Rsum Th (fun k => << (vv k) , (cxu k) >>)
  = - Rsum Th (fun k => << (uu k) , (cxv k) >>).

(*  eq:globalibp (identity (ii)).  *)
Hypothesis H_ibp_vp :
  Rsum Th (fun k => << (vv k) , (gpu k) >>)
  = - Rsum Th (fun k => << (pp k) , (divv k) >>).
Hypothesis H_ibp_qu :
  Rsum Th (fun k => << (qq k) , (divu k) >>)
  = - Rsum Th (fun k => << (uu k) , (gpv k) >>).

(*  eq:elemibp (identity (iii)) -- now a THEOREM, by the definition of FBc_e. *)
Lemma H_elem_conv_ibp :
  forall k, << (cxu k) , (vv k) >> = - << (cxv k) , (uu k) >> + FBc_e k.
Proof. intro k. unfold FBc_e. ring. Qed.

(*  Elemental pressure integration by parts (Step 6b) -- likewise a theorem.  *)
Lemma H_elem_p_ibp :
  forall k, << (vv k) , (gpu k) >> = - << (divv k) , (pp k) >> + FBp_e k.
Proof. intro k. unfold FBp_e. ring. Qed.

(*  Facewise assembly (Steps 6b, 6d): the elemental boundary contributions,  *)
(*  weighted by the elementwise-constant tau_1, assemble into jump-weighted  *)
(*  single-valued face integrals; boundary faces vanish (u = v = 0 there).   *)
Hypothesis H_assemble_p :
  Rsum Th (fun k => t1 k * FBp_e k)
  = Rsum Fl (fun f => (t1 (e1 f) - t1 (e2 f)) * FBp f).
Hypothesis H_assemble_c :
  Rsum Th (fun k => t1 k * FBc_e k)
  = Rsum Fl (fun f => (t1 (e1 f) - t1 (e2 f)) * FBc f).

(*  Face-integral estimates (eq:jumppart, eq:IIIface without the jump        *)
(*  factor): Hoelder on the face, meas <= C h^(d-1), the L-infinity inverse  *)
(*  estimate eq:inverse on both neighbours, alpha <= alpha_K on the face,    *)
(*  and comparability of neighbouring diameters (H:mesh).                    *)
(*  The first-argument L-infinity content is the interpolation estimate      *)
(*  eq:interpinftyE; the V-side L-infinity inverse estimate is kept.          *)
Hypothesis H_face_p :
  forall f, Rabs (FBp f)
  <= Cface * (  aK (e1 f) / hK (e1 f) * (nrm (vv (e1 f)) * IP (e1 f))
              + aK (e2 f) / hK (e2 f) * (nrm (vv (e1 f)) * IP (e2 f))
              + aK (e1 f) / hK (e1 f) * (nrm (vv (e2 f)) * IP (e1 f))
              + aK (e2 f) / hK (e2 f) * (nrm (vv (e2 f)) * IP (e2 f))).
Hypothesis H_face_c :
  forall f, Rabs (FBc f)
  <= Cface * (  aK (e1 f) * am (e1 f) / hK (e1 f)
                  * (IU (e1 f) * nrm (vv (e1 f)))
              + aK (e2 f) * am (e2 f) / hK (e2 f)
                  * (IU (e1 f) * nrm (vv (e2 f)))
              + aK (e1 f) * am (e1 f) / hK (e1 f)
                  * (IU (e2 f) * nrm (vv (e1 f)))
              + aK (e2 f) * am (e2 f) / hK (e2 f)
                  * (IU (e2 f) * nrm (vv (e2 f)))).

(*  H:jump.  *)
Hypothesis H_jump :
  forall f, cJ * ph (e1 f) <= ph (e2 f) /\ ph (e2 f) <= cJ' * ph (e1 f).

(*  Bounded face multiplicity (H:mesh).  *)
Hypothesis H_mult1 :
  forall g : K -> R, (forall k, 0 <= g k) ->
    Rsum Fl (fun f => g (e1 f)) <= Nf * Rsum Th g.
Hypothesis H_mult2 :
  forall g : K -> R, (forall k, 0 <= g k) ->
    Rsum Fl (fun f => g (e2 f)) <= Nf * Rsum Th g.

(*  V-side: the weighted inverse estimates of lem:winv (discrete argument).  *)
Hypothesis Hw_gv :
  forall k, nrm (gv k) <= Cinv / hK k * sqrt (aK k) * nrm (vv k).
Hypothesis Hw_dv :
  forall k, nrm (dv k) <= 2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k).
Hypothesis Hw_cxv :
  forall k, nrm (cxv k) <= Cinv / hK k * aK k * am k * nrm (vv k).

(*  E-side: the interpolation estimates of the appendix's replacement table  *)
(*  (eq:interpdivvisc--eq:interpzero), one constant CI for all of them.      *)
Hypothesis HI_gu :
  forall k, nrm (gu k) <= CI * sqrt (aK k) / hK k * IU k.
Hypothesis HI_du :
  forall k, nrm (du k) <= 2 * nu * CI * aK k / (hK k)^2 * IU k.
Hypothesis HI_cxu :
  forall k, nrm (cxu k) <= CI * aK k * am k / hK k * IU k.
Hypothesis HI_gpu :
  forall k, nrm (gpu k) <= CI * aK k / hK k * IP k.
Hypothesis HI_divu :
  forall k, nrm (divu k) <= CI * aK k / hK k * IU k.
Hypothesis HI_uu :
  forall k, nrm (uu k) <= CI * IU k.
Hypothesis HI_pp :
  forall k, nrm (pp k) <= CI * IP k.

(* ------------------------------------------------------------------------- *)
(*  IU >= 0 and IP >= 0 are THEOREMS, not hypotheses.                         *)
(*                                                                           *)
(*  The pre-Hilbert axioms already give nrm x >= 0 for every x (nrm_nonneg,  *)
(*  InnerSpace.v), so HI_uu reads 0 <= nrm (uu k) <= CI * IU k.  With CI > 0 *)
(*  (CI_pos) the product CI * IU k >= 0 forces IU k >= 0; likewise for IP    *)
(*  via HI_pp.  Nothing else is needed, so the two former hypotheses drop    *)
(*  out of the trusted base -- at the price of strengthening CI_nonneg to    *)
(*  CI_pos, which is where the sign transfer happens.  The names are kept    *)
(*  verbatim so every downstream use site is unchanged.                      *)
(* ------------------------------------------------------------------------- *)

Lemma IU_nonneg : forall k, 0 <= IU k.
Proof.
  intro k. pose proof (nrm_nonneg Hs (uu k)) as Hn.
  pose proof (HI_uu k) as Hi. nra.
Qed.

Lemma IP_nonneg : forall k, 0 <= IP k.
Proof.
  intro k. pose proof (nrm_nonneg Hs (pp k)) as Hn.
  pose proof (HI_pp k) as Hi. nra.
Qed.

(*  eq:epscond, elementwise.  *)
Hypothesis H_eps :
  forall k, eps <= C2 * c1 * (aK k)^2 * t1 k / (hK k)^2.

(* ========================================================================= *)
(*  Step 0: the eighteen terms (eq:T1--eq:T18) and B_S.                      *)
(* ========================================================================= *)

Definition T1  : R := 2 * nu * Rsum Th (fun k => << (gv k) , (gu k) >>).
Definition T2  : R := Rsum Th (fun k => << (vv k) , (xu k) >>).
Definition T3  : R := sigma * Rsum Th (fun k => << (vv k) , (uu k) >>).
Definition T4  : R := Rsum Th (fun k => << (qq k) , (divu k) >>).
Definition T5  : R := eps * Rsum Th (fun k => << (qq k) , (pp k) >>).
Definition T6  : R := - Rsum Th (fun k => t1 k * << (dv k) , (du k) >>).
Definition T7  : R := - Rsum Th (fun k => t1 k * << (xv k) , (du k) >>).
Definition T8  : R := sigma * Rsum Th (fun k => t1 k * << (vv k) , (du k) >>).
Definition T9  : R := Rsum Th (fun k => t1 k * << (dv k) , (xu k) >>).
Definition T10 : R := Rsum Th (fun k => t1 k * << (xv k) , (xu k) >>).
Definition T11 : R := - sigma * Rsum Th (fun k => t1 k * << (xu k) , (vv k) >>).
Definition T12 : R := sigma * Rsum Th (fun k => t1 k * << (dv k) , (uu k) >>).
Definition T13 : R := sigma * Rsum Th (fun k => t1 k * << (uu k) , (xv k) >>).
Definition T14 : R := - sigma^2 * Rsum Th (fun k => t1 k * << (uu k) , (vv k) >>).
Definition T15 : R := - eps^2 * Rsum Th (fun k => t2 k * << (pp k) , (qq k) >>).
Definition T16 : R := eps * Rsum Th (fun k => t2 k * << (divv k) , (pp k) >>).
Definition T17 : R := - eps * Rsum Th (fun k => t2 k * << (divu k) , (qq k) >>).
Definition T18 : R := Rsum Th (fun k => t2 k * << (divv k) , (divu k) >>).

Definition BS : R :=
  T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8 + T9
  + T10 + T11 + T12 + T13 + T14 + T15 + T16 + T17 + T18.

(* ========================================================================= *)
(*  Working norms and bracket norms (eq:triplenorm and the two brackets).    *)
(* ========================================================================= *)

Definition perU (k : K) : R :=
  nu * << (gu k) , (gu k) >> + sg k * << (uu k) , (uu k) >>
  + eps * << (pp k) , (pp k) >> + t1 k * << (xu k) , (xu k) >>
  + t2 k * << (divu k) , (divu k) >>.

Definition perV (k : K) : R :=
  nu * << (gv k) , (gv k) >> + sg k * << (vv k) , (vv k) >>
  + eps * << (qq k) , (qq k) >> + t1 k * << (xv k) , (xv k) >>
  + t2 k * << (divv k) , (divv k) >>.

Definition NU2 : R := Rsum Th perU.   Definition NU : R := sqrt NU2.
Definition NV2 : R := Rsum Th perV.   Definition NV : R := sqrt NV2.

(*  The l2 form of the appendix's psi(h) (eq:psih / eq:Enorm).  *)
Definition PsU2 : R :=
  Rsum Th (fun k => (aK k)^2 * t2 k / (hK k)^2 * (IU k)^2).
Definition PsU : R := sqrt PsU2.
Definition PsP2 : R :=
  Rsum Th (fun k => (aK k)^2 * t1 k / (hK k)^2 * (IP k)^2).
Definition PsP : R := sqrt PsP2.

(* ========================================================================= *)
(*  Elementary bridges to the machine-checked parameter algebra.             *)
(* ========================================================================= *)

Lemma ph_pos : forall k, 0 < ph k.
Proof. intro k. unfold ph. apply phi1_pos; auto. Qed.

Lemma t1_pos' : forall k, 0 < t1 k.
Proof. intro k. unfold t1. apply tau1_pos; auto. Qed.

Lemma t2_pos' : forall k, 0 < t2 k.
Proof. intro k. unfold t2. apply tau2_pos; auto. Qed.

Lemma sg_nonneg' : forall k, 0 <= sg k.
Proof. intro k. unfold sg. apply sigt_nonneg; auto. Qed.

Lemma L_pht1 : forall k, ph k * t1 k <= 1.
Proof. intro k. unfold ph, t1. apply P1_phi1_tau1; auto. Qed.

Lemma L_st1 : forall k, sigma * t1 k <= 1.
Proof. intro k. unfold t1. apply P1_sigma_tau1; auto. Qed.

Lemma L_one_minus : forall k, 1 - sigma * t1 k = ph k * t1 k.
Proof. intro k. unfold ph, t1. apply one_minus_sigma_tau1; auto. Qed.

Lemma L_sg_sub : forall k, sg k = sigma - sigma^2 * t1 k.
Proof. intro k. unfold sg, t1. apply sigt_id_sub; auto. Qed.

Lemma L_P3 : forall k, sqrt (ph k) * hK k = sqrt c1 * aK k * sqrt (t2 k).
Proof. intro k. unfold ph, t2. apply P3_sqrt; auto. Qed.

Lemma L_P5 : forall k, sqrt eps * hK k <= sqrt c1 * aK k * sqrt (t1 k).
Proof.
  intro k. unfold t1.
  apply (P5_sqrt nu (hK k) (aK k) sigma (am k) c1 c2
           nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
           c1_pos c2_pos eps C2 eps_nonneg C2_lt_1 (H_eps k)).
Qed.

Lemma L_epst2 : forall k, eps * t2 k <= C2.
Proof.
  intro k. unfold t2.
  apply (P5_eps_tau2_C2 nu (hK k) (aK k) sigma (am k) c1 c2
           nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
           c1_pos c2_pos eps C2 C2_nonneg (H_eps k)).
Qed.

Lemma L_step7 : forall k, eps * sqrt (t2 k) <= sqrt C2 * sqrt eps.
Proof.
  intro k. unfold t2.
  apply (step7_scalar nu (hK k) (aK k) sigma (am k) c1 c2
           nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
           c1_pos c2_pos eps C2 eps_nonneg C2_nonneg (H_eps k)).
Qed.

Lemma L_keysq : forall k, nu * t1 k * aK k * c1 <= (hK k)^2.
Proof. intro k. unfold t1. apply keyvisc_sq; auto. Qed.

Lemma L_T8c : forall k,
  sigma * (c1 * nu * aK k / (hK k)^2) * t1 k <= sg k.
Proof. intro k. unfold t1, sg. apply T8_chain; auto. Qed.

Lemma L_T13c : forall k,
  sigma * t1 k * (c2 * aK k * am k / hK k) <= sg k.
Proof. intro k. unfold t1, sg. apply T13_chain; auto. Qed.

Lemma L_vol : forall k, ph k * t1 k <= sqrt (ph k) * sqrt (t1 k).
Proof. intro k. unfold ph, t1. apply volpart_scalar; auto. Qed.

Lemma L_spt : forall k, sqrt (ph k) * t1 k <= sqrt (t1 k).
Proof. intro k. unfold ph, t1. apply sqrtphi_tau_le; auto. Qed.

Lemma L_pst : forall k, ph k * sqrt (t1 k) <= sqrt (ph k).
Proof. intro k. unfold ph, t1. apply phi_sqrttau_le; auto. Qed.

Lemma L_abs1 : forall k, sqrt nu * sqrt (aK k) <= aK k * sqrt (t2 k).
Proof. intro k. unfold t2. apply absorb1_sqrt; auto. Qed.

Lemma L_abs2 : forall k, sqrt (sg k) <= sqrt (ph k).
Proof. intro k. unfold sg, ph. apply absorb2_sqrt; auto. Qed.

Lemma L_convph : forall k, c2 * aK k * am k / hK k <= ph k.
Proof. intro k. unfold ph. apply conv_le_phi1; auto. Qed.

Lemma L_sgt1 : forall k, sg k * t1 k <= 1.
Proof.
  intro k.
  exact (sgA_tA_le_1 (ph k) sigma (ph_pos k) sigma_nonneg).
Qed.

(*  sqrt(ph)/h in the bracket form: sqrt(ph k) = sqrt c1 * aK * sqrt t2 / h. *)
Lemma L_P3div : forall k, sqrt (ph k) = sqrt c1 * aK k * sqrt (t2 k) / hK k.
Proof.
  intro k. pose proof (hK_pos k) as Hh. pose proof (L_P3 k) as HP.
  apply (Rmult_eq_reg_r (hK k)); [| lra].
  rewrite HP. field. lra.
Qed.

Lemma L_P5div : forall k, sqrt eps <= sqrt c1 * aK k * sqrt (t1 k) / hK k.
Proof.
  intro k. pose proof (hK_pos k) as Hh. pose proof (L_P5 k) as HP.
  apply (Rmult_le_reg_r (hK k)); [lra |].
  replace (sqrt c1 * aK k * sqrt (t1 k) / hK k * hK k)
    with (sqrt c1 * aK k * sqrt (t1 k)) by (field; lra).
  exact HP.
Qed.

(*  sg t1 <= 1 in square-root form (Step 6d, eq:jumpsplit).  *)
Lemma L_sgt_sqrt : forall k, sg k * t1 k <= sqrt (sg k) * sqrt (t1 k).
Proof.
  intro k.
  pose proof (sg_nonneg' k) as Hsg0. pose proof (t1_pos' k) as Ht.
  pose proof (L_sgt1 k) as H1.
  assert (H0 : 0 <= sg k * t1 k) by nra.
  pose proof (x_le_sqrt (sg k * t1 k) (conj H0 H1)) as Hx.
  rewrite sqrt_mult_alt in Hx by lra.
  exact Hx.
Qed.

(*  c1 nu aK / h^2 <= phi1 : the viscous part of tau_NS^{-1}.  *)
Lemma L_viscph : forall k, c1 * nu * aK k / (hK k)^2 <= ph k.
Proof.
  intro k. unfold ph.
  apply visc_le_phi1; auto.
Qed.

(*  sg = sigma * ph * t1 : the sigma-absorption identity eq:sigmatilde.  *)
Lemma L_sgtau : forall k, sg k = sigma * ph k * t1 k.
Proof.
  intro k.
  pose proof (L_one_minus k) as H1.
  pose proof (L_sg_sub k) as H2.
  rewrite H2.
  replace (sigma * ph k * t1 k) with (sigma * (ph k * t1 k)) by ring.
  rewrite <- H1. ring.
Qed.

(*  sg <= ph, squared form of L_abs2.  *)
Lemma L_sgph : forall k, sg k <= ph k.
Proof.
  intro k.
  pose proof (L_abs2 k) as HA.
  pose proof (sg_nonneg' k) as Hsg.
  pose proof (ph_pos k) as Hph.
  assert (Hs2 : sqrt (sg k) * sqrt (sg k) = sg k) by (apply sqrt_sqrt; lra).
  assert (Hp2 : sqrt (ph k) * sqrt (ph k) = ph k) by (apply sqrt_sqrt; lra).
  assert (Hs0 : 0 <= sqrt (sg k)) by apply sqrt_pos.
  assert (Hp0 : 0 <= sqrt (ph k)) by apply sqrt_pos.
  nra.
Qed.

(*  ph * h^2 = c1 * aK^2 * t2 : the squared form of (P3).  *)
Lemma L_phid : forall k, ph k * (hK k)^2 = c1 * (aK k)^2 * t2 k.
Proof.
  intro k.
  pose proof (L_P3 k) as HP3.
  pose proof (ph_pos k) as Hph.
  pose proof (t2_pos' k) as Ht2.
  assert (Hp2 : sqrt (ph k) * sqrt (ph k) = ph k) by (apply sqrt_sqrt; lra).
  assert (Hc2 : sqrt c1 * sqrt c1 = c1) by (apply sqrt_sqrt; lra).
  assert (Ht2s : sqrt (t2 k) * sqrt (t2 k) = t2 k) by (apply sqrt_sqrt; lra).
  assert (Ea : (sqrt (ph k) * hK k) * (sqrt (ph k) * hK k)
               = (sqrt (ph k) * sqrt (ph k)) * (hK k * hK k)) by ring.
  rewrite Hp2 in Ea.
  assert (Eb : (sqrt c1 * aK k * sqrt (t2 k)) * (sqrt c1 * aK k * sqrt (t2 k))
               = (sqrt c1 * sqrt c1)
                 * ((aK k * aK k) * (sqrt (t2 k) * sqrt (t2 k)))) by ring.
  rewrite Hc2, Ht2s in Eb.
  assert (Ec : (sqrt (ph k) * hK k) * (sqrt (ph k) * hK k)
               = (sqrt c1 * aK k * sqrt (t2 k))
                 * (sqrt c1 * aK k * sqrt (t2 k))) by (rewrite HP3; reflexivity).
  nra.
Qed.


(* ========================================================================= *)
(*  Jump machinery (lem:jump), per face, with explicit constants.            *)
(* ========================================================================= *)

Definition Dt1 (f : F) : R := t1 (e1 f) - t1 (e2 f).
Definition kJ : R := CJ cJ cJ' / cJ.

Lemma kJ_nonneg : 0 <= kJ.
Proof.
  unfold kJ. apply div_nonneg; [apply CJ_nonneg; lra | lra].
Qed.

Lemma F_base : forall f,
  sigma * Rabs (Dt1 f) <= kJ * (sg (e1 f) * t1 (e1 f)).
Proof.
  intro f. unfold Dt1, kJ.
  exact (jump_bound_A (ph (e1 f)) (ph (e2 f)) sigma cJ cJ'
           (ph_pos (e1 f)) sigma_nonneg cJ_pos cJ_le_1 one_le_cJ'
           (proj1 (H_jump f)) (proj2 (H_jump f))).
Qed.

Lemma F_tau_AB : forall f, t1 (e1 f) <= cJ' * t1 (e2 f).
Proof.
  intro f.
  exact (tau_comp_AB (ph (e1 f)) (ph (e2 f)) sigma cJ cJ'
           (ph_pos (e1 f)) sigma_nonneg cJ_pos one_le_cJ'
           (proj1 (H_jump f)) (proj2 (H_jump f))).
Qed.

Lemma F_sg_AB : forall f, sg (e1 f) <= cJ' / cJ * sg (e2 f).
Proof.
  intro f.
  exact (sg_comp_A_le_B (ph (e1 f)) (ph (e2 f)) sigma cJ cJ'
           (ph_pos (e1 f)) sigma_nonneg cJ_pos one_le_cJ'
           (proj1 (H_jump f)) (proj2 (H_jump f))).
Qed.

(*  Square-root moves across a face.  *)
Lemma F_sqt_move : forall f, sqrt (t1 (e1 f)) <= sqrt cJ' * sqrt (t1 (e2 f)).
Proof.
  intro f.
  pose proof (F_tau_AB f) as HT.
  assert (H := sqrt_le_1_alt _ _ HT).
  rewrite sqrt_mult_alt in H by lra.
  exact H.
Qed.

Lemma F_sqsg_move : forall f,
  sqrt (sg (e1 f)) <= sqrt (cJ' / cJ) * sqrt (sg (e2 f)).
Proof.
  intro f.
  pose proof (F_sg_AB f) as HT.
  assert (H := sqrt_le_1_alt _ _ HT).
  rewrite sqrt_mult_alt in H
    by (apply div_nonneg; lra).
  exact H.
Qed.

(*  The four jump-coefficient bounds of Step 6b (eq:jumpsplit resolved       *)
(*  toward each pair of neighbours).                                         *)
Definition kb11 : R := kJ.
Definition kb12 : R := kJ * sqrt cJ'.
Definition kb21 : R := kJ * sqrt (cJ' / cJ).
Definition kb22 : R := kJ * sqrt cJ' * sqrt (cJ' / cJ).

Lemma Jb11 : forall f,
  sigma * Rabs (Dt1 f) <= kb11 * (sqrt (sg (e1 f)) * sqrt (t1 (e1 f))).
Proof.
  intro f. unfold kb11.
  pose proof (F_base f) as HB.
  pose proof (L_sgt_sqrt (e1 f)) as HS.
  pose proof kJ_nonneg as HkJ.
  nra.
Qed.

Lemma Jb12 : forall f,
  sigma * Rabs (Dt1 f) <= kb12 * (sqrt (sg (e1 f)) * sqrt (t1 (e2 f))).
Proof.
  intro f. unfold kb12.
  pose proof (Jb11 f) as HB. unfold kb11 in HB.
  pose proof (F_sqt_move f) as HM.
  pose proof kJ_nonneg as HkJ.
  assert (Hsg : 0 <= sqrt (sg (e1 f))) by apply sqrt_pos.
  assert (Hk : 0 <= kJ * sqrt (sg (e1 f))) by nra.
  nra.
Qed.

Lemma Jb21 : forall f,
  sigma * Rabs (Dt1 f) <= kb21 * (sqrt (sg (e2 f)) * sqrt (t1 (e1 f))).
Proof.
  intro f. unfold kb21.
  pose proof (Jb11 f) as HB. unfold kb11 in HB.
  pose proof (F_sqsg_move f) as HM.
  pose proof kJ_nonneg as HkJ.
  assert (Ht : 0 <= sqrt (t1 (e1 f))) by apply sqrt_pos.
  assert (Hk : 0 <= kJ * sqrt (t1 (e1 f))) by nra.
  nra.
Qed.

Lemma Jb22 : forall f,
  sigma * Rabs (Dt1 f) <= kb22 * (sqrt (sg (e2 f)) * sqrt (t1 (e2 f))).
Proof.
  intro f. unfold kb22.
  pose proof (Jb12 f) as HB. unfold kb12 in HB.
  pose proof (F_sqsg_move f) as HM.
  pose proof kJ_nonneg as HkJ.
  assert (Hcj : 0 <= sqrt cJ') by apply sqrt_pos.
  assert (Ht : 0 <= sqrt (t1 (e2 f))) by apply sqrt_pos.
  assert (Hk0 : 0 <= kJ * sqrt cJ') by nra.
  assert (Hk : 0 <= kJ * sqrt cJ' * sqrt (t1 (e2 f))) by nra.
  assert (S1 : kJ * sqrt cJ' * (sqrt (sg (e1 f)) * sqrt (t1 (e2 f)))
               <= kJ * sqrt cJ' * sqrt (cJ' / cJ)
                  * (sqrt (sg (e2 f)) * sqrt (t1 (e2 f)))) by nra.
  lra.
Qed.

(*  The four jump-coefficient bounds of Step 6d (eq:jumpsplit combined with  *)
(*  alpha |n.a| <= phi_1 h / c2 and phi_1 tau_1 <= 1).                       *)
Definition kd11 : R := kJ / c2.
Definition kd12 : R := kJ * cJ' / c2 * sqrt (cJ' / cJ).
Definition kd21 : R := kJ / c2 * sqrt (cJ' / cJ).
Definition kd22 : R := kJ * cJ' / c2 * (sqrt (cJ' / cJ) * sqrt (cJ' / cJ)).

(*  Core: sigma |[tau1]| ph(e1) <= kJ sqrt(sg1) sqrt(sg1).  *)
Lemma Jd_core : forall f,
  sigma * Rabs (Dt1 f) * ph (e1 f)
  <= kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))).
Proof.
  intro f.
  pose proof (F_base f) as HB.
  pose proof (L_pht1 (e1 f)) as HP.
  pose proof (ph_pos (e1 f)) as Hph.
  pose proof (sg_nonneg' (e1 f)) as Hsg.
  pose proof (t1_pos' (e1 f)) as Ht.
  pose proof kJ_nonneg as HkJ.
  pose proof (Rabs_pos (Dt1 f)) as HD.
  rewrite sqrt_sqrt by exact Hsg.
  (*  sigma|D| ph1 <= kJ sg1 t11 ph1 <= kJ sg1  *)
  assert (S1 : sigma * Rabs (Dt1 f) * ph (e1 f)
               <= kJ * (sg (e1 f) * t1 (e1 f)) * ph (e1 f)) by nra.
  assert (Hk1 : 0 <= kJ * sg (e1 f)) by nra.
  assert (S2 : kJ * (sg (e1 f) * t1 (e1 f)) * ph (e1 f)
               <= kJ * sg (e1 f)) by nra.
  lra.
Qed.

Lemma Jd_j1 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e1 f) * am (e1 f) / hK (e1 f))
  <= kd11 * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))).
Proof.
  intro f. unfold kd11.
  pose proof (Jd_core f) as HC.
  pose proof (L_convph (e1 f)) as HP.
  pose proof (ph_pos (e1 f)) as Hph.
  pose proof (hK_pos (e1 f)) as Hh.
  pose proof (aK_pos (e1 f)) as Ha.
  pose proof (am_nonneg (e1 f)) as Hm.
  pose proof (Rabs_pos (Dt1 f)) as HD.
  (*  aK am / h <= ph / c2 *)
  assert (Hcoef : aK (e1 f) * am (e1 f) / hK (e1 f) <= ph (e1 f) / c2).
  { apply (Rmult_le_reg_l c2); [lra |].
    replace (c2 * (aK (e1 f) * am (e1 f) / hK (e1 f)))
      with (c2 * aK (e1 f) * am (e1 f) / hK (e1 f)) by (field; lra).
    replace (c2 * (ph (e1 f) / c2)) with (ph (e1 f)) by (field; lra).
    exact HP. }
  assert (HsD : 0 <= sigma * Rabs (Dt1 f)) by nra.
  assert (S1 : sigma * Rabs (Dt1 f) * (aK (e1 f) * am (e1 f) / hK (e1 f))
               <= sigma * Rabs (Dt1 f) * (ph (e1 f) / c2)) by nra.
  assert (S2 : sigma * Rabs (Dt1 f) * (ph (e1 f) / c2)
               = (sigma * Rabs (Dt1 f) * ph (e1 f)) / c2)
    by (field; lra).
  assert (S3 : (sigma * Rabs (Dt1 f) * ph (e1 f)) / c2
               <= (kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f)))) / c2).
  { apply (Rmult_le_reg_r c2); [lra |].
    replace ((sigma * Rabs (Dt1 f) * ph (e1 f)) / c2 * c2)
      with (sigma * Rabs (Dt1 f) * ph (e1 f)) by (field; lra).
    replace ((kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f)))) / c2 * c2)
      with (kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f)))) by (field; lra).
    exact HC. }
  assert (S4 : (kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f)))) / c2
               = kJ / c2 * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))))
    by (field; lra).
  lra.
Qed.

Lemma Jd_j2 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e2 f) * am (e2 f) / hK (e2 f))
  <= kJ * cJ' / c2 * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))).
Proof.
  intro f.
  pose proof (Jd_core f) as HC.
  pose proof (L_convph (e2 f)) as HP.
  pose proof (ph_pos (e1 f)) as Hph1.
  pose proof (ph_pos (e2 f)) as Hph2.
  pose proof (hK_pos (e2 f)) as Hh.
  pose proof (aK_pos (e2 f)) as Ha.
  pose proof (am_nonneg (e2 f)) as Hm.
  pose proof (Rabs_pos (Dt1 f)) as HD.
  pose proof (proj2 (H_jump f)) as HJ2.
  pose proof kJ_nonneg as HkJ.
  assert (Hcoef : aK (e2 f) * am (e2 f) / hK (e2 f) <= cJ' * ph (e1 f) / c2).
  { apply (Rmult_le_reg_l c2); [lra |].
    replace (c2 * (aK (e2 f) * am (e2 f) / hK (e2 f)))
      with (c2 * aK (e2 f) * am (e2 f) / hK (e2 f)) by (field; lra).
    replace (c2 * (cJ' * ph (e1 f) / c2)) with (cJ' * ph (e1 f))
      by (field; lra).
    lra. }
  assert (HsD : 0 <= sigma * Rabs (Dt1 f)) by nra.
  assert (S1 : sigma * Rabs (Dt1 f) * (aK (e2 f) * am (e2 f) / hK (e2 f))
               <= sigma * Rabs (Dt1 f) * (cJ' * ph (e1 f) / c2)) by nra.
  assert (S2 : sigma * Rabs (Dt1 f) * (cJ' * ph (e1 f) / c2)
               = cJ' / c2 * (sigma * Rabs (Dt1 f) * ph (e1 f)))
    by (field; lra).
  assert (Hcc : 0 <= cJ' / c2) by (apply div_nonneg; lra).
  assert (S3 : cJ' / c2 * (sigma * Rabs (Dt1 f) * ph (e1 f))
               <= cJ' / c2 * (kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f)))))
    by nra.
  assert (S4 : cJ' / c2 * (kJ * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))))
               = kJ * cJ' / c2 * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))))
    by (field; lra).
  lra.
Qed.

Lemma JD11 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e1 f) * am (e1 f) / hK (e1 f))
  <= kd11 * (sqrt (sg (e1 f)) * sqrt (sg (e1 f))).
Proof. exact Jd_j1. Qed.

Lemma JD12 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e2 f) * am (e2 f) / hK (e2 f))
  <= kd12 * (sqrt (sg (e1 f)) * sqrt (sg (e2 f))).
Proof.
  intro f. unfold kd12.
  pose proof (Jd_j2 f) as HB.
  pose proof (F_sqsg_move f) as HM.
  pose proof kJ_nonneg as HkJ.
  assert (Hk : 0 <= kJ * cJ' / c2)
    by (apply div_nonneg; [nra | lra]).
  assert (Hs1 : 0 <= sqrt (sg (e1 f))) by apply sqrt_pos.
  assert (Hk2 : 0 <= kJ * cJ' / c2 * sqrt (sg (e1 f))) by nra.
  nra.
Qed.

Lemma JD21 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e1 f) * am (e1 f) / hK (e1 f))
  <= kd21 * (sqrt (sg (e2 f)) * sqrt (sg (e1 f))).
Proof.
  intro f. unfold kd21.
  pose proof (Jd_j1 f) as HB. unfold kd11 in HB.
  pose proof (F_sqsg_move f) as HM.
  assert (Hk : 0 <= kJ / c2) by (apply div_nonneg; [apply kJ_nonneg | lra]).
  assert (Hs1 : 0 <= sqrt (sg (e1 f))) by apply sqrt_pos.
  assert (Hk2 : 0 <= kJ / c2 * sqrt (sg (e1 f))) by nra.
  nra.
Qed.

Lemma JD22 : forall f,
  sigma * Rabs (Dt1 f) * (aK (e2 f) * am (e2 f) / hK (e2 f))
  <= kd22 * (sqrt (sg (e2 f)) * sqrt (sg (e2 f))).
Proof.
  intro f. unfold kd22.
  pose proof (Jd_j2 f) as HB.
  pose proof (F_sqsg_move f) as HM.
  pose proof kJ_nonneg as HkJ.
  assert (Hk : 0 <= kJ * cJ' / c2)
    by (apply div_nonneg; [nra | lra]).
  assert (Hs1 : 0 <= sqrt (sg (e1 f))) by apply sqrt_pos.
  assert (Hs2 : 0 <= sqrt (sg (e2 f))) by apply sqrt_pos.
  assert (Hm : 0 <= sqrt (cJ' / cJ)) by apply sqrt_pos.
  (*  apply the move to both factors of sqrt(sg1)*sqrt(sg1)  *)
  assert (S1 : sqrt (sg (e1 f)) * sqrt (sg (e1 f))
               <= (sqrt (cJ' / cJ) * sqrt (sg (e2 f)))
                  * (sqrt (cJ' / cJ) * sqrt (sg (e2 f)))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Generic estimation kit.                                                  *)
(* ========================================================================= *)

Lemma sq_weight : forall (w : R) (x : V),
  0 <= w -> (sqrt w * nrm x)^2 = w * << x , x >>.
Proof.
  intros w x Hw.
  replace ((sqrt w * nrm x)^2)
    with ((sqrt w * sqrt w) * (nrm x * nrm x)) by ring.
  rewrite sqrt_sqrt by exact Hw.
  rewrite (nrm_sq Hs). reflexivity.
Qed.

Lemma sq_weight2s : forall (a w h s : R),
  0 <= w -> 0 < h ->
  (a * sqrt w / h * s)^2 = a^2 * w / h^2 * s^2.
Proof.
  intros a w h s Hw Hh.
  replace ((a * sqrt w / h * s)^2)
    with (a^2 * (sqrt w * sqrt w) / h^2 * s^2) by (field; lra).
  rewrite sqrt_sqrt by exact Hw. reflexivity.
Qed.

Lemma sq_weight2 : forall (a w h : R) (x : V),
  0 <= w -> 0 < h ->
  (a * sqrt w / h * nrm x)^2 = a^2 * w / h^2 * << x , x >>.
Proof.
  intros a w h x Hw Hh.
  replace ((a * sqrt w / h * nrm x)^2)
    with (a^2 * (sqrt w * sqrt w) / h^2 * (nrm x * nrm x)) by (field; lra).
  rewrite sqrt_sqrt by exact Hw.
  rewrite (nrm_sq Hs). reflexivity.
Qed.

(*  |sum_l c_j <x_j, y_j>|  <=  sum_l c_j ||x_j|| ||y_j||.                    *)
Lemma sum_ip_abs :
  forall (J : Type) (l : list J) (c : J -> R) (x y : J -> V),
    (forall j, 0 <= c j) ->
    Rabs (Rsum l (fun j => c j * << (x j) , (y j) >>))
    <= Rsum l (fun j => c j * (nrm (x j) * nrm (y j))).
Proof.
  intros J l c x y Hc.
  eapply Rle_trans; [apply Rsum_abs_le |].
  apply Rsum_le. intro j.
  rewrite Rabs_mult.
  rewrite (Rabs_right (c j)) by (apply Rle_ge, Hc).
  pose proof (CS Hs (x j) (y j)) as Hcs.
  pose proof (Hc j) as Hcj.
  pose proof (Rabs_pos (<< (x j) , (y j) >>)) as Hp.
  nra.
Qed.

(*  Workhorse: every Cauchy--Schwarz step of the appendix (elementwise CS,   *)
(*  then discrete CS for the sums over elements, then a bound of each        *)
(*  square-root sum by a norm) is an instance of this lemma.                 *)
Lemma abs_sum_bound_c :
  forall (cc : R) (co : K -> R) (x y : K -> V) (Aq Bq : K -> R) (SA SB : R),
    0 <= cc ->
    (forall k, 0 <= co k) ->
    (forall k, 0 <= Aq k) -> (forall k, 0 <= Bq k) ->
    (forall k, co k * (nrm (x k) * nrm (y k)) <= cc * (Aq k * Bq k)) ->
    sqrt (Rsum Th (fun k => (Aq k)^2)) <= SA ->
    sqrt (Rsum Th (fun k => (Bq k)^2)) <= SB ->
    Rabs (Rsum Th (fun k => co k * << (x k) , (y k) >>)) <= cc * (SA * SB).
Proof.
  intros cc co x y Aq Bq SA SB Hcc Hc HA HB Hpt HSA HSB.
  eapply Rle_trans; [apply sum_ip_abs; exact Hc |].
  eapply Rle_trans; [apply Rsum_le; exact Hpt |].
  assert (E : Rsum Th (fun k => cc * (Aq k * Bq k))
              = cc * Rsum Th (fun k => Aq k * Bq k))
    by apply Rsum_scal.
  rewrite E.
  pose proof (Rsum_CS K Th Aq Bq HA HB) as HCS.
  assert (P1 : 0 <= sqrt (Rsum Th (fun k => (Aq k)^2))) by apply sqrt_pos.
  assert (P2 : 0 <= sqrt (Rsum Th (fun k => (Bq k)^2))) by apply sqrt_pos.
  assert (HSA0 : 0 <= SA) by lra.
  assert (HSB0 : 0 <= SB) by lra.
  assert (HS0 : 0 <= Rsum Th (fun k => Aq k * Bq k)).
  { apply Rsum_nonneg. intro k.
    pose proof (HA k); pose proof (HB k). nra. }
  assert (T1 : sqrt (Rsum Th (fun k => (Aq k)^2))
               * sqrt (Rsum Th (fun k => (Bq k)^2)) <= SA * SB) by nra.
  nra.
Qed.

(* ---------- Nonnegativity of the norm components --------------------------- *)

Lemma perU_parts : forall k,
  0 <= nu * << (gu k) , (gu k) >> /\ 0 <= sg k * << (uu k) , (uu k) >>
  /\ 0 <= eps * << (pp k) , (pp k) >> /\ 0 <= t1 k * << (xu k) , (xu k) >>
  /\ 0 <= t2 k * << (divu k) , (divu k) >>.
Proof.
  intro k.
  pose proof (ip_pos Hs (gu k)). pose proof (ip_pos Hs (uu k)).
  pose proof (ip_pos Hs (pp k)). pose proof (ip_pos Hs (xu k)).
  pose proof (ip_pos Hs (divu k)).
  pose proof (sg_nonneg' k). pose proof (t1_pos' k). pose proof (t2_pos' k).
  repeat split; nra.
Qed.

Lemma perV_parts : forall k,
  0 <= nu * << (gv k) , (gv k) >> /\ 0 <= sg k * << (vv k) , (vv k) >>
  /\ 0 <= eps * << (qq k) , (qq k) >> /\ 0 <= t1 k * << (xv k) , (xv k) >>
  /\ 0 <= t2 k * << (divv k) , (divv k) >>.
Proof.
  intro k.
  pose proof (ip_pos Hs (gv k)). pose proof (ip_pos Hs (vv k)).
  pose proof (ip_pos Hs (qq k)). pose proof (ip_pos Hs (xv k)).
  pose proof (ip_pos Hs (divv k)).
  pose proof (sg_nonneg' k). pose proof (t1_pos' k). pose proof (t2_pos' k).
  repeat split; nra.
Qed.

Lemma NU_nonneg : 0 <= NU.  Proof. apply sqrt_pos. Qed.
Lemma NV_nonneg : 0 <= NV.  Proof. apply sqrt_pos. Qed.
Lemma PsU_nonneg : 0 <= PsU.  Proof. apply sqrt_pos. Qed.
Lemma PsP_nonneg : 0 <= PsP.  Proof. apply sqrt_pos. Qed.

(* ---------- Component-to-norm bounds ---------------------------------------- *)

Lemma comp_U :
  forall (w : K -> R) (z : K -> V),
    (forall k, 0 <= w k) ->
    (forall k, w k * << (z k) , (z k) >> <= perU k) ->
    sqrt (Rsum Th (fun k => (sqrt (w k) * nrm (z k))^2)) <= NU.
Proof.
  intros w z Hw Hle.
  rewrite (Rsum_ext K Th _ _ (fun k => sq_weight (w k) (z k) (Hw k))).
  unfold NU, NU2.
  apply sqrt_le_1_alt, Rsum_le, Hle.
Qed.

Lemma comp_V :
  forall (w : K -> R) (z : K -> V),
    (forall k, 0 <= w k) ->
    (forall k, w k * << (z k) , (z k) >> <= perV k) ->
    sqrt (Rsum Th (fun k => (sqrt (w k) * nrm (z k))^2)) <= NV.
Proof.
  intros w z Hw Hle.
  rewrite (Rsum_ext K Th _ _ (fun k => sq_weight (w k) (z k) (Hw k))).
  unfold NV, NV2.
  apply sqrt_le_1_alt, Rsum_le, Hle.
Qed.

Lemma cmpU_g : sqrt (Rsum Th (fun k => (sqrt nu * nrm (gu k))^2)) <= NU.
Proof.
  apply (comp_U (fun _ => nu) gu); intro k;
    [lra | destruct (perU_parts k) as [P1 [P2 [P3 [P4 P5]]]];
           unfold perU; lra].
Qed.

Lemma cmpU_s : sqrt (Rsum Th (fun k => (sqrt (sg k) * nrm (uu k))^2)) <= NU.
Proof.
  apply (comp_U sg uu); intro k;
    [apply sg_nonneg' | destruct (perU_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                        unfold perU; lra].
Qed.

Lemma cmpU_e : sqrt (Rsum Th (fun k => (sqrt eps * nrm (pp k))^2)) <= NU.
Proof.
  apply (comp_U (fun _ => eps) pp); intro k;
    [lra | destruct (perU_parts k) as [P1 [P2 [P3 [P4 P5]]]];
           unfold perU; lra].
Qed.

Lemma cmpU_x : sqrt (Rsum Th (fun k => (sqrt (t1 k) * nrm (xu k))^2)) <= NU.
Proof.
  apply (comp_U t1 xu); intro k;
    [apply Rlt_le, t1_pos' | destruct (perU_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                             unfold perU; lra].
Qed.

Lemma cmpU_d : sqrt (Rsum Th (fun k => (sqrt (t2 k) * nrm (divu k))^2)) <= NU.
Proof.
  apply (comp_U t2 divu); intro k;
    [apply Rlt_le, t2_pos' | destruct (perU_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                             unfold perU; lra].
Qed.

Lemma cmpV_g : sqrt (Rsum Th (fun k => (sqrt nu * nrm (gv k))^2)) <= NV.
Proof.
  apply (comp_V (fun _ => nu) gv); intro k;
    [lra | destruct (perV_parts k) as [P1 [P2 [P3 [P4 P5]]]];
           unfold perV; lra].
Qed.

Lemma cmpV_s : sqrt (Rsum Th (fun k => (sqrt (sg k) * nrm (vv k))^2)) <= NV.
Proof.
  apply (comp_V sg vv); intro k;
    [apply sg_nonneg' | destruct (perV_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                        unfold perV; lra].
Qed.

Lemma cmpV_e : sqrt (Rsum Th (fun k => (sqrt eps * nrm (qq k))^2)) <= NV.
Proof.
  apply (comp_V (fun _ => eps) qq); intro k;
    [lra | destruct (perV_parts k) as [P1 [P2 [P3 [P4 P5]]]];
           unfold perV; lra].
Qed.

Lemma cmpV_x : sqrt (Rsum Th (fun k => (sqrt (t1 k) * nrm (xv k))^2)) <= NV.
Proof.
  apply (comp_V t1 xv); intro k;
    [apply Rlt_le, t1_pos' | destruct (perV_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                             unfold perV; lra].
Qed.

Lemma cmpV_d : sqrt (Rsum Th (fun k => (sqrt (t2 k) * nrm (divv k))^2)) <= NV.
Proof.
  apply (comp_V t2 divv); intro k;
    [apply Rlt_le, t2_pos' | destruct (perV_parts k) as [P1 [P2 [P3 [P4 P5]]]];
                             unfold perV; lra].
Qed.

(* ---------- The bracket norms as square-root sums ---------------------------- *)

Lemma PsP_le :
  sqrt (Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * IP k)^2)) <= PsP.
Proof.
  unfold PsP, PsP2.
  rewrite (Rsum_ext K Th _ _
    (fun k => sq_weight2s (aK k) (t1 k) (hK k) (IP k)
                (Rlt_le 0 (t1 k) (t1_pos' k)) (hK_pos k))).
  apply Rle_refl.
Qed.

Lemma PsU_le :
  sqrt (Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * IU k)^2)) <= PsU.
Proof.
  unfold PsU, PsU2.
  rewrite (Rsum_ext K Th _ _
    (fun k => sq_weight2s (aK k) (t2 k) (hK k) (IU k)
                (Rlt_le 0 (t2 k) (t2_pos' k)) (hK_pos k))).
  apply Rle_refl.
Qed.

(* ---------- Face-sum Cauchy--Schwarz with bounded multiplicity --------------- *)

Lemma face_CS_pair :
  forall (Pa Pb : K -> R) (ea eb : F -> K),
    (forall k, 0 <= Pa k) -> (forall k, 0 <= Pb k) ->
    (forall g : K -> R, (forall k, 0 <= g k) ->
       Rsum Fl (fun f => g (ea f)) <= Nf * Rsum Th g) ->
    (forall g : K -> R, (forall k, 0 <= g k) ->
       Rsum Fl (fun f => g (eb f)) <= Nf * Rsum Th g) ->
    Rsum Fl (fun f => Pa (ea f) * Pb (eb f))
    <= Nf * (sqrt (Rsum Th (fun k => (Pa k)^2))
             * sqrt (Rsum Th (fun k => (Pb k)^2))).
Proof.
  intros Pa Pb ea eb HPa HPb Hma Hmb.
  eapply Rle_trans.
  { apply (Rsum_CS F Fl (fun f => Pa (ea f)) (fun f => Pb (eb f)));
      intro f; auto. }
  assert (HA : Rsum Fl (fun f => (Pa (ea f))^2)
               <= Nf * Rsum Th (fun k => (Pa k)^2))
    by (apply (Hma (fun k => (Pa k)^2)); intro k; apply pow2_ge_0).
  assert (HB : Rsum Fl (fun f => (Pb (eb f))^2)
               <= Nf * Rsum Th (fun k => (Pb k)^2))
    by (apply (Hmb (fun k => (Pb k)^2)); intro k; apply pow2_ge_0).
  assert (HTa : 0 <= Rsum Th (fun k => (Pa k)^2))
    by (apply Rsum_nonneg; intro k; apply pow2_ge_0).
  assert (HTb : 0 <= Rsum Th (fun k => (Pb k)^2))
    by (apply Rsum_nonneg; intro k; apply pow2_ge_0).
  assert (SA : sqrt (Rsum Fl (fun f => (Pa (ea f))^2))
               <= sqrt Nf * sqrt (Rsum Th (fun k => (Pa k)^2))).
  { rewrite <- sqrt_mult_alt by exact Nf_nonneg.
    apply sqrt_le_1_alt. exact HA. }
  assert (SB : sqrt (Rsum Fl (fun f => (Pb (eb f))^2))
               <= sqrt Nf * sqrt (Rsum Th (fun k => (Pb k)^2))).
  { rewrite <- sqrt_mult_alt by exact Nf_nonneg.
    apply sqrt_le_1_alt. exact HB. }
  assert (H2 : sqrt Nf * sqrt Nf = Nf) by (apply sqrt_sqrt; exact Nf_nonneg).
  assert (P1 : 0 <= sqrt (Rsum Fl (fun f => (Pa (ea f))^2))) by apply sqrt_pos.
  assert (P2 : 0 <= sqrt (Rsum Fl (fun f => (Pb (eb f))^2))) by apply sqrt_pos.
  assert (P3 : 0 <= sqrt Nf) by apply sqrt_pos.
  assert (P4 : 0 <= sqrt (Rsum Th (fun k => (Pa k)^2))) by apply sqrt_pos.
  assert (P5 : 0 <= sqrt (Rsum Th (fun k => (Pb k)^2))) by apply sqrt_pos.
  assert (T1 : sqrt (Rsum Fl (fun f => (Pa (ea f))^2))
               * sqrt (Rsum Fl (fun f => (Pb (eb f))^2))
               <= (sqrt Nf * sqrt (Rsum Th (fun k => (Pa k)^2)))
                  * sqrt (Rsum Fl (fun f => (Pb (eb f))^2))) by nra.
  assert (Pk : 0 <= sqrt Nf * sqrt (Rsum Th (fun k => (Pa k)^2))) by nra.
  assert (T2 : (sqrt Nf * sqrt (Rsum Th (fun k => (Pa k)^2)))
               * sqrt (Rsum Fl (fun f => (Pb (eb f))^2))
               <= (sqrt Nf * sqrt (Rsum Th (fun k => (Pa k)^2)))
                  * (sqrt Nf * sqrt (Rsum Th (fun k => (Pb k)^2)))) by nra.
  assert (P6 : 0 <= sqrt (Rsum Th (fun k => (Pa k)^2))
                    * sqrt (Rsum Th (fun k => (Pb k)^2))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Steps 1--5, 7: the directly bounded terms.                               *)
(* ========================================================================= *)

Lemma sqrtc1_pos : 0 < sqrt c1.
Proof. apply sqrt_lt_R0; exact c1_pos. Qed.

(*  Step 1: T1, T10, T18.  *)
Lemma bound_T1 : Rabs T1 <= 2 * (NU * NV).
Proof.
  assert (E : T1 = Rsum Th (fun k => (2 * nu) * << (gv k) , (gu k) >>)).
  { unfold T1. rewrite Rsum_scal. ring. }
  rewrite E.
  apply (abs_sum_bound_c 2 (fun _ => 2 * nu) gv gu
           (fun k => sqrt nu * nrm (gu k)) (fun k => sqrt nu * nrm (gv k))
           NU NV); try lra.
  - intro k. lra.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gu k)). nra.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gv k)). nra.
  - intro k.
    assert (Hn2 : sqrt nu * sqrt nu = nu) by (apply sqrt_sqrt; lra).
    pose proof (nrm_nonneg Hs (gu k)) as Hgu.
    pose proof (nrm_nonneg Hs (gv k)) as Hgv.
    assert (Hnn : 0 <= nrm (gu k) * nrm (gv k)) by nra.
    assert (E2 : 2 * ((sqrt nu * nrm (gu k)) * (sqrt nu * nrm (gv k)))
                 = 2 * (sqrt nu * sqrt nu) * (nrm (gv k) * nrm (gu k)))
      by ring.
    rewrite Hn2 in E2.
    lra.
  - exact cmpU_g.
  - exact cmpV_g.
Qed.

(*  Step 2: T3 + T14 = (sigmatilde u, v)_h exactly.  *)
Lemma bound_T3_T14 : Rabs (T3 + T14) <= 1 * (NU * NV).
Proof.
  assert (E : T3 + T14 = Rsum Th (fun k => sg k * << (uu k) , (vv k) >>)).
  { unfold T3, T14.
    assert (E3 : sigma * Rsum Th (fun k => << (vv k) , (uu k) >>)
                 = Rsum Th (fun k => sigma * << (uu k) , (vv k) >>)).
    { rewrite Rsum_scal. f_equal. apply Rsum_ext. intro k.
      rewrite (ip_sym Hs (vv k) (uu k)). reflexivity. }
    assert (E14 : - sigma^2 * Rsum Th (fun k => t1 k * << (uu k) , (vv k) >>)
                  = Rsum Th (fun k => (- sigma^2 * t1 k)
                                       * << (uu k) , (vv k) >>)).
    { rewrite <- Rsum_scal. apply Rsum_ext. intro k. ring. }
    rewrite E3, E14, <- Rsum_plus.
    apply Rsum_ext. intro k.
    pose proof (L_sg_sub k). nra. }
  rewrite E.
  apply (abs_sum_bound_c 1 sg uu vv
           (fun k => sqrt (sg k) * nrm (uu k))
           (fun k => sqrt (sg k) * nrm (vv k)) NU NV); try lra.
  - exact sg_nonneg'.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (vv k)). nra.
  - intro k.
    assert (Hs2 : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt, sg_nonneg').
    pose proof (nrm_nonneg Hs (uu k)) as Hu.
    pose proof (nrm_nonneg Hs (vv k)) as Hv.
    assert (E2 : 1 * ((sqrt (sg k) * nrm (uu k)) * (sqrt (sg k) * nrm (vv k)))
                 = (sqrt (sg k) * sqrt (sg k)) * (nrm (uu k) * nrm (vv k)))
      by ring.
    rewrite Hs2 in E2.
    lra.
  - exact cmpU_s.
  - exact cmpV_s.
Qed.

(*  Step 3: T5 and T15.  *)
Lemma bound_T5 : Rabs T5 <= 1 * (NU * NV).
Proof.
  assert (E : T5 = Rsum Th (fun k => eps * << (pp k) , (qq k) >>)).
  { unfold T5. rewrite Rsum_scal. f_equal. apply Rsum_ext. intro k.
    rewrite (ip_sym Hs (qq k) (pp k)). reflexivity. }
  rewrite E.
  apply (abs_sum_bound_c 1 (fun _ => eps) pp qq
           (fun k => sqrt eps * nrm (pp k))
           (fun k => sqrt eps * nrm (qq k)) NU NV); try lra.
  - intro k. lra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (pp k)). nra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (qq k)). nra.
  - intro k.
    assert (He2 : sqrt eps * sqrt eps = eps) by (apply sqrt_sqrt; lra).
    assert (E2 : 1 * ((sqrt eps * nrm (pp k)) * (sqrt eps * nrm (qq k)))
                 = (sqrt eps * sqrt eps) * (nrm (pp k) * nrm (qq k)))
      by ring.
    rewrite He2 in E2.
    lra.
  - exact cmpU_e.
  - exact cmpV_e.
Qed.

Lemma bound_T15 : Rabs T15 <= C2 * (NU * NV).
Proof.
  assert (E : T15 = - Rsum Th (fun k => (eps^2 * t2 k)
                                        * << (pp k) , (qq k) >>)).
  { unfold T15. rewrite <- Rsum_scal.
    rewrite <- Rsum_opp. apply Rsum_ext. intro k. ring. }
  rewrite E, Rabs_Ropp.
  apply (abs_sum_bound_c C2 (fun k => eps^2 * t2 k) pp qq
           (fun k => sqrt eps * nrm (pp k))
           (fun k => sqrt eps * nrm (qq k)) NU NV); try lra.
  - intro k. pose proof (t2_pos' k). pose proof eps_nonneg. nra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (pp k)). nra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (qq k)). nra.
  - intro k.
    pose proof (L_epst2 k) as HC.
    pose proof (t2_pos' k) as Ht2.
    pose proof (nrm_nonneg Hs (pp k)) as Hp.
    pose proof (nrm_nonneg Hs (qq k)) as Hq.
    assert (He2 : sqrt eps * sqrt eps = eps) by (apply sqrt_sqrt; lra).
    assert (Hnn : 0 <= nrm (pp k) * nrm (qq k)) by nra.
    assert (S1 : eps^2 * t2 k <= C2 * eps) by nra.
    assert (S2 : eps^2 * t2 k * (nrm (pp k) * nrm (qq k))
                 <= C2 * eps * (nrm (pp k) * nrm (qq k))) by nra.
    assert (E2 : C2 * ((sqrt eps * nrm (pp k)) * (sqrt eps * nrm (qq k)))
                 = C2 * (sqrt eps * sqrt eps) * (nrm (pp k) * nrm (qq k)))
      by ring.
    rewrite He2 in E2.
    lra.
  - exact cmpU_e.
  - exact cmpV_e.
Qed.

(*  Step 7: T16 and T17.  *)
Lemma bound_T16 : Rabs T16 <= sqrt C2 * (NU * NV).
Proof.
  assert (E : T16 = Rsum Th (fun k => (eps * t2 k)
                                       * << (pp k) , (divv k) >>)).
  { unfold T16. rewrite <- Rsum_scal. apply Rsum_ext. intro k.
    rewrite (ip_sym Hs (divv k) (pp k)). ring. }
  rewrite E.
  apply (abs_sum_bound_c (sqrt C2) (fun k => eps * t2 k) pp divv
           (fun k => sqrt eps * nrm (pp k))
           (fun k => sqrt (t2 k) * nrm (divv k)) NU NV).
  - apply sqrt_pos.
  - intro k. pose proof (t2_pos' k). nra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (pp k)). nra.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divv k)). nra.
  - intro k.
    pose proof (L_step7 k) as H7.
    pose proof (t2_pos' k) as Ht2.
    pose proof (nrm_nonneg Hs (pp k)) as Hp.
    pose proof (nrm_nonneg Hs (divv k)) as Hd.
    assert (Ht2s : sqrt (t2 k) * sqrt (t2 k) = t2 k)
      by (apply sqrt_sqrt; lra).
    assert (Hst : 0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (Hfac : 0 <= nrm (pp k) * (sqrt (t2 k) * nrm (divv k))).
    { assert (0 <= sqrt (t2 k) * nrm (divv k)) by nra. nra. }
    assert (S1 : (eps * sqrt (t2 k))
                   * (nrm (pp k) * (sqrt (t2 k) * nrm (divv k)))
                 = eps * (sqrt (t2 k) * sqrt (t2 k))
                   * (nrm (pp k) * nrm (divv k))) by ring.
    rewrite Ht2s in S1.
    assert (S2 : (eps * sqrt (t2 k))
                   * (nrm (pp k) * (sqrt (t2 k) * nrm (divv k)))
                 <= (sqrt C2 * sqrt eps)
                    * (nrm (pp k) * (sqrt (t2 k) * nrm (divv k)))) by nra.
    assert (E3 : sqrt C2 * ((sqrt eps * nrm (pp k)) * (sqrt (t2 k) * nrm (divv k)))
                 = (sqrt C2 * sqrt eps)
                   * (nrm (pp k) * (sqrt (t2 k) * nrm (divv k)))) by ring.
    lra.
  - exact cmpU_e.
  - exact cmpV_d.
Qed.

Lemma bound_T17 : Rabs T17 <= sqrt C2 * (NU * NV).
Proof.
  assert (E : T17 = - Rsum Th (fun k => (eps * t2 k)
                                        * << (divu k) , (qq k) >>)).
  { unfold T17. rewrite <- Rsum_scal, <- Rsum_opp.
    apply Rsum_ext. intro k. ring. }
  rewrite E, Rabs_Ropp.
  apply (abs_sum_bound_c (sqrt C2) (fun k => eps * t2 k) divu qq
           (fun k => sqrt (t2 k) * nrm (divu k))
           (fun k => sqrt eps * nrm (qq k)) NU NV).
  - apply sqrt_pos.
  - intro k. pose proof (t2_pos' k). nra.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divu k)). nra.
  - intro k. assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (qq k)). nra.
  - intro k.
    pose proof (L_step7 k) as H7.
    pose proof (t2_pos' k) as Ht2.
    pose proof (nrm_nonneg Hs (divu k)) as Hd.
    pose proof (nrm_nonneg Hs (qq k)) as Hq.
    assert (Ht2s : sqrt (t2 k) * sqrt (t2 k) = t2 k)
      by (apply sqrt_sqrt; lra).
    assert (Hst : 0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (Hfac : 0 <= (sqrt (t2 k) * nrm (divu k)) * nrm (qq k)).
    { assert (0 <= sqrt (t2 k) * nrm (divu k)) by nra. nra. }
    assert (S1 : (eps * sqrt (t2 k))
                   * ((sqrt (t2 k) * nrm (divu k)) * nrm (qq k))
                 = eps * (sqrt (t2 k) * sqrt (t2 k))
                   * (nrm (divu k) * nrm (qq k))) by ring.
    rewrite Ht2s in S1.
    assert (S2 : (eps * sqrt (t2 k))
                   * ((sqrt (t2 k) * nrm (divu k)) * nrm (qq k))
                 <= (sqrt C2 * sqrt eps)
                   * ((sqrt (t2 k) * nrm (divu k)) * nrm (qq k))) by nra.
    assert (E3 : sqrt C2 * ((sqrt (t2 k) * nrm (divu k)) * (sqrt eps * nrm (qq k)))
                 = (sqrt C2 * sqrt eps)
                   * ((sqrt (t2 k) * nrm (divu k)) * nrm (qq k))) by ring.
    lra.
  - exact cmpU_d.
  - exact cmpV_e.
Qed.

Lemma bound_T10 : Rabs T10 <= 1 * (NU * NV).
Proof.
  unfold T10.
  apply (abs_sum_bound_c 1 t1 xv xu
           (fun k => sqrt (t1 k) * nrm (xu k))
           (fun k => sqrt (t1 k) * nrm (xv k)) NU NV); try lra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xu k)). nra.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xv k)). nra.
  - intro k.
    assert (Ht2 : sqrt (t1 k) * sqrt (t1 k) = t1 k)
      by (apply sqrt_sqrt, Rlt_le, t1_pos').
    nra.
  - exact cmpU_x.
  - exact cmpV_x.
Qed.

Lemma bound_T18 : Rabs T18 <= 1 * (NU * NV).
Proof.
  unfold T18.
  apply (abs_sum_bound_c 1 t2 divv divu
           (fun k => sqrt (t2 k) * nrm (divu k))
           (fun k => sqrt (t2 k) * nrm (divv k)) NU NV); try lra.
  - intro k. apply Rlt_le, t2_pos'.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divu k)). nra.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divv k)). nra.
  - intro k.
    assert (Ht2 : sqrt (t2 k) * sqrt (t2 k) = t2 k)
      by (apply sqrt_sqrt, Rlt_le, t2_pos').
    nra.
  - exact cmpU_d.
  - exact cmpV_d.
Qed.

(* ---------- Step 4: the viscous-operator terms ------------------------------ *)

(*  The elemental core of eq:keyvisc, in coefficient form.  *)
Lemma keyvisc_coef : forall k,
  sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
  <= 2 * Cb / sqrt c1 * sqrt nu.
Proof.
  intro k.
  pose proof (L_keysq k) as HK.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (t1_pos' k) as Ht.
  pose proof sqrtc1_pos as Hsc.
  assert (Hsn : 0 < sqrt nu) by (apply sqrt_lt_R0; lra).
  assert (Hst : 0 < sqrt (t1 k)) by (apply sqrt_lt_R0; lra).
  assert (Hsa : 0 < sqrt (aK k)) by (apply sqrt_lt_R0; lra).
  (*  reduce to  sqrt nu sqrt t1 sqrt aK sqrt c1 <= h  (keyvisc_sqrt)  *)
  assert (HKs : sqrt nu * sqrt (t1 k) * sqrt (aK k) * sqrt c1 <= hK k).
  { unfold t1. apply keyvisc_sqrt; auto. }
  assert (Hpos : 0 < hK k * (sqrt c1 * sqrt nu)).
  { assert (0 < sqrt c1 * sqrt nu) by nra. nra. }
  apply (Rmult_le_reg_r (hK k * (sqrt c1 * sqrt nu))); [exact Hpos |].
  replace (sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
           * (hK k * (sqrt c1 * sqrt nu)))
    with (2 * nu * Cb
          * (sqrt nu * sqrt (t1 k) * sqrt (aK k) * sqrt c1))
    by (field; lra).
  replace (2 * Cb / sqrt c1 * sqrt nu * (hK k * (sqrt c1 * sqrt nu)))
    with (2 * Cb * (sqrt nu * sqrt nu) * hK k) by (field; lra).
  rewrite sqrt_sqrt by lra.
  assert (Hm : 0 <= 2 * nu * Cb) by nra.
  nra.
Qed.

Lemma bound_T6 : Rabs T6 <= 4 * Cb * CI / c1 * (PsU * NV).
Proof.
  unfold T6. rewrite Rabs_Ropp.
  apply (abs_sum_bound_c (4 * Cb * CI / c1) t1 dv du
           (fun k => aK k * sqrt (t2 k) / hK k * IU k)
           (fun k => sqrt nu * nrm (gv k)) PsU NV).
  - apply div_nonneg; nra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k.
    pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IU_nonneg k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gv k)). nra.
  - intro k.
    pose proof (Hw_dv k) as HDv.
    pose proof (HI_du k) as HDu.
    pose proof (L_viscph k) as HV.
    pose proof (L_pst k) as HPst.
    pose proof (keyvisc_coef k) as HKC.
    pose proof (L_P3div k) as HP3.
    pose proof (t1_pos' k) as Ht1.
    pose proof (ph_pos k) as Hph.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (IU_nonneg k) as HIU.
    pose proof (nrm_nonneg Hs (dv k)) as Hndv.
    pose proof (nrm_nonneg Hs (du k)) as Hndu.
    pose proof (nrm_nonneg Hs (gv k)) as Hngv.
    assert (Hst1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (Hsa  : 0 <= sqrt (aK k)) by apply sqrt_pos.
    assert (Hsnu : 0 <= sqrt nu) by apply sqrt_pos.
    assert (Hsph : 0 <= sqrt (ph k)) by apply sqrt_pos.
    assert (Ht1s : sqrt (t1 k) * sqrt (t1 k) = t1 k) by (apply sqrt_sqrt; lra).
    (*  nonnegativity of the two coefficients  *)
    assert (Hcdv : 0 <= 2 * nu * Cb / hK k * sqrt (aK k)).
    { assert (H1 : 0 < 2 * nu * Cb) by nra.
      assert (H2 : 0 < 2 * nu * Cb / hK k) by (apply Rdiv_lt_0_compat; lra).
      nra. }
    assert (Hh2 : 0 < (hK k)^2) by nra.
    assert (Hcdu : 0 <= 2 * nu * CI * aK k / (hK k)^2).
    { assert (H0 : 0 <= nu * CI) by nra.
      assert (H1 : 0 <= 2 * nu * CI * aK k) by nra.
      apply div_nonneg; lra. }
    (*  product of the two field bounds  *)
    assert (Hfac : 0 <= 2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k)) by nra.
    assert (S1a : nrm (dv k) * nrm (du k)
                  <= (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k))
                     * nrm (du k)) by nra.
    assert (S1 : (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k)) * nrm (du k)
                 <= (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k))
                    * (2 * nu * CI * aK k / (hK k)^2 * IU k)) by nra.
    (*  coefficient ladder  *)
    assert (C1 : 2 * nu * CI * aK k / (hK k)^2 <= 2 * CI / c1 * ph k).
    { assert (E : 2 * nu * CI * aK k / (hK k)^2
                  = 2 * CI / c1 * (c1 * nu * aK k / (hK k)^2)) by (field; lra).
      rewrite E.
      assert (Hc : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
      nra. }
    assert (K1 : t1 k * (2 * nu * CI * aK k / (hK k)^2)
                 <= 2 * CI / c1 * (ph k * t1 k)) by nra.
    assert (K2a : ph k * t1 k * (2 * nu * Cb / hK k * sqrt (aK k))
                  = (ph k * sqrt (t1 k))
                    * (sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k)))).
    { assert (E : (ph k * sqrt (t1 k))
                  * (sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k)))
                  = ph k * (sqrt (t1 k) * sqrt (t1 k))
                    * (2 * nu * Cb / hK k * sqrt (aK k))) by ring.
      rewrite Ht1s in E. lra. }
    assert (Hphs : 0 <= ph k * sqrt (t1 k)) by nra.
    assert (Hkc0 : 0 <= 2 * Cb / sqrt c1 * sqrt nu).
    { pose proof sqrtc1_pos.
      assert (0 <= 2 * Cb / sqrt c1) by (apply div_nonneg; nra).
      nra. }
    assert (K2b : (ph k * sqrt (t1 k))
                    * (sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k)))
                  <= (ph k * sqrt (t1 k)) * (2 * Cb / sqrt c1 * sqrt nu))
      by nra.
    assert (K2c : (ph k * sqrt (t1 k)) * (2 * Cb / sqrt c1 * sqrt nu)
                  <= sqrt (ph k) * (2 * Cb / sqrt c1 * sqrt nu)) by nra.
    assert (K2 : ph k * t1 k * (2 * nu * Cb / hK k * sqrt (aK k))
                 <= sqrt (ph k) * (2 * Cb / sqrt c1 * sqrt nu)) by lra.
    assert (Hc21 : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
    assert (K3a : t1 k * (2 * nu * Cb / hK k * sqrt (aK k))
                    * (2 * nu * CI * aK k / (hK k)^2)
                  <= (2 * nu * Cb / hK k * sqrt (aK k))
                     * (2 * CI / c1 * (ph k * t1 k))) by nra.
    assert (K3b : (2 * nu * Cb / hK k * sqrt (aK k))
                    * (2 * CI / c1 * (ph k * t1 k))
                  <= 2 * CI / c1 * (sqrt (ph k) * (2 * Cb / sqrt c1 * sqrt nu)))
      by nra.
    assert (K4 : 2 * CI / c1 * (sqrt (ph k) * (2 * Cb / sqrt c1 * sqrt nu))
                 = 4 * Cb * CI / c1 * (aK k * sqrt (t2 k) / hK k * sqrt nu)).
    { rewrite HP3. pose proof sqrtc1_pos. field. repeat split; lra. }
    assert (Hnn : 0 <= nrm (gv k) * IU k) by nra.
    assert (S2 : t1 k * (nrm (dv k) * nrm (du k))
                 <= t1 k * ((2 * nu * Cb / hK k * sqrt (aK k))
                            * (2 * nu * CI * aK k / (hK k)^2))
                    * (nrm (gv k) * IU k)) by nra.
    assert (K5 : t1 k * ((2 * nu * Cb / hK k * sqrt (aK k))
                         * (2 * nu * CI * aK k / (hK k)^2))
                 <= 4 * Cb * CI / c1 * (aK k * sqrt (t2 k) / hK k * sqrt nu))
      by lra.
    nra.
  - exact PsU_le.
  - exact cmpV_g.
Qed.

Lemma bound_T7 : Rabs T7 <= 2 * CI / sqrt c1 * (PsU * NV).
Proof.
  unfold T7. rewrite Rabs_Ropp.
  apply (abs_sum_bound_c (2 * CI / sqrt c1) t1 xv du
           (fun k => aK k * sqrt (t2 k) / hK k * IU k)
           (fun k => sqrt (t1 k) * nrm (xv k)) PsU NV).
  - pose proof sqrtc1_pos. apply div_nonneg; nra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k.
    pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IU_nonneg k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xv k)). nra.
  - intro k.
    pose proof (HI_du k) as HDu.
    pose proof (L_viscph k) as HV.
    pose proof (L_pst k) as HPst.
    pose proof (L_P3div k) as HP3.
    pose proof (t1_pos' k) as Ht1.
    pose proof (ph_pos k) as Hph.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (IU_nonneg k) as HIU.
    pose proof (nrm_nonneg Hs (du k)) as Hndu.
    pose proof (nrm_nonneg Hs (xv k)) as Hnxv.
    pose proof sqrtc1_pos as Hsc.
    assert (Hst1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (Hst2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (Hsph : 0 <= sqrt (ph k)) by apply sqrt_pos.
    assert (Ht1s : sqrt (t1 k) * sqrt (t1 k) = t1 k) by (apply sqrt_sqrt; lra).
    assert (Hh2 : 0 < (hK k)^2) by nra.
    assert (Hcdu : 0 <= 2 * nu * CI * aK k / (hK k)^2).
    { assert (H0 : 0 <= nu * CI) by nra.
      assert (H1 : 0 <= 2 * nu * CI * aK k) by nra.
      apply div_nonneg; lra. }
    assert (C1 : 2 * nu * CI * aK k / (hK k)^2 <= 2 * CI / c1 * ph k).
    { assert (E : 2 * nu * CI * aK k / (hK k)^2
                  = 2 * CI / c1 * (c1 * nu * aK k / (hK k)^2)) by (field; lra).
      rewrite E.
      assert (Hc : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
      nra. }
    assert (Hc21 : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
    assert (HX : 0 <= aK k * sqrt (t2 k) / hK k).
    { assert (0 <= aK k * sqrt (t2 k)) by nra.
      apply div_nonneg; lra. }
    assert (Hcoef : 2 * CI / c1 * sqrt c1 <= 2 * CI / sqrt c1).
    { assert (Hc2 : sqrt c1 * sqrt c1 = c1) by (apply sqrt_sqrt; lra).
      apply (Rmult_le_reg_r (sqrt c1)); [lra |].
      replace (2 * CI / sqrt c1 * sqrt c1) with (2 * CI) by (field; lra).
      replace (2 * CI / c1 * sqrt c1 * sqrt c1)
        with (2 * CI * (sqrt c1 * sqrt c1) / c1) by (field; lra).
      rewrite Hc2.
      replace (2 * CI * c1 / c1) with (2 * CI) by (field; lra).
      lra. }
    assert (K3 : 2 * CI / c1 * sqrt (ph k)
                 <= 2 * CI / sqrt c1 * (aK k * sqrt (t2 k) / hK k)).
    { rewrite HP3.
      assert (E : 2 * CI / c1 * (sqrt c1 * aK k * sqrt (t2 k) / hK k)
                  = (2 * CI / c1 * sqrt c1) * (aK k * sqrt (t2 k) / hK k))
        by (field; lra).
      rewrite E. nra. }
    assert (Hb : 0 <= sqrt (t1 k) * IU k) by nra.
    assert (Hc3 : 0 <= 2 * CI / c1 * IU k) by nra.
    assert (S1 : sqrt (t1 k) * nrm (du k)
                 <= sqrt (t1 k) * (2 * nu * CI * aK k / (hK k)^2) * IU k)
      by nra.
    assert (S2 : sqrt (t1 k) * (2 * nu * CI * aK k / (hK k)^2) * IU k
                 <= 2 * CI / c1 * (ph k * sqrt (t1 k)) * IU k) by nra.
    assert (S3 : 2 * CI / c1 * (ph k * sqrt (t1 k)) * IU k
                 <= 2 * CI / c1 * sqrt (ph k) * IU k) by nra.
    assert (S4 : 2 * CI / c1 * sqrt (ph k) * IU k
                 <= 2 * CI / sqrt c1 * (aK k * sqrt (t2 k) / hK k) * IU k)
      by nra.
    assert (Schain : sqrt (t1 k) * nrm (du k)
                     <= 2 * CI / sqrt c1 * (aK k * sqrt (t2 k) / hK k) * IU k)
      by lra.
    assert (Hxfac : 0 <= sqrt (t1 k) * nrm (xv k)) by nra.
    assert (E2 : t1 k * (nrm (xv k) * nrm (du k))
                 = (sqrt (t1 k) * nrm (xv k)) * (sqrt (t1 k) * nrm (du k))).
    { assert (E : (sqrt (t1 k) * nrm (xv k)) * (sqrt (t1 k) * nrm (du k))
                  = (sqrt (t1 k) * sqrt (t1 k))
                    * (nrm (xv k) * nrm (du k))) by ring.
      rewrite Ht1s in E. lra. }
    rewrite E2. nra.
  - exact PsU_le.
  - exact cmpV_x.
Qed.

Lemma bound_T9 : Rabs T9 <= 2 * Cb / sqrt c1 * (NU * NV).
Proof.
  unfold T9.
  apply (abs_sum_bound_c (2 * Cb / sqrt c1) t1 dv xu
           (fun k => sqrt (t1 k) * nrm (xu k))
           (fun k => sqrt nu * nrm (gv k)) NU NV).
  - pose proof sqrtc1_pos. apply div_nonneg; nra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xu k)). nra.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gv k)). nra.
  - intro k.
    pose proof (Hw_dv k) as HDv.
    pose proof (keyvisc_coef k) as HC.
    pose proof (t1_pos' k) as Ht1.
    pose proof (nrm_nonneg Hs (dv k)) as Hnd.
    pose proof (nrm_nonneg Hs (xu k)) as Hnx.
    pose proof (nrm_nonneg Hs (gv k)) as Hgv.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    assert (Hst : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (Ht1s : sqrt (t1 k) * sqrt (t1 k) = t1 k)
      by (apply sqrt_sqrt; lra).
    assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
    assert (Hcoef : 0 <= 2 * nu * Cb / hK k * sqrt (aK k)).
    { assert (H1 : 0 < 2 * nu * Cb) by nra.
      assert (H2 : 0 < 2 * nu * Cb / hK k)
        by (apply Rdiv_lt_0_compat; lra).
      nra. }
    assert (S1 : sqrt (t1 k) * nrm (dv k)
                 <= sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
                    * nrm (gv k)) by nra.
    assert (Hgap : 0 <= 2 * Cb / sqrt c1 * sqrt nu).
    { pose proof sqrtc1_pos.
      assert (0 <= sqrt nu) by apply sqrt_pos.
      assert (0 <= 2 * Cb / sqrt c1) by (apply div_nonneg; nra).
      nra. }
    assert (S2 : sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
                   * nrm (gv k)
                 <= (2 * Cb / sqrt c1 * sqrt nu) * nrm (gv k)) by nra.
    assert (Hfac : 0 <= sqrt (t1 k) * nrm (xu k)) by nra.
    assert (S3 : (sqrt (t1 k) * nrm (xu k)) * (sqrt (t1 k) * nrm (dv k))
                 <= (sqrt (t1 k) * nrm (xu k))
                    * ((2 * Cb / sqrt c1 * sqrt nu) * nrm (gv k))) by nra.
    nra.
  - exact cmpU_x.
  - exact cmpV_g.
Qed.

(*  The double inverse estimate eq:doubleinv, folded with the T8/T12 chain.  *)
Lemma double_inv_v : forall k,
  nrm (dv k) <= 2 * nu * Cb * Cinv * aK k / (hK k)^2 * nrm (vv k).
Proof.
  intro k.
  pose proof (Hw_dv k) as H1. pose proof (Hw_gv k) as H2.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (nrm_nonneg Hs (gv k)) as Hg.
  pose proof (nrm_nonneg Hs (vv k)) as Hu.
  assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
  assert (Hcoef : 0 <= 2 * nu * Cb / hK k * sqrt (aK k)).
  { assert (H3 : 0 < 2 * nu * Cb) by nra.
    assert (H4 : 0 < 2 * nu * Cb / hK k) by (apply Rdiv_lt_0_compat; lra).
    nra. }
  assert (S1 : nrm (dv k)
               <= (2 * nu * Cb / hK k * sqrt (aK k))
                  * (Cinv / hK k * sqrt (aK k) * nrm (vv k))) by nra.
  assert (E : (2 * nu * Cb / hK k * sqrt (aK k))
              * (Cinv / hK k * sqrt (aK k) * nrm (vv k))
              = 2 * nu * Cb * Cinv * (sqrt (aK k) * sqrt (aK k)) / (hK k)^2
                * nrm (vv k)) by (field; lra).
  rewrite E in S1. rewrite sqrt_sqrt in S1 by lra.
  exact S1.
Qed.

Lemma bound_T8 : Rabs T8 <= 2 * CI / sqrt c1 * (PsU * NV).
Proof.
  assert (E : T8 = Rsum Th (fun k => (sigma * t1 k)
                                      * << (vv k) , (du k) >>)).
  { unfold T8. rewrite <- Rsum_scal. apply Rsum_ext. intro k. ring. }
  rewrite E.
  apply (abs_sum_bound_c (2 * CI / sqrt c1) (fun k => sigma * t1 k) vv du
           (fun k => aK k * sqrt (t2 k) / hK k * IU k)
           (fun k => sqrt (sg k) * nrm (vv k)) PsU NV).
  - pose proof sqrtc1_pos. apply div_nonneg; nra.
  - intro k. pose proof (t1_pos' k). nra.
  - intro k.
    pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IU_nonneg k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (vv k)). nra.
  - intro k.
    pose proof (HI_du k) as HDu.
    pose proof (L_viscph k) as HV.
    pose proof (L_sgtau k) as HSg.
    pose proof (L_abs2 k) as HA2.
    pose proof (L_P3div k) as HP3.
    pose proof (t1_pos' k) as Ht1.
    pose proof (ph_pos k) as Hph.
    pose proof (sg_nonneg' k) as Hsg.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (IU_nonneg k) as HIU.
    pose proof (nrm_nonneg Hs (du k)) as Hndu.
    pose proof (nrm_nonneg Hs (vv k)) as Hnvv.
    pose proof sqrtc1_pos as Hsc.
    assert (Hssg : 0 <= sqrt (sg k)) by apply sqrt_pos.
    assert (Hsph : 0 <= sqrt (ph k)) by apply sqrt_pos.
    assert (Hsgs : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt; lra).
    assert (Hh2 : 0 < (hK k)^2) by nra.
    assert (C1 : 2 * nu * CI * aK k / (hK k)^2 <= 2 * CI / c1 * ph k).
    { assert (E1 : 2 * nu * CI * aK k / (hK k)^2
                   = 2 * CI / c1 * (c1 * nu * aK k / (hK k)^2)) by (field; lra).
      rewrite E1.
      assert (Hc : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
      nra. }
    (*  sigma t1 * coef_du <= (2CI/c1) * (sigma t1 ph) = (2CI/c1) sg  *)
    assert (Hst : 0 <= sigma * t1 k) by nra.
    assert (K1 : sigma * t1 k * (2 * nu * CI * aK k / (hK k)^2)
                 <= 2 * CI / c1 * (sigma * ph k * t1 k)) by nra.
    assert (K2 : 2 * CI / c1 * (sigma * ph k * t1 k) = 2 * CI / c1 * sg k)
      by (rewrite HSg; ring).
    (*  sqrt sg <= sqrt ph = sqrt c1 * aK sqrt t2 / h  *)
    assert (K3 : sqrt (sg k) <= sqrt c1 * (aK k * sqrt (t2 k) / hK k)).
    { assert (E1 : sqrt c1 * aK k * sqrt (t2 k) / hK k
                   = sqrt c1 * (aK k * sqrt (t2 k) / hK k)) by (field; lra).
      rewrite <- E1, <- HP3. exact HA2. }
    (*  pointwise assembly  *)
    assert (Hnn : 0 <= nrm (vv k) * IU k) by nra.
    assert (Hsv : 0 <= sigma * t1 k * nrm (vv k)) by nra.
    assert (S1 : sigma * t1 k * (nrm (vv k) * nrm (du k))
                 <= sigma * t1 k * (2 * nu * CI * aK k / (hK k)^2)
                    * (nrm (vv k) * IU k)) by nra.
    assert (S2 : sigma * t1 k * (2 * nu * CI * aK k / (hK k)^2)
                   * (nrm (vv k) * IU k)
                 <= 2 * CI / c1 * sg k * (nrm (vv k) * IU k)).
    { assert (K1' : sigma * t1 k * (2 * nu * CI * aK k / (hK k)^2)
                    <= 2 * CI / c1 * sg k) by lra.
      nra. }
    (*  2CI/c1 * sg * nvv * IU = (2CI/c1) * (sqrt sg * nvv) * (sqrt sg * IU)
        <= (2CI/c1) * (sqrt sg * nvv) * (sqrt c1 * (aK sqrt t2/h) * IU)     *)
    assert (E3 : 2 * CI / c1 * sg k * (nrm (vv k) * IU k)
                 = 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                  * (sqrt (sg k) * IU k))).
    { assert (E4 : 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                  * (sqrt (sg k) * IU k))
                   = 2 * CI / c1 * (sqrt (sg k) * sqrt (sg k))
                     * (nrm (vv k) * IU k)) by ring.
      rewrite Hsgs in E4. lra. }
    assert (Hc21 : 0 <= 2 * CI / c1) by (apply div_nonneg; lra).
    assert (Hsn : 0 <= sqrt (sg k) * nrm (vv k)) by nra.
    assert (Hfac : 0 <= 2 * CI / c1 * (sqrt (sg k) * nrm (vv k))) by nra.
    assert (S3 : 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                * (sqrt (sg k) * IU k))
                 <= 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                   * (sqrt c1 * (aK k * sqrt (t2 k) / hK k)
                                      * IU k))).
    { assert (K3' : sqrt (sg k) * IU k
                    <= sqrt c1 * (aK k * sqrt (t2 k) / hK k) * IU k) by nra.
      nra. }
    (*  coefficient: (2CI/c1) * sqrt c1 <= 2CI/sqrt c1  *)
    assert (Hcoef : 2 * CI / c1 * sqrt c1 <= 2 * CI / sqrt c1).
    { assert (Hc2 : sqrt c1 * sqrt c1 = c1) by (apply sqrt_sqrt; lra).
      apply (Rmult_le_reg_r (sqrt c1)); [lra |].
      replace (2 * CI / sqrt c1 * sqrt c1) with (2 * CI) by (field; lra).
      replace (2 * CI / c1 * sqrt c1 * sqrt c1)
        with (2 * CI * (sqrt c1 * sqrt c1) / c1) by (field; lra).
      rewrite Hc2.
      replace (2 * CI * c1 / c1) with (2 * CI) by (field; lra).
      lra. }
    assert (HXI : 0 <= (sqrt (sg k) * nrm (vv k))
                       * (aK k * sqrt (t2 k) / hK k * IU k)).
    { assert (Hst2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
      assert (H1 : 0 <= aK k * sqrt (t2 k)) by nra.
      assert (H2 : 0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
      assert (H3 : 0 <= sqrt (sg k) * nrm (vv k)) by nra.
      assert (H4 : 0 <= aK k * sqrt (t2 k) / hK k * IU k) by nra.
      nra. }
    assert (S4 : 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                * (sqrt c1 * (aK k * sqrt (t2 k) / hK k)
                                   * IU k))
                 <= 2 * CI / sqrt c1
                    * ((aK k * sqrt (t2 k) / hK k * IU k)
                       * (sqrt (sg k) * nrm (vv k)))).
    { assert (E5 : 2 * CI / c1 * ((sqrt (sg k) * nrm (vv k))
                                  * (sqrt c1 * (aK k * sqrt (t2 k) / hK k)
                                     * IU k))
                   = (2 * CI / c1 * sqrt c1)
                     * ((sqrt (sg k) * nrm (vv k))
                        * (aK k * sqrt (t2 k) / hK k * IU k))) by ring.
      assert (E6 : 2 * CI / sqrt c1
                     * ((aK k * sqrt (t2 k) / hK k * IU k)
                        * (sqrt (sg k) * nrm (vv k)))
                   = (2 * CI / sqrt c1)
                     * ((sqrt (sg k) * nrm (vv k))
                        * (aK k * sqrt (t2 k) / hK k * IU k))) by ring.
      rewrite E5, E6. nra. }
    lra.
  - exact PsU_le.
  - exact cmpV_s.
Qed.

Lemma bound_T12 : Rabs T12 <= 2 * Cb * Cinv / c1 * (NU * NV).
Proof.
  assert (E : T12 = Rsum Th (fun k => (sigma * t1 k)
                                       * << (dv k) , (uu k) >>)).
  { unfold T12. rewrite <- Rsum_scal. apply Rsum_ext. intro k. ring. }
  rewrite E.
  apply (abs_sum_bound_c (2 * Cb * Cinv / c1) (fun k => sigma * t1 k) dv uu
           (fun k => sqrt (sg k) * nrm (uu k))
           (fun k => sqrt (sg k) * nrm (vv k)) NU NV).
  - apply div_nonneg; nra.
  - intro k. pose proof (t1_pos' k). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (vv k)). nra.
  - intro k.
    pose proof (double_inv_v k) as HD.
    pose proof (L_T8c k) as H8.
    pose proof (t1_pos' k) as Ht1.
    pose proof (sg_nonneg' k) as Hsg.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (nrm_nonneg Hs (vv k)) as Hnv.
    pose proof (nrm_nonneg Hs (uu k)) as Hnu.
    pose proof (nrm_nonneg Hs (dv k)) as Hnd.
    assert (Hsgs : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt; lra).
    assert (Hst : 0 <= sigma * t1 k) by nra.
    assert (Hnn : 0 <= sigma * t1 k * nrm (uu k)) by nra.
    assert (S1 : sigma * t1 k * (nrm (dv k) * nrm (uu k))
                 <= sigma * t1 k
                    * ((2 * nu * Cb * Cinv * aK k / (hK k)^2 * nrm (vv k))
                       * nrm (uu k))) by nra.
    assert (E2 : sigma * t1 k * (2 * nu * Cb * Cinv * aK k / (hK k)^2)
                 = 2 * Cb * Cinv / c1
                   * (sigma * (c1 * nu * aK k / (hK k)^2) * t1 k))
      by (field; lra).
    assert (Hcc : 0 <= 2 * Cb * Cinv / c1) by (apply div_nonneg; nra).
    assert (Hnn2 : 0 <= nrm (vv k) * nrm (uu k)) by nra.
    assert (S2 : sigma * t1 k * (2 * nu * Cb * Cinv * aK k / (hK k)^2)
                   * (nrm (vv k) * nrm (uu k))
                 <= 2 * Cb * Cinv / c1 * sg k
                   * (nrm (vv k) * nrm (uu k))).
    { rewrite E2.
      assert (Hgap : 0 <= 2 * Cb * Cinv / c1 * (nrm (vv k) * nrm (uu k)))
        by nra.
      nra. }
    nra.
  - exact cmpU_s.
  - exact cmpV_s.
Qed.

(* ---------- Step 5: the split of T13 ------------------------------------------ *)

Definition T13c : R :=
  sigma * Rsum Th (fun k => t1 k * << (uu k) , (cxv k) >>).
Definition Rterm : R :=
  sigma * Rsum Th (fun k => t1 k * << (uu k) , (gpv k) >>).

Lemma T13_split : T13 = T13c + Rterm.
Proof.
  unfold T13, T13c, Rterm.
  rewrite <- Rmult_plus_distr_l. f_equal.
  rewrite <- Rsum_plus. apply Rsum_ext. intro k.
  unfold xv. rewrite ip_add_r. ring.
Qed.

Lemma bound_T13c : Rabs T13c <= Cinv / c2 * (NU * NV).
Proof.
  assert (E : T13c = Rsum Th (fun k => (sigma * t1 k)
                                        * << (uu k) , (cxv k) >>)).
  { unfold T13c. rewrite <- Rsum_scal. apply Rsum_ext. intro k. ring. }
  rewrite E.
  apply (abs_sum_bound_c (Cinv / c2) (fun k => sigma * t1 k) uu cxv
           (fun k => sqrt (sg k) * nrm (uu k))
           (fun k => sqrt (sg k) * nrm (vv k)) NU NV).
  - apply div_nonneg; nra.
  - intro k. pose proof (t1_pos' k). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (vv k)). nra.
  - intro k.
    pose proof (Hw_cxv k) as HC.
    pose proof (L_T13c k) as H13.
    pose proof (t1_pos' k) as Ht1.
    pose proof (sg_nonneg' k) as Hsg.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (am_nonneg k) as Hm.
    pose proof (nrm_nonneg Hs (uu k)) as Hnu.
    pose proof (nrm_nonneg Hs (vv k)) as Hnv.
    pose proof (nrm_nonneg Hs (cxv k)) as Hnc.
    assert (Hsgs : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt; lra).
    assert (Hst : 0 <= sigma * t1 k) by nra.
    assert (Hnn : 0 <= sigma * t1 k * nrm (uu k)) by nra.
    assert (S1 : sigma * t1 k * (nrm (uu k) * nrm (cxv k))
                 <= sigma * t1 k
                    * (nrm (uu k)
                       * (Cinv / hK k * aK k * am k * nrm (vv k)))) by nra.
    (*  sigma t1 (Cinv aK am / h) = (Cinv/c2) * (sigma t1 (c2 aK am / h))
        <= (Cinv/c2) sg  *)
    assert (E2 : sigma * t1 k * (Cinv / hK k * aK k * am k)
                 = Cinv / c2 * (sigma * t1 k * (c2 * aK k * am k / hK k)))
      by (field; lra).
    assert (Hcc : 0 <= Cinv / c2) by (apply div_nonneg; nra).
    assert (Hnn2 : 0 <= nrm (uu k) * nrm (vv k)) by nra.
    assert (S2 : sigma * t1 k * (Cinv / hK k * aK k * am k)
                   * (nrm (uu k) * nrm (vv k))
                 <= Cinv / c2 * sg k * (nrm (uu k) * nrm (vv k))).
    { rewrite E2.
      assert (Hgap : 0 <= Cinv / c2 * (nrm (uu k) * nrm (vv k))) by nra.
      nra. }
    nra.
  - exact cmpU_s.
  - exact cmpV_s.
Qed.

(* ========================================================================= *)
(*  Step 6a: exact rewriting of N = R + T2 + T4 + T11 (eq:fiveterms).        *)
(* ========================================================================= *)

Definition Iterm : R :=
  - sigma * Rsum Th (fun k => t1 k * << (vv k) , (gpu k) >>).
Definition IIterm : R :=
  sigma * Rsum Th (fun k => t1 k * << (uu k) , (xv k) >>).
Definition IIIterm : R :=
  - sigma * Rsum Th (fun k => t1 k * FBc_e k).
Definition IVterm : R := - Rsum Th (fun k => << (pp k) , (divv k) >>).
Definition Vterm : R := - Rsum Th (fun k => << (uu k) , (xv k) >>).

Lemma step6a : Rterm + T2 + T4 + T11 = Iterm + IIterm + IIIterm + IVterm + Vterm.
Proof.
  unfold Rterm, T2, T4, T11, Iterm, IIterm, IIIterm, IVterm, Vterm.
  (*  split the xu / xv pairings  *)
  assert (E2 : Rsum Th (fun k => << (vv k) , (xu k) >>)
               = Rsum Th (fun k => << (vv k) , (cxu k) >>)
                 + Rsum Th (fun k => << (vv k) , (gpu k) >>)).
  { rewrite <- Rsum_plus. apply Rsum_ext. intro k.
    unfold xu. rewrite ip_add_r. reflexivity. }
  assert (E11 : Rsum Th (fun k => t1 k * << (xu k) , (vv k) >>)
                = Rsum Th (fun k => t1 k * << (cxu k) , (vv k) >>)
                  + Rsum Th (fun k => t1 k * << (gpu k) , (vv k) >>)).
  { rewrite <- Rsum_plus. apply Rsum_ext. intro k.
    unfold xu. rewrite (ip_add_l Hs). ring. }
  assert (EII : Rsum Th (fun k => t1 k * << (uu k) , (xv k) >>)
                = Rsum Th (fun k => t1 k * << (uu k) , (cxv k) >>)
                  + Rsum Th (fun k => t1 k * << (uu k) , (gpv k) >>)).
  { rewrite <- Rsum_plus. apply Rsum_ext. intro k.
    unfold xv. rewrite ip_add_r. ring. }
  assert (EV : Rsum Th (fun k => << (uu k) , (xv k) >>)
               = Rsum Th (fun k => << (uu k) , (cxv k) >>)
                 + Rsum Th (fun k => << (uu k) , (gpv k) >>)).
  { rewrite <- Rsum_plus. apply Rsum_ext. intro k.
    unfold xv. rewrite ip_add_r. reflexivity. }
  (*  elemental convective integration by parts, weighted and summed  *)
  assert (Ec : Rsum Th (fun k => t1 k * << (cxu k) , (vv k) >>)
               = - Rsum Th (fun k => t1 k * << (cxv k) , (uu k) >>)
                 + Rsum Th (fun k => t1 k * FBc_e k)).
  { rewrite <- Rsum_opp, <- Rsum_plus. apply Rsum_ext. intro k.
    rewrite (H_elem_conv_ibp k). ring. }
  (*  symmetry swaps  *)
  assert (Esym1 : Rsum Th (fun k => t1 k * << (cxv k) , (uu k) >>)
                  = Rsum Th (fun k => t1 k * << (uu k) , (cxv k) >>)).
  { apply Rsum_ext. intro k. rewrite (ip_sym Hs (cxv k) (uu k)). ring. }
  assert (Esym2 : Rsum Th (fun k => t1 k * << (gpu k) , (vv k) >>)
                  = Rsum Th (fun k => t1 k * << (vv k) , (gpu k) >>)).
  { apply Rsum_ext. intro k. rewrite (ip_sym Hs (gpu k) (vv k)). ring. }
  rewrite E2, E11, EII, EV, Ec, Esym1, Esym2, H_skew, H_ibp_vp, H_ibp_qu.
  ring.
Qed.

(* ========================================================================= *)
(*  Step 6b: the pressure terms [I] + [IV].                                  *)
(* ========================================================================= *)

Definition VolP : R :=
  Rsum Th (fun k => (ph k * t1 k) * << (pp k) , (divv k) >>).
Definition JmpP : R := sigma * Rsum Fl (fun f => Dt1 f * FBp f).

Lemma step6b_identity : Iterm + IVterm = - VolP - JmpP.
Proof.
  unfold Iterm, IVterm, VolP, JmpP.
  (*  elemental pressure integration by parts, weighted and summed  *)
  assert (Ep : Rsum Th (fun k => t1 k * << (vv k) , (gpu k) >>)
               = - Rsum Th (fun k => t1 k * << (pp k) , (divv k) >>)
                 + Rsum Th (fun k => t1 k * FBp_e k)).
  { rewrite <- Rsum_opp, <- Rsum_plus. apply Rsum_ext. intro k.
    rewrite (H_elem_p_ibp k).
    rewrite (ip_sym Hs (divv k) (pp k)). ring. }
  rewrite Ep, H_assemble_p.
  (*  (1 - sigma tau1) = ph tau1 elementwise  *)
  assert (EW : Rsum Th (fun k => (ph k * t1 k) * << (pp k) , (divv k) >>)
               = Rsum Th (fun k => << (pp k) , (divv k) >>)
                 - sigma * Rsum Th (fun k => t1 k * << (pp k) , (divv k) >>)).
  { rewrite <- (Rsum_scal K Th sigma), <- Rsum_minus.
    apply Rsum_ext. intro k.
    pose proof (L_one_minus k) as H1.
    rewrite <- H1. ring. }
  unfold Dt1. lra.
Qed.

Lemma bound_VolP : Rabs VolP <= sqrt c1 * CI * (PsP * NV).
Proof.
  unfold VolP.
  apply (abs_sum_bound_c (sqrt c1 * CI) (fun k => ph k * t1 k) pp divv
           (fun k => aK k * sqrt (t1 k) / hK k * IP k)
           (fun k => sqrt (t2 k) * nrm (divv k)) PsP NV).
  - assert (0 <= sqrt c1) by apply sqrt_pos. nra.
  - intro k. pose proof (ph_pos k). pose proof (t1_pos' k). nra.
  - intro k. pose proof (hK_pos k). pose proof (aK_pos k).
    pose proof (IP_nonneg k).
    assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t1 k)) by nra.
    assert (0 <= aK k * sqrt (t1 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divv k)). nra.
  - intro k.
    pose proof (L_vol k) as HV.
    pose proof (L_P3div k) as HP3.
    pose proof (HI_pp k) as HPp.
    pose proof (IP_nonneg k) as HIP.
    pose proof (nrm_nonneg Hs (pp k)) as Hnp.
    pose proof (nrm_nonneg Hs (divv k)) as Hnd.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    assert (Hst1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (Hnn : 0 <= nrm (pp k) * nrm (divv k)) by nra.
    assert (S1 : ph k * t1 k * (nrm (pp k) * nrm (divv k))
                 <= sqrt (ph k) * sqrt (t1 k)
                    * (nrm (pp k) * nrm (divv k))) by nra.
    rewrite HP3 in S1.
    (*  replace nrm pp by CI * IP  *)
    assert (Hfac : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                        * nrm (divv k)).
    { assert (Hst2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
      assert (Hsc : 0 <= sqrt c1) by apply sqrt_pos.
      assert (H0 : 0 <= sqrt c1 * aK k) by nra.
      assert (H1 : 0 <= sqrt c1 * aK k * sqrt (t2 k)) by nra.
      assert (H2 : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k)
        by (apply div_nonneg; lra).
      assert (H3 : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k))
        by nra.
      nra. }
    assert (S2 : sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                   * (nrm (pp k) * nrm (divv k))
                 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                   * (CI * IP k * nrm (divv k))) by nra.
    (*  sqrt(t2)*sqrt(t1)*... regrouped into cc*(Aq*Bq)  *)
    assert (E : sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                  * (CI * IP k * nrm (divv k))
                = sqrt c1 * CI
                  * ((aK k * sqrt (t1 k) / hK k * IP k)
                     * (sqrt (t2 k) * nrm (divv k)))) by (field; lra).
    lra.
  - exact PsP_le.
  - exact cmpV_d.
Qed.

(*  The jump part: per-face estimate combining the four Jb bounds with the   *)
(*  face-integral hypothesis, then Cauchy--Schwarz over the faces.           *)
Definition PIk (k : K) : R := aK k * sqrt (t1 k) / hK k * IP k.
Definition VVs (k : K) : R := sqrt (sg k) * nrm (vv k).
Definition UIs (k : K) : R := sqrt (sg k) * IU k.

Lemma PIk_nonneg : forall k, 0 <= PIk k.
Proof.
  intro k. unfold PIk.
  pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IP_nonneg k).
  assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
  assert (0 <= aK k * sqrt (t1 k)) by nra.
  assert (0 <= aK k * sqrt (t1 k) / hK k) by (apply div_nonneg; lra).
  nra.
Qed.

Lemma VVs_nonneg : forall k, 0 <= VVs k.
Proof.
  intro k. unfold VVs.
  assert (0 <= sqrt (sg k)) by apply sqrt_pos.
  pose proof (nrm_nonneg Hs (vv k)). nra.
Qed.

Lemma UIs_nonneg : forall k, 0 <= UIs k.
Proof.
  intro k. unfold UIs.
  assert (0 <= sqrt (sg k)) by apply sqrt_pos.
  pose proof (IU_nonneg k). nra.
Qed.

(*  The sigma-weighted interpolation column is dominated by sqrt(c1) PsU :  *)
(*  per element  sg * IU^2 <= ph * IU^2 = c1 * aK^2 t2 / h^2 * IU^2.        *)
Lemma UIs_le : sqrt (Rsum Th (fun k => (UIs k)^2)) <= sqrt c1 * PsU.
Proof.
  assert (Hle : Rsum Th (fun k => (UIs k)^2) <= c1 * PsU2).
  { unfold PsU2. rewrite <- (Rsum_scal K Th c1).
    apply Rsum_le. intro k.
    unfold UIs.
    pose proof (L_sgph k) as Hsp.
    pose proof (L_phid k) as Hpd.
    pose proof (sg_nonneg' k) as Hsg.
    pose proof (ph_pos k) as Hph.
    pose proof (hK_pos k) as Hh.
    pose proof (IU_nonneg k) as HIU.
    assert (Hsgs : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt; lra).
    assert (E1 : (sqrt (sg k) * IU k)^2
                 = (sqrt (sg k) * sqrt (sg k)) * (IU k)^2) by ring.
    rewrite Hsgs in E1. rewrite E1.
    (*  sg * IU^2 <= ph * IU^2  and  ph = c1 aK^2 t2 / h^2 (divided form) *)
    assert (Hiu2 : 0 <= (IU k)^2) by (apply pow2_ge_0).
    assert (S1 : sg k * (IU k)^2 <= ph k * (IU k)^2) by nra.
    assert (E2 : c1 * ((aK k)^2 * t2 k / (hK k)^2 * (IU k)^2)
                 = (c1 * (aK k)^2 * t2 k) * (IU k)^2 / (hK k)^2)
      by (field; lra).
    assert (E3 : (c1 * (aK k)^2 * t2 k) * (IU k)^2 / (hK k)^2
                 = ph k * (IU k)^2).
    { rewrite <- Hpd. field. lra. }
    lra. }
  assert (H0 : 0 <= c1 * PsU2).
  { assert (0 <= PsU2).
    { apply Rsum_nonneg. intro k.
      pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
      pose proof (t2_pos' k) as Ht.
      assert (0 <= (IU k)^2) by apply pow2_ge_0.
      assert (0 <= (aK k)^2 * t2 k) by nra.
      assert (0 <= (aK k)^2 * t2 k / (hK k)^2).
      { apply div_nonneg; nra. }
      nra. }
    nra. }
  eapply Rle_trans; [apply sqrt_le_1_alt; exact Hle |].
  rewrite sqrt_mult_alt by lra.
  apply Rle_refl.
Qed.

Definition K6b : R := Cface * Nf * (kb11 + kb12 + kb21 + kb22).

Lemma kb_nonneg : 0 <= kb11 /\ 0 <= kb12 /\ 0 <= kb21 /\ 0 <= kb22.
Proof.
  pose proof kJ_nonneg as HkJ.
  assert (H1 : 0 <= sqrt cJ') by apply sqrt_pos.
  assert (H2 : 0 <= sqrt (cJ' / cJ)) by apply sqrt_pos.
  unfold kb11, kb12, kb21, kb22.
  assert (H3 : 0 <= kJ * sqrt cJ') by nra.
  repeat split; nra.
Qed.

(*  Per-face estimate of the jump term.  *)
Lemma face6b : forall f,
  sigma * Rabs (Dt1 f) * Rabs (FBp f)
  <= Cface * (  kb11 * (PIk (e1 f) * VVs (e1 f))
              + kb12 * (PIk (e2 f) * VVs (e1 f))
              + kb21 * (PIk (e1 f) * VVs (e2 f))
              + kb22 * (PIk (e2 f) * VVs (e2 f))).
Proof.
  intro f.
  pose proof (H_face_p f) as HF.
  pose proof (Jb11 f) as J11. pose proof (Jb12 f) as J12.
  pose proof (Jb21 f) as J21. pose proof (Jb22 f) as J22.
  pose proof (Rabs_pos (Dt1 f)) as HD.
  pose proof (Rabs_pos (FBp f)) as HFp.
  assert (HsD : 0 <= sigma * Rabs (Dt1 f)) by nra.
  (*  multiply the face bound by sigma |Dt1|  *)
  assert (S0 : sigma * Rabs (Dt1 f) * Rabs (FBp f)
               <= sigma * Rabs (Dt1 f)
                  * (Cface
                     * (aK (e1 f) / hK (e1 f)
                          * (nrm (vv (e1 f)) * IP (e1 f))
                        + aK (e2 f) / hK (e2 f)
                          * (nrm (vv (e1 f)) * IP (e2 f))
                        + aK (e1 f) / hK (e1 f)
                          * (nrm (vv (e2 f)) * IP (e1 f))
                        + aK (e2 f) / hK (e2 f)
                          * (nrm (vv (e2 f)) * IP (e2 f))))) by nra.
  (*  bound each of the four pieces via the corresponding Jb lemma  *)
  set (w11 := aK (e1 f) / hK (e1 f) * (nrm (vv (e1 f)) * IP (e1 f))).
  set (w12 := aK (e2 f) / hK (e2 f) * (nrm (vv (e1 f)) * IP (e2 f))).
  set (w21 := aK (e1 f) / hK (e1 f) * (nrm (vv (e2 f)) * IP (e1 f))).
  set (w22 := aK (e2 f) / hK (e2 f) * (nrm (vv (e2 f)) * IP (e2 f))).
  assert (Hah1 : 0 <= aK (e1 f) / hK (e1 f))
    by (apply div_nonneg; [apply Rlt_le, aK_pos | apply hK_pos]).
  assert (Hah2 : 0 <= aK (e2 f) / hK (e2 f))
    by (apply div_nonneg; [apply Rlt_le, aK_pos | apply hK_pos]).
  assert (Hn1 : 0 <= nrm (vv (e1 f))) by apply nrm_nonneg.
  assert (Hn2 : 0 <= nrm (vv (e2 f))) by apply nrm_nonneg.
  assert (Hp1 : 0 <= IP (e1 f)) by apply IP_nonneg.
  assert (Hp2 : 0 <= IP (e2 f)) by apply IP_nonneg.
  assert (Hq11 : 0 <= nrm (vv (e1 f)) * IP (e1 f)) by nra.
  assert (Hq12 : 0 <= nrm (vv (e1 f)) * IP (e2 f)) by nra.
  assert (Hq21 : 0 <= nrm (vv (e2 f)) * IP (e1 f)) by nra.
  assert (Hq22 : 0 <= nrm (vv (e2 f)) * IP (e2 f)) by nra.
  assert (Hw11 : 0 <= w11) by (unfold w11; nra).
  assert (Hw12 : 0 <= w12) by (unfold w12; nra).
  assert (Hw21 : 0 <= w21) by (unfold w21; nra).
  assert (Hw22 : 0 <= w22) by (unfold w22; nra).
  (*  each product: sigma|D| * wij <= kbij * PIk(ej) * VVs(ei)  *)
  assert (B11 : sigma * Rabs (Dt1 f) * w11
                <= kb11 * (PIk (e1 f) * VVs (e1 f))).
  { unfold PIk, VVs. unfold w11 in *. nra. }
  assert (B12 : sigma * Rabs (Dt1 f) * w12
                <= kb12 * (PIk (e2 f) * VVs (e1 f))).
  { unfold PIk, VVs. unfold w12 in *. nra. }
  assert (B21 : sigma * Rabs (Dt1 f) * w21
                <= kb21 * (PIk (e1 f) * VVs (e2 f))).
  { unfold PIk, VVs. unfold w21 in *. nra. }
  assert (B22 : sigma * Rabs (Dt1 f) * w22
                <= kb22 * (PIk (e2 f) * VVs (e2 f))).
  { unfold PIk, VVs. unfold w22 in *. nra. }
  fold w11 w12 w21 w22 in S0.
  assert (HCf : 0 <= Cface) by exact Cface_nonneg.
  nra.
Qed.

Lemma bound_JmpP : Rabs JmpP <= K6b * (PsP * NV).
Proof.
  unfold JmpP.
  rewrite Rabs_mult, (Rabs_right sigma) by lra.
  apply Rle_trans
    with (r2 := sigma * Rsum Fl (fun f => Rabs (Dt1 f * FBp f))).
  { apply Rmult_le_compat_l; [lra | apply Rsum_abs_le]. }
  assert (E : sigma * Rsum Fl (fun f => Rabs (Dt1 f * FBp f))
              = Rsum Fl (fun f => sigma * Rabs (Dt1 f) * Rabs (FBp f))).
  { rewrite <- Rsum_scal. apply Rsum_ext. intro f.
    rewrite Rabs_mult. ring. }
  rewrite E.
  eapply Rle_trans.
  { apply Rsum_le. intro f. apply face6b. }
  (*  distribute the face sum into the four (i,j) sums  *)
  assert (Esplit :
    Rsum Fl (fun f =>
      Cface * (  kb11 * (PIk (e1 f) * VVs (e1 f))
               + kb12 * (PIk (e2 f) * VVs (e1 f))
               + kb21 * (PIk (e1 f) * VVs (e2 f))
               + kb22 * (PIk (e2 f) * VVs (e2 f))))
    = Cface * (kb11 * Rsum Fl (fun f => PIk (e1 f) * VVs (e1 f))
               + kb12 * Rsum Fl (fun f => PIk (e2 f) * VVs (e1 f))
               + kb21 * Rsum Fl (fun f => PIk (e1 f) * VVs (e2 f))
               + kb22 * Rsum Fl (fun f => PIk (e2 f) * VVs (e2 f)))).
  { rewrite <- (Rsum_scal F Fl kb11), <- (Rsum_scal F Fl kb12),
            <- (Rsum_scal F Fl kb21), <- (Rsum_scal F Fl kb22).
    rewrite <- !Rsum_plus.
    rewrite <- (Rsum_scal F Fl Cface).
    apply Rsum_ext. intro f. ring. }
  rewrite Esplit.
  (*  each face sum by Cauchy--Schwarz with bounded multiplicity  *)
  assert (HB11 : Rsum Fl (fun f => PIk (e1 f) * VVs (e1 f))
                 <= Nf * (PsP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PIk VVs e1 e1 PIk_nonneg VVs_nonneg
               H_mult1 H_mult1). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PIk k)^2)) <= PsP)
      by (unfold PIk; exact PsP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PIk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof PsP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PIk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= PsP * NV) by nra.
    nra. }
  assert (HB12 : Rsum Fl (fun f => PIk (e2 f) * VVs (e1 f))
                 <= Nf * (PsP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PIk VVs e2 e1 PIk_nonneg VVs_nonneg
               H_mult2 H_mult1). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PIk k)^2)) <= PsP)
      by (unfold PIk; exact PsP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PIk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof PsP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PIk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= PsP * NV) by nra.
    nra. }
  assert (HB21 : Rsum Fl (fun f => PIk (e1 f) * VVs (e2 f))
                 <= Nf * (PsP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PIk VVs e1 e2 PIk_nonneg VVs_nonneg
               H_mult1 H_mult2). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PIk k)^2)) <= PsP)
      by (unfold PIk; exact PsP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PIk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof PsP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PIk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= PsP * NV) by nra.
    nra. }
  assert (HB22 : Rsum Fl (fun f => PIk (e2 f) * VVs (e2 f))
                 <= Nf * (PsP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PIk VVs e2 e2 PIk_nonneg VVs_nonneg
               H_mult2 H_mult2). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PIk k)^2)) <= PsP)
      by (unfold PIk; exact PsP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PIk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof PsP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PIk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= PsP * NV) by nra. 
    nra. }
  unfold K6b.
  destruct kb_nonneg as [Hb11 [Hb12 [Hb21 Hb22]]].
  pose proof PsP_nonneg. pose proof NV_nonneg.
  assert (HPN : 0 <= PsP * NV) by nra.
  assert (M11 : kb11 * Rsum Fl (fun f => PIk (e1 f) * VVs (e1 f))
                <= kb11 * (Nf * (PsP * NV))) by nra.
  assert (M12 : kb12 * Rsum Fl (fun f => PIk (e2 f) * VVs (e1 f))
                <= kb12 * (Nf * (PsP * NV))) by nra.
  assert (M21 : kb21 * Rsum Fl (fun f => PIk (e1 f) * VVs (e2 f))
                <= kb21 * (Nf * (PsP * NV))) by nra.
  assert (M22 : kb22 * Rsum Fl (fun f => PIk (e2 f) * VVs (e2 f))
                <= kb22 * (Nf * (PsP * NV))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Step 6c: the terms [II] + [V].                                           *)
(* ========================================================================= *)

Lemma bound_II_V : Rabs (IIterm + Vterm) <= sqrt c1 * CI * (PsU * NV).
Proof.
  assert (E : IIterm + Vterm
              = - Rsum Th (fun k => (ph k * t1 k) * << (uu k) , (xv k) >>)).
  { unfold IIterm, Vterm.
    rewrite <- (Rsum_scal K Th sigma).
    rewrite <- Rsum_opp.
    rewrite <- Rsum_plus.
    rewrite <- Rsum_opp.
    apply Rsum_ext. intro k.
    pose proof (L_one_minus k) as H1.
    rewrite <- H1. ring. }
  rewrite E, Rabs_Ropp.
  apply (abs_sum_bound_c (sqrt c1 * CI) (fun k => ph k * t1 k) uu xv
           (fun k => aK k * sqrt (t2 k) / hK k * IU k)
           (fun k => sqrt (t1 k) * nrm (xv k)) PsU NV).
  - assert (0 <= sqrt c1) by apply sqrt_pos. nra.
  - intro k. pose proof (ph_pos k). pose proof (t1_pos' k). nra.
  - intro k. pose proof (hK_pos k). pose proof (aK_pos k).
    pose proof (IU_nonneg k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xv k)). nra.
  - intro k.
    pose proof (L_vol k) as HV.
    pose proof (L_P3div k) as HP3.
    pose proof (HI_uu k) as HUu.
    pose proof (IU_nonneg k) as HIU.
    pose proof (nrm_nonneg Hs (uu k)) as Hnu.
    pose proof (nrm_nonneg Hs (xv k)) as Hnx.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    assert (Hst1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (Hnn : 0 <= nrm (uu k) * nrm (xv k)) by nra.
    assert (S1 : ph k * t1 k * (nrm (uu k) * nrm (xv k))
                 <= sqrt (ph k) * sqrt (t1 k)
                    * (nrm (uu k) * nrm (xv k))) by nra.
    rewrite HP3 in S1.
    assert (Hfac : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                        * nrm (xv k)).
    { assert (Hst2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
      assert (Hsc : 0 <= sqrt c1) by apply sqrt_pos.
      assert (H0 : 0 <= sqrt c1 * aK k) by nra.
      assert (H1 : 0 <= sqrt c1 * aK k * sqrt (t2 k)) by nra.
      assert (H2 : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k)
        by (apply div_nonneg; lra).
      assert (H3 : 0 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k))
        by nra.
      nra. }
    assert (S2 : sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                   * (nrm (uu k) * nrm (xv k))
                 <= sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                   * (CI * IU k * nrm (xv k))) by nra.
    assert (E2 : sqrt c1 * aK k * sqrt (t2 k) / hK k * sqrt (t1 k)
                   * (CI * IU k * nrm (xv k))
                 = sqrt c1 * CI
                   * ((aK k * sqrt (t2 k) / hK k * IU k)
                      * (sqrt (t1 k) * nrm (xv k)))) by (field; lra).
    lra.
  - exact PsU_le.
  - exact cmpV_x.
Qed.

(* ========================================================================= *)
(*  Step 6d: the inter-element term [III].                                   *)
(* ========================================================================= *)

Definition K6d : R := Cface * Nf * (kd11 + kd12 + kd21 + kd22).

Lemma kd_nonneg : 0 <= kd11 /\ 0 <= kd12 /\ 0 <= kd21 /\ 0 <= kd22.
Proof.
  pose proof kJ_nonneg as HkJ.
  assert (H1 : 0 <= sqrt (cJ' / cJ)) by apply sqrt_pos.
  unfold kd11, kd12, kd21, kd22.
  assert (H2 : 0 <= kJ / c2) by (apply div_nonneg; lra).
  assert (H3 : 0 <= kJ * cJ') by nra.
  assert (H4 : 0 <= kJ * cJ' / c2) by (apply div_nonneg; lra).
  assert (H5 : 0 <= sqrt (cJ' / cJ) * sqrt (cJ' / cJ)) by nra.
  repeat split; nra.
Qed.

Lemma face6d : forall f,
  sigma * Rabs (Dt1 f) * Rabs (FBc f)
  <= Cface * (  kd11 * (UIs (e1 f) * VVs (e1 f))
              + kd12 * (UIs (e1 f) * VVs (e2 f))
              + kd21 * (UIs (e2 f) * VVs (e1 f))
              + kd22 * (UIs (e2 f) * VVs (e2 f))).
Proof.
  intro f.
  pose proof (H_face_c f) as HF.
  pose proof (JD11 f) as J11. pose proof (JD12 f) as J12.
  pose proof (JD21 f) as J21. pose proof (JD22 f) as J22.
  pose proof (Rabs_pos (Dt1 f)) as HD.
  pose proof (Rabs_pos (FBc f)) as HFc.
  assert (HsD : 0 <= sigma * Rabs (Dt1 f)) by nra.
  assert (S0 : sigma * Rabs (Dt1 f) * Rabs (FBc f)
               <= sigma * Rabs (Dt1 f)
                  * (Cface
                     * (aK (e1 f) * am (e1 f) / hK (e1 f)
                          * (IU (e1 f) * nrm (vv (e1 f)))
                        + aK (e2 f) * am (e2 f) / hK (e2 f)
                          * (IU (e1 f) * nrm (vv (e2 f)))
                        + aK (e1 f) * am (e1 f) / hK (e1 f)
                          * (IU (e2 f) * nrm (vv (e1 f)))
                        + aK (e2 f) * am (e2 f) / hK (e2 f)
                          * (IU (e2 f) * nrm (vv (e2 f)))))) by nra.
  assert (Hn1 : 0 <= IU (e1 f)) by apply IU_nonneg.
  assert (Hn2 : 0 <= IU (e2 f)) by apply IU_nonneg.
  assert (Hv1 : 0 <= nrm (vv (e1 f))) by apply nrm_nonneg.
  assert (Hv2 : 0 <= nrm (vv (e2 f))) by apply nrm_nonneg.
  assert (Hq11 : 0 <= IU (e1 f) * nrm (vv (e1 f))) by nra.
  assert (Hq12 : 0 <= IU (e1 f) * nrm (vv (e2 f))) by nra.
  assert (Hq21 : 0 <= IU (e2 f) * nrm (vv (e1 f))) by nra.
  assert (Hq22 : 0 <= IU (e2 f) * nrm (vv (e2 f))) by nra.
  assert (B11 : sigma * Rabs (Dt1 f)
                  * (aK (e1 f) * am (e1 f) / hK (e1 f)
                     * (IU (e1 f) * nrm (vv (e1 f))))
                <= kd11 * (UIs (e1 f) * VVs (e1 f))).
  { unfold UIs, VVs. nra. }
  assert (B12 : sigma * Rabs (Dt1 f)
                  * (aK (e2 f) * am (e2 f) / hK (e2 f)
                     * (IU (e1 f) * nrm (vv (e2 f))))
                <= kd12 * (UIs (e1 f) * VVs (e2 f))).
  { unfold UIs, VVs. nra. }
  assert (B21 : sigma * Rabs (Dt1 f)
                  * (aK (e1 f) * am (e1 f) / hK (e1 f)
                     * (IU (e2 f) * nrm (vv (e1 f))))
                <= kd21 * (UIs (e2 f) * VVs (e1 f))).
  { unfold UIs, VVs. nra. }
  assert (B22 : sigma * Rabs (Dt1 f)
                  * (aK (e2 f) * am (e2 f) / hK (e2 f)
                     * (IU (e2 f) * nrm (vv (e2 f))))
                <= kd22 * (UIs (e2 f) * VVs (e2 f))).
  { unfold UIs, VVs. nra. }
  assert (HCf : 0 <= Cface) by exact Cface_nonneg.
  nra.
Qed.

Lemma bound_III : Rabs IIIterm <= K6d * (sqrt c1 * (PsU * NV)).
Proof.
  unfold IIIterm.
  rewrite Rabs_mult, Rabs_Ropp, (Rabs_right sigma) by lra.
  rewrite H_assemble_c.
  apply Rle_trans
    with (r2 := sigma
                * Rsum Fl (fun f => Rabs ((t1 (e1 f) - t1 (e2 f)) * FBc f))).
  { apply Rmult_le_compat_l; [lra | apply Rsum_abs_le]. }
  assert (E : sigma * Rsum Fl (fun f => Rabs ((t1 (e1 f) - t1 (e2 f)) * FBc f))
              = Rsum Fl (fun f => sigma * Rabs (Dt1 f) * Rabs (FBc f))).
  { rewrite <- Rsum_scal. apply Rsum_ext. intro f.
    unfold Dt1. rewrite Rabs_mult. ring. }
  rewrite E.
  eapply Rle_trans.
  { apply Rsum_le. intro f. apply face6d. }
  assert (Esplit :
    Rsum Fl (fun f =>
      Cface * (  kd11 * (UIs (e1 f) * VVs (e1 f))
               + kd12 * (UIs (e1 f) * VVs (e2 f))
               + kd21 * (UIs (e2 f) * VVs (e1 f))
               + kd22 * (UIs (e2 f) * VVs (e2 f))))
    = Cface * (kd11 * Rsum Fl (fun f => UIs (e1 f) * VVs (e1 f))
               + kd12 * Rsum Fl (fun f => UIs (e1 f) * VVs (e2 f))
               + kd21 * Rsum Fl (fun f => UIs (e2 f) * VVs (e1 f))
               + kd22 * Rsum Fl (fun f => UIs (e2 f) * VVs (e2 f)))).
  { rewrite <- (Rsum_scal F Fl kd11), <- (Rsum_scal F Fl kd12),
            <- (Rsum_scal F Fl kd21), <- (Rsum_scal F Fl kd22).
    rewrite <- !Rsum_plus.
    rewrite <- (Rsum_scal F Fl Cface).
    apply Rsum_ext. intro f. ring. }
  rewrite Esplit.
  assert (HU2 : sqrt (Rsum Th (fun k => (UIs k)^2)) <= sqrt c1 * PsU)
    by exact UIs_le.
  assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV).
  { unfold VVs. exact cmpV_s. }
  assert (Hprod : sqrt (Rsum Th (fun k => (UIs k)^2))
                  * sqrt (Rsum Th (fun k => (VVs k)^2))
                  <= sqrt c1 * PsU * NV).
  { assert (P1 : 0 <= sqrt (Rsum Th (fun k => (UIs k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    assert (Hsc : 0 <= sqrt c1) by apply sqrt_pos.
    pose proof PsU_nonneg. pose proof NV_nonneg.
    assert (Hscp : 0 <= sqrt c1 * PsU) by nra.
    nra. }
  assert (HB11 : Rsum Fl (fun f => UIs (e1 f) * VVs (e1 f))
                 <= Nf * (sqrt c1 * PsU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UIs VVs e1 e1 UIs_nonneg VVs_nonneg
               H_mult1 H_mult1). }
    nra. }
  assert (HB12 : Rsum Fl (fun f => UIs (e1 f) * VVs (e2 f))
                 <= Nf * (sqrt c1 * PsU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UIs VVs e1 e2 UIs_nonneg VVs_nonneg
               H_mult1 H_mult2). }
    nra. }
  assert (HB21 : Rsum Fl (fun f => UIs (e2 f) * VVs (e1 f))
                 <= Nf * (sqrt c1 * PsU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UIs VVs e2 e1 UIs_nonneg VVs_nonneg
               H_mult2 H_mult1). }
    nra. }
  assert (HB22 : Rsum Fl (fun f => UIs (e2 f) * VVs (e2 f))
                 <= Nf * (sqrt c1 * PsU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UIs VVs e2 e2 UIs_nonneg VVs_nonneg
               H_mult2 H_mult2). }
    nra. }
  unfold K6d.
  destruct kd_nonneg as [Hd11 [Hd12 [Hd21 Hd22]]].
  pose proof PsU_nonneg. pose proof NV_nonneg.
  assert (Hsc : 0 <= sqrt c1) by apply sqrt_pos.
  assert (Hsp : 0 <= sqrt c1 * PsU) by nra.
  assert (HPN : 0 <= sqrt c1 * PsU * NV) by nra.
  assert (M11 : kd11 * Rsum Fl (fun f => UIs (e1 f) * VVs (e1 f))
                <= kd11 * (Nf * (sqrt c1 * PsU * NV))) by nra.
  assert (M12 : kd12 * Rsum Fl (fun f => UIs (e1 f) * VVs (e2 f))
                <= kd12 * (Nf * (sqrt c1 * PsU * NV))) by nra.
  assert (M21 : kd21 * Rsum Fl (fun f => UIs (e2 f) * VVs (e1 f))
                <= kd21 * (Nf * (sqrt c1 * PsU * NV))) by nra.
  assert (M22 : kd22 * Rsum Fl (fun f => UIs (e2 f) * VVs (e2 f))
                <= kd22 * (Nf * (sqrt c1 * PsU * NV))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Step 8: assembly (eq:assembly, first-argument-interpolated form).       *)
(* ========================================================================= *)

Definition KUV_I : R :=
  2 + 1 + 1 + C2 + 2 * Cb / sqrt c1 + 1 + 2 * Cb * Cinv / c1
  + Cinv / c2 + sqrt C2 + sqrt C2 + 1.
Definition KPU_I : R :=
  4 * Cb * CI / c1 + 2 * CI / sqrt c1 + 2 * CI / sqrt c1
  + sqrt c1 * CI + K6d * sqrt c1.
Definition KPP_I : R := sqrt c1 * CI + K6b.

Theorem abstract_continterp_sharp :
  Rabs BS <= KUV_I * (NU * NV) + KPU_I * (PsU * NV) + KPP_I * (PsP * NV).
Proof.
  pose proof T13_split as E13.
  pose proof step6a as E6a.
  pose proof step6b_identity as E6b.
  assert (EBS : BS = T1 + (T3 + T14) + T5 + T15 + T6 + T7 + T8 + T9 + T10
                     + T12 + T16 + T17 + T18 + T13c
                     + (IIterm + Vterm) + IIIterm + (- VolP) + (- JmpP)).
  { unfold BS. lra. }
  rewrite EBS.
  pose proof bound_T1 as B1.
  pose proof bound_T3_T14 as B3.
  pose proof bound_T5 as B5.
  pose proof bound_T15 as B15.
  pose proof bound_T6 as B6.
  pose proof bound_T7 as B7.
  pose proof bound_T8 as B8.
  pose proof bound_T9 as B9.
  pose proof bound_T10 as B10.
  pose proof bound_T12 as B12.
  pose proof bound_T16 as B16.
  pose proof bound_T17 as B17.
  pose proof bound_T18 as B18.
  pose proof bound_T13c as B13.
  pose proof bound_II_V as B2V.
  pose proof bound_III as B3d.
  pose proof bound_VolP as BVp.
  pose proof bound_JmpP as BJp.
  (*  two-sided control of every group  *)
  pose proof (Rle_abs T1) as U1.
  pose proof (Rle_abs (- T1)) as W1. rewrite Rabs_Ropp in W1.
  pose proof (Rle_abs (T3 + T14)) as U2.
  pose proof (Rle_abs (- (T3 + T14))) as W2. rewrite Rabs_Ropp in W2.
  pose proof (Rle_abs T5) as U3.
  pose proof (Rle_abs (- T5)) as W3. rewrite Rabs_Ropp in W3.
  pose proof (Rle_abs T15) as U4.
  pose proof (Rle_abs (- T15)) as W4. rewrite Rabs_Ropp in W4.
  pose proof (Rle_abs T6) as U5.
  pose proof (Rle_abs (- T6)) as W5. rewrite Rabs_Ropp in W5.
  pose proof (Rle_abs T7) as U6.
  pose proof (Rle_abs (- T7)) as W6. rewrite Rabs_Ropp in W6.
  pose proof (Rle_abs T8) as U7.
  pose proof (Rle_abs (- T8)) as W7. rewrite Rabs_Ropp in W7.
  pose proof (Rle_abs T9) as U8.
  pose proof (Rle_abs (- T9)) as W8. rewrite Rabs_Ropp in W8.
  pose proof (Rle_abs T10) as U9.
  pose proof (Rle_abs (- T10)) as W9. rewrite Rabs_Ropp in W9.
  pose proof (Rle_abs T12) as U10.
  pose proof (Rle_abs (- T12)) as W10. rewrite Rabs_Ropp in W10.
  pose proof (Rle_abs T16) as U11.
  pose proof (Rle_abs (- T16)) as W11. rewrite Rabs_Ropp in W11.
  pose proof (Rle_abs T17) as U12.
  pose proof (Rle_abs (- T17)) as W12. rewrite Rabs_Ropp in W12.
  pose proof (Rle_abs T18) as U13.
  pose proof (Rle_abs (- T18)) as W13. rewrite Rabs_Ropp in W13.
  pose proof (Rle_abs T13c) as U14.
  pose proof (Rle_abs (- T13c)) as W14. rewrite Rabs_Ropp in W14.
  pose proof (Rle_abs (IIterm + Vterm)) as U15.
  pose proof (Rle_abs (- (IIterm + Vterm))) as W15. rewrite Rabs_Ropp in W15.
  pose proof (Rle_abs IIIterm) as U16.
  pose proof (Rle_abs (- IIIterm)) as W16. rewrite Rabs_Ropp in W16.
  pose proof (Rle_abs VolP) as U17.
  pose proof (Rle_abs (- VolP)) as W17. rewrite Rabs_Ropp in W17.
  pose proof (Rle_abs JmpP) as U18.
  pose proof (Rle_abs (- JmpP)) as W18. rewrite Rabs_Ropp in W18.
  apply Rabs_le. split.
  - unfold KUV_I, KPU_I, KPP_I. lra.
  - unfold KUV_I, KPU_I, KPP_I. lra.
Qed.

(* ========================================================================= *)
(*  Step 9 analogue: the triple norm of the interpolation error is bounded  *)
(*  by the l2 psi(h)  (eq:absorb1--eq:Enorm with the interpolation table).   *)
(* ========================================================================= *)

Definition KaggI : R :=
  sqrt (CI^2 + (sqrt c1 * CI)^2 + (sqrt c1 * CI)^2
        + (CI * sqrt c1 / c2 + CI)^2 + CI^2).

Lemma KaggI_nonneg : 0 <= KaggI.
Proof. apply sqrt_pos. Qed.

Lemma absorb_elem : forall k,
  perU k
  <= 2 * KaggI^2
     * ((aK k * sqrt (t2 k) / hK k * IU k)^2
        + (aK k * sqrt (t1 k) / hK k * IP k)^2).
Proof.
  intro k.
  set (Ak := aK k * sqrt (t2 k) / hK k * IU k).
  set (Bk := aK k * sqrt (t1 k) / hK k * IP k).
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (t1_pos' k) as Ht1. pose proof (t2_pos' k) as Ht2.
  pose proof (IU_nonneg k) as HIU. pose proof (IP_nonneg k) as HIP.
  pose proof sqrtc1_pos as Hsc.
  assert (Hst1 : 0 <= sqrt (t1 k)) by apply sqrt_pos.
  assert (Hst2 : 0 <= sqrt (t2 k)) by apply sqrt_pos.
  assert (HAk : 0 <= Ak).
  { unfold Ak.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra. }
  assert (HBk : 0 <= Bk).
  { unfold Bk.
    assert (0 <= aK k * sqrt (t1 k)) by nra.
    assert (0 <= aK k * sqrt (t1 k) / hK k) by (apply div_nonneg; lra).
    nra. }
  (*  P1: eq:interpgrad in place of eq:absorb1  *)
  assert (P1 : sqrt nu * nrm (gu k) <= CI * Ak).
  { pose proof (HI_gu k) as HG.
    pose proof (L_abs1 k) as HA1.
    assert (Hsn : 0 <= sqrt nu) by apply sqrt_pos.
    assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
    assert (S1 : sqrt nu * nrm (gu k)
                 <= sqrt nu * (CI * sqrt (aK k) / hK k * IU k)) by nra.
    assert (E1 : sqrt nu * (CI * sqrt (aK k) / hK k * IU k)
                 = CI / hK k * (sqrt nu * sqrt (aK k)) * IU k)
      by (field; lra).
    assert (Hci : 0 <= CI / hK k) by (apply div_nonneg; lra).
    assert (Hfac : 0 <= CI / hK k * IU k) by nra.
    assert (S2 : CI / hK k * (sqrt nu * sqrt (aK k)) * IU k
                 <= CI / hK k * (aK k * sqrt (t2 k)) * IU k) by nra.
    assert (E2 : CI / hK k * (aK k * sqrt (t2 k)) * IU k = CI * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  P2: eq:interpzero (m'=0, velocity) in place of eq:absorb2  *)
  assert (P2 : sqrt (sg k) * nrm (uu k) <= sqrt c1 * CI * Ak).
  { pose proof (HI_uu k) as HU.
    pose proof (L_abs2 k) as HA2.
    pose proof (L_P3div k) as HP3.
    assert (Hss : 0 <= sqrt (sg k)) by apply sqrt_pos.
    assert (Hsp : 0 <= sqrt (ph k)) by apply sqrt_pos.
    assert (S1 : sqrt (sg k) * nrm (uu k)
                 <= sqrt (sg k) * (CI * IU k)) by nra.
    assert (Hciu : 0 <= CI * IU k) by nra.
    assert (S2 : sqrt (sg k) * (CI * IU k)
                 <= sqrt (ph k) * (CI * IU k)) by nra.
    rewrite HP3 in S2.
    assert (E1 : sqrt c1 * aK k * sqrt (t2 k) / hK k * (CI * IU k)
                 = sqrt c1 * CI * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  P3: eq:interpzero (m'=0, pressure) in place of eq:absorb3  *)
  assert (P3 : sqrt eps * nrm (pp k) <= sqrt c1 * CI * Bk).
  { pose proof (HI_pp k) as HPq.
    pose proof (L_P5div k) as HP5.
    assert (Hse : 0 <= sqrt eps) by apply sqrt_pos.
    assert (S1 : sqrt eps * nrm (pp k) <= sqrt eps * (CI * IP k)) by nra.
    assert (Hcip : 0 <= CI * IP k) by nra.
    assert (S2 : sqrt eps * (CI * IP k)
                 <= sqrt c1 * aK k * sqrt (t1 k) / hK k * (CI * IP k)) by nra.
    assert (E1 : sqrt c1 * aK k * sqrt (t1 k) / hK k * (CI * IP k)
                 = sqrt c1 * CI * Bk).
    { unfold Bk. field. lra. }
    lra. }
  (*  P4: eq:interpfirst and the pressure-gradient replacement (eq:absorb4)  *)
  assert (P4 : sqrt (t1 k) * nrm (xu k)
               <= (CI * sqrt c1 / c2) * Ak + CI * Bk).
  { assert (Htri : nrm (xu k) <= nrm (cxu k) + nrm (gpu k)).
    { unfold xu. apply nrm_triangle. }
    pose proof (HI_cxu k) as HCx.
    pose proof (HI_gpu k) as HGp.
    pose proof (L_convph k) as HCp.
    pose proof (L_pst k) as HPst.
    pose proof (L_P3div k) as HP3.
    pose proof (nrm_nonneg Hs (cxu k)) as Hnc.
    pose proof (nrm_nonneg Hs (gpu k)) as Hng.
    pose proof (am_nonneg k) as Hm.
    pose proof (ph_pos k) as Hph.
    assert (Hci : 0 <= CI / c2) by (apply div_nonneg; lra).
    (*  velocity part  *)
    assert (V1 : sqrt (t1 k) * nrm (cxu k)
                 <= sqrt (t1 k) * (CI * aK k * am k / hK k * IU k)) by nra.
    assert (E1 : sqrt (t1 k) * (CI * aK k * am k / hK k * IU k)
                 = CI / c2 * (sqrt (t1 k) * (c2 * aK k * am k / hK k)) * IU k)
      by (field; lra).
    assert (Hfac : 0 <= CI / c2 * IU k) by nra.
    assert (Hfac2 : 0 <= CI / c2 * sqrt (t1 k) * IU k) by nra.
    assert (V2 : CI / c2 * (sqrt (t1 k) * (c2 * aK k * am k / hK k)) * IU k
                 <= CI / c2 * (sqrt (t1 k) * ph k) * IU k) by nra.
    assert (V3 : CI / c2 * (sqrt (t1 k) * ph k) * IU k
                 <= CI / c2 * sqrt (ph k) * IU k) by nra.
    assert (E2 : CI / c2 * sqrt (ph k) * IU k
                 = CI / c2 * sqrt (ph k) * IU k) by reflexivity.
    assert (V4 : CI / c2 * sqrt (ph k) * IU k
                 <= (CI * sqrt c1 / c2) * Ak).
    { rewrite HP3.
      assert (E3 : CI / c2 * (sqrt c1 * aK k * sqrt (t2 k) / hK k) * IU k
                   = (CI * sqrt c1 / c2) * Ak).
      { unfold Ak. field. lra. }
      lra. }
    (*  pressure part  *)
    assert (G1 : sqrt (t1 k) * nrm (gpu k)
                 <= sqrt (t1 k) * (CI * aK k / hK k * IP k)) by nra.
    assert (E4 : sqrt (t1 k) * (CI * aK k / hK k * IP k) = CI * Bk).
    { unfold Bk. field. lra. }
    nra. }
  (*  P5: the grad-div replacement (eq:absorb5 with interpolation)  *)
  assert (P5 : sqrt (t2 k) * nrm (divu k) <= CI * Ak).
  { pose proof (HI_divu k) as HD.
    assert (S1 : sqrt (t2 k) * nrm (divu k)
                 <= sqrt (t2 k) * (CI * aK k / hK k * IU k)) by nra.
    assert (E1 : sqrt (t2 k) * (CI * aK k / hK k * IU k) = CI * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  nonnegativity of the five pieces  *)
  assert (Hp1 : 0 <= sqrt nu * nrm (gu k)).
  { assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gu k)). nra. }
  assert (Hp2 : 0 <= sqrt (sg k) * nrm (uu k)).
  { assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)). nra. }
  assert (Hp3 : 0 <= sqrt eps * nrm (pp k)).
  { assert (0 <= sqrt eps) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (pp k)). nra. }
  assert (Hp4 : 0 <= sqrt (t1 k) * nrm (xu k)).
  { pose proof (nrm_nonneg Hs (xu k)). nra. }
  assert (Hp5 : 0 <= sqrt (t2 k) * nrm (divu k)).
  { pose proof (nrm_nonneg Hs (divu k)). nra. }
  assert (Hk1 : 0 <= CI) by exact CI_nonneg.
  assert (Hk2 : 0 <= sqrt c1 * CI).
  { assert (0 <= sqrt c1) by apply sqrt_pos. nra. }
  assert (Hk4 : 0 <= CI * sqrt c1 / c2).
  { assert (0 <= CI * sqrt c1) by nra.
    apply div_nonneg; lra. }
  pose proof (norm5_absorption
                (sqrt nu * nrm (gu k)) (sqrt (sg k) * nrm (uu k))
                (sqrt eps * nrm (pp k)) (sqrt (t1 k) * nrm (xu k))
                (sqrt (t2 k) * nrm (divu k))
                Ak Bk
                CI (sqrt c1 * CI) (sqrt c1 * CI) (CI * sqrt c1 / c2) CI CI
                Hp1 Hp2 Hp3 Hp4 Hp5 HAk HBk
                Hk1 Hk2 Hk2 Hk4 Hk1 Hk1
                P1 P2 P3 P4 P5) as HN5.
  fold KaggI in HN5.
  set (S5 := (sqrt nu * nrm (gu k))^2 + (sqrt (sg k) * nrm (uu k))^2
             + (sqrt eps * nrm (pp k))^2 + (sqrt (t1 k) * nrm (xu k))^2
             + (sqrt (t2 k) * nrm (divu k))^2).
  assert (HS5 : 0 <= S5).
  { unfold S5.
    pose proof (pow2_ge_0 (sqrt nu * nrm (gu k))).
    pose proof (pow2_ge_0 (sqrt (sg k) * nrm (uu k))).
    pose proof (pow2_ge_0 (sqrt eps * nrm (pp k))).
    pose proof (pow2_ge_0 (sqrt (t1 k) * nrm (xu k))).
    pose proof (pow2_ge_0 (sqrt (t2 k) * nrm (divu k))). lra. }
  assert (HKa : 0 <= KaggI) by apply sqrt_pos.
  assert (Hsq : S5 <= KaggI^2 * (Ak + Bk)^2).
  { fold S5 in HN5.
    assert (Hr : 0 <= KaggI * (Ak + Bk)) by nra.
    assert (Hss : sqrt S5 * sqrt S5
                  <= (KaggI * (Ak + Bk)) * (KaggI * (Ak + Bk))).
    { pose proof (sqrt_pos S5). nra. }
    rewrite sqrt_sqrt in Hss by exact HS5.
    nra. }
  assert (H2ab : (Ak + Bk)^2 <= 2 * Ak^2 + 2 * Bk^2).
  { pose proof (pow2_ge_0 (Ak - Bk)). nra. }
  assert (EperU : perU k = S5).
  { unfold perU, S5.
    rewrite (sq_weight nu (gu k)) by lra.
    rewrite (sq_weight (sg k) (uu k)) by apply sg_nonneg'.
    rewrite (sq_weight eps (pp k)) by lra.
    rewrite (sq_weight (t1 k) (xu k)) by (apply Rlt_le, t1_pos').
    rewrite (sq_weight (t2 k) (divu k)) by (apply Rlt_le, t2_pos').
    reflexivity. }
  rewrite EperU.
  fold Ak Bk.
  assert (HK2 : 0 <= KaggI^2) by (pose proof (pow2_ge_0 KaggI); lra).
  nra.
Qed.

(*  eq:Enorm, sharp l2 form:  |||E||| <= sqrt2 KaggI (PsU + PsP).            *)
Lemma interp_triple_norm : NU <= sqrt 2 * KaggI * (PsU + PsP).
Proof.
  unfold NU.
  assert (HKa : 0 <= KaggI) by apply sqrt_pos.
  assert (HPsU2 : 0 <= PsU2).
  { unfold PsU2. apply Rsum_nonneg. intro k.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t2_pos' k) as Ht.
    assert (0 <= (IU k)^2) by apply pow2_ge_0.
    assert (0 <= (aK k)^2 * t2 k) by nra.
    assert (0 <= (aK k)^2 * t2 k / (hK k)^2) by (apply div_nonneg; nra).
    nra. }
  assert (HPsP2 : 0 <= PsP2).
  { unfold PsP2. apply Rsum_nonneg. intro k.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t1_pos' k) as Ht.
    assert (0 <= (IP k)^2) by apply pow2_ge_0.
    assert (0 <= (aK k)^2 * t1 k) by nra.
    assert (0 <= (aK k)^2 * t1 k / (hK k)^2) by (apply div_nonneg; nra).
    nra. }
  assert (HN2 : NU2 <= 2 * KaggI^2 * (PsU2 + PsP2)).
  { unfold NU2.
    eapply Rle_trans.
    { apply Rsum_le. exact absorb_elem. }
    assert (E : Rsum Th (fun k =>
        2 * KaggI^2
        * ((aK k * sqrt (t2 k) / hK k * IU k)^2
           + (aK k * sqrt (t1 k) / hK k * IP k)^2))
        = 2 * KaggI^2
          * (Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * IU k)^2)
             + Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * IP k)^2))).
    { rewrite <- Rsum_plus, <- Rsum_scal.
      apply Rsum_ext. intro k. ring. }
    rewrite E.
    assert (EU : Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * IU k)^2)
                 = PsU2).
    { unfold PsU2. apply Rsum_ext. intro k.
      apply sq_weight2s; [apply Rlt_le, t2_pos' | apply hK_pos]. }
    assert (EP : Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * IP k)^2)
                 = PsP2).
    { unfold PsP2. apply Rsum_ext. intro k.
      apply sq_weight2s; [apply Rlt_le, t1_pos' | apply hK_pos]. }
    rewrite EU, EP. lra. }
  eapply Rle_trans.
  { apply sqrt_le_1_alt. exact HN2. }
  rewrite sqrt_mult_alt by (pose proof (pow2_ge_0 KaggI); lra).
  assert (E2 : sqrt (2 * KaggI^2) = sqrt 2 * KaggI).
  { rewrite sqrt_mult_alt by lra.
    rewrite sqrt_pow2 by exact HKa. reflexivity. }
  rewrite E2.
  assert (Hplus : sqrt (PsU2 + PsP2) <= PsU + PsP).
  { unfold PsU, PsP. apply sqrt_plus_le; assumption. }
  assert (Hm : 0 <= sqrt 2 * KaggI).
  { assert (0 <= sqrt 2) by apply sqrt_pos. nra. }
  nra.
Qed.

(* ========================================================================= *)
(*  lem:continterp, abstract form.                                           *)
(* ========================================================================= *)

Lemma KUV_I_nonneg : 0 <= KUV_I.
Proof.
  unfold KUV_I.
  pose proof sqrtc1_pos as Hsc.
  assert (H2 : 0 <= 2 * Cb / sqrt c1) by (apply div_nonneg; nra).
  assert (H3 : 0 <= 2 * Cb * Cinv / c1) by (apply div_nonneg; nra).
  assert (H4 : 0 <= Cinv / c2) by (apply div_nonneg; nra).
  assert (H5 : 0 <= sqrt C2) by apply sqrt_pos.
  lra.
Qed.

Lemma KPU_I_nonneg : 0 <= KPU_I.
Proof.
  unfold KPU_I.
  pose proof sqrtc1_pos as Hsc.
  assert (H1 : 0 <= 4 * Cb * CI / c1) by (apply div_nonneg; nra).
  assert (H2 : 0 <= 2 * CI / sqrt c1) by (apply div_nonneg; nra).
  assert (H3 : 0 <= sqrt c1 * CI) by nra.
  assert (H6 : 0 <= K6d).
  { unfold K6d.
    destruct kd_nonneg as [D1 [D2 [D3 D4]]].
    assert (0 <= Cface * Nf) by nra.
    nra. }
  assert (H7 : 0 <= K6d * sqrt c1) by nra.
  lra.
Qed.

Lemma KPP_I_nonneg : 0 <= KPP_I.
Proof.
  unfold KPP_I.
  pose proof sqrtc1_pos as Hsc.
  assert (H3 : 0 <= sqrt c1 * CI) by nra.
  assert (H6 : 0 <= K6b).
  { unfold K6b.
    destruct kb_nonneg as [D1 [D2 [D3 D4]]].
    assert (0 <= Cface * Nf) by nra.
    nra. }
  lra.
Qed.

Definition KabsI : R := sqrt 2 * KaggI.
Definition CtotI : R := KUV_I * KabsI + KPU_I + KPP_I.

Lemma KabsI_nonneg : 0 <= KabsI.
Proof.
  unfold KabsI.
  assert (0 <= sqrt 2) by apply sqrt_pos.
  pose proof KaggI_nonneg. nra.
Qed.

Lemma CtotI_nonneg : 0 <= CtotI.
Proof.
  unfold CtotI.
  pose proof KUV_I_nonneg. pose proof KPU_I_nonneg.
  pose proof KPP_I_nonneg. pose proof KabsI_nonneg.
  nra.
Qed.

Theorem abstract_continterp :
  Rabs BS <= CtotI * ((PsU + PsP) * NV).
Proof.
  pose proof abstract_continterp_sharp as HS.
  pose proof interp_triple_norm as H9.
  assert (H9' : NU <= KabsI * (PsU + PsP)) by (unfold KabsI; lra).
  pose proof KUV_I_nonneg as HK1.
  pose proof KPU_I_nonneg as HK2.
  pose proof KPP_I_nonneg as HK3.
  pose proof NV_nonneg as HNV.
  pose proof PsU_nonneg as HPU.
  pose proof PsP_nonneg as HPP.
  pose proof NU_nonneg as HNU.
  pose proof KabsI_nonneg as HKb.
  assert (HKN : 0 <= KUV_I * NV) by nra.
  assert (S1 : KUV_I * (NU * NV) <= KUV_I * (KabsI * (PsU + PsP) * NV))
    by nra.
  assert (HK2N : 0 <= KPU_I * NV) by nra.
  assert (S2 : KPU_I * (PsU * NV) <= KPU_I * ((PsU + PsP) * NV)) by nra.
  assert (HK3N : 0 <= KPP_I * NV) by nra.
  assert (S3 : KPP_I * (PsP * NV) <= KPP_I * ((PsU + PsP) * NV)) by nra.
  unfold CtotI. nra.
Qed.

(*  The appendix's l1 form of psi(h), via the discrete l2-in-l1 inequality.  *)
Corollary abstract_continterp_l1 :
  Rabs BS
  <= CtotI * (Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                                + aK k * sqrt (t1 k) / hK k * IP k) * NV).
Proof.
  pose proof abstract_continterp as HM.
  assert (HfU : forall k, 0 <= aK k * sqrt (t2 k) / hK k * IU k).
  { intro k.
    pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IU_nonneg k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra. }
  assert (HfP : forall k, 0 <= aK k * sqrt (t1 k) / hK k * IP k).
  { intro k.
    pose proof (hK_pos k). pose proof (aK_pos k). pose proof (IP_nonneg k).
    assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    assert (0 <= aK k * sqrt (t1 k)) by nra.
    assert (0 <= aK k * sqrt (t1 k) / hK k) by (apply div_nonneg; lra).
    nra. }
  assert (EU : Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * IU k)^2)
               = PsU2).
  { unfold PsU2. apply Rsum_ext. intro k.
    apply sq_weight2s; [apply Rlt_le, t2_pos' | apply hK_pos]. }
  assert (HU : PsU <= Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k)).
  { unfold PsU. rewrite <- EU. apply sqrt_sum_sq_le_sum. exact HfU. }
  assert (EP : Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * IP k)^2)
               = PsP2).
  { unfold PsP2. apply Rsum_ext. intro k.
    apply sq_weight2s; [apply Rlt_le, t1_pos' | apply hK_pos]. }
  assert (HP : PsP <= Rsum Th (fun k => aK k * sqrt (t1 k) / hK k * IP k)).
  { unfold PsP. rewrite <- EP. apply sqrt_sum_sq_le_sum. exact HfP. }
  assert (Esum : Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                                   + aK k * sqrt (t1 k) / hK k * IP k)
                 = Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k)
                   + Rsum Th (fun k => aK k * sqrt (t1 k) / hK k * IP k)).
  { apply Rsum_plus. }
  pose proof CtotI_nonneg as HCt.
  pose proof NV_nonneg as HNV.
  assert (HCN : 0 <= CtotI * NV) by nra.
  assert (Hpsi : PsU + PsP
                 <= Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                                      + aK k * sqrt (t1 k) / hK k * IP k))
    by lra.
  nra.
Qed.

End AbstractInterpolation.
