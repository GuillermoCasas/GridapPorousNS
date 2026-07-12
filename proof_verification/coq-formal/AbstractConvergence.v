(* ========================================================================= *)
(*  AbstractConvergence.v                                                    *)
(*                                                                           *)
(*  thm:convergence of the paper, proved IN FULL as the composition of the   *)
(*  two closed abstract theorems:                                            *)
(*                                                                           *)
(*    abstract_stability   (AbstractStability.v,  lemma:Stability /          *)
(*                          prop:stability)  applied to the discrete error   *)
(*                          W := Uhat_h - U_h, and                           *)
(*    abstract_continterp  (AbstractInterpolation.v, lem:continterp)         *)
(*                          applied to the pair (E, W), E := U - Uhat_h,     *)
(*                                                                           *)
(*  glued by (a) Galerkin orthogonality -- the single genuinely new trusted  *)
(*  item Horth, encoding consistency eq:consistency together with the        *)
(*  bilinearity of B_S in its first argument -- and (b) the triangle         *)
(*  inequality for the mesh-dependent triple norm, which is PROVED here      *)
(*  from the pre-Hilbert axioms (per-element five-fold Cauchy--Schwarz via   *)
(*  the discrete C--S over an explicit five-element index list, then the     *)
(*  discrete C--S over the mesh).                                            *)
(*                                                                           *)
(*  The appendix's proof is followed verbatim:                               *)
(*      C |||E_h|||^2 <= B_S(E_h,E_h) = B_S(Uhat-U, E_h) <= C psi |||E_h|||  *)
(*  and the triangle inequality; the conclusion is                           *)
(*                                                                           *)
(*    abstract_convergence :                                                 *)
(*        |||U - U_h||| <= (KabsI + CtotI / C_stab) * (PsiU + PsiP)          *)
(*                                                                           *)
(*  (the sharp l2 form of psi(h) in eq:convergence, porosity-weighted as in  *)
(*  rem:sharperconv), with the l1 form of the appendix as a corollary.       *)
(*                                                                           *)
(*  The W-family carries the hypotheses of BOTH prior files; note that       *)
(*  stability's inverse-estimate input S3 is literally the V-side hypothesis *)
(*  Hw_dv, and stability's eps-condition is the same eq:epscond (the two     *)
(*  closed formulas are definitionally bridged below), so the ONLY trusted   *)
(*  additions over the union of the two prior files are HBS_W (the tested    *)
(*  identity for the W-pair, the same class-G item as the stability lemma's  *)
(*  own HBS) and Horth.                                                      *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import StabilityAlgebra ContinuityAlgebra InnerSpace
                              AbstractSums AbstractStability
                              AbstractInterpolation.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  The two algebra modules define the parameters by the same formulas       *)
(*  (tau1, tau2) or by algebraically equal ones (sigt); the bridges.         *)
(* ========================================================================= *)

Lemma tau1_agree : forall nu h alphaK sigma amag c1 c2 : R,
  StabilityAlgebra.tau1 nu h alphaK sigma amag c1 c2
  = ContinuityAlgebra.tau1 nu h alphaK sigma amag c1 c2.
Proof.
  intros.
  unfold StabilityAlgebra.tau1, StabilityAlgebra.tau1NS_inv,
         ContinuityAlgebra.tau1, ContinuityAlgebra.phi1,
         ContinuityAlgebra.tauNSinv.
  reflexivity.
Qed.

Lemma tau2_agree : forall nu h alphaK amag c1 c2 : R,
  StabilityAlgebra.tau2 nu h alphaK amag c1 c2
  = ContinuityAlgebra.tau2 nu h alphaK amag c1 c2.
Proof.
  intros.
  unfold StabilityAlgebra.tau2, StabilityAlgebra.tau1NS_inv,
         ContinuityAlgebra.tau2, ContinuityAlgebra.tauNSinv.
  reflexivity.
Qed.

Lemma sigt_agree : forall nu h alphaK sigma amag c1 c2 : R,
  0 < nu -> 0 < h -> 0 < alphaK -> 0 <= sigma -> 0 <= amag ->
  0 < c1 -> 0 < c2 ->
  StabilityAlgebra.sigt nu h alphaK sigma amag c1 c2
  = ContinuityAlgebra.sigt nu h alphaK sigma amag c1 c2.
Proof.
  intros nu h alphaK sigma amag c1 c2 Hnu Hh Ha Hs Ham Hc1 Hc2.
  unfold StabilityAlgebra.sigt, StabilityAlgebra.tau1NS_inv,
         ContinuityAlgebra.sigt, ContinuityAlgebra.phi1,
         ContinuityAlgebra.tauNSinv.
  assert (Hh2 : 0 < h^2) by nra.
  assert (HX1 : 0 < c1 * nu / h^2)
    by (apply Rdiv_lt_0_compat; nra).
  assert (HX2 : 0 <= c2 * amag / h) by (apply div_nonneg; nra).
  assert (HX : 0 < c1 * nu / h^2 + c2 * amag / h) by lra.
  assert (HsA : 0 <= sigma / alphaK) by (apply div_nonneg; lra).
  assert (HD1 : 0 < c1 * nu / h^2 + c2 * amag / h + sigma / alphaK) by lra.
  assert (HD2 : 0 < alphaK * (c1 * nu / h^2 + c2 * amag / h) + sigma) by nra.
  assert (HG1 : 0 <= c2 * amag) by nra.
  assert (HG2 : 0 <= c2 * amag * h) by nra.
  assert (HG3 : 0 < c1 * nu + c2 * amag * h) by nra.
  assert (HG4 : 0 < alphaK * (c1 * nu + c2 * amag * h)) by nra.
  assert (HG5 : 0 <= sigma * (h * h)) by nra.
  assert (HG : 0 < alphaK * (c1 * nu + c2 * amag * h) + sigma * (h * h))
    by lra.
  field.
  repeat split; lra.
Qed.

Section AbstractConvergence.

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
Hypothesis CI_nonneg    : 0 <= CI.        (*  interpolation constant (eq:interp) *)

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

(*  The elemental interpolation error sizes (eq:Eint).  *)
Variables (IU IP : K -> R).
Hypothesis IU_nonneg : forall k, 0 <= IU k.
Hypothesis IP_nonneg : forall k, 0 <= IP k.

(* ---------- Face data --------------------------------------------------------- *)

Variable FBp : F -> R.   (*  int_{face} alpha (n . v) p                        *)
Variable FBc : F -> R.   (*  int_{face} alpha (n . a) (u . v)                  *)
Variable FBp_e : K -> R. (*  int_{bdry K} alpha (n_K . v) p                    *)
Variable FBc_e : K -> R. (*  int_{bdry K} alpha (n_K . a) (u . v)              *)

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

(*  eq:elemibp (identity (iii)).  *)
Hypothesis H_elem_conv_ibp :
  forall k, << (cxu k) , (vv k) >> = - << (cxv k) , (uu k) >> + FBc_e k.

(*  Elemental pressure integration by parts (Step 6b).  *)
Hypothesis H_elem_p_ibp :
  forall k, << (vv k) , (gpu k) >> = - << (divv k) , (pp k) >> + FBp_e k.

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

(*  eq:epscond, elementwise.  *)
Hypothesis H_eps :
  forall k, eps <= C2 * c1 * (aK k)^2 * t1 k / (hK k)^2.


(* ========================================================================= *)
(*  Additional data of the convergence theorem.                              *)
(* ========================================================================= *)

(*  The Young parameter and the strict positivity required by the stability  *)
(*  lemma (prop:stability).                                                  *)
Variable xi : R.
Hypothesis xi_large  : xi > 2.
Hypothesis c1_large  : c1 > 2 * xi * Cb^2.
Hypothesis sigma_pos : 0 < sigma.
Hypothesis eps_pos   : 0 < eps.
Hypothesis C2_pos    : 0 < C2.

Lemma xi_pos_l : 0 < xi.
Proof. lra. Qed.

(*  Positivity of the parameters, elementwise.  *)
Lemma t1_pos' : forall k, 0 < t1 k.
Proof. intro k. unfold t1. apply tau1_pos; auto. Qed.
Lemma t2_pos' : forall k, 0 < t2 k.
Proof. intro k. unfold t2. apply tau2_pos; auto. Qed.
Lemma sg_nonneg' : forall k, 0 <= sg k.
Proof. intro k. unfold sg. apply sigt_nonneg; auto. Qed.

(*  The x-components, as the closed partial applications of the              *)
(*  interpolation file (so that all norms below share their atoms).          *)
Definition xe : K -> V := AbstractInterpolation.xu Hs K cxu gpu.
Definition xw : K -> V := AbstractInterpolation.xv Hs K cxv gpv.

(* ========================================================================= *)
(*  The trusted additions.                                                   *)
(* ========================================================================= *)

(*  The tested identity for the discrete-error pair (eq:StabilityEstimate    *)
(*  with U_h := W): the same class-G item as the stability lemma's HBS.      *)
Variable BSWW : R.
Hypothesis HBS_W :
  BSWW =
  2 * nu * Rsum Th (fun k => << (gv k) , (gv k) >>)
  + sigma * Rsum Th (fun k => << (vv k) , (vv k) >>)
  + eps * Rsum Th (fun k => << (qq k) , (qq k) >>)
  - Rsum Th (fun k =>
      AbstractStability.t1 K nu sigma c1 c2 hK aK am k
      * << (AbstractStability.Lstar_m Hs K sigma vv xw dv k)
         , (AbstractStability.L_m Hs K sigma vv xw dv k) >>)
  - Rsum Th (fun k =>
      AbstractStability.t2 K nu c1 c2 hK aK am k
      * << (AbstractStability.Lstar_c Hs K eps qq divv k)
         , (AbstractStability.L_c Hs K eps qq divv k) >>).

(*  Galerkin orthogonality (eq:consistency) together with bilinearity of     *)
(*  B_S in its first argument:  B_S(W,W) = B_S(W - (U - U_h), W) = -B_S(E,W).*)
Definition BSEW : R :=
  AbstractInterpolation.BS Hs K Th nu sigma eps c1 c2 hK aK am
    gu gv du dv cxu cxv gpu gpv uu vv pp qq divu divv.
Hypothesis Horth : BSWW = - BSEW.

(* ========================================================================= *)
(*  The quantities of the convergence estimate.                              *)
(* ========================================================================= *)

Definition Psi : R :=
  AbstractInterpolation.PsU K Th nu c1 c2 hK aK am IU
  + AbstractInterpolation.PsP K Th nu sigma c1 c2 hK aK am IP.
Definition NE : R :=
  AbstractInterpolation.NU Hs K Th nu sigma eps c1 c2 hK aK am
    gu cxu gpu uu pp divu.
Definition NW : R :=
  AbstractInterpolation.NV Hs K Th nu sigma eps c1 c2 hK aK am
    gv cxv gpv vv qq divv.
Definition Cst : R := StabilityAlgebra.C_stab c1 Cb xi C2.
Definition Ct : R :=
  AbstractInterpolation.CtotI c1 c2 C2 Cinv Cb cJ cJ' Cface Nf CI.
Definition Kb : R := AbstractInterpolation.KabsI c1 c2 CI.

Lemma Psi_nonneg : 0 <= Psi.
Proof.
  unfold Psi.
  pose proof (AbstractInterpolation.PsU_nonneg K Th nu c1 c2 hK aK am IU).
  pose proof (AbstractInterpolation.PsP_nonneg K Th nu sigma c1 c2 hK aK am IP).
  lra.
Qed.

Lemma Ct_nonneg : 0 <= Ct.
Proof.
  unfold Ct.
  exact (AbstractInterpolation.CtotI_nonneg c1 c2 C2 Cinv Cb cJ cJ' Cface Nf CI
           c1_pos c2_pos C2_nonneg Cinv_pos Cb_pos cJ_pos one_le_cJ'
           Cface_nonneg Nf_nonneg CI_nonneg).
Qed.

(* ========================================================================= *)
(*  Per-element energies and their nonnegativity.                            *)
(* ========================================================================= *)

Definition perE : K -> R :=
  AbstractInterpolation.perU Hs K nu sigma eps c1 c2 hK aK am
    gu cxu gpu uu pp divu.
Definition perW : K -> R :=
  AbstractInterpolation.perV Hs K nu sigma eps c1 c2 hK aK am
    gv cxv gpv vv qq divv.

Lemma perE_nonneg : forall k, 0 <= perE k.
Proof.
  intro k.
  unfold perE, AbstractInterpolation.perU. cbv beta.
  unfold AbstractInterpolation.sg, AbstractInterpolation.t1,
         AbstractInterpolation.t2, AbstractInterpolation.xu. cbv beta.
  pose proof (tau1_pos nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Ht1.
  pose proof (tau2_pos nu (hK k) (aK k) (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) (am_nonneg k)
                c1_pos c2_pos) as Ht2.
  pose proof (sigt_nonneg nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Hsg.
  pose proof (ip_pos Hs (gu k)) as P1.
  pose proof (ip_pos Hs (uu k)) as P2.
  pose proof (ip_pos Hs (pp k)) as P3.
  pose proof (ip_pos Hs (vadd Hs (cxu k) (gpu k))) as P4.
  pose proof (ip_pos Hs (divu k)) as P5.
  assert (Q1 : 0 <= nu * ip Hs (gu k) (gu k)) by nra.
  assert (Q2 : 0 <= ContinuityAlgebra.sigt nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (uu k) (uu k)) by nra.
  assert (Q3 : 0 <= eps * ip Hs (pp k) (pp k)) by nra.
  assert (Q4 : 0 <= ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (vadd Hs (cxu k) (gpu k))
                           (vadd Hs (cxu k) (gpu k))) by nra.
  assert (Q5 : 0 <= ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2
                    * ip Hs (divu k) (divu k)) by nra.
  lra.
Qed.

Lemma perW_nonneg : forall k, 0 <= perW k.
Proof.
  intro k.
  unfold perW, AbstractInterpolation.perV. cbv beta.
  unfold AbstractInterpolation.sg, AbstractInterpolation.t1,
         AbstractInterpolation.t2, AbstractInterpolation.xv. cbv beta.
  pose proof (tau1_pos nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Ht1.
  pose proof (tau2_pos nu (hK k) (aK k) (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) (am_nonneg k)
                c1_pos c2_pos) as Ht2.
  pose proof (sigt_nonneg nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Hsg.
  pose proof (ip_pos Hs (gv k)) as P1.
  pose proof (ip_pos Hs (vv k)) as P2.
  pose proof (ip_pos Hs (qq k)) as P3.
  pose proof (ip_pos Hs (vadd Hs (cxv k) (gpv k))) as P4.
  pose proof (ip_pos Hs (divv k)) as P5.
  assert (Q1 : 0 <= nu * ip Hs (gv k) (gv k)) by nra.
  assert (Q2 : 0 <= ContinuityAlgebra.sigt nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (vv k) (vv k)) by nra.
  assert (Q3 : 0 <= eps * ip Hs (qq k) (qq k)) by nra.
  assert (Q4 : 0 <= ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (vadd Hs (cxv k) (gpv k))
                           (vadd Hs (cxv k) (gpv k))) by nra.
  assert (Q5 : 0 <= ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2
                    * ip Hs (divv k) (divv k)) by nra.
  lra.
Qed.

(* ========================================================================= *)
(*  Coercivity of the W-energy: prop:stability applied to W.                 *)
(* ========================================================================= *)

Lemma Heps_W :
  forall k, eps <= StabilityAlgebra.eps_max nu (hK k) (aK k) sigma (am k)
                     c1 c2 C2.
Proof.
  intro k.
  pose proof (H_eps k) as H. unfold t1 in H.
  unfold StabilityAlgebra.eps_max.
  rewrite (tau1_agree nu (hK k) (aK k) sigma (am k) c1 c2).
  exact H.
Qed.

Lemma NW2_bridge :
  AbstractStability.NormSq Hs K Th nu sigma eps c1 c2 hK aK am gv vv qq xw divv
  = NW ^ 2.
Proof.
  assert (HNV2 : 0 <= AbstractInterpolation.NV2 Hs K Th nu sigma eps c1 c2
                        hK aK am gv cxv gpv vv qq divv).
  { unfold AbstractInterpolation.NV2. apply Rsum_nonneg. intro k.
    exact (perW_nonneg k). }
  assert (E2 : NW ^ 2 = AbstractInterpolation.NV2 Hs K Th nu sigma eps c1 c2
                          hK aK am gv cxv gpv vv qq divv).
  { unfold NW, AbstractInterpolation.NV.
    replace ((sqrt (AbstractInterpolation.NV2 Hs K Th nu sigma eps c1 c2
                      hK aK am gv cxv gpv vv qq divv)) ^ 2)
      with (sqrt (AbstractInterpolation.NV2 Hs K Th nu sigma eps c1 c2
                    hK aK am gv cxv gpv vv qq divv)
            * sqrt (AbstractInterpolation.NV2 Hs K Th nu sigma eps c1 c2
                      hK aK am gv cxv gpv vv qq divv)) by ring.
    apply sqrt_sqrt. exact HNV2. }
  rewrite E2.
  unfold AbstractStability.NormSq, AbstractInterpolation.NV2.
  apply Rsum_ext. intro k.
  unfold AbstractStability.perNorm,
         AbstractStability.Pn, AbstractStability.Vn, AbstractStability.Un,
         AbstractStability.Xn, AbstractStability.Dn,
         AbstractInterpolation.perV. cbv beta.
  unfold AbstractStability.sg, AbstractStability.t1, AbstractStability.t2,
         AbstractInterpolation.sg, AbstractInterpolation.t1,
         AbstractInterpolation.t2, AbstractInterpolation.xv, xw,
         AbstractInterpolation.xv. cbv beta.
  rewrite (tau1_agree nu (hK k) (aK k) sigma (am k) c1 c2).
  rewrite (tau2_agree nu (hK k) (aK k) (am k) c1 c2).
  rewrite (sigt_agree nu (hK k) (aK k) sigma (am k) c1 c2
             nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
             c1_pos c2_pos).
  ring.
Qed.

Lemma stab_W : 0 < Cst /\ BSWW >= Cst * NW ^ 2.
Proof.
  pose proof (abstract_stability Hs K Th nu sigma eps c1 c2 Cb xi C2
                nu_pos sigma_pos eps_pos c1_pos c2_pos Cb_pos xi_pos_l C2_pos
                C2_lt_1 c1_large xi_large
                hK aK am hK_pos aK_pos am_nonneg
                gv vv qq xw divv dv BSWW HBS_W Hw_dv Heps_W) as [H1 H2].
  split; [exact H1 |].
  rewrite NW2_bridge in H2. exact H2.
Qed.

Lemma Cst_pos : 0 < Cst.
Proof. destruct stab_W; assumption. Qed.

(* ========================================================================= *)
(*  Continuity at the interpolation error: lem:continterp applied to (E,W).  *)
(* ========================================================================= *)

Lemma cont_EW : Rabs BSEW <= Ct * (Psi * NW).
Proof.
  unfold BSEW, Ct, Psi, NW.
  exact (abstract_continterp Hs K Th F Fl e1 e2
           nu sigma eps c1 c2 C2 Cinv Cb cJ cJ' Cface Nf CI
           nu_pos sigma_nonneg eps_nonneg c1_pos c2_pos C2_nonneg C2_lt_1
           Cinv_pos Cb_pos cJ_pos cJ_le_1 one_le_cJ' Cface_nonneg Nf_nonneg
           CI_nonneg
           hK aK am hK_pos aK_pos am_nonneg
           gu gv du dv cxu cxv gpu gpv uu vv pp qq divu divv
           IU IP IU_nonneg IP_nonneg FBp FBc FBp_e FBc_e
           H_skew H_ibp_vp H_ibp_qu H_elem_conv_ibp H_elem_p_ibp
           H_assemble_p H_assemble_c H_face_p H_face_c H_jump H_mult1 H_mult2
           Hw_gv Hw_dv Hw_cxv HI_gu HI_du HI_cxu HI_gpu HI_divu HI_uu HI_pp
           H_eps).
Qed.

(* ========================================================================= *)
(*  The discrete-error bound:  |||W||| <= (CtotI / C_stab) psi.              *)
(* ========================================================================= *)

Lemma NW_nonneg : 0 <= NW.
Proof.
  unfold NW.
  exact (AbstractInterpolation.NV_nonneg Hs K Th nu sigma eps c1 c2
           hK aK am gv cxv gpv vv qq divv).
Qed.

Lemma NW_bound : NW <= Ct / Cst * Psi.
Proof.
  destruct stab_W as [HCst HCoer].
  pose proof cont_EW as HC.
  pose proof Psi_nonneg as HPsi.
  pose proof NW_nonneg as HNW0.
  pose proof Ct_nonneg as HCt.
  assert (Hch : Cst * NW ^ 2 <= Ct * (Psi * NW)).
  { pose proof (Rle_abs (- BSEW)) as HA. rewrite Rabs_Ropp in HA.
    rewrite <- Horth in HA. lra. }
  destruct (Rle_lt_or_eq_dec 0 NW HNW0) as [Hpos | Heq0].
  - assert (H1 : Cst * NW <= Ct * Psi).
    { apply (Rmult_le_reg_r NW); [exact Hpos |].
      replace (Cst * NW * NW) with (Cst * NW ^ 2) by ring.
      replace (Ct * Psi * NW) with (Ct * (Psi * NW)) by ring.
      exact Hch. }
    apply (Rmult_le_reg_l Cst); [exact HCst |].
    replace (Cst * (Ct / Cst * Psi)) with (Ct * Psi) by (field; lra).
    exact H1.
  - rewrite <- Heq0.
    assert (0 <= Ct / Cst) by (apply div_nonneg; lra).
    nra.
Qed.

(* ========================================================================= *)
(*  The interpolation-error bound: eq:Enorm.                                 *)
(* ========================================================================= *)

Lemma NE_bound : NE <= Kb * Psi.
Proof.
  pose proof (interp_triple_norm Hs K Th nu sigma eps c1 c2 C2 CI
                nu_pos sigma_nonneg eps_nonneg c1_pos c2_pos C2_lt_1 CI_nonneg
                hK aK am hK_pos aK_pos am_nonneg
                gu cxu gpu uu pp divu IU IP IU_nonneg IP_nonneg
                HI_gu HI_cxu HI_gpu HI_divu HI_uu HI_pp H_eps) as H.
  unfold Kb, AbstractInterpolation.KabsI, NE, Psi.
  exact H.
Qed.

(* ========================================================================= *)
(*  The triangle inequality for the triple norm, proved.                     *)
(* ========================================================================= *)

Definition Qk (k : K) : R :=
  nu * << (gu k) , (gv k) >>
  + sg k * << (uu k) , (vv k) >>
  + eps * << (pp k) , (qq k) >>
  + t1 k * << (xe k) , (xw k) >>
  + t2 k * << (divu k) , (divv k) >>.

Definition perErr (k : K) : R :=
  nu * << (gu k) +v (gv k) , (gu k) +v (gv k) >>
  + sg k * << (uu k) +v (vv k) , (uu k) +v (vv k) >>
  + eps * << (pp k) +v (qq k) , (pp k) +v (qq k) >>
  + t1 k * << (xe k) +v (xw k) , (xe k) +v (xw k) >>
  + t2 k * << (divu k) +v (divv k) , (divu k) +v (divv k) >>.

Definition NErr2 : R := Rsum Th perErr.
Definition NErr : R := sqrt NErr2.

Lemma perErr_expand : forall k, perErr k = perE k + 2 * Qk k + perW k.
Proof.
  intro k.
  unfold perErr, perE, perW, Qk,
         AbstractInterpolation.perU, AbstractInterpolation.perV. cbv beta.
  unfold sg, t1, t2,
         AbstractInterpolation.sg, AbstractInterpolation.t1,
         AbstractInterpolation.t2,
         xe, xw, AbstractInterpolation.xu, AbstractInterpolation.xv. cbv beta.
  rewrite (ip_expand_add Hs (gu k) (gv k)).
  rewrite (ip_expand_add Hs (uu k) (vv k)).
  rewrite (ip_expand_add Hs (pp k) (qq k)).
  rewrite (ip_expand_add Hs (vadd Hs (cxu k) (gpu k))
                            (vadd Hs (cxv k) (gpv k))).
  rewrite (ip_expand_add Hs (divu k) (divv k)).
  ring.
Qed.

Lemma perErr_nonneg : forall k, 0 <= perErr k.
Proof.
  intro k.
  unfold perErr.
  unfold sg, t1, t2.
  pose proof (tau1_pos nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Ht1.
  pose proof (tau2_pos nu (hK k) (aK k) (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) (am_nonneg k)
                c1_pos c2_pos) as Ht2.
  pose proof (sigt_nonneg nu (hK k) (aK k) sigma (am k) c1 c2
                nu_pos (hK_pos k) (aK_pos k) sigma_nonneg (am_nonneg k)
                c1_pos c2_pos) as Hsg.
  pose proof (ip_pos Hs (vadd Hs (gu k) (gv k))) as P1.
  pose proof (ip_pos Hs (vadd Hs (uu k) (vv k))) as P2.
  pose proof (ip_pos Hs (vadd Hs (pp k) (qq k))) as P3.
  pose proof (ip_pos Hs (vadd Hs (xe k) (xw k))) as P4.
  pose proof (ip_pos Hs (vadd Hs (divu k) (divv k))) as P5.
  assert (Q1 : 0 <= nu * ip Hs (vadd Hs (gu k) (gv k))
                         (vadd Hs (gu k) (gv k))) by nra.
  assert (Q2 : 0 <= ContinuityAlgebra.sigt nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (vadd Hs (uu k) (vv k))
                            (vadd Hs (uu k) (vv k))) by nra.
  assert (Q3 : 0 <= eps * ip Hs (vadd Hs (pp k) (qq k))
                          (vadd Hs (pp k) (qq k))) by nra.
  assert (Q4 : 0 <= ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2
                    * ip Hs (vadd Hs (xe k) (xw k))
                            (vadd Hs (xe k) (xw k))) by nra.
  assert (Q5 : 0 <= ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2
                    * ip Hs (vadd Hs (divu k) (divv k))
                            (vadd Hs (divu k) (divv k))) by nra.
  lra.
Qed.

Lemma Qk_CS : forall k, Qk k <= sqrt (perE k) * sqrt (perW k).
Proof.
  intro k.
  pose proof (t1_pos' k) as Ht1.
  pose proof (t2_pos' k) as Ht2.
  pose proof (sg_nonneg' k) as Hsg.
  set (fE := fun i : nat => match i with
    | 0%nat => sqrt nu * nrm (gu k)
    | 1%nat => sqrt (sg k) * nrm (uu k)
    | 2%nat => sqrt eps * nrm (pp k)
    | 3%nat => sqrt (t1 k) * nrm (xe k)
    | 4%nat => sqrt (t2 k) * nrm (divu k)
    | _ => 0 end).
  set (fW := fun i : nat => match i with
    | 0%nat => sqrt nu * nrm (gv k)
    | 1%nat => sqrt (sg k) * nrm (vv k)
    | 2%nat => sqrt eps * nrm (qq k)
    | 3%nat => sqrt (t1 k) * nrm (xw k)
    | 4%nat => sqrt (t2 k) * nrm (divv k)
    | _ => 0 end).
  assert (HfE : forall i, 0 <= fE i).
  { intro i. unfold fE.
    destruct i as [|[|[|[|[|i]]]]];
      try (assert (0 <= sqrt nu) by apply sqrt_pos;
           assert (0 <= sqrt (sg k)) by apply sqrt_pos;
           assert (0 <= sqrt eps) by apply sqrt_pos;
           assert (0 <= sqrt (t1 k)) by apply sqrt_pos;
           assert (0 <= sqrt (t2 k)) by apply sqrt_pos;
           pose proof (nrm_nonneg Hs (gu k));
           pose proof (nrm_nonneg Hs (uu k));
           pose proof (nrm_nonneg Hs (pp k));
           pose proof (nrm_nonneg Hs (xe k));
           pose proof (nrm_nonneg Hs (divu k));
           nra);
      lra. }
  assert (HfW : forall i, 0 <= fW i).
  { intro i. unfold fW.
    destruct i as [|[|[|[|[|i]]]]];
      try (assert (0 <= sqrt nu) by apply sqrt_pos;
           assert (0 <= sqrt (sg k)) by apply sqrt_pos;
           assert (0 <= sqrt eps) by apply sqrt_pos;
           assert (0 <= sqrt (t1 k)) by apply sqrt_pos;
           assert (0 <= sqrt (t2 k)) by apply sqrt_pos;
           pose proof (nrm_nonneg Hs (gv k));
           pose proof (nrm_nonneg Hs (vv k));
           pose proof (nrm_nonneg Hs (qq k));
           pose proof (nrm_nonneg Hs (xw k));
           pose proof (nrm_nonneg Hs (divv k));
           nra);
      lra. }
  pose proof (Rsum_CS nat (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat fE fW HfE HfW)
    as HCS5.
  (*  pointwise:  each  w <a,b>  <=  (sqrt w |a|)(sqrt w |b|)  *)
  assert (Q1 : nu * << (gu k) , (gv k) >> <= fE 0%nat * fW 0%nat).
  { unfold fE, fW.
    pose proof (CS_le Hs (gu k) (gv k)) as HC.
    assert (Hw2 : sqrt nu * sqrt nu = nu) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt nu * nrm (gu k)) * (sqrt nu * nrm (gv k))
                = (sqrt nu * sqrt nu) * (nrm (gu k) * nrm (gv k))) by ring.
    rewrite Hw2 in E. nra. }
  assert (Q2 : sg k * << (uu k) , (vv k) >> <= fE 1%nat * fW 1%nat).
  { unfold fE, fW.
    pose proof (CS_le Hs (uu k) (vv k)) as HC.
    assert (Hw2 : sqrt (sg k) * sqrt (sg k) = sg k) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (sg k) * nrm (uu k)) * (sqrt (sg k) * nrm (vv k))
                = (sqrt (sg k) * sqrt (sg k)) * (nrm (uu k) * nrm (vv k)))
      by ring.
    rewrite Hw2 in E. nra. }
  assert (Q3 : eps * << (pp k) , (qq k) >> <= fE 2%nat * fW 2%nat).
  { unfold fE, fW.
    pose proof (CS_le Hs (pp k) (qq k)) as HC.
    assert (Hw2 : sqrt eps * sqrt eps = eps) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt eps * nrm (pp k)) * (sqrt eps * nrm (qq k))
                = (sqrt eps * sqrt eps) * (nrm (pp k) * nrm (qq k))) by ring.
    rewrite Hw2 in E. nra. }
  assert (Q4 : t1 k * << (xe k) , (xw k) >> <= fE 3%nat * fW 3%nat).
  { unfold fE, fW.
    pose proof (CS_le Hs (xe k) (xw k)) as HC.
    assert (Hw2 : sqrt (t1 k) * sqrt (t1 k) = t1 k) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (t1 k) * nrm (xe k)) * (sqrt (t1 k) * nrm (xw k))
                = (sqrt (t1 k) * sqrt (t1 k)) * (nrm (xe k) * nrm (xw k)))
      by ring.
    rewrite Hw2 in E. nra. }
  assert (Q5 : t2 k * << (divu k) , (divv k) >> <= fE 4%nat * fW 4%nat).
  { unfold fE, fW.
    pose proof (CS_le Hs (divu k) (divv k)) as HC.
    assert (Hw2 : sqrt (t2 k) * sqrt (t2 k) = t2 k) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (t2 k) * nrm (divu k)) * (sqrt (t2 k) * nrm (divv k))
                = (sqrt (t2 k) * sqrt (t2 k)) * (nrm (divu k) * nrm (divv k)))
      by ring.
    rewrite Hw2 in E. nra. }
  assert (HA1 : Qk k
                <= Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat
                     (fun i => fE i * fW i)).
  { cbn [Rsum]. unfold Qk. lra. }
  assert (HE2 : Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat
                  (fun i => (fE i)^2) = perE k).
  { cbn [Rsum]. unfold fE.
    rewrite (AbstractInterpolation.sq_weight Hs nu (gu k)
               (Rlt_le 0 nu nu_pos)).
    rewrite (AbstractInterpolation.sq_weight Hs (sg k) (uu k) Hsg).
    rewrite (AbstractInterpolation.sq_weight Hs eps (pp k) eps_nonneg).
    rewrite (AbstractInterpolation.sq_weight Hs (t1 k) (xe k)
               (Rlt_le 0 (t1 k) Ht1)).
    rewrite (AbstractInterpolation.sq_weight Hs (t2 k) (divu k)
               (Rlt_le 0 (t2 k) Ht2)).
    unfold perE, AbstractInterpolation.perU. cbv beta.
    unfold sg, t1, t2,
           AbstractInterpolation.sg, AbstractInterpolation.t1,
           AbstractInterpolation.t2,
           xe, AbstractInterpolation.xu. cbv beta.
    ring. }
  assert (HW2 : Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat
                  (fun i => (fW i)^2) = perW k).
  { cbn [Rsum]. unfold fW.
    rewrite (AbstractInterpolation.sq_weight Hs nu (gv k)
               (Rlt_le 0 nu nu_pos)).
    rewrite (AbstractInterpolation.sq_weight Hs (sg k) (vv k) Hsg).
    rewrite (AbstractInterpolation.sq_weight Hs eps (qq k) eps_nonneg).
    rewrite (AbstractInterpolation.sq_weight Hs (t1 k) (xw k)
               (Rlt_le 0 (t1 k) Ht1)).
    rewrite (AbstractInterpolation.sq_weight Hs (t2 k) (divv k)
               (Rlt_le 0 (t2 k) Ht2)).
    unfold perW, AbstractInterpolation.perV. cbv beta.
    unfold sg, t1, t2,
           AbstractInterpolation.sg, AbstractInterpolation.t1,
           AbstractInterpolation.t2,
           xw, AbstractInterpolation.xv. cbv beta.
    ring. }
  rewrite HE2, HW2 in HCS5.
  lra.
Qed.

Lemma sumQ_CS : Rsum Th Qk <= NE * NW.
Proof.
  eapply Rle_trans.
  { apply Rsum_le. intro k. exact (Qk_CS k). }
  assert (HfE : forall k, 0 <= sqrt (perE k)) by (intro; apply sqrt_pos).
  assert (HfW : forall k, 0 <= sqrt (perW k)) by (intro; apply sqrt_pos).
  pose proof (Rsum_CS K Th (fun k => sqrt (perE k)) (fun k => sqrt (perW k))
                HfE HfW) as HCS.
  cbv beta in HCS.
  assert (EE : Rsum Th (fun k => (sqrt (perE k))^2) = Rsum Th perE).
  { apply Rsum_ext. intro k.
    replace ((sqrt (perE k))^2) with (sqrt (perE k) * sqrt (perE k)) by ring.
    apply sqrt_sqrt. exact (perE_nonneg k). }
  assert (EW : Rsum Th (fun k => (sqrt (perW k))^2) = Rsum Th perW).
  { apply Rsum_ext. intro k.
    replace ((sqrt (perW k))^2) with (sqrt (perW k) * sqrt (perW k)) by ring.
    apply sqrt_sqrt. exact (perW_nonneg k). }
  rewrite EE, EW in HCS.
  assert (ENE : NE = sqrt (Rsum Th perE)) by reflexivity.
  assert (ENW : NW = sqrt (Rsum Th perW)) by reflexivity.
  rewrite <- ENE, <- ENW in HCS.
  exact HCS.
Qed.

Lemma NE_nonneg : 0 <= NE.
Proof.
  assert (ENE : NE = sqrt (Rsum Th perE)) by reflexivity.
  rewrite ENE. apply sqrt_pos.
Qed.

Lemma NErr_triangle : NErr <= NE + NW.
Proof.
  assert (Hsplit : NErr2 = Rsum Th perE + 2 * Rsum Th Qk + Rsum Th perW).
  { unfold NErr2.
    assert (E1 : Rsum Th perErr
                 = Rsum Th (fun k => perE k + (2 * Qk k + perW k))).
    { apply Rsum_ext. intro k. rewrite (perErr_expand k). ring. }
    rewrite E1.
    assert (S1 : Rsum Th (fun k => perE k + (2 * Qk k + perW k))
                 = Rsum Th perE + Rsum Th (fun k => 2 * Qk k + perW k))
      by exact (Rsum_plus K Th perE (fun k => 2 * Qk k + perW k)).
    assert (S2 : Rsum Th (fun k => 2 * Qk k + perW k)
                 = Rsum Th (fun k => 2 * Qk k) + Rsum Th perW)
      by exact (Rsum_plus K Th (fun k => 2 * Qk k) perW).
    assert (S3 : Rsum Th (fun k => 2 * Qk k) = 2 * Rsum Th Qk)
      by exact (Rsum_scal K Th 2 Qk).
    rewrite S1, S2, S3. lra. }
  assert (HPE0 : 0 <= Rsum Th perE)
    by (apply Rsum_nonneg; exact perE_nonneg).
  assert (HPW0 : 0 <= Rsum Th perW)
    by (apply Rsum_nonneg; exact perW_nonneg).
  assert (HNE2 : NE ^ 2 = Rsum Th perE).
  { assert (ENE : NE = sqrt (Rsum Th perE)) by reflexivity.
    rewrite ENE.
    replace ((sqrt (Rsum Th perE))^2)
      with (sqrt (Rsum Th perE) * sqrt (Rsum Th perE)) by ring.
    apply sqrt_sqrt. exact HPE0. }
  assert (HNW2 : NW ^ 2 = Rsum Th perW).
  { assert (ENW : NW = sqrt (Rsum Th perW)) by reflexivity.
    rewrite ENW.
    replace ((sqrt (Rsum Th perW))^2)
      with (sqrt (Rsum Th perW) * sqrt (Rsum Th perW)) by ring.
    apply sqrt_sqrt. exact HPW0. }
  pose proof sumQ_CS as HQ.
  pose proof NE_nonneg as HNE0.
  pose proof NW_nonneg as HNW0.
  apply nonneg_le_of_sqr.
  - unfold NErr. apply sqrt_pos.
  - lra.
  - assert (HNErr2 : NErr ^ 2 = NErr2).
    { unfold NErr.
      replace ((sqrt NErr2)^2) with (sqrt NErr2 * sqrt NErr2) by ring.
      apply sqrt_sqrt.
      unfold NErr2. apply Rsum_nonneg. exact perErr_nonneg. }
    rewrite HNErr2, Hsplit.
    replace ((NE + NW)^2) with (NE^2 + 2 * (NE * NW) + NW^2) by ring.
    lra.
Qed.

(* ========================================================================= *)
(*  thm:convergence, abstract form.                                          *)
(* ========================================================================= *)

Definition Cconv : R := Kb + Ct / Cst.

Theorem abstract_convergence : NErr <= Cconv * Psi.
Proof.
  pose proof NErr_triangle as HT.
  pose proof NE_bound as HE.
  pose proof NW_bound as HW.
  unfold Cconv. lra.
Qed.

Lemma Kb_nonneg : 0 <= Kb.
Proof.
  unfold Kb, AbstractInterpolation.KabsI.
  assert (0 <= sqrt 2) by apply sqrt_pos.
  assert (0 <= AbstractInterpolation.KaggI c1 c2 CI)
    by apply AbstractInterpolation.KaggI_nonneg.
  nra.
Qed.

Lemma Cconv_nonneg : 0 <= Cconv.
Proof.
  unfold Cconv.
  pose proof Kb_nonneg. pose proof Ct_nonneg. pose proof Cst_pos.
  assert (0 <= Ct / Cst) by (apply div_nonneg; lra).
  lra.
Qed.

(*  The appendix's l1 form of psi(h) (eq:convergence, second line).          *)
Corollary abstract_convergence_l1 :
  NErr <= Cconv
          * Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                              + aK k * sqrt (t1 k) / hK k * IP k).
Proof.
  pose proof abstract_convergence as HM.
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
               = AbstractInterpolation.PsU2 K Th nu c1 c2 hK aK am IU).
  { unfold AbstractInterpolation.PsU2. apply Rsum_ext. intro k.
    unfold t2, AbstractInterpolation.t2.
    apply AbstractInterpolation.sq_weight2s.
    - apply Rlt_le.
      apply tau2_pos; auto.
    - apply hK_pos. }
  assert (HU : AbstractInterpolation.PsU K Th nu c1 c2 hK aK am IU
               <= Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k)).
  { unfold AbstractInterpolation.PsU. rewrite <- EU.
    apply sqrt_sum_sq_le_sum. exact HfU. }
  assert (EP : Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * IP k)^2)
               = AbstractInterpolation.PsP2 K Th nu sigma c1 c2 hK aK am IP).
  { unfold AbstractInterpolation.PsP2. apply Rsum_ext. intro k.
    unfold t1, AbstractInterpolation.t1.
    apply AbstractInterpolation.sq_weight2s.
    - apply Rlt_le.
      apply tau1_pos; auto.
    - apply hK_pos. }
  assert (HP : AbstractInterpolation.PsP K Th nu sigma c1 c2 hK aK am IP
               <= Rsum Th (fun k => aK k * sqrt (t1 k) / hK k * IP k)).
  { unfold AbstractInterpolation.PsP. rewrite <- EP.
    apply sqrt_sum_sq_le_sum. exact HfP. }
  assert (Esum : Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                                   + aK k * sqrt (t1 k) / hK k * IP k)
                 = Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k)
                   + Rsum Th (fun k => aK k * sqrt (t1 k) / hK k * IP k)).
  { exact (Rsum_plus K Th
             (fun k => aK k * sqrt (t2 k) / hK k * IU k)
             (fun k => aK k * sqrt (t1 k) / hK k * IP k)). }
  pose proof Cconv_nonneg as HCc.
  pose proof Psi_nonneg as HPsi.
  assert (Hpsi : Psi <= Rsum Th (fun k => aK k * sqrt (t2 k) / hK k * IU k
                                          + aK k * sqrt (t1 k) / hK k * IP k)).
  { unfold Psi. lra. }
  nra.
Qed.

End AbstractConvergence.
