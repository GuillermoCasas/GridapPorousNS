(* ========================================================================= *)
(*  AbstractContinuity.v                                                     *)
(*                                                                           *)
(*  lemma:Continuity of the paper (Appendix B, lem:continuity), proved IN    *)
(*  FULL -- the eighteen-term decomposition of Step 0, the estimates of      *)
(*  Steps 1--5 and 7, the exact rewritings of Step 6a--6b, the jump          *)
(*  treatment of Steps 6b and 6d with explicit constants, the absorption     *)
(*  chain of Step 9, and the assembly over elements and interior faces --    *)
(*  from a named, quantitative trusted base consisting exactly of:           *)
(*                                                                           *)
(*   * the three integration-by-parts identities of Step 6a                  *)
(*     (eq:skew, eq:globalibp, eq:elemibp), the elemental pressure           *)
(*     integration by parts of Step 6b, and the two facewise assembly        *)
(*     identities (continuity of the integrands across interior faces plus   *)
(*     vanishing of the boundary faces);                                     *)
(*   * the two face-integral estimates (Hoelder on the face,                 *)
(*     meas(face) <= C h^(d-1), the L-infinity inverse estimates on the two  *)
(*     neighbouring elements, alpha <= alpha_K, and the comparability of     *)
(*     neighbouring diameters), packaged at the level the appendix reaches   *)
(*     in eq:jumppart / eq:IIIface;                                          *)
(*   * the jump condition H:jump;                                            *)
(*   * the bounded face-multiplicity of the mesh (H:mesh);                   *)
(*   * the eight weighted inverse estimates of lem:winv                      *)
(*     (eq:winv-grad, eq:winv-conv both pieces, eq:winv-divu,                *)
(*     eq:winv-divvisc, on the arguments where the proof uses them);         *)
(*   * the epsilon condition eq:epscond, elementwise.                        *)
(*                                                                           *)
(*  The elemental Cauchy--Schwarz inequalities are NOT assumed: they are     *)
(*  proved from the inner-product axioms (InnerSpace.v).  The discrete       *)
(*  Cauchy--Schwarz over element and face sums is proved by induction        *)
(*  (AbstractSums.v).  All per-element parameter algebra is imported from   *)
(*  the machine-checked ContinuityAlgebra.v.                                 *)
(*                                                                           *)
(*  Modelling convention: the elemental restriction of each field is an      *)
(*  abstract vector; weighted L^2(K) pairings such as (P grad v,             *)
(*  alpha P grad u)_K are represented as << gv , gu >> with                  *)
(*  gu ~ alpha^(1/2) P grad u|_K, i.e. the square-root weighting             *)
(*  rewrites of the appendix are baked into the choice of representatives    *)
(*  (this is the appendix's own first move in each step).                    *)
(*                                                                           *)
(*  Conclusions (with fully explicit constants):                             *)
(*    abstract_continuity_sharp :                                            *)
(*      BS <= KUV (NU NV) + KPV (BrP NV) + sqrt c1 (BrU NV)                  *)
(*      -- eq:assembly / the sharp form eq:sharpcont;                        *)
(*    abstract_continuity :                                                  *)
(*      BS <= Ctot (BrU + BrP) NV                                            *)
(*      -- lemma:Continuity for discrete arguments, Step 9 included.         *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

Section AbstractContinuity.

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

Variables (nu sigma eps c1 c2 C2 Cinv Cb cJ cJ' Cface Nf : R).
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

(*  For U = [u_h; p_h] and V = [v_h; q_h]:                                    *)
(*    gu/gv     ~  (alpha^(1/2) P grad .)|_K                                  *)
(*    du/dv     ~  (2 nu div(alpha P grad .))|_K                              *)
(*    cxu/cxv   ~  (alpha a . grad .)|_K                                      *)
(*    gpu/gpv   ~  (alpha grad p)|_K, (alpha grad q)|_K                       *)
(*    uu/vv     ~  u_h|_K, v_h|_K ;  pp/qq ~ p_h|_K, q_h|_K                   *)
(*    divu/divv ~  (div(alpha .))|_K                                          *)
Variables (gu gv du dv cxu cxv gpu gpv uu vv pp qq divu divv : K -> V).

Definition xu (k : K) : V := (cxu k) +v (gpu k).   (*  (alpha X(U))|_K  *)
Definition xv (k : K) : V := (cxv k) +v (gpv k).   (*  (alpha X(V))|_K  *)

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
Hypothesis H_face_p :
  forall f, Rabs (FBp f)
  <= Cface * (  aK (e1 f) / hK (e1 f) * (nrm (vv (e1 f)) * nrm (pp (e1 f)))
              + aK (e2 f) / hK (e2 f) * (nrm (vv (e1 f)) * nrm (pp (e2 f)))
              + aK (e1 f) / hK (e1 f) * (nrm (vv (e2 f)) * nrm (pp (e1 f)))
              + aK (e2 f) / hK (e2 f) * (nrm (vv (e2 f)) * nrm (pp (e2 f)))).
Hypothesis H_face_c :
  forall f, Rabs (FBc f)
  <= Cface * (  aK (e1 f) * am (e1 f) / hK (e1 f)
                  * (nrm (uu (e1 f)) * nrm (vv (e1 f)))
              + aK (e2 f) * am (e2 f) / hK (e2 f)
                  * (nrm (uu (e1 f)) * nrm (vv (e2 f)))
              + aK (e1 f) * am (e1 f) / hK (e1 f)
                  * (nrm (uu (e2 f)) * nrm (vv (e1 f)))
              + aK (e2 f) * am (e2 f) / hK (e2 f)
                  * (nrm (uu (e2 f)) * nrm (vv (e2 f)))).

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

(*  The weighted inverse estimates of lem:winv, on the arguments used.       *)
Hypothesis Hw_gu :
  forall k, nrm (gu k) <= Cinv / hK k * sqrt (aK k) * nrm (uu k).
Hypothesis Hw_gv :
  forall k, nrm (gv k) <= Cinv / hK k * sqrt (aK k) * nrm (vv k).
Hypothesis Hw_du :
  forall k, nrm (du k) <= 2 * nu * Cb / hK k * sqrt (aK k) * nrm (gu k).
Hypothesis Hw_dv :
  forall k, nrm (dv k) <= 2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k).
Hypothesis Hw_cxu :
  forall k, nrm (cxu k) <= Cinv / hK k * aK k * am k * nrm (uu k).
Hypothesis Hw_cxv :
  forall k, nrm (cxv k) <= Cinv / hK k * aK k * am k * nrm (vv k).
Hypothesis Hw_gpu :
  forall k, nrm (gpu k) <= Cinv / hK k * aK k * nrm (pp k).
Hypothesis Hw_divu :
  forall k, nrm (divu k) <= Cb * aK k / hK k * nrm (uu k).

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

Definition BrU2 : R :=
  Rsum Th (fun k => (aK k)^2 * t2 k / (hK k)^2 * << (uu k) , (uu k) >>).
Definition BrU : R := sqrt BrU2.
Definition BrP2 : R :=
  Rsum Th (fun k => (aK k)^2 * t1 k / (hK k)^2 * << (pp k) , (pp k) >>).
Definition BrP : R := sqrt BrP2.

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
Lemma BrU_nonneg : 0 <= BrU.  Proof. apply sqrt_pos. Qed.
Lemma BrP_nonneg : 0 <= BrP.  Proof. apply sqrt_pos. Qed.

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

Lemma BrP_le :
  sqrt (Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * nrm (pp k))^2)) <= BrP.
Proof.
  unfold BrP, BrP2.
  rewrite (Rsum_ext K Th _ _
    (fun k => sq_weight2 (aK k) (t1 k) (hK k) (pp k)
                (Rlt_le 0 (t1 k) (t1_pos' k)) (hK_pos k))).
  apply Rle_refl.
Qed.

Lemma BrU_le :
  sqrt (Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * nrm (uu k))^2)) <= BrU.
Proof.
  unfold BrU, BrU2.
  rewrite (Rsum_ext K Th _ _
    (fun k => sq_weight2 (aK k) (t2 k) (hK k) (uu k)
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

Lemma bound_T6 : Rabs T6 <= 4 * Cb^2 / c1 * (NU * NV).
Proof.
  unfold T6. rewrite Rabs_Ropp.
  apply (abs_sum_bound_c (4 * Cb^2 / c1) t1 dv du
           (fun k => sqrt nu * nrm (gu k))
           (fun k => sqrt nu * nrm (gv k)) NU NV).
  - apply div_nonneg; nra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gu k)). nra.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gv k)). nra.
  - intro k.
    pose proof (Hw_du k) as HDu. pose proof (Hw_dv k) as HDv.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t1_pos' k) as Ht1.
    pose proof (nrm_nonneg Hs (du k)) as Hnd.
    pose proof (nrm_nonneg Hs (dv k)) as Hnv.
    pose proof (nrm_nonneg Hs (gu k)) as Hgu.
    pose proof (nrm_nonneg Hs (gv k)) as Hgv.
    pose proof (L_keysq k) as HK.
    assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
    assert (Hcoef : 0 <= 2 * nu * Cb / hK k * sqrt (aK k)).
    { assert (H1 : 0 < 2 * nu * Cb) by nra.
      assert (H2 : 0 < 2 * nu * Cb / hK k)
        by (apply Rdiv_lt_0_compat; lra).
      nra. }
    assert (A1 : nrm (dv k) * nrm (du k)
                 <= (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k))
                    * nrm (du k)) by nra.
    assert (Hc2 : 0 <= 2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k)) by nra.
    assert (A2 : (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k)) * nrm (du k)
                 <= (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gv k))
                    * (2 * nu * Cb / hK k * sqrt (aK k) * nrm (gu k)))
      by nra.
    assert (S2 : t1 k * (2 * nu * Cb / hK k * sqrt (aK k))^2
                 <= 4 * Cb^2 / c1 * nu).
    { replace ((2 * nu * Cb / hK k * sqrt (aK k))^2)
        with (4 * nu^2 * Cb^2 * (sqrt (aK k) * sqrt (aK k)) / (hK k)^2)
        by (field; lra).
      rewrite sqrt_sqrt by lra.
      assert (Hh2 : 0 < (hK k)^2) by nra.
      assert (Hh2c : 0 < (hK k)^2 * c1) by nra.
      apply (Rmult_le_reg_r ((hK k)^2 * c1)); [exact Hh2c |].
      replace (t1 k * (4 * nu^2 * Cb^2 * aK k / (hK k)^2) * ((hK k)^2 * c1))
        with (4 * nu * Cb^2 * (nu * t1 k * aK k * c1)) by (field; lra).
      replace (4 * Cb^2 / c1 * nu * ((hK k)^2 * c1))
        with (4 * nu * Cb^2 * (hK k)^2) by (field; lra).
      assert (Hm : 0 <= 4 * nu * Cb^2) by nra.
      nra. }
    assert (Hnn : 0 <= nrm (gv k) * nrm (gu k)) by nra.
    assert (S3 : t1 k * (nrm (dv k) * nrm (du k))
                 <= t1 k * ((2 * nu * Cb / hK k * sqrt (aK k))^2
                            * (nrm (gv k) * nrm (gu k)))) by nra.
    assert (S4 : t1 k * ((2 * nu * Cb / hK k * sqrt (aK k))^2
                         * (nrm (gv k) * nrm (gu k)))
                 <= (4 * Cb^2 / c1 * nu) * (nrm (gv k) * nrm (gu k)))
      by nra.
    assert (Hnu2 : sqrt nu * sqrt nu = nu) by (apply sqrt_sqrt; lra).
    nra.
  - exact cmpU_g.
  - exact cmpV_g.
Qed.

Lemma bound_T7 : Rabs T7 <= 2 * Cb / sqrt c1 * (NU * NV).
Proof.
  unfold T7. rewrite Rabs_Ropp.
  apply (abs_sum_bound_c (2 * Cb / sqrt c1) t1 xv du
           (fun k => sqrt nu * nrm (gu k))
           (fun k => sqrt (t1 k) * nrm (xv k)) NU NV).
  - pose proof sqrtc1_pos. apply div_nonneg; nra.
  - intro k. apply Rlt_le, t1_pos'.
  - intro k. assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gu k)). nra.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xv k)). nra.
  - intro k.
    pose proof (Hw_du k) as HDu.
    pose proof (keyvisc_coef k) as HC.
    pose proof (t1_pos' k) as Ht1.
    pose proof (nrm_nonneg Hs (du k)) as Hnd.
    pose proof (nrm_nonneg Hs (xv k)) as Hnx.
    pose proof (nrm_nonneg Hs (gu k)) as Hgu.
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
    (*  t1 nxv ndu = (sqrt t1 nxv)(sqrt t1 ndu)
        <= (sqrt t1 nxv)(sqrt t1 (2nuCb/h sqrt aK) ngu)
        <= (sqrt t1 nxv)((2Cb/sqrt c1) sqrt nu ngu)  *)
    assert (S1 : sqrt (t1 k) * nrm (du k)
                 <= sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
                    * nrm (gu k)) by nra.
    assert (Hgap : 0 <= 2 * Cb / sqrt c1 * sqrt nu).
    { pose proof sqrtc1_pos.
      assert (0 <= sqrt nu) by apply sqrt_pos.
      assert (0 <= 2 * Cb / sqrt c1) by (apply div_nonneg; nra).
      nra. }
    assert (S2 : sqrt (t1 k) * (2 * nu * Cb / hK k * sqrt (aK k))
                   * nrm (gu k)
                 <= (2 * Cb / sqrt c1 * sqrt nu) * nrm (gu k)) by nra.
    assert (Hfac : 0 <= sqrt (t1 k) * nrm (xv k)) by nra.
    assert (S3 : (sqrt (t1 k) * nrm (xv k)) * (sqrt (t1 k) * nrm (du k))
                 <= (sqrt (t1 k) * nrm (xv k))
                    * ((2 * Cb / sqrt c1 * sqrt nu) * nrm (gu k))) by nra.
    nra.
  - exact cmpU_g.
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
Lemma double_inv_u : forall k,
  nrm (du k) <= 2 * nu * Cb * Cinv * aK k / (hK k)^2 * nrm (uu k).
Proof.
  intro k.
  pose proof (Hw_du k) as H1. pose proof (Hw_gu k) as H2.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (nrm_nonneg Hs (gu k)) as Hg.
  pose proof (nrm_nonneg Hs (uu k)) as Hu.
  assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
  assert (Hcoef : 0 <= 2 * nu * Cb / hK k * sqrt (aK k)).
  { assert (H3 : 0 < 2 * nu * Cb) by nra.
    assert (H4 : 0 < 2 * nu * Cb / hK k) by (apply Rdiv_lt_0_compat; lra).
    nra. }
  assert (S1 : nrm (du k)
               <= (2 * nu * Cb / hK k * sqrt (aK k))
                  * (Cinv / hK k * sqrt (aK k) * nrm (uu k))) by nra.
  assert (E : (2 * nu * Cb / hK k * sqrt (aK k))
              * (Cinv / hK k * sqrt (aK k) * nrm (uu k))
              = 2 * nu * Cb * Cinv * (sqrt (aK k) * sqrt (aK k)) / (hK k)^2
                * nrm (uu k)) by (field; lra).
  rewrite E in S1. rewrite sqrt_sqrt in S1 by lra.
  exact S1.
Qed.

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

Lemma bound_T8 : Rabs T8 <= 2 * Cb * Cinv / c1 * (NU * NV).
Proof.
  assert (E : T8 = Rsum Th (fun k => (sigma * t1 k)
                                      * << (vv k) , (du k) >>)).
  { unfold T8. rewrite <- Rsum_scal. apply Rsum_ext. intro k. ring. }
  rewrite E.
  apply (abs_sum_bound_c (2 * Cb * Cinv / c1) (fun k => sigma * t1 k) vv du
           (fun k => sqrt (sg k) * nrm (uu k))
           (fun k => sqrt (sg k) * nrm (vv k)) NU NV).
  - apply div_nonneg; nra.
  - intro k. pose proof (t1_pos' k). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)). nra.
  - intro k. assert (0 <= sqrt (sg k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (vv k)). nra.
  - intro k.
    pose proof (double_inv_u k) as HD.
    pose proof (L_T8c k) as H8.
    pose proof (t1_pos' k) as Ht1.
    pose proof (sg_nonneg' k) as Hsg.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (nrm_nonneg Hs (vv k)) as Hnv.
    pose proof (nrm_nonneg Hs (uu k)) as Hnu.
    pose proof (nrm_nonneg Hs (du k)) as Hnd.
    assert (Hsgs : sqrt (sg k) * sqrt (sg k) = sg k)
      by (apply sqrt_sqrt; lra).
    assert (Hst : 0 <= sigma * t1 k) by nra.
    (*  sigma t1 nvv ndu <= sigma t1 (2nuCbCinv aK/h^2) nvv nuu  *)
    assert (Hnn : 0 <= sigma * t1 k * nrm (vv k)) by nra.
    assert (S1 : sigma * t1 k * (nrm (vv k) * nrm (du k))
                 <= sigma * t1 k
                    * (nrm (vv k)
                       * (2 * nu * Cb * Cinv * aK k / (hK k)^2
                          * nrm (uu k)))) by nra.
    (*  coefficient:  sigma t1 (2nuCbCinv aK/h^2)
                      = (2CbCinv/c1) * (sigma (c1 nu aK/h^2) t1) <= (2CbCinv/c1) sg  *)
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

Lemma bound_VolP : Rabs VolP <= sqrt c1 * (BrP * NV).
Proof.
  unfold VolP.
  apply (abs_sum_bound_c (sqrt c1) (fun k => ph k * t1 k) pp divv
           (fun k => aK k * sqrt (t1 k) / hK k * nrm (pp k))
           (fun k => sqrt (t2 k) * nrm (divv k)) BrP NV).
  - apply sqrt_pos.
  - intro k. pose proof (ph_pos k). pose proof (t1_pos' k). nra.
  - intro k. pose proof (hK_pos k). pose proof (aK_pos k).
    assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (pp k)).
    assert (0 <= aK k * sqrt (t1 k)) by nra.
    assert (0 <= aK k * sqrt (t1 k) / hK k)
      by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (divv k)). nra.
  - intro k.
    pose proof (L_vol k) as HV.
    pose proof (L_P3div k) as HP3.
    pose proof (nrm_nonneg Hs (pp k)) as Hnp.
    pose proof (nrm_nonneg Hs (divv k)) as Hnd.
    assert (Hnn : 0 <= nrm (pp k) * nrm (divv k)) by nra.
    assert (S1 : ph k * t1 k * (nrm (pp k) * nrm (divv k))
                 <= sqrt (ph k) * sqrt (t1 k)
                    * (nrm (pp k) * nrm (divv k))) by nra.
    rewrite HP3 in S1.
    nra.
  - exact BrP_le.
  - exact cmpV_d.
Qed.

(*  The jump part: per-face estimate combining the four Jb bounds with the   *)
(*  face-integral hypothesis, then Cauchy--Schwarz over the faces.           *)
Definition PPk (k : K) : R := aK k * sqrt (t1 k) / hK k * nrm (pp k).
Definition VVs (k : K) : R := sqrt (sg k) * nrm (vv k).
Definition UUs (k : K) : R := sqrt (sg k) * nrm (uu k).

Lemma PPk_nonneg : forall k, 0 <= PPk k.
Proof.
  intro k. unfold PPk.
  pose proof (hK_pos k). pose proof (aK_pos k).
  assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
  pose proof (nrm_nonneg Hs (pp k)).
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

Lemma UUs_nonneg : forall k, 0 <= UUs k.
Proof.
  intro k. unfold UUs.
  assert (0 <= sqrt (sg k)) by apply sqrt_pos.
  pose proof (nrm_nonneg Hs (uu k)). nra.
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
  <= Cface * (  kb11 * (PPk (e1 f) * VVs (e1 f))
              + kb12 * (PPk (e2 f) * VVs (e1 f))
              + kb21 * (PPk (e1 f) * VVs (e2 f))
              + kb22 * (PPk (e2 f) * VVs (e2 f))).
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
                          * (nrm (vv (e1 f)) * nrm (pp (e1 f)))
                        + aK (e2 f) / hK (e2 f)
                          * (nrm (vv (e1 f)) * nrm (pp (e2 f)))
                        + aK (e1 f) / hK (e1 f)
                          * (nrm (vv (e2 f)) * nrm (pp (e1 f)))
                        + aK (e2 f) / hK (e2 f)
                          * (nrm (vv (e2 f)) * nrm (pp (e2 f)))))) by nra.
  (*  bound each of the four pieces via the corresponding Jb lemma  *)
  set (w11 := aK (e1 f) / hK (e1 f) * (nrm (vv (e1 f)) * nrm (pp (e1 f)))).
  set (w12 := aK (e2 f) / hK (e2 f) * (nrm (vv (e1 f)) * nrm (pp (e2 f)))).
  set (w21 := aK (e1 f) / hK (e1 f) * (nrm (vv (e2 f)) * nrm (pp (e1 f)))).
  set (w22 := aK (e2 f) / hK (e2 f) * (nrm (vv (e2 f)) * nrm (pp (e2 f)))).
  assert (Hah1 : 0 <= aK (e1 f) / hK (e1 f))
    by (apply div_nonneg; [apply Rlt_le, aK_pos | apply hK_pos]).
  assert (Hah2 : 0 <= aK (e2 f) / hK (e2 f))
    by (apply div_nonneg; [apply Rlt_le, aK_pos | apply hK_pos]).
  assert (Hn1 : 0 <= nrm (vv (e1 f))) by apply nrm_nonneg.
  assert (Hn2 : 0 <= nrm (vv (e2 f))) by apply nrm_nonneg.
  assert (Hp1 : 0 <= nrm (pp (e1 f))) by apply nrm_nonneg.
  assert (Hp2 : 0 <= nrm (pp (e2 f))) by apply nrm_nonneg.
  assert (Hq11 : 0 <= nrm (vv (e1 f)) * nrm (pp (e1 f))) by nra.
  assert (Hq12 : 0 <= nrm (vv (e1 f)) * nrm (pp (e2 f))) by nra.
  assert (Hq21 : 0 <= nrm (vv (e2 f)) * nrm (pp (e1 f))) by nra.
  assert (Hq22 : 0 <= nrm (vv (e2 f)) * nrm (pp (e2 f))) by nra.
  assert (Hw11 : 0 <= w11) by (unfold w11; nra).
  assert (Hw12 : 0 <= w12) by (unfold w12; nra).
  assert (Hw21 : 0 <= w21) by (unfold w21; nra).
  assert (Hw22 : 0 <= w22) by (unfold w22; nra).
  (*  each product: sigma|D| * wij <= kbij * PPk(ej) * VVs(ei)  *)
  assert (B11 : sigma * Rabs (Dt1 f) * w11
                <= kb11 * (PPk (e1 f) * VVs (e1 f))).
  { unfold PPk, VVs. unfold w11 in *. nra. }
  assert (B12 : sigma * Rabs (Dt1 f) * w12
                <= kb12 * (PPk (e2 f) * VVs (e1 f))).
  { unfold PPk, VVs. unfold w12 in *. nra. }
  assert (B21 : sigma * Rabs (Dt1 f) * w21
                <= kb21 * (PPk (e1 f) * VVs (e2 f))).
  { unfold PPk, VVs. unfold w21 in *. nra. }
  assert (B22 : sigma * Rabs (Dt1 f) * w22
                <= kb22 * (PPk (e2 f) * VVs (e2 f))).
  { unfold PPk, VVs. unfold w22 in *. nra. }
  fold w11 w12 w21 w22 in S0.
  assert (HCf : 0 <= Cface) by exact Cface_nonneg.
  nra.
Qed.

Lemma bound_JmpP : Rabs JmpP <= K6b * (BrP * NV).
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
      Cface * (  kb11 * (PPk (e1 f) * VVs (e1 f))
               + kb12 * (PPk (e2 f) * VVs (e1 f))
               + kb21 * (PPk (e1 f) * VVs (e2 f))
               + kb22 * (PPk (e2 f) * VVs (e2 f))))
    = Cface * (kb11 * Rsum Fl (fun f => PPk (e1 f) * VVs (e1 f))
               + kb12 * Rsum Fl (fun f => PPk (e2 f) * VVs (e1 f))
               + kb21 * Rsum Fl (fun f => PPk (e1 f) * VVs (e2 f))
               + kb22 * Rsum Fl (fun f => PPk (e2 f) * VVs (e2 f)))).
  { rewrite <- (Rsum_scal F Fl kb11), <- (Rsum_scal F Fl kb12),
            <- (Rsum_scal F Fl kb21), <- (Rsum_scal F Fl kb22).
    rewrite <- !Rsum_plus.
    rewrite <- (Rsum_scal F Fl Cface).
    apply Rsum_ext. intro f. ring. }
  rewrite Esplit.
  (*  each face sum by Cauchy--Schwarz with bounded multiplicity  *)
  assert (HB11 : Rsum Fl (fun f => PPk (e1 f) * VVs (e1 f))
                 <= Nf * (BrP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PPk VVs e1 e1 PPk_nonneg VVs_nonneg
               H_mult1 H_mult1). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PPk k)^2)) <= BrP)
      by (unfold PPk; exact BrP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PPk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof BrP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PPk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= BrP * NV) by nra.
    nra. }
  assert (HB12 : Rsum Fl (fun f => PPk (e2 f) * VVs (e1 f))
                 <= Nf * (BrP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PPk VVs e2 e1 PPk_nonneg VVs_nonneg
               H_mult2 H_mult1). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PPk k)^2)) <= BrP)
      by (unfold PPk; exact BrP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PPk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof BrP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PPk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= BrP * NV) by nra.
    nra. }
  assert (HB21 : Rsum Fl (fun f => PPk (e1 f) * VVs (e2 f))
                 <= Nf * (BrP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PPk VVs e1 e2 PPk_nonneg VVs_nonneg
               H_mult1 H_mult2). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PPk k)^2)) <= BrP)
      by (unfold PPk; exact BrP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PPk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof BrP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PPk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= BrP * NV) by nra.
    nra. }
  assert (HB22 : Rsum Fl (fun f => PPk (e2 f) * VVs (e2 f))
                 <= Nf * (BrP * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair PPk VVs e2 e2 PPk_nonneg VVs_nonneg
               H_mult2 H_mult2). }
    assert (HP2 : sqrt (Rsum Th (fun k => (PPk k)^2)) <= BrP)
      by (unfold PPk; exact BrP_le).
    assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV)
      by (unfold VVs; exact cmpV_s).
    assert (P1 : 0 <= sqrt (Rsum Th (fun k => (PPk k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof BrP_nonneg. pose proof NV_nonneg.
    assert (T1 : sqrt (Rsum Th (fun k => (PPk k)^2))
                 * sqrt (Rsum Th (fun k => (VVs k)^2)) <= BrP * NV) by nra. 
    nra. }
  unfold K6b.
  destruct kb_nonneg as [Hb11 [Hb12 [Hb21 Hb22]]].
  pose proof BrP_nonneg. pose proof NV_nonneg.
  assert (HPN : 0 <= BrP * NV) by nra.
  assert (M11 : kb11 * Rsum Fl (fun f => PPk (e1 f) * VVs (e1 f))
                <= kb11 * (Nf * (BrP * NV))) by nra.
  assert (M12 : kb12 * Rsum Fl (fun f => PPk (e2 f) * VVs (e1 f))
                <= kb12 * (Nf * (BrP * NV))) by nra.
  assert (M21 : kb21 * Rsum Fl (fun f => PPk (e1 f) * VVs (e2 f))
                <= kb21 * (Nf * (BrP * NV))) by nra.
  assert (M22 : kb22 * Rsum Fl (fun f => PPk (e2 f) * VVs (e2 f))
                <= kb22 * (Nf * (BrP * NV))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Step 6c: the terms [II] + [V].                                           *)
(* ========================================================================= *)

Lemma bound_II_V : Rabs (IIterm + Vterm) <= sqrt c1 * (BrU * NV).
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
  apply (abs_sum_bound_c (sqrt c1) (fun k => ph k * t1 k) uu xv
           (fun k => aK k * sqrt (t2 k) / hK k * nrm (uu k))
           (fun k => sqrt (t1 k) * nrm (xv k)) BrU NV).
  - apply sqrt_pos.
  - intro k. pose proof (ph_pos k). pose proof (t1_pos' k). nra.
  - intro k. pose proof (hK_pos k). pose proof (aK_pos k).
    assert (0 <= sqrt (t2 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (uu k)).
    assert (0 <= aK k * sqrt (t2 k)) by nra.
    assert (0 <= aK k * sqrt (t2 k) / hK k) by (apply div_nonneg; lra).
    nra.
  - intro k. assert (0 <= sqrt (t1 k)) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (xv k)). nra.
  - intro k.
    pose proof (L_vol k) as HV.
    pose proof (L_P3div k) as HP3.
    pose proof (nrm_nonneg Hs (uu k)) as Hnu.
    pose proof (nrm_nonneg Hs (xv k)) as Hnx.
    assert (Hnn : 0 <= nrm (uu k) * nrm (xv k)) by nra.
    assert (S1 : ph k * t1 k * (nrm (uu k) * nrm (xv k))
                 <= sqrt (ph k) * sqrt (t1 k)
                    * (nrm (uu k) * nrm (xv k))) by nra.
    rewrite HP3 in S1.
    nra.
  - exact BrU_le.
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
  <= Cface * (  kd11 * (UUs (e1 f) * VVs (e1 f))
              + kd12 * (UUs (e1 f) * VVs (e2 f))
              + kd21 * (UUs (e2 f) * VVs (e1 f))
              + kd22 * (UUs (e2 f) * VVs (e2 f))).
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
                          * (nrm (uu (e1 f)) * nrm (vv (e1 f)))
                        + aK (e2 f) * am (e2 f) / hK (e2 f)
                          * (nrm (uu (e1 f)) * nrm (vv (e2 f)))
                        + aK (e1 f) * am (e1 f) / hK (e1 f)
                          * (nrm (uu (e2 f)) * nrm (vv (e1 f)))
                        + aK (e2 f) * am (e2 f) / hK (e2 f)
                          * (nrm (uu (e2 f)) * nrm (vv (e2 f)))))) by nra.
  assert (Hn1 : 0 <= nrm (uu (e1 f))) by apply nrm_nonneg.
  assert (Hn2 : 0 <= nrm (uu (e2 f))) by apply nrm_nonneg.
  assert (Hv1 : 0 <= nrm (vv (e1 f))) by apply nrm_nonneg.
  assert (Hv2 : 0 <= nrm (vv (e2 f))) by apply nrm_nonneg.
  assert (Hq11 : 0 <= nrm (uu (e1 f)) * nrm (vv (e1 f))) by nra.
  assert (Hq12 : 0 <= nrm (uu (e1 f)) * nrm (vv (e2 f))) by nra.
  assert (Hq21 : 0 <= nrm (uu (e2 f)) * nrm (vv (e1 f))) by nra.
  assert (Hq22 : 0 <= nrm (uu (e2 f)) * nrm (vv (e2 f))) by nra.
  assert (B11 : sigma * Rabs (Dt1 f)
                  * (aK (e1 f) * am (e1 f) / hK (e1 f)
                     * (nrm (uu (e1 f)) * nrm (vv (e1 f))))
                <= kd11 * (UUs (e1 f) * VVs (e1 f))).
  { unfold UUs, VVs. nra. }
  assert (B12 : sigma * Rabs (Dt1 f)
                  * (aK (e2 f) * am (e2 f) / hK (e2 f)
                     * (nrm (uu (e1 f)) * nrm (vv (e2 f))))
                <= kd12 * (UUs (e1 f) * VVs (e2 f))).
  { unfold UUs, VVs. nra. }
  assert (B21 : sigma * Rabs (Dt1 f)
                  * (aK (e1 f) * am (e1 f) / hK (e1 f)
                     * (nrm (uu (e2 f)) * nrm (vv (e1 f))))
                <= kd21 * (UUs (e2 f) * VVs (e1 f))).
  { unfold UUs, VVs. nra. }
  assert (B22 : sigma * Rabs (Dt1 f)
                  * (aK (e2 f) * am (e2 f) / hK (e2 f)
                     * (nrm (uu (e2 f)) * nrm (vv (e2 f))))
                <= kd22 * (UUs (e2 f) * VVs (e2 f))).
  { unfold UUs, VVs. nra. }
  assert (HCf : 0 <= Cface) by exact Cface_nonneg.
  nra.
Qed.

Lemma bound_III : Rabs IIIterm <= K6d * (NU * NV).
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
      Cface * (  kd11 * (UUs (e1 f) * VVs (e1 f))
               + kd12 * (UUs (e1 f) * VVs (e2 f))
               + kd21 * (UUs (e2 f) * VVs (e1 f))
               + kd22 * (UUs (e2 f) * VVs (e2 f))))
    = Cface * (kd11 * Rsum Fl (fun f => UUs (e1 f) * VVs (e1 f))
               + kd12 * Rsum Fl (fun f => UUs (e1 f) * VVs (e2 f))
               + kd21 * Rsum Fl (fun f => UUs (e2 f) * VVs (e1 f))
               + kd22 * Rsum Fl (fun f => UUs (e2 f) * VVs (e2 f)))).
  { rewrite <- (Rsum_scal F Fl kd11), <- (Rsum_scal F Fl kd12),
            <- (Rsum_scal F Fl kd21), <- (Rsum_scal F Fl kd22).
    rewrite <- !Rsum_plus.
    rewrite <- (Rsum_scal F Fl Cface).
    apply Rsum_ext. intro f. ring. }
  rewrite Esplit.
  assert (HU2 : sqrt (Rsum Th (fun k => (UUs k)^2)) <= NU).
  { unfold UUs. exact cmpU_s. }
  assert (HV2 : sqrt (Rsum Th (fun k => (VVs k)^2)) <= NV).
  { unfold VVs. exact cmpV_s. }
  assert (Hprod : sqrt (Rsum Th (fun k => (UUs k)^2))
                  * sqrt (Rsum Th (fun k => (VVs k)^2)) <= NU * NV).
  { assert (P1 : 0 <= sqrt (Rsum Th (fun k => (UUs k)^2))) by apply sqrt_pos.
    assert (P2 : 0 <= sqrt (Rsum Th (fun k => (VVs k)^2))) by apply sqrt_pos.
    pose proof NU_nonneg. pose proof NV_nonneg. nra. }
  assert (HB11 : Rsum Fl (fun f => UUs (e1 f) * VVs (e1 f))
                 <= Nf * (NU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UUs VVs e1 e1 UUs_nonneg VVs_nonneg
               H_mult1 H_mult1). }
    nra. }
  assert (HB12 : Rsum Fl (fun f => UUs (e1 f) * VVs (e2 f))
                 <= Nf * (NU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UUs VVs e1 e2 UUs_nonneg VVs_nonneg
               H_mult1 H_mult2). }
    nra. }
  assert (HB21 : Rsum Fl (fun f => UUs (e2 f) * VVs (e1 f))
                 <= Nf * (NU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UUs VVs e2 e1 UUs_nonneg VVs_nonneg
               H_mult2 H_mult1). }
    nra. }
  assert (HB22 : Rsum Fl (fun f => UUs (e2 f) * VVs (e2 f))
                 <= Nf * (NU * NV)).
  { eapply Rle_trans.
    { apply (face_CS_pair UUs VVs e2 e2 UUs_nonneg VVs_nonneg
               H_mult2 H_mult2). }
    nra. }
  unfold K6d.
  destruct kd_nonneg as [Hd11 [Hd12 [Hd21 Hd22]]].
  pose proof NU_nonneg. pose proof NV_nonneg.
  assert (HPN : 0 <= NU * NV) by nra.
  assert (M11 : kd11 * Rsum Fl (fun f => UUs (e1 f) * VVs (e1 f))
                <= kd11 * (Nf * (NU * NV))) by nra.
  assert (M12 : kd12 * Rsum Fl (fun f => UUs (e1 f) * VVs (e2 f))
                <= kd12 * (Nf * (NU * NV))) by nra.
  assert (M21 : kd21 * Rsum Fl (fun f => UUs (e2 f) * VVs (e1 f))
                <= kd21 * (Nf * (NU * NV))) by nra.
  assert (M22 : kd22 * Rsum Fl (fun f => UUs (e2 f) * VVs (e2 f))
                <= kd22 * (Nf * (NU * NV))) by nra.
  nra.
Qed.

(* ========================================================================= *)
(*  Step 8: assembly (eq:assembly / eq:sharpcont).                           *)
(* ========================================================================= *)

Definition KUV : R :=
  2 + 1 + 1 + C2 + 4 * Cb^2 / c1 + 2 * Cb / sqrt c1 + 2 * Cb / sqrt c1
  + 2 * Cb * Cinv / c1 + 2 * Cb * Cinv / c1 + 1 + 1 + Cinv / c2
  + sqrt C2 + sqrt C2 + K6d.
Definition KPV : R := sqrt c1 + K6b.

Theorem abstract_continuity_sharp :
  BS <= KUV * (NU * NV) + KPV * (BrP * NV) + sqrt c1 * (BrU * NV).
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
  pose proof (Rle_abs T1). pose proof (Rle_abs (T3 + T14)).
  pose proof (Rle_abs T5). pose proof (Rle_abs T15).
  pose proof (Rle_abs T6). pose proof (Rle_abs T7).
  pose proof (Rle_abs T8). pose proof (Rle_abs T9).
  pose proof (Rle_abs T10). pose proof (Rle_abs T12).
  pose proof (Rle_abs T16). pose proof (Rle_abs T17).
  pose proof (Rle_abs T18). pose proof (Rle_abs T13c).
  pose proof (Rle_abs (IIterm + Vterm)). pose proof (Rle_abs IIIterm).
  pose proof (Rle_abs (- VolP)) as HrV. rewrite Rabs_Ropp in HrV.
  pose proof (Rle_abs (- JmpP)) as HrJ. rewrite Rabs_Ropp in HrJ.
  unfold KUV, KPV. lra.
Qed.

(* ========================================================================= *)
(*  Step 9: absorption of the working norm of U (eq:absorb1--eq:normconv).   *)
(* ========================================================================= *)

Definition Kagg : R :=
  sqrt (Cinv^2 + (sqrt c1)^2 + (sqrt c1)^2
        + (Cinv * sqrt c1 / c2 + Cinv)^2 + Cb^2).
Definition Kabs : R := sqrt 2 * Kagg.

Lemma absorb_elem : forall k,
  perU k
  <= 2 * Kagg^2
     * ((aK k * sqrt (t2 k) / hK k * nrm (uu k))^2
        + (aK k * sqrt (t1 k) / hK k * nrm (pp k))^2).
Proof.
  intro k.
  set (Ak := aK k * sqrt (t2 k) / hK k * nrm (uu k)).
  set (Bk := aK k * sqrt (t1 k) / hK k * nrm (pp k)).
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (t1_pos' k) as Ht1. pose proof (t2_pos' k) as Ht2.
  pose proof (nrm_nonneg Hs (uu k)) as Hnu.
  pose proof (nrm_nonneg Hs (pp k)) as Hnp.
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
  (*  eq:absorb1  *)
  assert (P1 : sqrt nu * nrm (gu k) <= Cinv * Ak).
  { pose proof (Hw_gu k) as HG.
    pose proof (L_abs1 k) as HA1.
    assert (Hsn : 0 <= sqrt nu) by apply sqrt_pos.
    assert (Hsa : 0 <= sqrt (aK k)) by apply sqrt_pos.
    assert (Hci : 0 <= Cinv / hK k)
      by (apply div_nonneg; lra).
    (*  sqrt nu ngu <= sqrt nu (Cinv/h) sqrt aK nuu
                    <= (Cinv/h) (aK sqrt t2) nuu = Cinv Ak  *)
    assert (S1 : sqrt nu * nrm (gu k)
                 <= Cinv / hK k * (sqrt nu * sqrt (aK k)) * nrm (uu k)) by nra.
    assert (S2 : Cinv / hK k * (sqrt nu * sqrt (aK k)) * nrm (uu k)
                 <= Cinv / hK k * (aK k * sqrt (t2 k)) * nrm (uu k)).
    { assert (Hg : 0 <= Cinv / hK k * nrm (uu k)) by nra. nra. }
    assert (E : Cinv / hK k * (aK k * sqrt (t2 k)) * nrm (uu k) = Cinv * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  eq:absorb2  *)
  assert (P2 : sqrt (sg k) * nrm (uu k) <= sqrt c1 * Ak).
  { pose proof (L_abs2 k) as HA2.
    pose proof (L_P3div k) as HP3.
    assert (S1 : sqrt (sg k) * nrm (uu k)
                 <= sqrt (ph k) * nrm (uu k)) by nra.
    rewrite HP3 in S1.
    assert (E : sqrt c1 * aK k * sqrt (t2 k) / hK k * nrm (uu k)
                = sqrt c1 * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  eq:absorb3  *)
  assert (P3 : sqrt eps * nrm (pp k) <= sqrt c1 * Bk).
  { pose proof (L_P5div k) as HP5.
    assert (S1 : sqrt eps * nrm (pp k)
                 <= sqrt c1 * aK k * sqrt (t1 k) / hK k * nrm (pp k)) by nra.
    assert (E : sqrt c1 * aK k * sqrt (t1 k) / hK k * nrm (pp k)
                = sqrt c1 * Bk).
    { unfold Bk. field. lra. }
    lra. }
  (*  eq:absorb4  *)
  assert (P4 : sqrt (t1 k) * nrm (xu k)
               <= (Cinv * sqrt c1 / c2) * Ak + Cinv * Bk).
  { assert (Htri : nrm (xu k) <= nrm (cxu k) + nrm (gpu k)).
    { unfold xu. apply nrm_triangle. }
    pose proof (Hw_cxu k) as HCx.
    pose proof (Hw_gpu k) as HGp.
    pose proof (L_convph k) as HCp.
    pose proof (L_pst k) as HPst.
    pose proof (L_P3div k) as HP3.
    pose proof (nrm_nonneg Hs (cxu k)) as Hnc.
    pose proof (nrm_nonneg Hs (gpu k)) as Hng.
    pose proof (am_nonneg k) as Hm.
    pose proof sqrtc1_pos as Hsc.
    (*  velocity part:  sqrt t1 * (Cinv aK am / h)
          = (Cinv/c2) sqrt t1 (c2 aK am / h) <= (Cinv/c2) sqrt t1 ph
          <= (Cinv/c2) sqrt ph = (Cinv/c2) sqrt c1 aK sqrt t2 / h  *)
    assert (Hci : 0 <= Cinv / c2) by (apply div_nonneg; lra).
    assert (V1 : sqrt (t1 k) * (c2 * aK k * am k / hK k)
                 <= sqrt (t1 k) * ph k) by nra.
    assert (V2 : sqrt (t1 k) * ph k <= sqrt (ph k)) by nra.
    assert (V3 : sqrt (t1 k) * (Cinv / hK k * aK k * am k)
                 <= Cinv / c2 * sqrt (ph k)).
    { assert (E1 : sqrt (t1 k) * (Cinv / hK k * aK k * am k)
                   = Cinv / c2 * (sqrt (t1 k) * (c2 * aK k * am k / hK k)))
        by (field; lra).
      rewrite E1. nra. }
    rewrite HP3 in V3.
    (*  assemble  *)
    assert (S1 : sqrt (t1 k) * nrm (xu k)
                 <= sqrt (t1 k) * nrm (cxu k)
                    + sqrt (t1 k) * nrm (gpu k)) by nra.
    assert (S2 : sqrt (t1 k) * nrm (cxu k)
                 <= sqrt (t1 k) * (Cinv / hK k * aK k * am k) * nrm (uu k))
      by nra.
    assert (S3 : sqrt (t1 k) * (Cinv / hK k * aK k * am k) * nrm (uu k)
                 <= Cinv / c2 * (sqrt c1 * aK k * sqrt (t2 k) / hK k)
                    * nrm (uu k)) by nra.
    assert (E2 : Cinv / c2 * (sqrt c1 * aK k * sqrt (t2 k) / hK k)
                   * nrm (uu k)
                 = (Cinv * sqrt c1 / c2) * Ak).
    { unfold Ak. field. lra. }
    assert (S4 : sqrt (t1 k) * nrm (gpu k)
                 <= sqrt (t1 k) * (Cinv / hK k * aK k) * nrm (pp k)) by nra.
    assert (E3 : sqrt (t1 k) * (Cinv / hK k * aK k) * nrm (pp k)
                 = Cinv * Bk).
    { unfold Bk. field. lra. }
    lra. }
  (*  eq:absorb5  *)
  assert (P5 : sqrt (t2 k) * nrm (divu k) <= Cb * Ak).
  { pose proof (Hw_divu k) as HD.
    assert (S1 : sqrt (t2 k) * nrm (divu k)
                 <= sqrt (t2 k) * (Cb * aK k / hK k) * nrm (uu k)) by nra.
    assert (E : sqrt (t2 k) * (Cb * aK k / hK k) * nrm (uu k) = Cb * Ak).
    { unfold Ak. field. lra. }
    lra. }
  (*  nonnegativity of the pieces  *)
  assert (Hp1 : 0 <= sqrt nu * nrm (gu k)).
  { assert (0 <= sqrt nu) by apply sqrt_pos.
    pose proof (nrm_nonneg Hs (gu k)). nra. }
  assert (Hp2 : 0 <= sqrt (sg k) * nrm (uu k)).
  { assert (0 <= sqrt (sg k)) by apply sqrt_pos. nra. }
  assert (Hp3 : 0 <= sqrt eps * nrm (pp k)).
  { assert (0 <= sqrt eps) by apply sqrt_pos. nra. }
  assert (Hp4 : 0 <= sqrt (t1 k) * nrm (xu k)).
  { pose proof (nrm_nonneg Hs (xu k)). nra. }
  assert (Hp5 : 0 <= sqrt (t2 k) * nrm (divu k)).
  { pose proof (nrm_nonneg Hs (divu k)). nra. }
  assert (Hk1 : 0 <= Cinv) by lra.
  assert (Hk2 : 0 <= sqrt c1) by apply sqrt_pos.
  assert (Hk4 : 0 <= Cinv * sqrt c1 / c2).
  { apply div_nonneg; [nra | lra]. }
  assert (Hk6 : 0 <= Cb) by lra.
  (*  the aggregation lemma of ContinuityAlgebra  *)
  pose proof (norm5_absorption
                (sqrt nu * nrm (gu k)) (sqrt (sg k) * nrm (uu k))
                (sqrt eps * nrm (pp k)) (sqrt (t1 k) * nrm (xu k))
                (sqrt (t2 k) * nrm (divu k))
                Ak Bk
                Cinv (sqrt c1) (sqrt c1) (Cinv * sqrt c1 / c2) Cinv Cb
                Hp1 Hp2 Hp3 Hp4 Hp5 HAk HBk
                Hk1 Hk2 Hk2 Hk4 Hk1 Hk6
                P1 P2 P3 P4 P5) as HN5.
  fold Kagg in HN5.
  (*  square the aggregated bound  *)
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
  assert (HKa : 0 <= Kagg) by apply sqrt_pos.
  assert (Hsq : S5 <= Kagg^2 * (Ak + Bk)^2).
  { fold S5 in HN5.
    assert (Hr : 0 <= Kagg * (Ak + Bk)) by nra.
    assert (Hss : sqrt S5 * sqrt S5 <= (Kagg * (Ak + Bk)) * (Kagg * (Ak + Bk))).
    { pose proof (sqrt_pos S5). nra. }
    rewrite sqrt_sqrt in Hss by exact HS5.
    nra. }
  (*  (A+B)^2 <= 2 A^2 + 2 B^2  *)
  assert (H2ab : (Ak + Bk)^2 <= 2 * Ak^2 + 2 * Bk^2).
  { pose proof (pow2_ge_0 (Ak - Bk)). nra. }
  (*  perU k equals S5  *)
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
  assert (HK2 : 0 <= Kagg^2) by (pose proof (pow2_ge_0 Kagg); lra).
  nra.
Qed.

Lemma step9 : NU <= Kabs * (BrU + BrP).
Proof.
  unfold NU, Kabs.
  assert (HKa : 0 <= Kagg) by apply sqrt_pos.
  assert (HBrU2 : 0 <= BrU2).
  { unfold BrU2. apply Rsum_nonneg. intro k.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t2_pos' k) as Ht. pose proof (ip_pos Hs (uu k)) as Hi.
    assert (0 <= (aK k)^2 * t2 k) by nra.
    assert (0 <= (aK k)^2 * t2 k / (hK k)^2)
      by (apply div_nonneg; nra).
    nra. }
  assert (HBrP2 : 0 <= BrP2).
  { unfold BrP2. apply Rsum_nonneg. intro k.
    pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
    pose proof (t1_pos' k) as Ht. pose proof (ip_pos Hs (pp k)) as Hi.
    assert (0 <= (aK k)^2 * t1 k) by nra.
    assert (0 <= (aK k)^2 * t1 k / (hK k)^2)
      by (apply div_nonneg; nra).
    nra. }
  (*  NU2 <= 2 Kagg^2 (BrU2 + BrP2)  *)
  assert (HN2 : NU2 <= 2 * Kagg^2 * (BrU2 + BrP2)).
  { unfold NU2.
    eapply Rle_trans.
    { apply Rsum_le. exact absorb_elem. }
    assert (E : Rsum Th (fun k =>
        2 * Kagg^2
        * ((aK k * sqrt (t2 k) / hK k * nrm (uu k))^2
           + (aK k * sqrt (t1 k) / hK k * nrm (pp k))^2))
        = 2 * Kagg^2
          * (Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * nrm (uu k))^2)
             + Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * nrm (pp k))^2))).
    { rewrite <- Rsum_plus, <- Rsum_scal.
      apply Rsum_ext. intro k. ring. }
    rewrite E.
    assert (EU : Rsum Th (fun k => (aK k * sqrt (t2 k) / hK k * nrm (uu k))^2)
                 = BrU2).
    { unfold BrU2. apply Rsum_ext. intro k.
      apply sq_weight2; [apply Rlt_le, t2_pos' | apply hK_pos]. }
    assert (EP : Rsum Th (fun k => (aK k * sqrt (t1 k) / hK k * nrm (pp k))^2)
                 = BrP2).
    { unfold BrP2. apply Rsum_ext. intro k.
      apply sq_weight2; [apply Rlt_le, t1_pos' | apply hK_pos]. }
    rewrite EU, EP. lra. }
  (*  take square roots  *)
  eapply Rle_trans.
  { apply sqrt_le_1_alt. exact HN2. }
  rewrite sqrt_mult_alt by (pose proof (pow2_ge_0 Kagg); lra).
  assert (E2 : sqrt (2 * Kagg^2) = sqrt 2 * Kagg).
  { rewrite sqrt_mult_alt by lra.
    rewrite sqrt_pow2 by exact HKa. reflexivity. }
  rewrite E2.
  assert (Hplus : sqrt (BrU2 + BrP2) <= BrU + BrP).
  { unfold BrU, BrP. apply sqrt_plus_le; assumption. }
  assert (Hm : 0 <= sqrt 2 * Kagg).
  { assert (0 <= sqrt 2) by apply sqrt_pos. nra. }
  nra.
Qed.

(* ========================================================================= *)
(*  lemma:Continuity, abstract form.                                         *)
(* ========================================================================= *)

Lemma KUV_nonneg : 0 <= KUV.
Proof.
  unfold KUV.
  pose proof sqrtc1_pos as Hsc.
  assert (H1 : 0 <= 4 * Cb^2 / c1) by (apply div_nonneg; nra).
  assert (H2 : 0 <= 2 * Cb / sqrt c1) by (apply div_nonneg; nra).
  assert (H3 : 0 <= 2 * Cb * Cinv / c1) by (apply div_nonneg; nra).
  assert (H4 : 0 <= Cinv / c2) by (apply div_nonneg; nra).
  assert (H5 : 0 <= sqrt C2) by apply sqrt_pos.
  assert (H6 : 0 <= K6d).
  { unfold K6d.
    destruct kd_nonneg as [D1 [D2 [D3 D4]]].
    assert (0 <= Cface * Nf) by nra.
    nra. }
  lra.
Qed.

Lemma KPV_nonneg : 0 <= KPV.
Proof.
  unfold KPV.
  pose proof sqrtc1_pos as Hsc.
  assert (H6 : 0 <= K6b).
  { unfold K6b.
    destruct kb_nonneg as [D1 [D2 [D3 D4]]].
    assert (0 <= Cface * Nf) by nra.
    nra. }
  lra.
Qed.

Definition Ctot : R := KUV * Kabs + KPV + sqrt c1.

Theorem abstract_continuity : BS <= Ctot * ((BrU + BrP) * NV).
Proof.
  pose proof abstract_continuity_sharp as HS.
  pose proof step9 as H9.
  pose proof KUV_nonneg as HK1.
  pose proof KPV_nonneg as HK2.
  pose proof NV_nonneg as HNV.
  pose proof BrU_nonneg as HBU.
  pose proof BrP_nonneg as HBP.
  pose proof NU_nonneg as HNU.
  pose proof sqrtc1_pos as Hsc.
  (*  KUV NU NV <= KUV Kabs (BrU + BrP) NV  *)
  assert (HKN : 0 <= KUV * NV) by nra.
  assert (S1 : KUV * (NU * NV) <= KUV * (Kabs * (BrU + BrP) * NV)) by nra.
  (*  KPV BrP NV <= KPV (BrU + BrP) NV  *)
  assert (HKP : 0 <= KPV * NV) by nra.
  assert (S2 : KPV * (BrP * NV) <= KPV * ((BrU + BrP) * NV)) by nra.
  (*  sqrt c1 BrU NV <= sqrt c1 (BrU + BrP) NV  *)
  assert (HSC : 0 <= sqrt c1 * NV) by nra.
  assert (S3 : sqrt c1 * (BrU * NV) <= sqrt c1 * ((BrU + BrP) * NV)) by nra.
  unfold Ctot. nra.
Qed.

End AbstractContinuity.
