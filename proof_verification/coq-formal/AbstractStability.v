(* ========================================================================= *)
(*  AbstractStability.v                                                      *)
(*                                                                           *)
(*  lemma:Stability of the paper, proved IN FULL -- including the elemental  *)
(*  inner-product manipulations, the parametrized Young step, and the        *)
(*  summation over the mesh -- from a named, quantitative trusted base of    *)
(*  exactly TWO analytic facts:                                              *)
(*                                                                           *)
(*    (HBS)  the Galerkin-testing / adjoint-expansion identity: testing      *)
(*           B_S with U_h itself gives                                       *)
(*             B_S(U_h,U_h) = 2 nu ||a^(1/2) P grad u||^2 + sigma ||u||^2    *)
(*                            + eps ||p||^2                                  *)
(*                            - sum_K tau_1 < L*_m , L_m >_K                 *)
(*                            - sum_K tau_2 < L*_c , L_c >_K ,               *)
(*           where the momentum/mass residual vectors are the paper's        *)
(*           (eq:strongop, eq:XG):  L_m = alpha X(U) - G(u),                 *)
(*           L*_m = -alpha X(U) - G(u),  L_c = eps p + div(alpha u),         *)
(*           L*_c = eps p - div(alpha u).  This packages: integration by     *)
(*           parts with u_h = 0 on the boundary, div(alpha a) = 0            *)
(*           (H:advection), and idempotence/symmetry of the projector.       *)
(*                                                                           *)
(*    (S3)   the weighted inverse estimate eq:winv-divvisc (lem:winv):       *)
(*             || 2 div(alpha nu P grad u) ||_K                              *)
(*               <= (2 nu Cbar / h_K) alpha_K^(1/2)                          *)
(*                  || alpha^(1/2) P grad u ||_K .                           *)
(*                                                                           *)
(*  Everything else -- the difference-of-squares cancellation, the           *)
(*  expansion of || G(u) ||^2, Cauchy--Schwarz, the Young step with          *)
(*  parameter xi Cbar^2 nu alpha_K / h_K^2, the exact cancellation of the    *)
(*  inverse-estimate constant, the coefficient collection, the elemental     *)
(*  coercivity, and the assembly over elements -- is machine-checked here,   *)
(*  reusing the closed algebraic theorems of StabilityAlgebra.v.             *)
(*                                                                           *)
(*  Conclusion:  B_S(U_h,U_h) >= C_stab ||| U_h |||^2  with                  *)
(*  C_stab = min{ C_visc, C_u, 1 - C_2, 1 } > 0 explicit, under              *)
(*  c1 > 2 xi Cbar^2, xi > 2, C_2 < 1 and eq:UpperBoundOnEpsilon.            *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import StabilityAlgebra InnerSpace AbstractSums.
Local Open Scope R_scope.

Section AbstractStability.

(* ---------- The ambient inner-product space and the mesh ------------------- *)

Variable Hs : PreHilbert.
Notation V := (carrier Hs).
Notation "'<<' x , y '>>'" := (ip Hs x y) (at level 0).
Notation "x '+v' y" := (vadd Hs x y) (at level 50, left associativity).
Notation "a '*v' x" := (vscal Hs a x) (at level 40).

Variable K : Type.
Variable Th : list K.          (*  the triangulation  *)

(* ---------- Physical constants and the numerical parameters ---------------- *)

Variables (nu sigma eps c1 c2 Cb xi C2 : R).
Hypothesis nu_pos  : 0 < nu.
Hypothesis sigma_pos : 0 < sigma.
Hypothesis eps_pos : 0 < eps.
Hypothesis c1_pos  : 0 < c1.
Hypothesis c2_pos  : 0 < c2.
Hypothesis Cb_pos  : 0 < Cb.        (*  Cbar of rem:winvconst / lem:winv  *)
Hypothesis xi_pos  : 0 < xi.
Hypothesis C2_pos  : 0 < C2.
Hypothesis C2_lt_1 : C2 < 1.
Hypothesis c1_large : c1 > 2 * xi * Cb^2.   (*  eq:conditions_on_num_param  *)
Hypothesis xi_large : xi > 2.

(* ---------- Mesh data ------------------------------------------------------- *)

Variables (hK aK am : K -> R).       (*  h_K, alpha_K, |a|_{infty,K}  *)
Hypothesis hK_pos  : forall k, 0 < hK k.
Hypothesis aK_pos  : forall k, 0 < aK k.
Hypothesis am_nonneg : forall k, 0 <= am k.

(*  The stabilization parameters, per element, via the paper's formulas      *)
(*  (eq:TauNavierStokes, eq:Tau1Final, eq:Tau2Final, eq:SigmaAlpha) --       *)
(*  reusing the closed definitions of StabilityAlgebra.v verbatim.           *)
Definition t1 (k : K) : R := tau1 nu (hK k) (aK k) sigma (am k) c1 c2.
Definition t2 (k : K) : R := tau2 nu (hK k) (aK k) (am k) c1 c2.
Definition sg (k : K) : R := sigt nu (hK k) (aK k) sigma (am k) c1 c2.

(* ---------- The discrete fields, elementwise ------------------------------- *)

(*  For U_h = [u_h, p_h] in the discrete space:                               *)
(*    gu k   ~  (alpha^(1/2) P grad u_h)|_K                                   *)
(*    uu k   ~  u_h|_K                                                        *)
(*    pp k   ~  p_h|_K                                                        *)
(*    xu k   ~  (alpha X(U_h))|_K = (alpha a . grad u_h + alpha grad p_h)|_K  *)
(*    divu k ~  (div(alpha u_h))|_K                                           *)
(*    du k   ~  (2 div(alpha nu P grad u_h))|_K                               *)
Variables (gu uu pp xu divu du : K -> V).

(* ---------- The residual vectors (eq:strongop, eq:XG) ----------------------- *)

Definition Bv (k : K) : V := vsub (du k) (sigma *v (uu k)).      (*  G(u_h)|_K  *)
Definition L_m (k : K) : V := (xu k) +v (vopp (Bv k)).           (*  alpha X - G *)
Definition Lstar_m (k : K) : V := (vopp (xu k)) +v (vopp (Bv k)).   (* -alpha X - G *)
Definition L_c (k : K) : V := (divu k) +v (eps *v (pp k)).
Definition Lstar_c (k : K) : V := (vopp (divu k)) +v (eps *v (pp k)).

(* ---------- The trusted base ------------------------------------------------ *)

Variable BS : R.    (*  the number B_S(U_h, U_h)  *)

(*  (HBS): Galerkin testing + adjoint expansion (see header).  *)
Hypothesis HBS :
  BS = 2 * nu * Rsum Th (fun k => << (gu k) , (gu k) >>)
       + sigma * Rsum Th (fun k => << (uu k) , (uu k) >>)
       + eps * Rsum Th (fun k => << (pp k) , (pp k) >>)
       - Rsum Th (fun k => t1 k * << (Lstar_m k) , (L_m k) >>)
       - Rsum Th (fun k => t2 k * << (Lstar_c k) , (L_c k) >>).

(*  (S3): the weighted inverse estimate eq:winv-divvisc.  *)
Hypothesis S3 :
  forall k, nrm (du k) <= (2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k).

(*  eq:UpperBoundOnEpsilon, in its elementwise (infimum) form.  *)
Hypothesis Heps :
  forall k, eps <= eps_max nu (hK k) (aK k) sigma (am k) c1 c2 C2.

(* ========================================================================= *)
(*  The proof.                                                               *)
(* ========================================================================= *)

(*  Elemental squared norms, as inner products.  *)
Definition Vn (k : K) : R := << (gu k) , (gu k) >>.
Definition Un (k : K) : R := << (uu k) , (uu k) >>.
Definition Pn (k : K) : R := << (pp k) , (pp k) >>.
Definition Xn (k : K) : R := << (xu k) , (xu k) >>.
Definition Dn (k : K) : R := << (divu k) , (divu k) >>.

Lemma Vn_nonneg : forall k, 0 <= Vn k. Proof. intro k. apply (ip_pos Hs). Qed.
Lemma Un_nonneg : forall k, 0 <= Un k. Proof. intro k. apply (ip_pos Hs). Qed.
Lemma Pn_nonneg : forall k, 0 <= Pn k. Proof. intro k. apply (ip_pos Hs). Qed.
Lemma Xn_nonneg : forall k, 0 <= Xn k. Proof. intro k. apply (ip_pos Hs). Qed.
Lemma Dn_nonneg : forall k, 0 <= Dn k. Proof. intro k. apply (ip_pos Hs). Qed.

(*  Positivity of the elemental parameters, from the closed lemmas.  *)
Lemma t1_pos : forall k, 0 < t1 k.
Proof.
  intro k. unfold t1.
  apply tau1_pos; auto.
Qed.

Lemma t2_pos : forall k, 0 < t2 k.
Proof.
  intro k. unfold t2.
  apply tau2_pos; auto.
Qed.

(* ---------- Step 1: the difference-of-squares expansion --------------------- *)

(*  < L*_m , L_m >_K = ||G(u)||^2 - ||alpha X||^2   (eq:StabilityEstimate).   *)
Lemma momentum_pairing :
  forall k, << (Lstar_m k) , (L_m k) >> = << (Bv k) , (Bv k) >> - Xn k.
Proof.
  intro k. unfold Lstar_m, L_m, Xn.
  rewrite (diff_of_squares Hs (xu k) (vopp (Bv k))).
  rewrite (ip_opp_opp Hs).
  reflexivity.
Qed.

(*  < L*_c , L_c >_K = eps^2 ||p||^2 - ||div(alpha u)||^2.                    *)
Lemma mass_pairing :
  forall k, << (Lstar_c k) , (L_c k) >> = eps^2 * Pn k - Dn k.
Proof.
  intro k. unfold Lstar_c, L_c, Pn, Dn.
  rewrite (diff_of_squares Hs (divu k) (eps *v (pp k))).
  rewrite ip_scal_r, (ip_scal_l Hs).
  ring.
Qed.

(*  || G(u) ||^2 expanded.  *)
Lemma Bv_expand :
  forall k,
    << (Bv k) , (Bv k) >>
    = << (du k) , (du k) >> - 2 * sigma * << (du k) , (uu k) >>
      + sigma^2 * Un k.
Proof.
  intro k. unfold Bv, Un.
  rewrite (ip_expand_sub Hs).
  rewrite ip_scal_r, (ip_scal_l Hs), ip_scal_r.
  ring.
Qed.

(* ---------- Step 2: the inverse estimate, squared --------------------------- *)

Lemma du_sq_bound :
  forall k, << (du k) , (du k) >> <= 4 * nu^2 * Cb^2 * (aK k / (hK k)^2) * Vn k.
Proof.
  intro k.
  pose proof (S3 k) as HS.
  pose proof (nrm_nonneg Hs (du k)) as Hd0.
  pose proof (nrm_nonneg Hs (gu k)) as Hg0.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  assert (Hrhs0 : 0 <= (2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k)).
  { assert (H1 : 0 < 2 * nu * Cb) by nra.
    assert (H2 : 0 < 2 * nu * Cb / hK k) by (apply Rdiv_lt_0_compat; lra).
    assert (H3 : 0 <= sqrt (aK k)) by apply sqrt_pos.
    nra. }
  (*  square both sides  *)
  assert (Hsq : nrm (du k) * nrm (du k)
                <= ((2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k))
                   * ((2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k))).
  { nra. }
  rewrite (nrm_sq Hs) in Hsq.
  replace (((2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k))
           * ((2 * nu * Cb / hK k) * sqrt (aK k) * nrm (gu k)))
    with ((2 * nu * Cb / hK k) * (2 * nu * Cb / hK k)
          * (sqrt (aK k) * sqrt (aK k)) * (nrm (gu k) * nrm (gu k)))
    in Hsq by ring.
  rewrite sqrt_sqrt in Hsq by lra.
  rewrite (nrm_sq Hs) in Hsq.
  unfold Vn.
  replace (4 * nu^2 * Cb^2 * (aK k / (hK k)^2) * << (gu k) , (gu k) >>)
    with ((2 * nu * Cb / hK k) * (2 * nu * Cb / hK k) * aK k
          * << (gu k) , (gu k) >>) by (field; lra).
  exact Hsq.
Qed.

(* ---------- Step 3: the Young step and the coefficient collection ----------- *)

(*  The per-element lower bound: the left-hand-side elemental contribution    *)
(*  dominates the coefficient-collected form of eq:StabilityEstimateFinal.    *)
Lemma element_lower_bound :
  forall k,
    2 * nu * Vn k + sigma * Un k + eps * Pn k
    - t1 k * << (Lstar_m k) , (L_m k) >>
    - t2 k * << (Lstar_c k) , (L_c k) >>
    >= eps * (1 - eps * t2 k) * Pn k
       + visc_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi * Vn k
       + u_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi * Un k
       + t1 k * Xn k + t2 k * Dn k.
Proof.
  intro k.
  rewrite momentum_pairing, mass_pairing, Bv_expand.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  pose proof (hK_pos k) as Hh. pose proof (aK_pos k) as Ha.
  pose proof (Vn_nonneg k) as HV. pose proof (Un_nonneg k) as HU.
  pose proof (du_sq_bound k) as Hdu.
  pose proof (ip_pos Hs (du k)) as Hdu0.
  (*  Young with parameter lam = xi Cb^2 nu aK / h^2  *)
  set (lam := xi * Cb^2 * nu * aK k / (hK k)^2).
  assert (Hlam : 0 < lam).
  { unfold lam.
    assert (H1 : 0 < xi * Cb^2) by nra.
    assert (H2 : 0 < xi * Cb^2 * nu) by nra.
    assert (H3 : 0 < xi * Cb^2 * nu * aK k) by nra.
    assert (H4 : 0 < (hK k)^2) by nra.
    apply Rdiv_lt_0_compat; lra. }
  pose proof (young_vector Hs (du k) (uu k) lam Hlam) as HY.
  (*  the cross term:  2 sigma t1 <du,uu> >= - sigma t1 <du,du>/lam           *)
  (*                                          - sigma t1 lam <uu,uu>          *)
  assert (Hst1 : 0 <= sigma * t1 k) by nra.
  assert (Hcross :
      2 * sigma * t1 k * << (du k) , (uu k) >>
      >= - sigma * t1 k * (<< (du k) , (du k) >> / lam)
         - sigma * t1 k * (lam * Un k)).
  { unfold Un. nra. }
  (*  inverse estimate through the Young denominator: exact cancellation      *)
  (*  (4 nu^2 Cb^2 aK / h^2) / lam = 4 nu / xi.                               *)
  assert (Hdu_lam :
      << (du k) , (du k) >> / lam <= (4 * nu / xi) * Vn k).
  { apply (Rmult_le_reg_r lam); [exact Hlam |].
    replace (<< (du k) , (du k) >> / lam * lam)
      with (<< (du k) , (du k) >>) by (field; lra).
    replace ((4 * nu / xi) * Vn k * lam)
      with (4 * nu^2 * Cb^2 * (aK k / (hK k)^2) * Vn k)
      by (unfold lam; field; lra).
    exact Hdu. }
  assert (Hdiv0 : 0 <= << (du k) , (du k) >> / lam).
  { apply Rmult_le_pos; [exact Hdu0 | left; apply Rinv_0_lt_compat; exact Hlam]. }
  assert (T1 : t1 k * << (du k) , (du k) >>
               <= t1 k * (4 * nu^2 * Cb^2 * (aK k / (hK k)^2) * Vn k)) by nra.
  assert (T2 : sigma * t1 k * (<< (du k) , (du k) >> / lam)
               <= sigma * t1 k * ((4 * nu / xi) * Vn k)) by nra.
  (*  collect: identify the coefficients with visc_final and u_final.         *)
  assert (Evisc :
      visc_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi
      = 2 * nu - 4 * nu^2 * Cb^2 * (aK k / (hK k)^2) * t1 k
        - sigma * t1 k * (4 * nu / xi)).
  { pose proof (t1_pos k) as Hp. unfold t1 in Hp.
    unfold visc_final, t1. field.
    repeat split; lra. }
  assert (Eu :
      u_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi
      = sigma - sigma^2 * t1 k
        - sigma * t1 k * (xi * Cb^2 * nu * aK k / (hK k)^2)).
  { pose proof (t1_pos k) as Hp. unfold t1 in Hp.
    unfold u_final, t1. field.
    repeat split; lra. }
  rewrite Evisc, Eu.
  unfold lam in Hcross, Hdu_lam, Hdiv0, T2.
  nra.
Qed.

(* ---------- Step 4: elemental coercivity, from StabilityAlgebra ------------- *)

Lemma element_coercive :
  forall k,
    eps * (1 - eps * t2 k) * Pn k
    + visc_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi * Vn k
    + u_final nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi * Un k
    + t1 k * Xn k + t2 k * Dn k
    >= C_stab c1 Cb xi C2
       * (eps * Pn k + nu * Vn k + sg k * Un k + t1 k * Xn k + t2 k * Dn k).
Proof.
  intro k.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  assert (HXn : 0 <= t1 k * Xn k)
    by (pose proof (Xn_nonneg k); nra).
  assert (HDn : 0 <= t2 k * Dn k)
    by (pose proof (Dn_nonneg k); nra).
  exact (elemental_coercivity nu (hK k) (aK k) sigma (am k) c1 c2 Cb xi
           nu_pos (hK_pos k) (aK_pos k) sigma_pos (am_nonneg k)
           c1_pos c2_pos Cb_pos xi_pos
           eps C2 eps_pos C2_pos
           (Pn k) (Vn k) (Un k) (t1 k * Xn k) (t2 k * Dn k)
           (Heps k) (Pn_nonneg k) (Vn_nonneg k) (Un_nonneg k) HXn HDn).
Qed.

(* ---------- Step 5: assembly over the mesh ----------------------------------- *)

(*  The triple norm squared (eq:triplenorm), and the per-element pieces.      *)
Definition perLHS (k : K) : R :=
  2 * nu * Vn k + sigma * Un k + eps * Pn k
  - t1 k * << (Lstar_m k) , (L_m k) >>
  - t2 k * << (Lstar_c k) , (L_c k) >>.

Definition perNorm (k : K) : R :=
  eps * Pn k + nu * Vn k + sg k * Un k + t1 k * Xn k + t2 k * Dn k.

Definition NormSq : R := Rsum Th perNorm.

Lemma BS_as_sum : BS = Rsum Th perLHS.
Proof.
  rewrite HBS.
  unfold perLHS.
  rewrite <- (Rsum_scal K Th (2 * nu) (fun k => << (gu k) , (gu k) >>)).
  rewrite <- (Rsum_scal K Th sigma (fun k => << (uu k) , (uu k) >>)).
  rewrite <- (Rsum_scal K Th eps (fun k => << (pp k) , (pp k) >>)).
  rewrite <- !Rsum_plus.
  rewrite <- !Rsum_minus.
  apply Rsum_ext. intro k.
  unfold Vn, Un, Pn. ring.
Qed.

(*  lemma:Stability, abstract form.  *)
Theorem abstract_stability :
  0 < C_stab c1 Cb xi C2 /\ BS >= C_stab c1 Cb xi C2 * NormSq.
Proof.
  split.
  - (*  positivity of the explicit constant  *)
    pose proof (stability_constants_positive c1 Cb xi
                  c1_pos Cb_pos xi_pos c1_large xi_large) as [HCv HCu].
    unfold C_stab.
    apply Rmin_pos; apply Rmin_pos; lra.
  - rewrite BS_as_sum.
    unfold NormSq.
    rewrite <- (Rsum_scal K Th (C_stab c1 Cb xi C2) perNorm).
    apply Rle_ge, Rsum_le.
    intro k.
    pose proof (element_lower_bound k) as H1.
    pose proof (element_coercive k) as H2.
    unfold perLHS, perNorm. lra.
Qed.

End AbstractStability.
