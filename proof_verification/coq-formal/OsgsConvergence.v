(* ========================================================================= *)
(*  OsgsConvergence.v                                                        *)
(*                                                                           *)
(*  Theorem 5.4 (th:convergence) of theory/osgs_a_priori/                    *)
(*  osgs_convergence.tex, proved as the composition of the four OSGS results, *)
(*  glued by a PROVED triangle inequality for the OSGS mesh-dependent triple  *)
(*  norm:                                                                    *)
(*                                                                           *)
(*    |||U - U_h|||  <=  C_conv * E(h),                                       *)
(*                                                                           *)
(*  the porosity-weighted l1 form eq:MainResult.  Following the note, write   *)
(*  U - U_h = Ehat + eta with eta = Uhat_h - U_h in X_h0; then                *)
(*                                                                           *)
(*    beta_stab |||eta|||^2 <= B_osgs(eta, W)              [th:stability]     *)
(*                           = B_osgs(U-U_h, W) - B_osgs(Ehat, W)  [bilinear] *)
(*                           <= (C_cons + C_int) E(h) |||W|||                 *)
(*                                    [lem:consistency + lem:interpolation]   *)
(*                           <= (C_cons + C_int) C_W E(h) |||eta|||,          *)
(*                                                                           *)
(*  so |||eta||| <= C E(h), and the triangle inequality with                  *)
(*  |||Ehat||| <= C E(h)  (lem:interpsize) concludes.                        *)
(*                                                                           *)
(*  The four inputs -- the stability lower bound and test-function            *)
(*  boundedness (abstract_osgs_stability applied to eta), interpolation       *)
(*  continuity (abstract_osgs_continterp at (Ehat,W)), the consistency error  *)
(*  (abstract_osgs_consistency at W), and the interpolation-error size        *)
(*  (lem:interpsize) -- enter as the glue hypotheses, exactly as              *)
(*  AbstractConvergence.v consumes the closed abstract_stability /            *)
(*  abstract_continterp; the one genuinely new algebraic item is the          *)
(*  bilinearity of B_osgs in its first argument (Hbilin), the analogue of the *)
(*  ASGS Horth.  Because the OSGS argument is inf--sup rather than            *)
(*  coercive, its test function W is NOT the trial function eta, so the four   *)
(*  inputs are carried as hypotheses about a common W rather than re-derived   *)
(*  internally.  The triangle inequality and the whole gluing arithmetic are  *)
(*  proved here from the real-number / pre-Hilbert axioms.                    *)
(*                                                                           *)
(*  No Admitted, no Axiom.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums OsgsNorm.
Local Open Scope R_scope.

Section OsgsConvergence.

Variable Hs : PreHilbert.
Notation V := (carrier Hs).
Notation "'<<' x , y '>>'" := (ip Hs x y) (at level 0).
Notation "x '+v' y" := (vadd Hs x y) (at level 50, left associativity).

Variable K : Type.
Variable Th : list K.

Variables (nu sigma eps c1 c2 : R).
Hypothesis nu_pos       : 0 < nu.
Hypothesis sigma_nonneg : 0 <= sigma.
Hypothesis eps_nonneg   : 0 <= eps.
Hypothesis c1_pos       : 0 < c1.
Hypothesis c2_pos       : 0 < c2.

Variables (hK aK am : K -> R).
Hypothesis hK_pos    : forall k, 0 < hK k.
Hypothesis aK_pos    : forall k, 0 < aK k.
Hypothesis am_nonneg : forall k, 0 <= am k.

Definition t1 (k : K) : R := ContinuityAlgebra.tau1 nu (hK k) (aK k) sigma (am k) c1 c2.
Definition t2 (k : K) : R := ContinuityAlgebra.tau2 nu (hK k) (aK k) (am k) c1 c2.

Lemma t1_pos : forall k, 0 < t1 k.
Proof. intro k. unfold t1. apply tau1_pos; auto. Qed.
Lemma t2_pos : forall k, 0 < t2 k.
Proof. intro k. unfold t2. apply tau2_pos; auto. Qed.

(* ---------- Atoms: the interpolation error E and the discrete error eta ---- *)

Variables (gE cxE gpE vE qE dvE : K -> V).   (*  Ehat = U - Uhat_h            *)
Variables (gN cxN gpN vN qN dvN : K -> V).   (*  eta  = Uhat_h - U_h          *)

Definition xE (k : K) : V := (cxE k) +v (gpE k).
Definition xN (k : K) : V := (cxN k) +v (gpN k).

(*  Per-element OSGS energies (eq:TripleNorm) of E, eta and their sum;       *)
(*  the sum E (+) eta represents U - U_h componentwise.                      *)
Definition perE (k : K) : R :=
  2 * nu * << (gE k) , (gE k) >> + sigma * << (vE k) , (vE k) >>
  + eps * << (qE k) , (qE k) >> + t1 k * << (xE k) , (xE k) >>
  + t2 k * << (dvE k) , (dvE k) >>.

Definition perN (k : K) : R :=
  2 * nu * << (gN k) , (gN k) >> + sigma * << (vN k) , (vN k) >>
  + eps * << (qN k) , (qN k) >> + t1 k * << (xN k) , (xN k) >>
  + t2 k * << (dvN k) , (dvN k) >>.

Definition perU (k : K) : R :=
  2 * nu * << (gE k) +v (gN k) , (gE k) +v (gN k) >>
  + sigma * << (vE k) +v (vN k) , (vE k) +v (vN k) >>
  + eps * << (qE k) +v (qN k) , (qE k) +v (qN k) >>
  + t1 k * << (xE k) +v (xN k) , (xE k) +v (xN k) >>
  + t2 k * << (dvE k) +v (dvN k) , (dvE k) +v (dvN k) >>.

Definition Qk (k : K) : R :=
  2 * nu * << (gE k) , (gN k) >> + sigma * << (vE k) , (vN k) >>
  + eps * << (qE k) , (qN k) >> + t1 k * << (xE k) , (xN k) >>
  + t2 k * << (dvE k) , (dvN k) >>.

Definition NE : R := sqrt (Rsum Th perE).
Definition Neta : R := sqrt (Rsum Th perN).
Definition NUU : R := sqrt (Rsum Th perU).

(* ---------- Nonnegativity of the per-element energies ---------------------- *)

Lemma perE_nonneg : forall k, 0 <= perE k.
Proof.
  intro k. pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  unfold perE.
  pose proof (ip_pos Hs (gE k)). pose proof (ip_pos Hs (vE k)).
  pose proof (ip_pos Hs (qE k)). pose proof (ip_pos Hs (xE k)).
  pose proof (ip_pos Hs (dvE k)).
  assert (0 <= 2*nu*<< (gE k),(gE k) >>) by nra.
  assert (0 <= sigma*<< (vE k),(vE k) >>) by nra.
  assert (0 <= eps*<< (qE k),(qE k) >>) by nra.
  assert (0 <= t1 k*<< (xE k),(xE k) >>) by nra.
  assert (0 <= t2 k*<< (dvE k),(dvE k) >>) by nra.
  lra.
Qed.

Lemma perN_nonneg : forall k, 0 <= perN k.
Proof.
  intro k. pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  unfold perN.
  pose proof (ip_pos Hs (gN k)). pose proof (ip_pos Hs (vN k)).
  pose proof (ip_pos Hs (qN k)). pose proof (ip_pos Hs (xN k)).
  pose proof (ip_pos Hs (dvN k)).
  assert (0 <= 2*nu*<< (gN k),(gN k) >>) by nra.
  assert (0 <= sigma*<< (vN k),(vN k) >>) by nra.
  assert (0 <= eps*<< (qN k),(qN k) >>) by nra.
  assert (0 <= t1 k*<< (xN k),(xN k) >>) by nra.
  assert (0 <= t2 k*<< (dvN k),(dvN k) >>) by nra.
  lra.
Qed.

Lemma perU_nonneg : forall k, 0 <= perU k.
Proof.
  intro k. pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  unfold perU.
  pose proof (ip_pos Hs ((gE k) +v (gN k))). pose proof (ip_pos Hs ((vE k) +v (vN k))).
  pose proof (ip_pos Hs ((qE k) +v (qN k))). pose proof (ip_pos Hs ((xE k) +v (xN k))).
  pose proof (ip_pos Hs ((dvE k) +v (dvN k))).
  assert (0 <= 2*nu*<< (gE k) +v (gN k),(gE k) +v (gN k) >>) by nra.
  assert (0 <= sigma*<< (vE k) +v (vN k),(vE k) +v (vN k) >>) by nra.
  assert (0 <= eps*<< (qE k) +v (qN k),(qE k) +v (qN k) >>) by nra.
  assert (0 <= t1 k*<< (xE k) +v (xN k),(xE k) +v (xN k) >>) by nra.
  assert (0 <= t2 k*<< (dvE k) +v (dvN k),(dvE k) +v (dvN k) >>) by nra.
  lra.
Qed.

(* ---------- perU = perE + 2 Qk + perN  (eq:perErr expansion) --------------- *)

Lemma perU_expand : forall k, perU k = perE k + 2 * Qk k + perN k.
Proof.
  intro k. unfold perU, perE, perN, Qk.
  rewrite (ip_expand_add Hs (gE k) (gN k)).
  rewrite (ip_expand_add Hs (vE k) (vN k)).
  rewrite (ip_expand_add Hs (qE k) (qN k)).
  rewrite (ip_expand_add Hs (xE k) (xN k)).
  rewrite (ip_expand_add Hs (dvE k) (dvN k)).
  ring.
Qed.

(* ---------- Per-element 5-fold Cauchy--Schwarz:  Qk <= sqrt perE sqrt perN - *)

Lemma sq_weight :
  forall (w : R) (x : V), 0 <= w -> (sqrt w * nrm x)^2 = w * << x , x >>.
Proof.
  intros w x Hw.
  replace ((sqrt w * nrm x)^2) with ((sqrt w * sqrt w) * (nrm x * nrm x)) by ring.
  rewrite (sqrt_sqrt w) by exact Hw. rewrite (nrm_sq Hs). reflexivity.
Qed.

Lemma Qk_CS : forall k, Qk k <= sqrt (perE k) * sqrt (perN k).
Proof.
  intro k.
  pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
  assert (H2nu : 0 <= 2 * nu) by nra.
  set (fE := fun i : nat => match i with
    | 0%nat => sqrt (2 * nu) * nrm (gE k)
    | 1%nat => sqrt sigma * nrm (vE k)
    | 2%nat => sqrt eps * nrm (qE k)
    | 3%nat => sqrt (t1 k) * nrm (xE k)
    | 4%nat => sqrt (t2 k) * nrm (dvE k)
    | _ => 0 end).
  set (fN := fun i : nat => match i with
    | 0%nat => sqrt (2 * nu) * nrm (gN k)
    | 1%nat => sqrt sigma * nrm (vN k)
    | 2%nat => sqrt eps * nrm (qN k)
    | 3%nat => sqrt (t1 k) * nrm (xN k)
    | 4%nat => sqrt (t2 k) * nrm (dvN k)
    | _ => 0 end).
  assert (HfE : forall i, 0 <= fE i).
  { intro i. unfold fE.
    destruct i as [|[|[|[|[|i]]]]];
      try (assert (0 <= sqrt (2*nu)) by apply sqrt_pos;
           assert (0 <= sqrt sigma) by apply sqrt_pos;
           assert (0 <= sqrt eps) by apply sqrt_pos;
           assert (0 <= sqrt (t1 k)) by apply sqrt_pos;
           assert (0 <= sqrt (t2 k)) by apply sqrt_pos;
           pose proof (nrm_nonneg Hs (gE k));
           pose proof (nrm_nonneg Hs (vE k));
           pose proof (nrm_nonneg Hs (qE k));
           pose proof (nrm_nonneg Hs (xE k));
           pose proof (nrm_nonneg Hs (dvE k));
           nra);
      lra. }
  assert (HfN : forall i, 0 <= fN i).
  { intro i. unfold fN.
    destruct i as [|[|[|[|[|i]]]]];
      try (assert (0 <= sqrt (2*nu)) by apply sqrt_pos;
           assert (0 <= sqrt sigma) by apply sqrt_pos;
           assert (0 <= sqrt eps) by apply sqrt_pos;
           assert (0 <= sqrt (t1 k)) by apply sqrt_pos;
           assert (0 <= sqrt (t2 k)) by apply sqrt_pos;
           pose proof (nrm_nonneg Hs (gN k));
           pose proof (nrm_nonneg Hs (vN k));
           pose proof (nrm_nonneg Hs (qN k));
           pose proof (nrm_nonneg Hs (xN k));
           pose proof (nrm_nonneg Hs (dvN k));
           nra);
      lra. }
  pose proof (Rsum_CS nat (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat fE fN HfE HfN) as HCS5.
  (*  pointwise: each  w <a,b>  <=  (sqrt w |a|)(sqrt w |b|)  *)
  assert (Q0 : 2 * nu * << (gE k) , (gN k) >> <= fE 0%nat * fN 0%nat).
  { unfold fE, fN.
    pose proof (CS_le Hs (gE k) (gN k)) as HC.
    assert (Hw : sqrt (2*nu) * sqrt (2*nu) = 2*nu) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (2*nu) * nrm (gE k)) * (sqrt (2*nu) * nrm (gN k))
                = (sqrt (2*nu) * sqrt (2*nu)) * (nrm (gE k) * nrm (gN k))) by ring.
    rewrite Hw in E. nra. }
  assert (Q1 : sigma * << (vE k) , (vN k) >> <= fE 1%nat * fN 1%nat).
  { unfold fE, fN.
    pose proof (CS_le Hs (vE k) (vN k)) as HC.
    assert (Hw : sqrt sigma * sqrt sigma = sigma) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt sigma * nrm (vE k)) * (sqrt sigma * nrm (vN k))
                = (sqrt sigma * sqrt sigma) * (nrm (vE k) * nrm (vN k))) by ring.
    rewrite Hw in E. nra. }
  assert (Q2 : eps * << (qE k) , (qN k) >> <= fE 2%nat * fN 2%nat).
  { unfold fE, fN.
    pose proof (CS_le Hs (qE k) (qN k)) as HC.
    assert (Hw : sqrt eps * sqrt eps = eps) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt eps * nrm (qE k)) * (sqrt eps * nrm (qN k))
                = (sqrt eps * sqrt eps) * (nrm (qE k) * nrm (qN k))) by ring.
    rewrite Hw in E. nra. }
  assert (Q3 : t1 k * << (xE k) , (xN k) >> <= fE 3%nat * fN 3%nat).
  { unfold fE, fN.
    pose proof (CS_le Hs (xE k) (xN k)) as HC.
    assert (Hw : sqrt (t1 k) * sqrt (t1 k) = t1 k) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (t1 k) * nrm (xE k)) * (sqrt (t1 k) * nrm (xN k))
                = (sqrt (t1 k) * sqrt (t1 k)) * (nrm (xE k) * nrm (xN k))) by ring.
    rewrite Hw in E. nra. }
  assert (Q4 : t2 k * << (dvE k) , (dvN k) >> <= fE 4%nat * fN 4%nat).
  { unfold fE, fN.
    pose proof (CS_le Hs (dvE k) (dvN k)) as HC.
    assert (Hw : sqrt (t2 k) * sqrt (t2 k) = t2 k) by (apply sqrt_sqrt; lra).
    assert (E : (sqrt (t2 k) * nrm (dvE k)) * (sqrt (t2 k) * nrm (dvN k))
                = (sqrt (t2 k) * sqrt (t2 k)) * (nrm (dvE k) * nrm (dvN k))) by ring.
    rewrite Hw in E. nra. }
  assert (HA1 : Qk k
                <= Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat (fun i => fE i * fN i)).
  { cbn [Rsum]. unfold Qk. lra. }
  assert (HE2 : Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat (fun i => (fE i)^2) = perE k).
  { cbn [Rsum]. unfold fE.
    rewrite (sq_weight (2 * nu) (gE k) H2nu).
    rewrite (sq_weight sigma (vE k) sigma_nonneg).
    rewrite (sq_weight eps (qE k) eps_nonneg).
    rewrite (sq_weight (t1 k) (xE k) (Rlt_le _ _ Ht1)).
    rewrite (sq_weight (t2 k) (dvE k) (Rlt_le _ _ Ht2)).
    unfold perE. ring. }
  assert (HN2 : Rsum (0 :: 1 :: 2 :: 3 :: 4 :: nil)%nat (fun i => (fN i)^2) = perN k).
  { cbn [Rsum]. unfold fN.
    rewrite (sq_weight (2 * nu) (gN k) H2nu).
    rewrite (sq_weight sigma (vN k) sigma_nonneg).
    rewrite (sq_weight eps (qN k) eps_nonneg).
    rewrite (sq_weight (t1 k) (xN k) (Rlt_le _ _ Ht1)).
    rewrite (sq_weight (t2 k) (dvN k) (Rlt_le _ _ Ht2)).
    unfold perN. ring. }
  rewrite HE2, HN2 in HCS5.
  lra.
Qed.

Lemma NE_nonneg : 0 <= NE.
Proof. unfold NE. apply sqrt_pos. Qed.
Lemma Neta_nonneg : 0 <= Neta.
Proof. unfold Neta. apply sqrt_pos. Qed.
Lemma NUU_nonneg : 0 <= NUU.
Proof. unfold NUU. apply sqrt_pos. Qed.

Lemma sumQ_CS : Rsum Th Qk <= NE * Neta.
Proof.
  eapply Rle_trans.
  { apply Rsum_le. intro k. exact (Qk_CS k). }
  assert (HfE : forall k, 0 <= sqrt (perE k)) by (intro; apply sqrt_pos).
  assert (HfN : forall k, 0 <= sqrt (perN k)) by (intro; apply sqrt_pos).
  pose proof (Rsum_CS K Th (fun k => sqrt (perE k)) (fun k => sqrt (perN k))
                HfE HfN) as HCS.
  cbv beta in HCS.
  assert (EE : Rsum Th (fun k => (sqrt (perE k))^2) = Rsum Th perE).
  { apply Rsum_ext. intro k.
    replace ((sqrt (perE k))^2) with (sqrt (perE k) * sqrt (perE k)) by ring.
    apply sqrt_sqrt. exact (perE_nonneg k). }
  assert (EN : Rsum Th (fun k => (sqrt (perN k))^2) = Rsum Th perN).
  { apply Rsum_ext. intro k.
    replace ((sqrt (perN k))^2) with (sqrt (perN k) * sqrt (perN k)) by ring.
    apply sqrt_sqrt. exact (perN_nonneg k). }
  rewrite EE, EN in HCS.
  unfold NE, Neta. exact HCS.
Qed.

(* ---------- The triangle inequality:  |||U - U_h||| <= |||Ehat||| + |||eta||| *)

Theorem NUU_triangle : NUU <= NE + Neta.
Proof.
  assert (Hsplit : Rsum Th perU = Rsum Th perE + 2 * Rsum Th Qk + Rsum Th perN).
  { assert (E1 : Rsum Th perU = Rsum Th (fun k => perE k + (2 * Qk k + perN k))).
    { apply Rsum_ext. intro k. rewrite (perU_expand k). ring. }
    rewrite E1.
    rewrite (Rsum_plus K Th perE (fun k => 2 * Qk k + perN k)).
    rewrite (Rsum_plus K Th (fun k => 2 * Qk k) perN).
    rewrite (Rsum_scal K Th 2 Qk). ring. }
  assert (HPE0 : 0 <= Rsum Th perE) by (apply Rsum_nonneg; exact perE_nonneg).
  assert (HPN0 : 0 <= Rsum Th perN) by (apply Rsum_nonneg; exact perN_nonneg).
  assert (HNE2 : NE ^ 2 = Rsum Th perE).
  { unfold NE. replace ((sqrt (Rsum Th perE))^2)
      with (sqrt (Rsum Th perE) * sqrt (Rsum Th perE)) by ring.
    apply sqrt_sqrt. exact HPE0. }
  assert (HNN2 : Neta ^ 2 = Rsum Th perN).
  { unfold Neta. replace ((sqrt (Rsum Th perN))^2)
      with (sqrt (Rsum Th perN) * sqrt (Rsum Th perN)) by ring.
    apply sqrt_sqrt. exact HPN0. }
  pose proof sumQ_CS as HQ.
  pose proof NE_nonneg as HNE0. pose proof Neta_nonneg as HNN0.
  apply nonneg_le_of_sqr.
  - unfold NUU. apply sqrt_pos.
  - lra.
  - assert (HNUU2 : NUU ^ 2 = Rsum Th perU).
    { unfold NUU. replace ((sqrt (Rsum Th perU))^2)
        with (sqrt (Rsum Th perU) * sqrt (Rsum Th perU)) by ring.
      apply sqrt_sqrt. apply Rsum_nonneg. exact perU_nonneg. }
    rewrite HNUU2, Hsplit.
    replace ((NE + Neta)^2) with (NE^2 + 2 * (NE * Neta) + Neta^2) by ring.
    lra.
Qed.

(* ========================================================================= *)
(*  The composition (glue).  The four inputs, applied to eta / (Ehat,W) / W,  *)
(*  enter as hypotheses about a common test function W with norm NW.          *)
(* ========================================================================= *)

Variables (Eh NW : R).
Hypothesis Eh_nonneg : 0 <= Eh.
Hypothesis NW_nonneg : 0 <= NW.

(*  The three bilinear evaluations at the common test function W.             *)
Variables (Beta BEW BcW : R).   (*  B_osgs(eta,W), B_osgs(Ehat,W), B_osgs(U-U_h,W) *)

(*  Constants of the four inputs.  *)
Variables (beta_stab Cw Cint Ccons Ke : R).
Hypothesis beta_stab_pos : 0 < beta_stab.
Hypothesis Cw_nn   : 0 <= Cw.
Hypothesis Cint_nn : 0 <= Cint.
Hypothesis Ccons_nn : 0 <= Ccons.
Hypothesis Ke_nn : 0 <= Ke.

(*  (i) th:stability applied to eta:  B_osgs(eta,W) >= beta |||eta|||^2,      *)
(*      |||W||| <= Cw |||eta|||.                                             *)
Hypothesis Hstab_lb : Beta >= beta_stab * Neta ^ 2.
Hypothesis HW_bound : NW <= Cw * Neta.
(*  (ii) lem:interpolation at (Ehat,W):  |B_osgs(Ehat,W)| <= Cint E(h) |||W|||. *)
Hypothesis Hinterp : Rabs BEW <= Cint * (Eh * NW).
(*  (iii) lem:consistency at W:  |B_osgs(U-U_h,W)| <= Ccons E(h) |||W|||.      *)
Hypothesis Hcons : Rabs BcW <= Ccons * (Eh * NW).
(*  (iv) bilinearity of B_osgs in the first argument (analogue of Horth):     *)
(*      B_osgs(eta,W) = B_osgs(U-U_h,W) - B_osgs(Ehat,W).                     *)
Hypothesis Hbilin : Beta = BcW - BEW.
(*  (v) lem:interpsize (Lemma 5.3):  |||Ehat||| <= Ke E(h).                   *)
Hypothesis Hinterpsize : NE <= Ke * Eh.

(* ---------- |||eta||| <= C E(h) ------------------------------------------- *)

Lemma Neta_bound : Neta <= (Ccons + Cint) * Cw / beta_stab * Eh.
Proof.
  pose proof Neta_nonneg as HNN0.
  (*  beta |||eta|||^2 <= B_osgs(eta,W) <= (Ccons+Cint) E(h) |||W|||          *)
  assert (Hchain : beta_stab * Neta ^ 2 <= (Ccons + Cint) * (Eh * NW)).
  { assert (HB : Beta <= Rabs BcW + Rabs BEW).
    { rewrite Hbilin.
      replace (BcW - BEW) with (BcW + - BEW) by ring.
      pose proof (Rabs_triang BcW (- BEW)) as Ht. rewrite Rabs_Ropp in Ht.
      pose proof (Rle_abs (BcW + - BEW)) as Hle. lra. }
    pose proof Hstab_lb as Hstb. pose proof Hinterp as Hi. pose proof Hcons as Hc.
    nra. }
  (*  |||W||| <= Cw |||eta||| and E(h),|||eta|||,|||W||| >= 0                 *)
  assert (Hstep : beta_stab * Neta ^ 2 <= (Ccons + Cint) * Eh * (Cw * Neta)).
  { assert (HEhNW : Eh * NW <= Eh * (Cw * Neta)).
    { apply Rmult_le_compat_l; [exact Eh_nonneg | exact HW_bound]. }
    assert (Hcc : 0 <= Ccons + Cint) by lra.
    assert (H1 : (Ccons + Cint) * (Eh * NW) <= (Ccons + Cint) * (Eh * (Cw * Neta))).
    { apply Rmult_le_compat_l; [exact Hcc | exact HEhNW]. }
    assert (E : (Ccons + Cint) * (Eh * (Cw * Neta)) = (Ccons + Cint) * Eh * (Cw * Neta))
      by ring.
    lra. }
  (*  cancel one factor of |||eta||| (or handle |||eta||| = 0)               *)
  destruct (Rle_lt_or_eq_dec 0 Neta HNN0) as [Hpos | Heq0].
  - apply (Rmult_le_reg_l beta_stab); [exact beta_stab_pos |].
    assert (Hgoal : beta_stab * Neta <= (Ccons + Cint) * Cw * Eh).
    { apply (Rmult_le_reg_r Neta); [exact Hpos |].
      replace (beta_stab * Neta * Neta) with (beta_stab * Neta ^ 2) by ring.
      replace ((Ccons + Cint) * Cw * Eh * Neta)
        with ((Ccons + Cint) * Eh * (Cw * Neta)) by ring.
      exact Hstep. }
    replace (beta_stab * ((Ccons + Cint) * Cw / beta_stab * Eh))
      with ((Ccons + Cint) * Cw * Eh) by (field; lra).
    exact Hgoal.
  - rewrite <- Heq0.
    assert (0 <= (Ccons + Cint) * Cw / beta_stab * Eh).
    { assert (0 <= (Ccons + Cint) * Cw) by (apply Rmult_le_pos; lra).
      assert (0 <= (Ccons + Cint) * Cw / beta_stab)
        by (apply div_nonneg; lra).
      apply Rmult_le_pos; [lra | exact Eh_nonneg]. }
    lra.
Qed.

(* ========================================================================= *)
(*  Theorem 5.4 (th:convergence), abstract form.                             *)
(* ========================================================================= *)

Definition Cconv : R := Ke + (Ccons + Cint) * Cw / beta_stab.

Theorem abstract_osgs_convergence : NUU <= Cconv * Eh.
Proof.
  pose proof NUU_triangle as HT.
  pose proof Neta_bound as HN.
  pose proof Hinterpsize as HE.
  unfold Cconv. lra.
Qed.

Lemma Cconv_nonneg : 0 <= Cconv.
Proof.
  unfold Cconv.
  assert (0 <= (Ccons + Cint) * Cw) by (apply Rmult_le_pos; lra).
  assert (0 <= (Ccons + Cint) * Cw / beta_stab) by (apply div_nonneg; lra).
  lra.
Qed.

(* ========================================================================= *)
(*  Corollary 5.5 (cor:asgsnorm): the estimate in the ASGS ("S") norm.        *)
(*  Since the OSGS norm dominates the ASGS norm (Lemma 3.5, here perSU_le_perU *)
(*  from sigmatilde <= sigma and nu <= 2 nu), the convergence estimate holds  *)
(*  verbatim in the ASGS norm.                                               *)
(* ========================================================================= *)

Definition sg (k : K) : R := ContinuityAlgebra.sigt nu (hK k) (aK k) sigma (am k) c1 c2.

Lemma sg_nonneg : forall k, 0 <= sg k.
Proof. intro k. unfold sg. apply sigt_nonneg; auto. Qed.

Lemma sg_le_sigma : forall k, sg k <= sigma.
Proof. intro k. unfold sg. apply P4_le_sigma; auto. Qed.

Definition perSU (k : K) : R :=
  nu * << (gE k) +v (gN k) , (gE k) +v (gN k) >>
  + sg k * << (vE k) +v (vN k) , (vE k) +v (vN k) >>
  + eps * << (qE k) +v (qN k) , (qE k) +v (qN k) >>
  + t1 k * << (xE k) +v (xN k) , (xE k) +v (xN k) >>
  + t2 k * << (dvE k) +v (dvN k) , (dvE k) +v (dvN k) >>.

Definition NSU : R := sqrt (Rsum Th perSU).

Lemma perSU_le_perU : forall k, perSU k <= perU k.
Proof.
  intro k. unfold perSU, perU.
  pose proof (ip_pos Hs ((gE k) +v (gN k))) as Pg.
  pose proof (ip_pos Hs ((vE k) +v (vN k))) as Pv.
  assert (Hnu : nu * << (gE k) +v (gN k) , (gE k) +v (gN k) >>
                <= 2 * nu * << (gE k) +v (gN k) , (gE k) +v (gN k) >>) by nra.
  assert (Hsg : sg k * << (vE k) +v (vN k) , (vE k) +v (vN k) >>
                <= sigma * << (vE k) +v (vN k) , (vE k) +v (vN k) >>).
  { apply Rmult_le_compat_r; [exact Pv | exact (sg_le_sigma k)]. }
  lra.
Qed.

Lemma NSU_le_NUU : NSU <= NUU.
Proof.
  unfold NSU, NUU. apply sqrt_le_1_alt. apply Rsum_le. exact perSU_le_perU.
Qed.

Corollary abstract_osgs_convergence_asgsnorm : NSU <= Cconv * Eh.
Proof.
  pose proof NSU_le_NUU. pose proof abstract_osgs_convergence. lra.
Qed.

(* ========================================================================= *)
(*  Corollary 5.6 (cor:Ltwo): the sigma-robust L^2 velocity estimate.         *)
(*  The OSGS triple norm controls the FULL reactive term                      *)
(*  ||sigma^{1/2}(u - u_h)||, so sqrt(sigma) ||u - u_h|| <= |||U - U_h|||;     *)
(*  combined with th:convergence this bounds the sigma^{1/2}-weighted L^2      *)
(*  velocity error by C E(h), uniformly in sigma (the reward for deleting the *)
(*  reactive subscale, rem:mechanism).                                       *)
(* ========================================================================= *)

Definition Nvel : R :=
  sqrt (Rsum Th (fun k => << (vE k) +v (vN k) , (vE k) +v (vN k) >>)).

Lemma sig_vel_le_NUU : sqrt sigma * Nvel <= NUU.
Proof.
  assert (Hvel2 : 0 <= Rsum Th (fun k => << (vE k) +v (vN k) , (vE k) +v (vN k) >>)).
  { apply Rsum_nonneg. intro k. apply (ip_pos Hs). }
  assert (HperU0 : 0 <= Rsum Th perU) by (apply Rsum_nonneg; exact perU_nonneg).
  apply nonneg_le_of_sqr.
  - assert (0 <= sqrt sigma) by apply sqrt_pos.
    assert (0 <= Nvel) by (unfold Nvel; apply sqrt_pos). nra.
  - unfold NUU. apply sqrt_pos.
  - (*  (sqrt sigma * Nvel)^2 = sigma * Rsum<vU,vU> <= Rsum perU = NUU^2  *)
    replace ((sqrt sigma * Nvel)^2)
      with ((sqrt sigma * sqrt sigma) * (Nvel * Nvel)) by ring.
    rewrite (sqrt_sqrt sigma) by exact sigma_nonneg.
    assert (HNvel2 : Nvel * Nvel
                     = Rsum Th (fun k => << (vE k) +v (vN k) , (vE k) +v (vN k) >>)).
    { unfold Nvel. apply sqrt_sqrt. exact Hvel2. }
    rewrite HNvel2.
    replace ((NUU)^2) with (NUU * NUU) by ring.
    assert (HNUU2 : NUU * NUU = Rsum Th perU).
    { unfold NUU. apply sqrt_sqrt. exact HperU0. }
    rewrite HNUU2.
    rewrite <- (Rsum_scal K Th sigma
                 (fun k => << (vE k) +v (vN k) , (vE k) +v (vN k) >>)).
    apply Rsum_le. intro k.
    pose proof (t1_pos k) as Ht1. pose proof (t2_pos k) as Ht2.
    unfold perU.
    pose proof (ip_pos Hs ((gE k) +v (gN k))).
    pose proof (ip_pos Hs ((qE k) +v (qN k))).
    pose proof (ip_pos Hs ((xE k) +v (xN k))).
    pose proof (ip_pos Hs ((dvE k) +v (dvN k))).
    assert (0 <= 2*nu*<< (gE k) +v (gN k),(gE k) +v (gN k) >>) by nra.
    assert (0 <= eps*<< (qE k) +v (qN k),(qE k) +v (qN k) >>) by nra.
    assert (0 <= t1 k*<< (xE k) +v (xN k),(xE k) +v (xN k) >>) by nra.
    assert (0 <= t2 k*<< (dvE k) +v (dvN k),(dvE k) +v (dvN k) >>) by nra.
    lra.
Qed.

Corollary abstract_osgs_convergence_Ltwo : sqrt sigma * Nvel <= Cconv * Eh.
Proof.
  pose proof sig_vel_le_NUU. pose proof abstract_osgs_convergence. lra.
Qed.

End OsgsConvergence.
