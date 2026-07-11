(* ========================================================================= *)
(*  TauDesign.v                                                              *)
(*                                                                           *)
(*  Machine-checked verification (Coq 8.18, stdlib only) of the Fourier      *)
(*  analysis design of the stabilization parameters tau, Section 4 /         *)
(*  Appendix C of *A stabilized finite element method for incompressible,    *)
(*  inertial flows in inhomogeneous porous media* (Casas, Gonzalez-Usua,     *)
(*  Codina, de-Pouplana).                                                    *)
(*                                                                           *)
(*  Mirrors theory/verification scripts/fourier_tau_verification.py:         *)
(*   [1] the viscous symbol  sum_ij k_i k_j K_ij                             *)
(*         = alpha nu (|k0|^2 I + (1 - 2/d) k0 k0^T)   (d = 3),              *)
(*       its parallel eigenpair with eigenvalue (4/3) alpha nu |k0|^2        *)
(*       (amendment A4) and a transverse eigenpair with alpha nu |k0|^2,     *)
(*       plus the general-d factor (2 - 2/d) at d = 3 and d = 2;             *)
(*   [2] the convective symbol collapses to i (alpha/h)(w.k0) I on the       *)
(*       velocity block, with modulus (alpha/h)|w.k0| (amendment B2);        *)
(*   [3] the coupling matrix C = [[0,k0],[k0^T,0]]: eigen-equations for      *)
(*       +|k0|, -|k0| and the two-dimensional kernel, and the                *)
(*       characteristic polynomial  det(C - lam I) = lam^2 (lam^2 - |k0|^2), *)
(*       so the spectrum is exactly { +|k0|, -|k0|, 0 };                     *)
(*   [4] the tau_b generalized-eigenproblem: the sqrt(lam)/(1/sqrt(lam))     *)
(*       pair is the unique positive solution of                             *)
(*         X*Y = (a|k0|/h)^2 ,  X/Y = lam ;                                  *)
(*   [5] the porosity-gradient design entry (well-defined);                  *)
(*   [6] the assembly with lam = h^2/(|k0|^2 tau_{1,NS}^2) recovers          *)
(*       eq:Tau1 (with C_alpha = alpha + (h/|k0|)|grad alpha|, eq:CAlpha)    *)
(*       and eq:Tau2.                                                        *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  [1]  The viscous symbol (d = 3).                                         *)
(* ========================================================================= *)

Section ViscousSymbol.

Variables (nu alpha : R).
Variables (k1 k2 k3 : R).                 (* the h-normalised wavenumber k0 *)
Hypothesis nu_pos    : 0 < nu.
Hypothesis alpha_pos : 0 < alpha.

(* Kronecker delta on nat indices. *)
Definition kron (p q : nat) : R := if Nat.eqb p q then 1 else 0.

(* Component p of k0, p in {0,1,2}. *)
Definition kv (p : nat) : R :=
  match p with 0 => k1 | 1 => k2 | _ => k3 end.

Definition knorm2 : R := k1^2 + k2^2 + k3^2.

(* Velocity block of the diffusion matrix K_ij                               *)
(* (eq:matrices_stationary_strong_problem), entries (a,b):                   *)
(*   a = b : nu alpha ( delta_ij + (1/3) delta_ai delta_aj )                 *)
(*   a <> b: nu alpha ( delta_bi delta_aj - (2/3) delta_ai delta_bj )        *)
Definition Kmat (i j a b : nat) : R :=
  if Nat.eqb a b
  then nu * alpha * (kron i j + / 3 * kron a i * kron a j)
  else nu * alpha * (kron b i * kron a j - 2 / 3 * kron a i * kron b j).

(* The contraction  M_ab := sum_{i,j} k_i k_j K_ij^{(a,b)}                   *)
(* (the overall 1/h^2 is pulled out front, as in the paper).                 *)
Definition Mvisc (a b : nat) : R :=
  kv 0 * kv 0 * Kmat 0 0 a b + kv 0 * kv 1 * Kmat 0 1 a b
  + kv 0 * kv 2 * Kmat 0 2 a b
  + kv 1 * kv 0 * Kmat 1 0 a b + kv 1 * kv 1 * Kmat 1 1 a b
  + kv 1 * kv 2 * Kmat 1 2 a b
  + kv 2 * kv 0 * Kmat 2 0 a b + kv 2 * kv 1 * Kmat 2 1 a b
  + kv 2 * kv 2 * Kmat 2 2 a b.

(* --- Closed form (amendment A4):                                          *)
(*     M = alpha nu ( |k0|^2 I + (1 - 2/d) k0 k0^T ),  1 - 2/3 = 1/3. ------ *)
Theorem viscous_symbol_closed_form :
  forall a b : nat, (a < 3)%nat -> (b < 3)%nat ->
    Mvisc a b = alpha * nu * (knorm2 * kron a b + / 3 * (kv a * kv b)).
Proof.
  intros a b Ha Hb.
  destruct a as [| [| [| a']]]; [| | | lia];
  destruct b as [| [| [| b']]]; [| | | lia | | | | lia | | | | lia];
  unfold Mvisc, Kmat, kron, kv, knorm2; simpl; field.
Qed.

(* --- Parallel eigenpair:  M k0 = (4/3) alpha nu |k0|^2 k0  (d = 3). ------ *)
Theorem viscous_parallel_eigenpair :
  forall a : nat, (a < 3)%nat ->
    Mvisc a 0 * kv 0 + Mvisc a 1 * kv 1 + Mvisc a 2 * kv 2
    = (4 / 3) * alpha * nu * knorm2 * kv a.
Proof.
  intros a Ha.
  destruct a as [| [| [| a']]]; [| | | lia];
  unfold Mvisc, Kmat, kron, kv, knorm2; simpl; field.
Qed.

(* --- Transverse eigenpair:  M vperp = alpha nu |k0|^2 vperp,               *)
(*     with vperp = (-k2, k1, 0) orthogonal to k0. ------------------------- *)
Definition vperp (p : nat) : R :=
  match p with 0 => - k2 | 1 => k1 | _ => 0 end.

Theorem viscous_transverse_eigenpair :
  forall a : nat, (a < 3)%nat ->
    Mvisc a 0 * vperp 0 + Mvisc a 1 * vperp 1 + Mvisc a 2 * vperp 2
    = alpha * nu * knorm2 * vperp a.
Proof.
  intros a Ha.
  destruct a as [| [| [| a']]]; [| | | lia];
  unfold Mvisc, Kmat, kron, kv, vperp, knorm2; simpl; field.
Qed.

Lemma vperp_orthogonal : kv 0 * vperp 0 + kv 1 * vperp 1 + kv 2 * vperp 2 = 0.
Proof. unfold kv, vperp; simpl; ring. Qed.

(* --- The general-d parallel factor (2 - 2/d): 4/3 at d = 3, 1 at d = 2. -- *)
Theorem parallel_factor_d3 : 2 - 2 / 3 = 4 / 3.
Proof. lra. Qed.

Theorem parallel_factor_d2 : 2 - 2 / 2 = 1.
Proof. lra. Qed.

End ViscousSymbol.

(* ========================================================================= *)
(*  [2]  The convective symbol.                                              *)
(*  L^_c(k0) = i (k_i/h) A_{v,i} = i (alpha/h)(w.k0) I on the velocity       *)
(*  block; the imaginary unit factors out, so the real content is the        *)
(*  scalar identity below, and the modulus of the (degenerate) eigenvalue    *)
(*  i mu is |mu| = (alpha/h)|w.k0|  (amendment B2).                          *)
(* ========================================================================= *)

Section ConvectiveSymbol.

Variables (alpha h : R).
Variables (k1 k2 k3 w1 w2 w3 : R).
Hypothesis alpha_pos : 0 < alpha.
Hypothesis h_pos     : 0 < h.

Theorem convective_symbol_scalar :
  (k1 / h) * (alpha * w1) + (k2 / h) * (alpha * w2) + (k3 / h) * (alpha * w3)
  = (alpha / h) * (w1 * k1 + w2 * k2 + w3 * k3).
Proof. field; lra. Qed.

(* The modulus of a purely imaginary number i*mu, computed as the            *)
(* Euclidean length of (0, mu), is |mu|.                                     *)
Lemma purely_imaginary_modulus :
  forall mu : R, sqrt (0^2 + mu^2) = Rabs mu.
Proof.
  intro mu.
  replace (0^2 + mu^2) with (Rsqr mu) by (unfold Rsqr; ring).
  apply sqrt_Rsqr_abs.
Qed.

Theorem convective_spectral_radius :
  forall wk : R, Rabs ((alpha / h) * wk) = (alpha / h) * Rabs wk.
Proof.
  intro wk.
  rewrite Rabs_mult. f_equal.
  apply Rabs_right.
  assert (0 < alpha / h) by (apply Rdiv_lt_0_compat; lra).
  lra.
Qed.

End ConvectiveSymbol.

(* ========================================================================= *)
(*  [3]  The pressure-gradient / divergence coupling matrix                  *)
(*         C = [[0, k0], [k0^T, 0]]  (4 x 4).                                *)
(*  Its action on (v, s) in R^3 x R is  C (v,s) = (s k0, k0 . v).            *)
(* ========================================================================= *)

Section CouplingMatrix.

Variables (k1 k2 k3 : R).
Definition kn2 : R := k1^2 + k2^2 + k3^2.
Definition s0  : R := sqrt kn2.           (* |k0| *)

Lemma kn2_nonneg : 0 <= kn2.
Proof.
  unfold kn2.
  pose proof (pow2_ge_0 k1). pose proof (pow2_ge_0 k2). pose proof (pow2_ge_0 k3).
  lra.
Qed.

Lemma s0_sq : s0 * s0 = kn2.
Proof. unfold s0. apply sqrt_sqrt, kn2_nonneg. Qed.

Lemma s0_nonneg : 0 <= s0.
Proof. unfold s0. apply sqrt_pos. Qed.

(* Action of C on a 4-vector (v1,v2,v3,s). *)
Definition Cact (v1 v2 v3 s : R) : R * R * R * R :=
  (s * k1, s * k2, s * k3, k1 * v1 + k2 * v2 + k3 * v3).

(* --- Eigen-equation for the eigenvalue +|k0|, eigenvector (k0, |k0|). ---- *)
Theorem coupling_eigen_plus :
  Cact k1 k2 k3 s0 = (s0 * k1, s0 * k2, s0 * k3, s0 * s0).
Proof.
  unfold Cact. rewrite s0_sq. unfold kn2.
  repeat f_equal; ring.
Qed.

(* --- Eigen-equation for the eigenvalue -|k0|, eigenvector (k0, -|k0|). --- *)
Theorem coupling_eigen_minus :
  Cact k1 k2 k3 (- s0) = (- s0 * k1, - s0 * k2, - s0 * k3, - s0 * - s0).
Proof.
  unfold Cact.
  assert (H : - s0 * - s0 = kn2) by (pose proof s0_sq; nra).
  rewrite H. unfold kn2.
  repeat f_equal; ring.
Qed.

(* --- Kernel: every (v, 0) with k0 . v = 0 is annihilated;                  *)
(*     for k0 <> 0 this kernel is two-dimensional, giving the eigenvalue     *)
(*     0 with multiplicity two.                                              *)
Theorem coupling_kernel :
  forall v1 v2 v3, k1 * v1 + k2 * v2 + k3 * v3 = 0 ->
    Cact v1 v2 v3 0 = (0, 0, 0, 0).
Proof.
  intros v1 v2 v3 Horth. unfold Cact.
  rewrite Horth. repeat f_equal; ring.
Qed.

(* Two explicit orthogonal vectors witnessing the kernel equations. *)
Example kernel_witness_1 : k1 * (- k2) + k2 * k1 + k3 * 0 = 0.
Proof. ring. Qed.
Example kernel_witness_2 : k1 * (- k3) + k2 * 0 + k3 * k1 = 0.
Proof. ring. Qed.

(* --- Characteristic polynomial.  Explicit 3x3 and 4x4 determinants. ------ *)
Definition det3 (a b c d e f g h i : R) : R :=
  a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g.

(* Cofactor expansion of a 4x4 determinant along the first row. *)
Definition det4
  (m00 m01 m02 m03 m10 m11 m12 m13
   m20 m21 m22 m23 m30 m31 m32 m33 : R) : R :=
    m00 * det3 m11 m12 m13 m21 m22 m23 m31 m32 m33
  - m01 * det3 m10 m12 m13 m20 m22 m23 m30 m32 m33
  + m02 * det3 m10 m11 m13 m20 m21 m23 m30 m31 m33
  - m03 * det3 m10 m11 m12 m20 m21 m22 m30 m31 m32.

(* det(C - lam I) = lam^4 - |k0|^2 lam^2 = lam^2 (lam^2 - |k0|^2). *)
Theorem coupling_characteristic_polynomial :
  forall lam : R,
    det4 (- lam) 0 0 k1
         0 (- lam) 0 k2
         0 0 (- lam) k3
         k1 k2 k3 (- lam)
    = lam^2 * (lam^2 - kn2).
Proof. intro lam. unfold det4, det3, kn2. ring. Qed.

(* Hence the spectrum is exactly { +|k0|, -|k0|, 0 }. *)
Theorem coupling_spectrum :
  forall lam : R,
    lam^2 * (lam^2 - kn2) = 0 <-> lam = 0 \/ lam = s0 \/ lam = - s0.
Proof.
  intro lam. split.
  - intro H.
    assert (Hfac : lam * lam * ((lam - s0) * (lam + s0)) = 0).
    { pose proof s0_sq as Hs. nra. }
    apply Rmult_integral in Hfac. destruct Hfac as [Hll | Hpm].
    + apply Rmult_integral in Hll. left. destruct Hll; lra.
    + apply Rmult_integral in Hpm. right. destruct Hpm; [left | right]; lra.
  - intro H. destruct H as [H | [H | H]]; subst; pose proof s0_sq; nra.
Qed.

(* Spectral radius: max of the moduli of {0, +|k0|, -|k0|} is |k0|. *)
Theorem coupling_spectral_radius :
  Rmax (Rabs 0) (Rmax (Rabs s0) (Rabs (- s0))) = s0.
Proof.
  pose proof s0_nonneg as Hs.
  rewrite Rabs_R0, Rabs_Ropp.
  rewrite (Rabs_right s0) by lra.
  unfold Rmax; repeat destruct Rle_dec; lra.
Qed.

End CouplingMatrix.

(* ========================================================================= *)
(*  [4]  The tau_b generalized-eigenproblem design pair.                     *)
(*  With c := a|k0|/h the coupling magnitude and lam > 0 the free            *)
(*  momentum/mass scale ratio, the system                                    *)
(*        X * Y = c^2 ,   X / Y = lam ,   X, Y > 0                           *)
(*  has the unique solution  X = c sqrt(lam),  Y = c / sqrt(lam).            *)
(* ========================================================================= *)

Section TaubPair.

Variables (lam c : R).
Hypothesis lam_pos : 0 < lam.
Hypothesis c_pos   : 0 < c.

Let sl := sqrt lam.

Lemma sl_pos : 0 < sl.
Proof. unfold sl. apply sqrt_lt_R0. exact lam_pos. Qed.

Lemma sl_sq : sl * sl = lam.
Proof. unfold sl. apply sqrt_sqrt. lra. Qed.

(* Existence: the paper's pair solves the design system. *)
Theorem taub_pair_solves :
  (c * sl) * (c / sl) = c^2 /\ (c * sl) / (c / sl) = lam.
Proof.
  pose proof sl_pos as Hsl. pose proof sl_sq as Hsq.
  split.
  - field; lra.
  - rewrite <- Hsq. field; lra.
Qed.

(* Uniqueness among positive solutions. *)
Theorem taub_pair_unique :
  forall X Y : R, 0 < X -> 0 < Y ->
    X * Y = c^2 -> X / Y = lam ->
    X = c * sl /\ Y = c / sl.
Proof.
  intros X Y HX HY Hprod Hratio.
  pose proof sl_pos as Hsl. pose proof sl_sq as Hsq.
  assert (HXlY : X = lam * Y).
  { apply (Rmult_eq_reg_r (/ Y)); [| apply Rinv_neq_0_compat; lra].
    unfold Rdiv in Hratio. rewrite Hratio. field. lra. }
  assert (HY2 : lam * (Y * Y) = c * c).
  { replace (c * c) with (c^2) by ring. rewrite <- Hprod, HXlY. ring. }
  rewrite <- Hsq in HY2.
  assert (HYY : Rsqr (sl * Y) = Rsqr c) by (unfold Rsqr; nra).
  assert (Hsly : sl * Y = c).
  { apply Rsqr_inj; [nra | lra | exact HYY]. }
  assert (HYval : Y = c / sl).
  { apply (Rmult_eq_reg_l sl); [| lra].
    rewrite Hsly. field. lra. }
  split; [| exact HYval].
  rewrite HXlY, HYval, <- Hsq. field. lra.
Qed.

End TaubPair.

(* ========================================================================= *)
(*  [5] + [6]  Assembly with the scaling  lam = h^2 / (|k0|^2 tau_{1,NS}^2), *)
(*  recovering eq:Tau1 (with eq:CAlpha) and eq:Tau2.                         *)
(* ========================================================================= *)

Section Assembly.

Variables (alpha h k0 tau1NS nu c2 wmag grada rho_sigma eps : R).
Hypothesis alpha_pos  : 0 < alpha.
Hypothesis h_pos      : 0 < h.
Hypothesis k0_pos     : 0 < k0.            (* |k0| of the design mode *)
Hypothesis tau1NS_pos : 0 < tau1NS.
Hypothesis nu_pos     : 0 < nu.
Hypothesis c2_pos     : 0 < c2.
Hypothesis wmag_nn    : 0 <= wmag.
Hypothesis grada_nn   : 0 <= grada.        (* |grad alpha| *)
Hypothesis rho_nn     : 0 <= rho_sigma.
Hypothesis eps_pos    : 0 < eps.

(* c1 := |k0|^2 (the paragraph after eq:TauNavierStokes). *)
Definition c1A : R := k0^2.

(* The lambda scaling (eq. for lambda). *)
Definition lam_choice : R := h^2 / (k0^2 * tau1NS^2).

(* --- sqrt(lam) collapses to h / (|k0| tau_{1,NS}). ----------------------- *)
Theorem sqrt_lam_choice : sqrt lam_choice = h / (k0 * tau1NS).
Proof.
  unfold lam_choice.
  replace (h^2 / (k0^2 * tau1NS^2)) with ((h / (k0 * tau1NS))^2)
    by (field; lra).
  apply sqrt_pow2.
  assert (0 < h / (k0 * tau1NS)) by (apply Rdiv_lt_0_compat; nra).
  lra.
Qed.

(* --- tau_{b,1}^{-1} = a (|k0|/h) sqrt(lam)  ->  a / tau_{1,NS}. ---------- *)
Theorem taub1_inv_assembled :
  alpha * (k0 / h) * sqrt lam_choice = alpha / tau1NS.
Proof. rewrite sqrt_lam_choice. field; nra. Qed.

(* --- tau_{b,2}^{-1} = a (|k0|/h) / sqrt(lam)  ->  c1 a tau_{1,NS} / h^2. - *)
Theorem taub2_inv_assembled :
  alpha * (k0 / h) / sqrt lam_choice = c1A * alpha * tau1NS / h^2.
Proof.
  rewrite sqrt_lam_choice. unfold c1A.
  field; repeat split; nra.
Qed.

(* --- tau_{grad a,1}^{-1} = sqrt(lam) |grad a|                              *)
(*       ->  (h/|k0|) |grad a| / tau_{1,NS}. ------------------------------- *)
Theorem tau_grada_inv_assembled :
  sqrt lam_choice * grada = (h / k0) * grada / tau1NS.
Proof. rewrite sqrt_lam_choice. field; nra. Qed.

(* --- Momentum row: viscous + convective collapse to a tau_{1,NS}^{-1}. --- *)
(* (Here tau_{1,NS}^{-1} = c1 nu/h^2 + c2 |w|/h with the 4/3 absorbed        *)
(*  into c1; eq:TauNavierStokes.)                                            *)
Definition tau1NS_inv_def : R := c1A * nu / h^2 + c2 * wmag / h.

Theorem visc_plus_conv_momentum :
  alpha * (c1A * nu / h^2) + alpha * (c2 * wmag / h) = alpha * tau1NS_inv_def.
Proof. unfold tau1NS_inv_def. ring. Qed.

(* --- eq:Tau1: tau_1^{-1} = C_alpha tau_{1,NS}^{-1} + rho(sigma),           *)
(*     with  C_alpha = alpha + (h/|k0|) |grad alpha|  (eq:CAlpha).           *)
(* (tau_{b,1} is the self-referential contribution absorbed into the         *)
(*  constants -- see the paragraph after eq:StabilizationParameters --       *)
(*  so it is NOT re-added here, exactly as in the SymPy script.)             *)
Definition C_alpha : R := alpha + (h / k0) * grada.

Theorem tau1_inv_assembled :
  forall t1NSi : R,
    alpha * t1NSi + ((h / k0) * grada) * t1NSi + rho_sigma
    = C_alpha * t1NSi + rho_sigma.
Proof. intro t1NSi. unfold C_alpha. ring. Qed.

(* --- eq:Tau2: the mass row  tau_{b,2}^{-1} + eps  inverts to               *)
(*       tau_2 = h^2 / (c1 a tau_{1,NS} + eps h^2). --------------------- --- *)
Theorem tau2_assembled :
  / (c1A * alpha * tau1NS / h^2 + eps)
  = h^2 / (c1A * alpha * tau1NS + eps * h^2).
Proof.
  unfold c1A.
  assert (Hk2  : 0 < k0^2) by (apply pow_lt; lra).
  assert (Hka  : 0 < k0^2 * alpha) by nra.
  assert (Hnum : 0 < k0^2 * alpha * tau1NS) by nra.
  assert (Hh2  : 0 < h^2) by (apply pow_lt; lra).
  assert (Hq   : 0 < k0^2 * alpha * tau1NS / h^2)
    by (apply Rdiv_lt_0_compat; lra).
  assert (Heh  : 0 < eps * h^2) by nra.
  field; repeat split; lra.
Qed.

End Assembly.
