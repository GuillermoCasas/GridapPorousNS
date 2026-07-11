(* ========================================================================= *)
(*  InnerSpace.v                                                             *)
(*                                                                           *)
(*  A minimal real pre-Hilbert space: a carrier with addition, scalar        *)
(*  multiplication and a symmetric, bilinear, positive-semidefinite form.    *)
(*  This models the elemental L^2(K) pairings of the paper.  From the four   *)
(*  axioms we DERIVE (rather than assume): full bilinearity, the             *)
(*  Cauchy--Schwarz inequality, the norm expansions of sums and              *)
(*  differences, the parametrized Young inequality at vector level, and the  *)
(*  difference-of-squares identity behind eq:StabilityEstimate.              *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz.
Local Open Scope R_scope.

Record PreHilbert : Type := mkPreHilbert {
  carrier :> Type;
  vadd  : carrier -> carrier -> carrier;
  vscal : R -> carrier -> carrier;
  ip    : carrier -> carrier -> R;
  ip_sym    : forall x y, ip x y = ip y x;
  ip_add_l  : forall x y z, ip (vadd x y) z = ip x z + ip y z;
  ip_scal_l : forall a x y, ip (vscal a x) y = a * ip x y;
  ip_pos    : forall x, 0 <= ip x x
}.

Lemma nonneg_le_of_sqr' :
  forall x y : R, 0 <= x -> 0 <= y -> x^2 <= y^2 -> x <= y.
Proof.
  intros x y Hx Hy Hs.
  destruct (Rle_or_lt x y) as [H | H]; [exact H |].
  exfalso.
  assert (Hxy : 0 < x + y) by lra.
  assert (Hp : 0 < (x - y) * (x + y)) by nra.
  nra.
Qed.

Section InnerSpaceFacts.

Variable H : PreHilbert.

Notation V := (carrier H).
Notation "x '+v' y" := (vadd H x y) (at level 50, left associativity).
Notation "a '*v' x" := (vscal H a x) (at level 40).
Notation "'<<' x , y '>>'" := (ip H x y) (at level 0).

Definition vopp (x : V) : V := (-1) *v x.
Definition vsub (x y : V) : V := x +v (vopp y).
Definition nrm (x : V) : R := sqrt (<< x , x >>).

(* ---------- Derived bilinearity -------------------------------------------- *)

Lemma ip_add_r : forall x y z : V, << x , (y +v z) >> = << x , y >> + << x , z >>.
Proof.
  intros x y z.
  rewrite (ip_sym H x (y +v z)), (ip_add_l H y z x),
          (ip_sym H y x), (ip_sym H z x).
  reflexivity.
Qed.

Lemma ip_scal_r : forall (a : R) (x y : V), << x , (a *v y) >> = a * << x , y >>.
Proof.
  intros a x y.
  rewrite (ip_sym H x (a *v y)), (ip_scal_l H a y x), (ip_sym H y x).
  reflexivity.
Qed.

Lemma ip_opp_l : forall x y : V, << (vopp x) , y >> = - << x , y >>.
Proof. intros x y. unfold vopp. rewrite (ip_scal_l H). lra. Qed.

Lemma ip_opp_r : forall x y : V, << x , (vopp y) >> = - << x , y >>.
Proof. intros x y. unfold vopp. rewrite ip_scal_r. lra. Qed.

Lemma ip_sub_l : forall x y z : V, << (vsub x y) , z >> = << x , z >> - << y , z >>.
Proof.
  intros x y z. unfold vsub.
  rewrite (ip_add_l H), ip_opp_l. lra.
Qed.

Lemma ip_sub_r : forall x y z : V, << x , (vsub y z) >> = << x , y >> - << x , z >>.
Proof.
  intros x y z. unfold vsub.
  rewrite ip_add_r, ip_opp_r. lra.
Qed.

(* ---------- Norm basics ----------------------------------------------------- *)

Lemma nrm_nonneg : forall x : V, 0 <= nrm x.
Proof. intro x. apply sqrt_pos. Qed.

Lemma nrm_sq : forall x : V, nrm x * nrm x = << x , x >>.
Proof. intro x. unfold nrm. apply sqrt_sqrt, (ip_pos H). Qed.

(*  Expansion of the squared norm of a sum and of a difference. *)
Lemma ip_expand_add :
  forall x y : V,
    << (x +v y) , (x +v y) >> = << x , x >> + 2 * << x , y >> + << y , y >>.
Proof.
  intros x y.
  rewrite (ip_add_l H), !ip_add_r, (ip_sym H y x). lra.
Qed.

Lemma ip_expand_sub :
  forall x y : V,
    << (vsub x y) , (vsub x y) >> = << x , x >> - 2 * << x , y >> + << y , y >>.
Proof.
  intros x y.
  rewrite ip_sub_l, !ip_sub_r, (ip_sym H y x). lra.
Qed.

Lemma ip_opp_opp : forall x : V, << (vopp x) , (vopp x) >> = << x , x >>.
Proof. intro x. rewrite ip_opp_l, ip_opp_r. lra. Qed.

(* ---------- Cauchy--Schwarz -------------------------------------------------- *)

(*  Squared form:  << x , y >>^2 <= << x , x >> * << y , y >>. *)
Lemma CS_sq :
  forall x y : V, (<< x , y >>)^2 <= << x , x >> * << y , y >>.
Proof.
  intros x y.
  set (A := << y , y >>). set (B := << x , y >>). set (C := << x , x >>).
  assert (HA : 0 <= A) by apply (ip_pos H).
  assert (HC : 0 <= C) by apply (ip_pos H).
  assert (Hq : forall t : R, 0 <= C + 2 * t * B + t^2 * A).
  { intro t.
    pose proof (ip_pos H (x +v (t *v y))) as Hp.
    rewrite ip_expand_add in Hp.
    rewrite ip_scal_r in Hp.
    rewrite (ip_scal_l H) in Hp.
    rewrite ip_scal_r in Hp.
    fold B C in Hp. fold A in Hp.
    nra. }
  destruct (Req_dec A 0) as [HA0 | HAne].
  - (*  degenerate direction: B must vanish *)
    assert (HB0 : B = 0).
    { destruct (Req_dec B 0) as [E | Hne]; [exact E |].
      exfalso.
      destruct (Rle_or_lt B 0) as [Hle | Hgt].
      - assert (HBneg : B < 0) by lra.
        pose proof (Hq ((C + 1) / (- 2 * B))) as Hbad.
        rewrite HA0 in Hbad.
        replace (C + 2 * ((C + 1) / (- 2 * B)) * B
                 + ((C + 1) / (- 2 * B))^2 * 0)
          with (- 1) in Hbad by (field; lra).
        lra.
      - pose proof (Hq (- (C + 1) / (2 * B))) as Hbad.
        rewrite HA0 in Hbad.
        replace (C + 2 * (- (C + 1) / (2 * B)) * B
                 + (- (C + 1) / (2 * B))^2 * 0)
          with (- 1) in Hbad by (field; lra).
        lra. }
    rewrite HA0, HB0. nra.
  - (*  A > 0: evaluate at the vertex t = - B / A *)
    assert (HApos : 0 < A) by lra.
    pose proof (Hq (- B / A)) as Hv.
    replace (C + 2 * (- B / A) * B + (- B / A)^2 * A)
      with (C - B^2 / A) in Hv by (field; lra).
    assert (Hfin : B^2 <= A * C).
    { apply (Rmult_le_reg_r (/ A)); [apply Rinv_0_lt_compat; lra |].
      replace (A * C * / A) with C by (field; lra).
      replace (B^2 * / A) with (B^2 / A) by (unfold Rdiv; ring).
      lra. }
    nra.
Qed.

(*  Absolute-value form. *)
Theorem CS : forall x y : V, Rabs (<< x , y >>) <= nrm x * nrm y.
Proof.
  intros x y.
  pose proof (CS_sq x y) as Hs.
  assert (HE : nrm x * nrm y = sqrt (<< x , x >> * << y , y >>)).
  { unfold nrm. rewrite sqrt_mult_alt; [reflexivity | apply (ip_pos H)]. }
  rewrite HE.
  rewrite <- sqrt_Rsqr_abs.
  apply sqrt_le_1_alt.
  unfold Rsqr. nra.
Qed.

Corollary CS_le : forall x y : V, << x , y >> <= nrm x * nrm y.
Proof.
  intros x y.
  pose proof (CS x y).
  pose proof (Rle_abs (<< x , y >>)).
  lra.
Qed.

Corollary CS_ge : forall x y : V, - (nrm x * nrm y) <= << x , y >>.
Proof.
  intros x y.
  pose proof (CS x y).
  pose proof (Rle_abs (- << x , y >>)) as Hr.
  rewrite Rabs_Ropp in Hr.
  lra.
Qed.

(*  Triangle inequality for the norm. *)
Theorem nrm_triangle :
  forall x y : V, nrm (x +v y) <= nrm x + nrm y.
Proof.
  intros x y.
  apply nonneg_le_of_sqr'.
  - apply sqrt_pos.
  - pose proof (nrm_nonneg x). pose proof (nrm_nonneg y). lra.
  - replace ((nrm (x +v y))^2) with (nrm (x +v y) * nrm (x +v y)) by ring.
    rewrite nrm_sq, ip_expand_add.
    pose proof (CS_le x y) as Hcs.
    pose proof (nrm_sq x) as Hx. pose proof (nrm_sq y) as Hy.
    nra.
Qed.

(* ---------- Parametrized Young inequality at vector level ------------------ *)

(*  Scalar Young:  2 x y <= x^2 / lam + lam y^2   for lam > 0. *)
Lemma young_scalar :
  forall x y lam : R, 0 < lam -> 2 * (x * y) <= x^2 / lam + lam * y^2.
Proof.
  intros x y lam Hl.
  apply (Rmult_le_reg_r lam); [exact Hl |].
  replace ((x^2 / lam + lam * y^2) * lam)
    with (x^2 + lam^2 * y^2) by (field; lra).
  pose proof (pow2_ge_0 (x - lam * y)) as Hp.
  nra.
Qed.

(*  Vector Young:  2 << x , y >> >= - ||x||^2 / lam - lam ||y||^2. *)
Theorem young_vector :
  forall (x y : V) (lam : R),
    0 < lam ->
    2 * << x , y >> >= - (<< x , x >> / lam) - lam * << y , y >>.
Proof.
  intros x y lam Hl.
  pose proof (CS_ge x y) as Hcs.
  pose proof (young_scalar (nrm x) (nrm y) lam Hl) as Hy.
  pose proof (nrm_sq x) as Hx2. pose proof (nrm_sq y) as Hy2.
  pose proof (nrm_nonneg x). pose proof (nrm_nonneg y).
  nra.
Qed.

(* ---------- The difference-of-squares identity ------------------------------ *)

(*  << -a + b , a + b >> = << b , b >> - << a , a >>: the exact mechanism     *)
(*  behind the ASGS cancellation in eq:StabilityEstimate.                     *)
Theorem diff_of_squares :
  forall a b : V,
    << ((vopp a) +v b) , (a +v b) >> = << b , b >> - << a , a >>.
Proof.
  intros a b.
  rewrite (ip_add_l H), !ip_add_r, !ip_opp_l, (ip_sym H b a).
  lra.
Qed.

End InnerSpaceFacts.

Arguments vopp {H}.
Arguments vsub {H}.
Arguments nrm {H}.
