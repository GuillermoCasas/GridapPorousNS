(* ========================================================================= *)
(*  AbstractSums.v                                                           *)
(*                                                                           *)
(*  Finite sums over a list of mesh elements (or faces), with the summation  *)
(*  toolkit needed by the abstract stability and continuity proofs:          *)
(*  linearity, monotonicity, the triangle inequality, the Cauchy--Schwarz    *)
(*  inequality for sums (the discrete l2 pairing used every time the paper   *)
(*  invokes the Cauchy--Schwarz inequality for the sums over elements), and  *)
(*  subadditivity of the square root.                                        *)
(*                                                                           *)
(*  Coq 8.18, stdlib only.                                                   *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
Local Open Scope R_scope.

(* ---------- Square-comparison helpers (shared pattern) -------------------- *)

Lemma nonneg_le_of_sqr :
  forall x y : R, 0 <= x -> 0 <= y -> x^2 <= y^2 -> x <= y.
Proof.
  intros x y Hx Hy Hs.
  destruct (Rle_or_lt x y) as [H | H]; [exact H |].
  exfalso.
  assert (Hxy : 0 < x + y) by lra.
  assert (Hp : 0 < (x - y) * (x + y)) by nra.
  nra.
Qed.

Section Sums.

Variable K : Type.

Fixpoint Rsum (l : list K) (f : K -> R) : R :=
  match l with
  | nil => 0
  | k :: t => f k + Rsum t f
  end.

Lemma Rsum_ext :
  forall (l : list K) (f g : K -> R),
    (forall k, f k = g k) -> Rsum l f = Rsum l g.
Proof.
  intros l f g H. induction l as [| a t IH]; simpl.
  - reflexivity.
  - rewrite H, IH. reflexivity.
Qed.

Lemma Rsum_nonneg :
  forall (l : list K) (f : K -> R),
    (forall k, 0 <= f k) -> 0 <= Rsum l f.
Proof.
  intros l f H. induction l as [| a t IH]; simpl.
  - lra.
  - pose proof (H a). lra.
Qed.

Lemma Rsum_le :
  forall (l : list K) (f g : K -> R),
    (forall k, f k <= g k) -> Rsum l f <= Rsum l g.
Proof.
  intros l f g H. induction l as [| a t IH]; simpl.
  - lra.
  - pose proof (H a). lra.
Qed.

Lemma Rsum_plus :
  forall (l : list K) (f g : K -> R),
    Rsum l (fun k => f k + g k) = Rsum l f + Rsum l g.
Proof.
  intros l f g. induction l as [| a t IH]; simpl.
  - lra.
  - rewrite IH. lra.
Qed.

Lemma Rsum_minus :
  forall (l : list K) (f g : K -> R),
    Rsum l (fun k => f k - g k) = Rsum l f - Rsum l g.
Proof.
  intros l f g. induction l as [| a t IH]; simpl.
  - lra.
  - rewrite IH. lra.
Qed.

Lemma Rsum_opp :
  forall (l : list K) (f : K -> R),
    Rsum l (fun k => - f k) = - Rsum l f.
Proof.
  intros l f. induction l as [| a t IH]; simpl.
  - lra.
  - rewrite IH. lra.
Qed.

Lemma Rsum_scal :
  forall (l : list K) (c : R) (f : K -> R),
    Rsum l (fun k => c * f k) = c * Rsum l f.
Proof.
  intros l c f. induction l as [| a t IH]; simpl.
  - lra.
  - rewrite IH. lra.
Qed.

(*  Triangle inequality:  |sum f| <= sum |f|. *)
Lemma Rsum_abs_le :
  forall (l : list K) (f : K -> R),
    Rabs (Rsum l f) <= Rsum l (fun k => Rabs (f k)).
Proof.
  intros l f. induction l as [| a t IH]; simpl.
  - rewrite Rabs_R0. lra.
  - pose proof (Rabs_triang (f a) (Rsum t f)). lra.
Qed.

(* ---------- Cauchy--Schwarz for sums -------------------------------------- *)

(*  Squared form, by induction (nonnegative entries suffice for all uses:    *)
(*  the entries are elemental norms).                                        *)
Lemma Rsum_CS_sq :
  forall (l : list K) (f g : K -> R),
    (forall k, 0 <= f k) -> (forall k, 0 <= g k) ->
    (Rsum l (fun k => f k * g k))^2
    <= Rsum l (fun k => (f k)^2) * Rsum l (fun k => (g k)^2).
Proof.
  intros l f g Hf Hg.
  induction l as [| a t IH]; cbn [Rsum].
  - nra.
  - set (s := Rsum t (fun k => f k * g k)) in *.
    set (P := Rsum t (fun k => (f k)^2)) in *.
    set (Q := Rsum t (fun k => (g k)^2)) in *.
    assert (Hs0 : 0 <= s).
    { apply Rsum_nonneg. intro k. pose proof (Hf k); pose proof (Hg k). nra. }
    assert (HP0 : 0 <= P).
    { apply Rsum_nonneg. intro k. apply pow2_ge_0. }
    assert (HQ0 : 0 <= Q).
    { apply Rsum_nonneg. intro k. apply pow2_ge_0. }
    assert (HsPQ : s <= sqrt P * sqrt Q).
    { apply nonneg_le_of_sqr; [exact Hs0 | |].
      - assert (0 <= sqrt P) by apply sqrt_pos.
        assert (0 <= sqrt Q) by apply sqrt_pos. nra.
      - replace ((sqrt P * sqrt Q)^2)
          with ((sqrt P * sqrt P) * (sqrt Q * sqrt Q)) by ring.
        rewrite !sqrt_sqrt by lra.
        exact IH. }
    assert (HP2 : sqrt P * sqrt P = P) by (apply sqrt_sqrt; lra).
    assert (HQ2 : sqrt Q * sqrt Q = Q) by (apply sqrt_sqrt; lra).
    assert (E1 : (f a * sqrt Q - g a * sqrt P)^2
                 = (f a)^2 * (sqrt Q * sqrt Q)
                   - 2 * (f a * g a) * (sqrt P * sqrt Q)
                   + (g a)^2 * (sqrt P * sqrt P)) by ring.
    rewrite HP2, HQ2 in E1.
    pose proof (pow2_ge_0 (f a * sqrt Q - g a * sqrt P)) as Hp2.
    rewrite E1 in Hp2.
    (*  2 (fa ga) s <= 2 (fa ga) sqrt P sqrt Q <= fa^2 Q + ga^2 P *)
    assert (Hfg0 : 0 <= f a * g a).
    { pose proof (Hf a); pose proof (Hg a). nra. }
    assert (Hcross1 : 2 * (f a * g a) * s
                      <= 2 * (f a * g a) * (sqrt P * sqrt Q)) by nra.
    assert (Hcross : 2 * (f a * g a) * s <= (f a)^2 * Q + (g a)^2 * P)
      by lra.
    replace ((f a * g a + s)^2)
      with ((f a)^2 * (g a)^2 + 2 * (f a * g a) * s + s^2) by ring.
    replace (((f a)^2 + P) * ((g a)^2 + Q))
      with ((f a)^2 * (g a)^2 + (f a)^2 * Q + P * (g a)^2 + P * Q) by ring.
    lra.
Qed.

(*  Square-root form. *)
Theorem Rsum_CS :
  forall (l : list K) (f g : K -> R),
    (forall k, 0 <= f k) -> (forall k, 0 <= g k) ->
    Rsum l (fun k => f k * g k)
    <= sqrt (Rsum l (fun k => (f k)^2)) * sqrt (Rsum l (fun k => (g k)^2)).
Proof.
  intros l f g Hf Hg.
  assert (HP0 : 0 <= Rsum l (fun k => (f k)^2))
    by (apply Rsum_nonneg; intro k; apply pow2_ge_0).
  assert (HQ0 : 0 <= Rsum l (fun k => (g k)^2))
    by (apply Rsum_nonneg; intro k; apply pow2_ge_0).
  apply nonneg_le_of_sqr.
  - apply Rsum_nonneg. intro k.
    pose proof (Hf k); pose proof (Hg k). nra.
  - assert (0 <= sqrt (Rsum l (fun k => (f k)^2))) by apply sqrt_pos.
    assert (0 <= sqrt (Rsum l (fun k => (g k)^2))) by apply sqrt_pos.
    nra.
  - replace ((sqrt (Rsum l (fun k => (f k)^2))
              * sqrt (Rsum l (fun k => (g k)^2)))^2)
      with ((sqrt (Rsum l (fun k => (f k)^2))
             * sqrt (Rsum l (fun k => (f k)^2)))
            * (sqrt (Rsum l (fun k => (g k)^2))
               * sqrt (Rsum l (fun k => (g k)^2)))) by ring.
    rewrite !sqrt_sqrt by lra.
    apply Rsum_CS_sq; assumption.
Qed.

(*  Monotone image bound for a sum with squared entries (used with the       *)
(*  square root outside).                                                    *)
Lemma Rsum_sq_mono :
  forall (l : list K) (f g : K -> R),
    (forall k, 0 <= f k) -> (forall k, f k <= g k) ->
    Rsum l (fun k => (f k)^2) <= Rsum l (fun k => (g k)^2).
Proof.
  intros l f g Hf Hle. apply Rsum_le. intro k.
  pose proof (Hf k). pose proof (Hle k). nra.
Qed.

End Sums.

Arguments Rsum {K}.

(* ---------- Square root subadditivity ------------------------------------- *)

Lemma sqrt_plus_le :
  forall x y : R, 0 <= x -> 0 <= y -> sqrt (x + y) <= sqrt x + sqrt y.
Proof.
  intros x y Hx Hy.
  apply nonneg_le_of_sqr.
  - apply sqrt_pos.
  - assert (0 <= sqrt x) by apply sqrt_pos.
    assert (0 <= sqrt y) by apply sqrt_pos. lra.
  - replace ((sqrt (x + y))^2) with (sqrt (x + y) * sqrt (x + y)) by ring.
    replace ((sqrt x + sqrt y)^2)
      with (sqrt x * sqrt x + 2 * (sqrt x * sqrt y) + sqrt y * sqrt y)
      by ring.
    rewrite !sqrt_sqrt by lra.
    assert (0 <= sqrt x) by apply sqrt_pos.
    assert (0 <= sqrt y) by apply sqrt_pos.
    assert (0 <= sqrt x * sqrt y) by nra.
    lra.
Qed.

(*  Monotonicity of the square root, restated for convenience. *)
Lemma sqrt_mono : forall x y : R, x <= y -> sqrt x <= sqrt y.
Proof. exact sqrt_le_1_alt. Qed.

(*  The discrete l2-in-l1 inequality:  sqrt (sum a_k^2) <= sum a_k  for
    nonnegative entries (the appendix's eq:Enorm step). *)
Lemma sqrt_sum_sq_le_sum :
  forall (K : Type) (l : list K) (f : K -> R),
    (forall k, 0 <= f k) ->
    sqrt (Rsum l (fun k => (f k)^2)) <= Rsum l f.
Proof.
  intros K l f Hf. induction l as [| a t IH]; cbn [Rsum].
  - rewrite sqrt_0. lra.
  - assert (HS0 : 0 <= Rsum t (fun k => (f k)^2))
      by (apply Rsum_nonneg; intro k; apply pow2_ge_0).
    assert (HT0 : 0 <= Rsum t f) by (apply Rsum_nonneg; exact Hf).
    pose proof (Hf a) as Ha.
    apply nonneg_le_of_sqr; [apply sqrt_pos | lra |].
    replace ((sqrt ((f a)^2 + Rsum t (fun k => (f k)^2)))^2)
      with (sqrt ((f a)^2 + Rsum t (fun k => (f k)^2))
            * sqrt ((f a)^2 + Rsum t (fun k => (f k)^2))) by ring.
    rewrite sqrt_sqrt by (pose proof (pow2_ge_0 (f a)); lra).
    (*  (f a + S)^2 >= f a ^2 + (sqrt of tail bound)^2  *)
    assert (Hsq : (sqrt (Rsum t (fun k => (f k)^2)))^2
                  = Rsum t (fun k => (f k)^2)).
    { replace ((sqrt (Rsum t (fun k => (f k)^2)))^2)
        with (sqrt (Rsum t (fun k => (f k)^2))
              * sqrt (Rsum t (fun k => (f k)^2))) by ring.
      apply sqrt_sqrt; exact HS0. }
    assert (Hst : 0 <= sqrt (Rsum t (fun k => (f k)^2))) by apply sqrt_pos.
    nra.
Qed.
