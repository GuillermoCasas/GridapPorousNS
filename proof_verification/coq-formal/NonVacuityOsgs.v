(* ========================================================================= *)
(*  NonVacuityOsgs.v                                                         *)
(*                                                                           *)
(*  Machine-checked NON-VACUITY of the OSGS abstract theorems: explicit       *)
(*  concrete instances that discharge EVERY hypothesis of                     *)
(*    - abstract_osgs_stability (Theorem 4.1, the inf--sup collection),       *)
(*    - abstract_osgs_continterp (Lemma 5.2, interpolation continuity),       *)
(*    - abstract_osgs_consistency (Lemma 5.1, the consistency error),         *)
(*  and apply the theorem inside the kernel, obtaining a NON-DEGENERATE       *)
(*  numeric conclusion with both sides strictly positive.  This proves the    *)
(*  hypothesis bundles are jointly satisfiable (consistent), the analogue of  *)
(*  NonVacuity.v / NonVacuityInterp.v for the ASGS chain.                     *)
(*                                                                           *)
(*  No Admitted, no Axiom.                                                    *)
(* ========================================================================= *)

From Coq Require Import Reals Lra Lia Psatz List.
Import ListNotations.
From PNSFormal Require Import ContinuityAlgebra InnerSpace AbstractSums
                              OsgsStability OsgsInterpolation.
Local Open Scope R_scope.

(* ========================================================================= *)
(*  Witness 1: abstract_osgs_stability (the special-test-function            *)
(*  inf--sup collection, Steps 4--5 of th:stability).                        *)
(*                                                                           *)
(*  A non-degenerate diagonal instance: beta0 = 1, M2 = 1, gamma = 20        *)
(*  (= gamma0 = 10(1+M2), the extremal admissible value), psi = 0, C2 = 0;   *)
(*  every scalar energy = 1 (Dtau = P2 + O2 = 2), and the seven term         *)
(*  estimates met with room.  The theorem then yields                        *)
(*    B_osgs(U_h, W) = 7  >=  beta_stab * |||U_h|||^2 = (1/16) * 6 = 3/8,     *)
(*  with beta_stab = min(1/8, 1/16) = 1/16 and |||U_h|||^2 = 6 > 0.          *)
(* ========================================================================= *)

Theorem witness_osgs_stability :
  0 < OsgsStability.beta_stab 1
  /\ (7:R) >= OsgsStability.beta_stab 1 * OsgsStability.NormSq 1 1 1 1 2
  /\ (8:R) <= 2 * (1 + 1) * OsgsStability.NormSq 1 1 1 1 2.
Proof.
  refine (OsgsStability.abstract_osgs_stability
           1 20 0 1 0
           _ _ _ _ _ _ _ _
           1 1 1 1 1 1 1 1 2 2 8 1
           _ _ _ _ _ _ _ _
           7 0 1 0 0 1 0 0
           _ _ _ _ _ _ _ _ _ _ _ _ _);
    unfold OsgsStability.NormSq; (lra || nra).
Qed.

(*  The headline: both sides of the inf--sup conclusion are strictly         *)
(*  positive and the strengthened triple norm is nonzero -- not vacuous.      *)
Theorem osgs_stability_not_vacuous :
  OsgsStability.NormSq 1 1 1 1 2 = 6
  /\ 0 < OsgsStability.beta_stab 1 * OsgsStability.NormSq 1 1 1 1 2
  /\ (7:R) >= OsgsStability.beta_stab 1 * OsgsStability.NormSq 1 1 1 1 2.
Proof.
  destruct witness_osgs_stability as [Hpos [Hge _]].
  assert (HN : OsgsStability.NormSq 1 1 1 1 2 = 6)
    by (unfold OsgsStability.NormSq; lra).
  split; [exact HN |].
  split; [ rewrite HN; nra | exact Hge ].
Qed.

(* ========================================================================= *)
(*  Witnesses 2 and 3: abstract_osgs_continterp and abstract_osgs_consistency. *)
(*                                                                           *)
(*  A one-element mesh (K = unit, Th = [tt]) with nu = 1, sigma = 0, eps = 0, *)
(*  c1 = c2 = 1, h = alpha = 1, |a| = 0, IU = IP = 1 makes tau1 = tau2 = 1,   *)
(*  Da_{h,K} = 0, so the error function is E(h) = 2 (clean rational).         *)
(*  The seven interpolation groups I1..I7 = 1 (BEV = 7) with unit constants,  *)
(*  and the two consistency slots S1 = S2 = 1 (BcU = 2), are met with room.   *)
(* ========================================================================= *)

Definition uTh  : list unit := [tt].
Definition uhK  (_ : unit) : R := 1.
Definition uaK  (_ : unit) : R := 1.
Definition uam  (_ : unit) : R := 0.
Definition uIU  (_ : unit) : R := 1.
Definition uIP  (_ : unit) : R := 1.

(*  E(h) = 2 on this instance (tau1 = tau2 = 1, Da_{h,K} = 0).  *)
Lemma wEh_val :
  OsgsInterpolation.Eh unit uTh 1 0 1 1 uhK uaK uam uIU uIP = 2.
Proof.
  unfold OsgsInterpolation.Eh, uTh. simpl Rsum.
  unfold OsgsInterpolation.ErrTerm, OsgsInterpolation.t1, OsgsInterpolation.t2,
         OsgsInterpolation.Dah, uhK, uaK, uam, uIU, uIP.
  unfold ContinuityAlgebra.tau1, ContinuityAlgebra.tau2, ContinuityAlgebra.phi1,
         ContinuityAlgebra.tauNSinv.
  replace (1 + 0 * 1^2 / (1 * 1)) with 1 by field.
  replace (1^2 * (1 * 1 / 1^2 + 1 * 0 / 1) / (1 * 1)) with 1 by field.
  replace (/ (1 * (1 * 1 / 1^2 + 1 * 0 / 1) + 0)) with 1 by field.
  rewrite !sqrt_1.
  field.
Qed.

Theorem witness_osgs_continterp :
  Rabs 7 <= OsgsInterpolation.CtotI 1 1 1 1 1 1 1
            * (OsgsInterpolation.Eh unit uTh 1 0 1 1 uhK uaK uam uIU uIP * 1).
Proof.
  refine (OsgsInterpolation.abstract_osgs_continterp
           unit uTh 1 0 1 1 uhK uaK uam uIU uIP
           1
           7 1 1 1 1 1 1 1
           1 1 1 1 1 1 1
           _ _ _ _ _ _ _ _);
    try lra;
    rewrite Rabs_R1; rewrite wEh_val; lra.
Qed.

Theorem witness_osgs_consistency :
  Rabs 2 <= OsgsInterpolation.CconsI 1 1
            * (OsgsInterpolation.Eh unit uTh 1 0 1 1 uhK uaK uam uIU uIP * 1).
Proof.
  refine (OsgsInterpolation.abstract_osgs_consistency
           unit uTh 1 0 1 1 uhK uaK uam uIU uIP
           1
           2 1 1
           1 1
           _ _ _);
    try lra;
    rewrite Rabs_R1; rewrite wEh_val; lra.
Qed.
