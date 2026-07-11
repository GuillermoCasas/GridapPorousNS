/-
  PorousNSToolbox.lean
  --------------------
  STATEMENT ROADMAP — NOT A PROOF ARTIFACT.

  This file states, as Lean 4 targets, the residual trusted base of the
  Coq development (AbstractStability.v / AbstractContinuity.v) for the
  porous Navier--Stokes stabilized FEM paper.  Every theorem ends in
  `sorry`; the file is expected to need API repair before it even
  elaborates against current Mathlib.  See LEAN_ROADMAP.md for the
  classification, strategy and loop protocol.  The intended semantics of
  each statement is pinned in the comment above it; when repairing the
  statement, preserve the semantics, not the syntax.

  Suggested first loop task: make this file compile with sorrys.
-/

import Mathlib

open MeasureTheory

noncomputable section
namespace PorousNSToolbox

variable {d : ℕ}

/-- A nondegenerate simplex in ℝ^d, carried as its vertex map.
    (Repair candidate: use `Affine.Simplex ℝ (EuclideanSpace ℝ (Fin d)) d`.) -/
def SimplexIn (d : ℕ) := Affine.Simplex ℝ (EuclideanSpace ℝ (Fin d)) d

/-- The closed convex hull of the vertices — the element K. -/
def elemSet (K : SimplexIn d) : Set (EuclideanSpace ℝ (Fin d)) :=
  convexHull ℝ (Set.range K.points)

/-- Diameter and inradius of the element.  h_K and rho_K of the paper;
    shape regularity is the hypothesis  h_K ≤ γ · rho_K. -/
def hK (K : SimplexIn d) : ℝ := Metric.diam (elemSet K)

/-- Inradius: radius of the largest inscribed ball.  (Repair candidate:
    define via `sSup { r | ∃ x, Metric.closedBall x r ⊆ elemSet K }`.) -/
def rhoK (K : SimplexIn d) : ℝ :=
  sSup { r : ℝ | ∃ x, Metric.closedBall x r ⊆ elemSet K }

/-- Scalar polynomials of total degree ≤ k on ℝ^d, as functions. -/
def PolyDeg (d k : ℕ) : Set (EuclideanSpace ℝ (Fin d) → ℝ) :=
  { f | ∃ p : MvPolynomial (Fin d) ℝ, p.totalDegree ≤ k ∧
        ∀ x, f x = MvPolynomial.eval (fun i => x i) p }

/- ======================================================================= -/
/-  T1.  Unweighted inverse estimate on a shape-regular simplex.           -/
/-  Paper: eq:inverse (L2 form); consumed by lem:winv.                     -/
/-  ||∇p||_{L²(K)} ≤ C(k,d,γ)/h_K ||p||_{L²(K)} for p ∈ P_k, h_K ≤ γ ρ_K.  -/
/- ======================================================================= -/

theorem inverse_estimate_grad (k : ℕ) (γ : ℝ) (hγ : 1 ≤ γ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (K : SimplexIn d) (p : EuclideanSpace ℝ (Fin d) → ℝ),
        p ∈ PolyDeg d k →
        hK K ≤ γ * rhoK K →
        Real.sqrt (∫ x in elemSet K, ‖fderiv ℝ p x‖ ^ 2)
          ≤ C / hK K * Real.sqrt (∫ x in elemSet K, (p x) ^ 2) := by
  sorry

/- ======================================================================= -/
/-  T2a.  L∞ inverse estimate.  Paper: eq:inverse (L∞ form).               -/
/-  ||p||_{L∞(K)} ≤ C(k,d,γ) h_K^{-d/2} ||p||_{L²(K)}.                     -/
/- ======================================================================= -/

theorem inverse_estimate_linf (k : ℕ) (γ : ℝ) (hγ : 1 ≤ γ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (K : SimplexIn d) (p : EuclideanSpace ℝ (Fin d) → ℝ),
        p ∈ PolyDeg d k →
        hK K ≤ γ * rhoK K →
        (∀ x ∈ elemSet K, |p x|
          ≤ C * (hK K) ^ (-(d : ℝ) / 2)
              * Real.sqrt (∫ y in elemSet K, (p y) ^ 2)) := by
  sorry

/- ======================================================================= -/
/-  T2b.  Face measure bound.  meas_{d-1}(face) ≤ C(d) h_K^{d-1}.          -/
/-  Intended measure: (d-1)-dimensional Hausdorff on the affine span of    -/
/-  the face.                                                              -/
/- ======================================================================= -/

theorem face_measure_bound :
    ∃ C : ℝ, 0 < C ∧
      ∀ (K : SimplexIn d) (i : Fin (d + 1)),
        (μH[(d : ℝ) - 1] (convexHull ℝ
            (Set.range (K.points ∘ (fun j : Fin d => (i.succAbove j)))))).toReal
          ≤ C * hK K ^ (d - 1) := by
  sorry

/- ======================================================================= -/
/-  T4.  Divergence theorem on a simplex, for the product data class       -/
/-  actually used: F = α · G with G elementwise polynomial (vector) and    -/
/-  α ∈ W^{1,∞}.  Stated here for Lipschitz α and componentwise-polynomial -/
/-  G; n is the outward unit normal, surface measure as in T2b.            -/
/-  Consumed by: H_elem_conv_ibp, H_elem_p_ibp (per element), and — after  -/
/-  summation with zero boundary trace — H_skew, H_ibp_vp, H_ibp_qu, HBS.  -/
/- ======================================================================= -/

theorem simplex_divergence (k : ℕ)
    (K : SimplexIn d)
    (α : EuclideanSpace ℝ (Fin d) → ℝ)
    (hα : LipschitzOnWith 1 α (elemSet K))   -- placeholder for W^{1,∞}
    (G : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (hG : ∀ i, (fun x => G x i) ∈ PolyDeg d k) :
    True := by
  -- Intended statement (to be made precise during the elaboration pass):
  --   ∫_K div (α • G) = ∫_{∂K} α (G ⬝ n) dS,
  -- with div taken a.e. (α is differentiable a.e. by Rademacher) and the
  -- boundary integral over the d+1 faces with T2b's surface measure.
  -- The `True` placeholder keeps the file compilable at the elaboration
  -- pass; replace it with the equation once the boundary-integral API is
  -- fixed.
  sorry

/- ======================================================================= -/
/-  T3.  Weighted reductions (lem:winv), given T1/T2 and the porosity      -/
/-  resolution.  The scalar chain is already machine-checked in Coq        -/
/-  (ContinuityAlgebra.delta_alpha_bound, winv_ratio); here only the       -/
/-  norm-level monotonicity  α₀ ∫ f² ≤ ∫ α f² ≤ α_∞ ∫ f²  is new.          -/
/- ======================================================================= -/

theorem weighted_norm_squeeze
    (K : SimplexIn d) (α f : EuclideanSpace ℝ (Fin d) → ℝ)
    (a0 aInf : ℝ)
    (hlo : ∀ x ∈ elemSet K, a0 ≤ α x)
    (hhi : ∀ x ∈ elemSet K, α x ≤ aInf)
    (hf : IntegrableOn (fun x => (f x) ^ 2) (elemSet K)) :
    a0 * ∫ x in elemSet K, (f x) ^ 2
      ≤ ∫ x in elemSet K, α x * (f x) ^ 2 ∧
    ∫ x in elemSet K, α x * (f x) ^ 2
      ≤ aInf * ∫ x in elemSet K, (f x) ^ 2 := by
  sorry

end PorousNSToolbox
