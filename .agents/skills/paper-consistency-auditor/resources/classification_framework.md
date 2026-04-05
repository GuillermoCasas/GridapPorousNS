# Issue Classification Framework

When evaluating code mapping to the reference mathematics, you must assign one of the following strict severity and consistency markers.

## Severity Labels

- 🚨 **CRITICAL INCONSISTENCY**: A direct mathematical violation of the paper's formulation that fundamentally breaks theoretical guarantees. (e.g., using a non-exact directional derivative in a Newton solver linesearch; discarding a matrix term incorrectly; dimension-specific hard-coding in a supposedly generic function).
- 🐛 **LIKELY BUG**: An algebraic mistake, transcription error, or variable typo where the intent was to follow the paper but the code executes incorrectly.
- ⚠️ **UNDOCUMENTED DEVIATION**: The implementation uses a heuristic, alternative choice, or different numerical framework than the text implies, but does so without explicit comments or architectural abstraction marking it as a deviation.
- 📝 **DOCUMENTED APPROXIMATION**: The codebase explicitly documents a departure from the paper (e.g., using a continuous `freeze_cusp` to avoid dividing by zero mapping at origin), and the logic maps appropriately to a known experimental track.
- ⏳ **NEEDS DERIVATION**: The mathematical claim from the paper is highly ambiguous or generic, and tracing its translation to Gridap-specific discrete finite elemental operators requires scratchpad derivation before a pass/fail claim can be confidently made.

## Consistency Status
Use these tags strictly when mapping to the matrix:
- **Checked & Passing**: Verified mathematically against the reference and isolated with a live fast test.
- **Checked & Failing**: Identified as wrong. Pending remediation.
- **Unchecked**: Identified as existing in code but currently lacking a rigorous mathematical review or mapping trace.
- **Needs Derivation**: Pending analytical derivation.

## Key Target Areas to Audit Carefully
- **Continuous PDE Model**: Does the code reflect the correct strong form balance equations?
- **Weak Form**: Proper testing pairings? Integration by parts sign orientations?
- **Strong Residual**: Is the stabilizing residual strictly matching the continuous momentum/mass definitions?
- **Adjoint Operator**: Used rigorously within OSGS/ASGS, does it mathematically mirror the implemented weak form boundary operators precisely?
- **Tau Logic**: Exact Fréchet limits, dimensional scaling, positivity constraints.
- **Projection Policies**: Linear vs. nonlinear law legalities.
- **Solver Globalization**: Are heuristic limit-cycles documented as alternatives? Is Exact Newton utilizing strict Armijo descent correctly?
