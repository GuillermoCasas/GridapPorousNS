using Pkg; Pkg.activate(".")
using Gridap
using PorousNSSolver

nls = PorousNSSolver.SafeNewtonSolver(LUSolver(), 15, 3, 1e-12, 1e-8, 1e-10, 1e-3, 1e-4)

# We need to simulate the EXACT FESolver call sequence!
struct DummyOp <: Gridap.Algebra.NonlinearOperator end
Gridap.Algebra.residual!(b, op::DummyOp, x) = (b .= x)
Gridap.Algebra.jacobian!(A, op::DummyOp, x) = fill!(A, 1.0)
Gridap.Algebra.allocate_residual(op::DummyOp, x) = copy(x)
Gridap.Algebra.allocate_jacobian(op::DummyOp, x) = zeros(2,2)

# But wait, we don't know if DummyOp will throw it. The exception comes from Gridap.FESpaces!
# So we need a dummy FEOperator.
# Or better yet, just run run_test.jl with a custom catch!
