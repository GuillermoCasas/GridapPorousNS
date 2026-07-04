> **ARCHIVED / SUPERSEDED (2026-07-04).** Raw 2026-05-23 transcript. Its central recommendation (untag/free the outlet-corner DOF to escape the singularity) was FALSIFIED — the ModifiedCorner experiment made structured convergence ~3× worse (see [investigation-synthesis.md](investigation-synthesis.md), H1). Retained for provenance only.

> **RAW CONVERSATION TRANSCRIPT (2026-05-23) — historical.** Corner-singularity Q&A from the Cocquet investigation; kept verbatim for traceability. Consolidated conclusions are in the canonical [docs/cocquet/investigation-synthesis.md](investigation-synthesis.md).

# you asked

message time: 2026-05-23 23:03:40

Must tgv used or is the topological manipulation sufficient?

---

# gemini response

### Summary

**No, you do not need to implement the TGV (Très Grande Valeur) algebraic penalty in Gridap.** The topological manipulation—manually stripping the Dirichlet tag from the exact corner vertices—is entirely sufficient. 

Both Gridap's topological condensation and FreeFem's TGV penalty are just different numerical machineries designed to achieve the exact same mathematical outcome: forcing a discrete nodal value to a prescribed number. The reason FreeFem escapes the singularity is not because TGV has special physical properties, but because its unstructured mesh tagging randomly fails to apply the TGV to the corner node. By intentionally untagging the corner in Gridap, you perfectly replicate this omission, freeing the node and releasing the unphysical pressure spike.

***

### The Equivalence of Constraint Machineries

To understand why topological manipulation is sufficient, it is necessary to look at the resulting algebraic systems produced by Gridap and FreeFem++ when a node is effectively constrained versus when it is released.

#### Gridap: Topological Condensation
When Gridap sees a node tagged as Dirichlet, it performs algebraic condensation. If node $i$ is constrained to $u_\text{bc}$, Gridap completely removes row $i$ and column $i$ from the global stiffness matrix $\mathbf{K}$. It moves the known quantity to the right-hand side, altering the load vector $\mathbf{F}$ for all adjacent interior nodes. The dimension of the linear system is physically reduced.

#### FreeFem++: Algebraic Penalization (TGV)
When FreeFem++ applies the `on()` command, it leaves the dimension of the matrix unchanged. Instead, for a constrained node $i$, it overwrites the matrix diagonal and the load vector with a massive penalty parameter (TGV, typically $10^{30}$):
$$\mathbf{K}_{ii} = 10^{30}, \quad \mathbf{F}_i = 10^{30} \cdot u_\text{bc}$$
Because $10^{30}$ dwarfs all other physical terms (like viscosity or permeability) in that row, the solver effectively sees the equation $10^{30} u_i = 10^{30} u_\text{bc}$, which trivially yields $u_i = u_\text{bc}$.

#### The Verdict on the Corner Node
If both methods successfully target the corner node, they will both lock it to $\mathbf{u} = \mathbf{0}$, and both will generate the exact same infinite pressure gradient. 

However, FreeFem's unstructured boundary tagging often results in the corner node being assigned exclusively to the Neumann boundary ($\Gamma_\text{N}$). When this happens, FreeFem simply **skips** applying the TGV to that node. The node behaves as a standard, unconstrained degree of freedom governed purely by the weak form of the Navier-Stokes and Darcy/Forchheimer operators. 

By manually changing the entity ID of the corner vertices in Gridap (the topological manipulation), you are instructing Gridap to skip the condensation of those specific nodes.  The resulting algebraic system in Gridap will be mathematically identical to the algebraic system FreeFem solves when it misses the corner: the exact corner node becomes a free degree of freedom, instantly resolving the geometric incompatibility and smoothing the pressure field.

### Why the Pressure Spike Vanishes

When the exact corner node is algebraically unconstrained, it acts as a microscopic "pressure release valve." 

In the fully constrained (singular) case, the fluid is mathematically forced to have zero velocity exactly at $(2,1)$. But infinitesimally close to that point, on the outlet edge, the fluid must exhibit a fully developed Poiseuille-like exiting profile. This creates an infinite shear rate.

When you untag the corner, the finite element shape functions (the $P_2$ velocity bubble) centered on that specific corner node are allowed to take on a non-zero value. The solver will naturally find a small, non-zero exit velocity for that exact geometric point that minimizes the global residual energy. Because the velocity is no longer strictly zero, the shear rate is finite, the pressure remains bounded, and the global $L^2$ error convergence is restored to the optimal $O(h^2)$ rate expected for a smooth, well-posed problem.

***

### Generalization: Subspace Restriction vs. Subspace Perturbation

This specific discrete phenomenon can be generalized to the broader framework of how we construct constrained functional spaces in Ritz-Galerkin methods. 

When solving an elliptic boundary value problem, the true mathematical solution $u$ resides in a highly specific function space, typically a Sobolev space like $H^1(\Omega)$, subject to a trace condition $\gamma_0(u) = g$ on $\Gamma_\text{D}$. The subspace of functions satisfying this essential boundary condition is an affine manifold. 

The two methods discussed represent two fundamentally different philosophies for restricting our discrete approximation $\mathbf{u}_h$ to this manifold:

1.  **Exact Subspace Restriction (Gridap's Condensation):** This is a purely topological operation. By mathematically eliminating degrees of freedom, we ensure that our discrete test and trial spaces $V_h$ and $U_h$ are strictly conforming subsets of the continuous spaces $H^1_0(\Omega)$ and $H^1_g(\Omega)$. The constraint is satisfied exactly, but if the continuous manifold contains a geometric singularity (due to trace incompatibility), the discrete space is forced to brutally approximate it, destroying global convergence.
2.  **Operator Perturbation (FreeFem's TGV):** This method does not alter the underlying finite element space; the space remains the fully unconstrained $H^1(\Omega)$. Instead, it aggressively perturbs the continuous operator (the bilinear form) with an algebraic penalty. While it mimics subspace restriction as the penalty approaches infinity, its reliance on discrete nodal tagging makes it highly vulnerable to topological noise.

When dealing with boundary incompatibilities, Exact Subspace Restriction is mathematically pure but computationally unforgiving. It exposes the exact, unvarnished truth of the Boundary Value Problem, which, as you discovered, is sometimes a truth that the original authors themselves failed to see due to the topological noise of their own numerical framework.