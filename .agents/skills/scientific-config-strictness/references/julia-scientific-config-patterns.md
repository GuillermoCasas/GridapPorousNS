# Julia Scientific Config Patterns

This document showcases strict, structurally protective Julia idioms explicitly curated for high-stress numerical simulation, FEM pipelines, and multi-physics architecture handling configuration safely.

## 1. Validated Typed Structs (The Goal)

**Anti-Pattern:** Passing `Dict{String, Any}` dictionaries down into solving layers.
**Anti-Pattern:** Overloaded multi-argument parameter definitions offering keyword fallbacks: `function create_fluid_domain(mesh; nu=1e-3, reference_scale=1.0)`

**Pro-Pattern:**
Apply struct validation upon immediate creation.
```julia
Base.@kwdef struct PhysicalParameters
    viscosity::Float64
    density::Float64
    function PhysicalParameters(nu, rho)
        nu > 0 || throw(ArgumentError("Viscosity must be strictly > 0. Got: $nu."))
        rho > 0 || throw(ArgumentError("Density must be strictly > 0. Got: $rho."))
        new(nu, rho)
    end
end
```

## 2. Using `Enum` or Traits for Model Selection

String-based logic inside integration steps or solver pipelines is prone to silent fallbacks or performance issues.

**Anti-Pattern:**
```julia
if config["model"] == "Darcy"
    # Execute A
elseif config["model"] == "Forchheimer"
    # Execute B
else
    # Automatically defaults A, or drops out natively without warning.
end
```

**Pro-Pattern:** Use enums and explicit mapping.
```julia
@enum FlowModel Darcy Forchheimer Brinkman

function Base.parse(::Type{FlowModel}, s::String)
    if s == "Darcy"; return Darcy
    elseif s == "Forchheimer"; return Forchheimer
    elseif s == "Brinkman"; return Brinkman
    else
        throw(ArgumentError("Unrecognized FlowModel string: '$s'. Permitted choices: Darcy, Forchheimer, Brinkman."))
    end
end
```
Alternatively, multiple dispatch typing:
```julia
abstract type FlowModel end
struct Darcy <: FlowModel end
struct Forchheimer <: FlowModel end
```

## 3. Banishing `coalesce` and `get` out of Mathematical Contexts

**Anti-Pattern:**
```julia
kappa = get(params, "permeability", 1e-8)
```
**Pro-Pattern:**
Demand explicitly present keys during JSON/Dictionary ingestion.
```julia
if !haskey(params, "permeability")
    throw(KeyError("Required physical parameter 'permeability' is entirely missing. Rejecting implicit fallbacks."))
end
kappa = Float64(params["permeability"])
```

## 4. Abolishing Code-Side Silent Regularization

**Anti-Pattern:**
```julia
# Deep inside weak form integral formulations
regularized_porosity = max(cell_porosity, 1e-5) # Completely alters PDE!
```

**Pro-Pattern:**
Construct explicit physical policies at parse-time representing modifications to physical laws. 
```julia
struct RegularizedPorosityPolicy
    eps_floor::Float64
    function RegularizedPorosityPolicy(val)
        val > 0 || error("Regularization floor parameter cannot realistically be zero or negative.")
        new(val)
    end
end
struct ExactPorosityPolicy end

function apply_porosity_model(policy::RegularizedPorosityPolicy, native_val)
    return max(native_val, policy.eps_floor)
end

function apply_porosity_model(policy::ExactPorosityPolicy, native_val)
    native_val <= 0.0 && error("Exact numerical state generated <= 0.0 porosity with regularization officially disabled!")
    return native_val
end
```

## 5. Exposing Solver Tweaks Without Corrupting the Public API

Underlying functions must strictly trace parameter derivation without obscuring variables under default `kwargs`.

**Anti-Pattern:**
```julia
function solve_nonlinear_system!(residual, jacobian, state; tol=1e-8, maxiters=10)
```

**Pro-Pattern:**
Consume a rigid options schema type specifically bound to configuration limits.
```julia
function solve_nonlinear_system!(residual, jacobian, state, options::NonlinearSolverConfig)
    # Strictly map convergence rules globally dictated through options.tol
end
```
