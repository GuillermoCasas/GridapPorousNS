# src/config.jl

using JSON3
using StructTypes

Base.@kwdef struct PhysicalProperties
    nu::Float64
    f_x::Float64
    f_y::Float64
    eps_val::Float64
    eps_floor::Float64
    reaction_model::String
    sigma_constant::Float64
    sigma_linear::Float64
    sigma_nonlinear::Float64
    u_base_floor_ref::Float64
    h_floor_weight::Float64
    epsilon_floor::Float64
    tau_regularization_limit::Float64
end

Base.@kwdef struct DomainConfig
    alpha_0::Float64
    r_1::Float64
    r_2::Float64
    bounding_box::Vector{Float64}
end

Base.@kwdef struct ElementSpacesConfig
    k_velocity::Int
    k_pressure::Int
end

Base.@kwdef struct AcceleratorConfig
    type::String
    m::Int
    relaxation_factor::Float64
end

Base.@kwdef struct StabilizationConfig
    method::String
    osgs_iterations::Int
    osgs_inner_newton_iters::Int
    osgs_tolerance::Float64
    osgs_stopping_mode::String
    osgs_projection_tolerance::Float64
    osgs_state_drift_scale::String
    osgs_warmup_iterations::Int
    osgs_warmup_tolerance::Float64
end

Base.@kwdef struct MeshConfig
    partition::Vector{Int}
    convergence_partitions::Vector{Int}
    element_type::String
end

Base.@kwdef struct SolverConfig
    picard_iterations::Int
    newton_iterations::Int
    ftol::Float64
    dynamic_ftol_ceiling::Float64
    dynamic_ftol_spatial_safety_factor::Float64
    xtol::Float64
    max_increases::Int
    freeze_jacobian_cusp::Bool
    armijo_c1::Float64
    divergence_merit_factor::Float64
    stagnation_noise_floor::Float64
    linesearch_alpha_min::Float64
    max_linesearch_iterations::Int
    linesearch_contraction_factor::Float64
    run_diagnostics::Bool
    ablation_mode::String
    experimental_reaction_mode::String
    accelerator::AcceleratorConfig
end

Base.@kwdef struct NumericalMethodConfig
    element_spaces::ElementSpacesConfig
    stabilization::StabilizationConfig
    solver::SolverConfig
    mesh::MeshConfig
    viscous_operator_type::String
end

Base.@kwdef struct OutputConfig
    directory::String
    basename::String
end

Base.@kwdef struct PorousNSConfig
    physical_properties::PhysicalProperties
    domain::DomainConfig
    numerical_method::NumericalMethodConfig
    output::OutputConfig
end

# StructTypes definitions
StructTypes.StructType(::Type{PhysicalProperties}) = StructTypes.Struct()
StructTypes.StructType(::Type{DomainConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{ElementSpacesConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{AcceleratorConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{StabilizationConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{MeshConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{SolverConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{NumericalMethodConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{OutputConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{PorousNSConfig}) = StructTypes.Struct()

function validate!(cfg::PorousNSConfig)
    # Physical
    @assert cfg.physical_properties.nu > 0 "Kinematic viscosity 'nu' must be > 0"
    @assert cfg.physical_properties.eps_val > 0 "eps_val must be > 0"
    
    # Solver
    sol = cfg.numerical_method.solver
    @assert sol.ftol > 0 "Solver ftol must be > 0"
    @assert sol.dynamic_ftol_ceiling >= sol.ftol "Solver dynamic_ftol_ceiling must be strictly >= base ftol"
    @assert 0.0 < sol.dynamic_ftol_spatial_safety_factor <= 1.0 "Solver dynamic_ftol_spatial_safety_factor must be in (0, 1]"
    @assert sol.xtol > 0 "Solver xtol must be > 0"
    @assert 0.0 < sol.armijo_c1 < 1.0 "Armijo c1 must be strictly between 0 and 1"
    @assert sol.divergence_merit_factor >= 1.0 "Divergence merit factor must be >= 1.0"
    @assert sol.newton_iterations >= 1 "Newton iterations must be >= 1"
    @assert sol.max_linesearch_iterations >= 1 "Linesearch iterations must be strictly bounded >= 1"
    @assert 0.0 < sol.linesearch_contraction_factor < 1.0 "Linesearch contraction map alpha must strictly be in (0, 1)"
    @assert sol.accelerator.m >= 1 "Accelerator history size m must be >= 1"
    @assert 0.0 < sol.accelerator.relaxation_factor <= 1.0 "Accelerator relaxation_factor must be in (0, 1]"
    
    # Stabilization
    stab = cfg.numerical_method.stabilization
    @assert stab.method in ("ASGS", "OSGS") "Stabilization method must be ASGS or OSGS"
    @assert stab.osgs_iterations >= 1
    @assert stab.osgs_tolerance > 0
    @assert stab.osgs_projection_tolerance > 0
    @assert stab.osgs_stopping_mode in ("state_drift", "projection_drift", "both") "osgs_stopping_mode must be 'state_drift', 'projection_drift', or 'both'"
    @assert stab.osgs_state_drift_scale in ("Linf", "L2_mass") "osgs_state_drift_scale must be 'Linf' or 'L2_mass'"
    
    # Formulation Operator validation
    @assert cfg.numerical_method.viscous_operator_type in ("DeviatoricSymmetric", "SymmetricGradient", "Laplacian") "viscous_operator_type strictly expects DeviatoricSymmetric, SymmetricGradient, or Laplacian"
    
    return cfg
end

function _check_unknown_keys(T::Type, dict::AbstractDict, path::String="")
    allowed_keys = string.(fieldnames(T))
    for key in keys(dict)
        if path == "root" && key == "\$schema"
            continue
        end
        if !(key in allowed_keys)
            @warn "Unknown JSON config key '$key' at $path in struct $(T). Please check for typos."
        end
    end
end

function _check_unknown_keys_hierarchical(dict::AbstractDict)
    _check_unknown_keys(PorousNSConfig, dict, "root")
    
    if haskey(dict, "physical_properties") && dict["physical_properties"] isa AbstractDict
        _check_unknown_keys(PhysicalProperties, dict["physical_properties"], "physical_properties")
    end
    if haskey(dict, "domain") && dict["domain"] isa AbstractDict
        _check_unknown_keys(DomainConfig, dict["domain"], "domain")
    end
    if haskey(dict, "numerical_method") && dict["numerical_method"] isa AbstractDict
        nm = dict["numerical_method"]
        _check_unknown_keys(NumericalMethodConfig, nm, "numerical_method")
        
        if haskey(nm, "element_spaces") && nm["element_spaces"] isa AbstractDict
            _check_unknown_keys(ElementSpacesConfig, nm["element_spaces"], "numerical_method.element_spaces")
        end
        if haskey(nm, "stabilization") && nm["stabilization"] isa AbstractDict
            _check_unknown_keys(StabilizationConfig, nm["stabilization"], "numerical_method.stabilization")
        end
        if haskey(nm, "solver") && nm["solver"] isa AbstractDict
            _check_unknown_keys(SolverConfig, nm["solver"], "numerical_method.solver")
            if haskey(nm["solver"], "accelerator") && nm["solver"]["accelerator"] isa AbstractDict
                _check_unknown_keys(AcceleratorConfig, nm["solver"]["accelerator"], "numerical_method.solver.accelerator")
            end
        end
        if haskey(nm, "mesh") && nm["mesh"] isa AbstractDict
            _check_unknown_keys(MeshConfig, nm["mesh"], "numerical_method.mesh")
        end
    end
    if haskey(dict, "output") && dict["output"] isa AbstractDict
        _check_unknown_keys(OutputConfig, dict["output"], "output")
    end
end

function deep_merge!(base::AbstractDict, override::AbstractDict)
    for (k, v) in override
        if haskey(base, k) && isa(base[k], AbstractDict) && isa(v, AbstractDict)
            deep_merge!(base[k], v)
        else
            base[k] = v
        end
    end
    return base
end

function load_config_with_defaults(override_path::String="")
    base_config_path = joinpath(@__DIR__, "..", "base_config.json")
    base_raw = read(base_config_path, String)
    base_dict = copy(JSON3.read(base_raw, Dict{String, Any}))
    
    if !isempty(override_path) && isfile(override_path)
        test_raw = read(override_path, String)
        test_dict = JSON3.read(test_raw, Dict{String, Any})
        deep_merge!(base_dict, test_dict)
    end
    
    _check_unknown_keys_hierarchical(base_dict)
    cfg = JSON3.read(JSON3.write(base_dict), PorousNSConfig)
    return validate!(cfg)
end

function load_frozen_config(exact_path::String)
    if !isfile(exact_path)
        error("Config file not found: $exact_path")
    end
    raw = read(exact_path, String)
    dict = JSON3.read(raw, Dict{String, Any})
    _check_unknown_keys_hierarchical(dict)
    cfg = JSON3.read(raw, PorousNSConfig)
    return validate!(cfg)
end

function load_config_from_dict(override::AbstractDict)
    base_config_path = joinpath(@__DIR__, "..", "base_config.json")
    base_raw = read(base_config_path, String)
    base_dict = copy(JSON3.read(base_raw, Dict{String, Any}))
    
    deep_merge!(base_dict, override)
    _check_unknown_keys_hierarchical(base_dict)
    cfg = JSON3.read(JSON3.write(base_dict), PorousNSConfig)
    return validate!(cfg)
end

# Backward compatibility shim
function load_config(path::String="")
    return load_config_with_defaults(path)
end
