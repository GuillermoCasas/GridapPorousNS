# src/config.jl

using JSON3
using StructTypes

Base.@kwdef struct PhysicalProperties
    nu::Float64 = 1.0
    f_x::Float64 = 0.0
    f_y::Float64 = 0.0
    eps_val::Float64 = 1e-8
    eps_floor::Float64 = 1e-8
    reaction_model::String = "Forchheimer"
    sigma_constant::Float64 = 1.0
    sigma_linear::Float64 = 150.0
    sigma_nonlinear::Float64 = 1.75
    u_base_floor_ref::Float64 = 1e-4
    h_floor_weight::Float64 = 0.1
    epsilon_floor::Float64 = 1e-12
    tau_regularization_limit::Float64 = 1e-12
end

Base.@kwdef struct DomainConfig
    alpha_0::Float64 = 0.4
    r_1::Float64 = 0.2
    r_2::Float64 = 0.5
    bounding_box::Vector{Float64} = [0.0, 1.0, 0.0, 1.0]
end

Base.@kwdef struct ElementSpacesConfig
    k_velocity::Int = 2
    k_pressure::Int = 1
end

Base.@kwdef struct StabilizationConfig
    method::String = "ASGS"
    osgs_iterations::Int = 3
    osgs_tolerance::Float64 = 1e-5
end

Base.@kwdef struct MeshConfig
    partition::Vector{Int} = [20, 20]
    convergence_partitions::Vector{Int} = [10, 20, 30]
    element_type::String = "QUAD"
end

Base.@kwdef struct SolverConfig
    picard_iterations::Int = 5
    newton_iterations::Int = 20
    ftol::Float64 = 1e-10
    xtol::Float64 = 1e-8
    max_increases::Int = 2
    freeze_jacobian_cusp::Bool = false
    armijo_c1::Float64 = 1e-4
    divergence_merit_factor::Float64 = 1.05
    stagnation_noise_floor::Float64 = 1e-2
    linesearch_alpha_min::Float64 = 1e-4
    run_diagnostics::Bool = false
    ablation_mode::String = "full"
    experimental_reaction_mode::String = "standard"
    # Note: stagnation_tol, use_linesearch, linesearch_tolerance were pruned unless requested.
end

Base.@kwdef struct NumericalMethodConfig
    element_spaces::ElementSpacesConfig = ElementSpacesConfig()
    stabilization::StabilizationConfig = StabilizationConfig()
    solver::SolverConfig = SolverConfig()
    mesh::MeshConfig = MeshConfig()
end

Base.@kwdef struct OutputConfig
    directory::String = "results"
    basename::String = "porous_ns"
end

Base.@kwdef struct PorousNSConfig
    physical_properties::PhysicalProperties = PhysicalProperties()
    domain::DomainConfig = DomainConfig()
    numerical_method::NumericalMethodConfig = NumericalMethodConfig()
    output::OutputConfig = OutputConfig()
end

# StructTypes definitions
StructTypes.StructType(::Type{PhysicalProperties}) = StructTypes.Struct()
StructTypes.StructType(::Type{DomainConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{ElementSpacesConfig}) = StructTypes.Struct()
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
    @assert sol.xtol > 0 "Solver xtol must be > 0"
    @assert 0.0 < sol.armijo_c1 < 1.0 "Armijo c1 must be strictly between 0 and 1"
    @assert sol.divergence_merit_factor >= 1.0 "Divergence merit factor must be >= 1.0"
    @assert sol.newton_iterations >= 1 "Newton iterations must be >= 1"
    
    # Stabilization
    stab = cfg.numerical_method.stabilization
    @assert stab.method in ("ASGS", "OSGS") "Stabilization method must be ASGS or OSGS"
    @assert stab.osgs_iterations >= 1
    @assert stab.osgs_tolerance > 0
    
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
