# src/config.jl

using Parameters
using JSON

@with_kw struct PhysicalProperties
    nu::Float64 = 1.0           # Kinematic viscosity
    f_x::Float64 = 0.0          # Forcing x
    f_y::Float64 = 0.0          # Forcing y
    eps_val::Float64 = 1e-8     # Absolute numerical stabilization limit for pressure
    eps_floor::Float64 = 1e-8   # Minimum limit applied to eps_val
    reaction_model::String = "Forchheimer"
    sigma_constant::Float64 = 1.0
    sigma_linear::Float64 = 150.0   # A
    sigma_nonlinear::Float64 = 1.75 # B
    u_base_floor_ref::Float64 = 1e-4
    h_floor_weight::Float64 = 0.1
    epsilon_floor::Float64 = 1e-12
    tau_regularization_limit::Float64 = 1e-12
end

@with_kw struct DomainConfig
    alpha_0::Float64 = 0.4
    r_1::Float64 = 0.2
    r_2::Float64 = 0.5
    bounding_box::Vector{Float64} = [0.0, 1.0, 0.0, 1.0]
end

@with_kw struct ElementSpacesConfig
    k_velocity::Int = 2
    k_pressure::Int = 1
end

@with_kw struct StabilizationConfig
    method::String = "ASGS"
    osgs_iterations::Int = 3
    osgs_tolerance::Float64 = 1e-5
end

@with_kw struct MeshConfig
    partition::Vector{Int} = [20, 20]
    convergence_partitions::Vector{Int} = [10, 20, 30]
    element_type::String = "QUAD"
end

@with_kw struct SolverConfig
    picard_iterations::Int = 5
    newton_iterations::Int = 20
    use_linesearch::Bool = true
    xtol::Float64 = 1e-8
    stagnation_tol::Float64 = 1e-5
    ftol::Float64 = 1e-10
    max_increases::Int = 2
    freeze_jacobian_cusp::Bool = false
    linesearch_tolerance::Float64 = 1.0001
    linesearch_alpha_min::Float64 = 1e-4
    armijo_c1::Float64 = 1e-4
    divergence_merit_factor::Float64 = 1.05
    stagnation_noise_floor::Float64 = 1e-2
    run_diagnostics::Bool = false
    ablation_mode::String = "full"
    experimental_reaction_mode::String = "standard"
end

@with_kw struct NumericalMethodConfig
    element_spaces::ElementSpacesConfig = ElementSpacesConfig()
    stabilization::StabilizationConfig = StabilizationConfig()
    solver::SolverConfig = SolverConfig()
    mesh::MeshConfig = MeshConfig()
end

@with_kw struct OutputConfig
    directory::String = "results"
    basename::String = "porous_ns"
end

@with_kw struct PorousNSConfig
    physical_properties::PhysicalProperties = PhysicalProperties()
    domain::DomainConfig = DomainConfig()
    numerical_method::NumericalMethodConfig = NumericalMethodConfig()
    output::OutputConfig = OutputConfig()
    
    # Keeping compatibility interfaces for internal tests if needed, but the primary API reflects the hierarchy
    phys::PhysicalProperties = physical_properties
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

function load_config(test_config_path::String="")
    base_config_path = joinpath(@__DIR__, "..", "base_config.json")
    base_dict = JSON.parsefile(base_config_path)
    
    if !isempty(test_config_path) && isfile(test_config_path)
        test_dict = JSON.parsefile(test_config_path)
        deep_merge!(base_dict, test_dict)
    end
    
    return _parse_dict_to_config(base_dict)
end

function load_config_from_dict(override::AbstractDict)
    base_config_path = joinpath(@__DIR__, "..", "base_config.json")
    base_dict = JSON.parsefile(base_config_path)
    deep_merge!(base_dict, override)
    return _parse_dict_to_config(base_dict)
end

function _parse_dict_to_config(dict::AbstractDict)
    local_phys = PhysicalProperties()
    if haskey(dict, "physical_properties")
        local_phys = PhysicalProperties(; (Symbol(k) => v for (k,v) in dict["physical_properties"] if Symbol(k) in fieldnames(PhysicalProperties))...)
    elseif haskey(dict, "physical_parameters") # backward compatibility
        local_phys = PhysicalProperties(; (Symbol(k) => v for (k,v) in dict["physical_parameters"] if Symbol(k) in fieldnames(PhysicalProperties))...)
    end
    
    local_domain = DomainConfig()
    if haskey(dict, "domain")
        local_domain = DomainConfig(; (Symbol(k) => v for (k,v) in dict["domain"] if Symbol(k) in fieldnames(DomainConfig))...)
    end
    
    local_elem = ElementSpacesConfig()
    local_stab = StabilizationConfig()
    local_solv = SolverConfig()
    local_mesh = MeshConfig()
    
    if haskey(dict, "numerical_method")
        nm_dict = dict["numerical_method"]
        if haskey(nm_dict, "element_spaces")
            local_elem = ElementSpacesConfig(; (Symbol(k) => v for (k,v) in nm_dict["element_spaces"] if Symbol(k) in fieldnames(ElementSpacesConfig))...)
        end
        if haskey(nm_dict, "stabilization")
            local_stab = StabilizationConfig(; (Symbol(k) => v for (k,v) in nm_dict["stabilization"] if Symbol(k) in fieldnames(StabilizationConfig))...)
        end
        if haskey(nm_dict, "solver")
            local_solv = SolverConfig(; (Symbol(k) => v for (k,v) in nm_dict["solver"] if Symbol(k) in fieldnames(SolverConfig))...)
        end
        if haskey(nm_dict, "mesh")
            local_mesh = MeshConfig(; (Symbol(k) => v for (k,v) in nm_dict["mesh"] if Symbol(k) in fieldnames(MeshConfig))...)
        end
    else
        # Backward compatibility fallback
        if haskey(dict, "discretization")
            local_elem = ElementSpacesConfig(; (Symbol(k) => v for (k,v) in dict["discretization"] if Symbol(k) in fieldnames(ElementSpacesConfig))...)
        end
        if haskey(dict, "mesh")
            local_mesh = MeshConfig(; (Symbol(k) => v for (k,v) in dict["mesh"] if Symbol(k) in fieldnames(MeshConfig))...)
        end
        if haskey(dict, "solver")
            local_solv = SolverConfig(; (Symbol(k) => v for (k,v) in dict["solver"] if Symbol(k) in fieldnames(SolverConfig))...)
            local_stab = StabilizationConfig(; (Symbol(k) => v for (k,v) in dict["solver"] if Symbol(k) in fieldnames(StabilizationConfig))...)
        end
    end
    
    local_nm = NumericalMethodConfig(element_spaces=local_elem, stabilization=local_stab, solver=local_solv, mesh=local_mesh)
    
    local_out = OutputConfig()
    if haskey(dict, "output")
        local_out = OutputConfig(; (Symbol(k) => v for (k,v) in dict["output"] if Symbol(k) in fieldnames(OutputConfig))...)
    end
    
    return PorousNSConfig(physical_properties=local_phys, phys=local_phys, domain=local_domain, numerical_method=local_nm, output=local_out)
end
