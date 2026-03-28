# src/config.jl

@with_kw struct PhysicalParameters
    Re::Float64 = 1.0
    Da::Float64 = 1.0
    physical_epsilon::Float64 = 0.0
    numerical_epsilon_coefficient::Float64 = 1e-4
    f_x::Float64 = 0.0
    f_y::Float64 = 0.0
end

@with_kw struct PorosityField
    alpha_0::Float64 = 0.4
    r_1::Float64 = 0.2
    r_2::Float64 = 0.5
end

@with_kw struct DiscretizationConfig
    k_velocity::Int = 2
    k_pressure::Int = 1
end

@with_kw struct MeshConfig
    domain::Vector{Float64} = [0.0, 1.0, 0.0, 1.0]
    partition::Vector{Int} = [20, 20]
    convergence_partitions::Vector{Int} = [10, 20, 30]
    element_type::String = "QUAD"
end

@with_kw struct OutputConfig
    directory::String = "results"
    basename::String = "porous_ns"
end

@with_kw struct SolverConfig
    picard_iterations::Int = 5
    newton_iterations::Int = 20
    use_linesearch::Bool = true
end

@with_kw struct PorousNSConfig
    phys::PhysicalParameters = PhysicalParameters()
    porosity::PorosityField = PorosityField()
    discretization::DiscretizationConfig = DiscretizationConfig()
    mesh::MeshConfig = MeshConfig()
    output::OutputConfig = OutputConfig()
    solver::SolverConfig = SolverConfig()
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
    local_phys = PhysicalParameters(; (Symbol(k) => v for (k,v) in dict["physical_parameters"])...)
    local_poro = PorosityField(; (Symbol(k) => v for (k,v) in dict["porosity_field"])...)
    local_disc = DiscretizationConfig(; (Symbol(k) => v for (k,v) in dict["discretization"])...)
    local_mesh = MeshConfig(; (Symbol(k) => v for (k,v) in dict["mesh"])...)
    local_out = OutputConfig(; (Symbol(k) => v for (k,v) in dict["output"])...)
    
    local_solver = SolverConfig()
    if haskey(dict, "solver")
        local_solver = SolverConfig(; (Symbol(k) => v for (k,v) in dict["solver"])...)
    end
    
    return PorousNSConfig(phys=local_phys, porosity=local_poro, discretization=local_disc, mesh=local_mesh, output=local_out, solver=local_solver)
end
