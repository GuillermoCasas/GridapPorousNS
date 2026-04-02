# src/io.jl

function export_results(config::PorousNSConfig, model, u_h, p_h, extra_fields::Pair...)
    outdir = config.output.directory
    if !isdir(outdir)
        mkdir(outdir)
    end
    
    base_name = joinpath(outdir, config.output.basename)
    if !endswith(base_name, ".vtu")
        base_name = base_name * ".vtu"
    end
    
    Ω = Triangulation(model)
    fields = Pair{String, Any}["u"=>u_h, "p"=>p_h]
    for f in extra_fields
        push!(fields, f)
    end
    
    writevtk(Ω, base_name, cellfields=fields)
    
    println("Results exported to: ", base_name)
end
