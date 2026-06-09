# src/io.jl
#
# Output stage of the pipeline. After `solve_system` produces the discrete
# solution, this module serializes it to disk for post-processing/visualization.
# It is the final step invoked by `run_simulation` (see src/run_simulation.jl).

"""
    export_results(config, model, u_h, p_h, extra_fields...)

Write the computed velocity and pressure fields to a VTK file for visualization
(e.g. in ParaView).

Arguments:
- `config`  — the frozen run configuration; `config.output.directory` and
  `config.output.basename` decide where and under what name the file is written.
- `model`   — the `DiscreteModel` the problem was solved on; its `Triangulation`
  supplies the mesh geometry the fields are sampled over.
- `u_h`     — the discrete velocity FE function (vector field), labeled `"u"`.
- `p_h`     — the discrete pressure FE function (scalar field), labeled `"p"`.
- `extra_fields...` — optional `name => field` pairs for diagnostics one may want
  alongside the solution (e.g. an exact MMS field or a stabilization quantity);
  each is appended as its own named cell field.

Mechanism: the output directory is created if absent, a `.vtu` extension is
enforced on the basename, and Gridap's `writevtk` dumps each named field as a
cell field on the mesh triangulation `Ω`.
"""
function export_results(config::PorousNSConfig, model, u_h, p_h, extra_fields::Pair...)
    outdir = config.output.directory
    if !isdir(outdir)
        mkdir(outdir)
    end

    # Ensure the output path carries the .vtu extension Gridap/ParaView expects.
    base_name = joinpath(outdir, config.output.basename)
    if !endswith(base_name, ".vtu")
        base_name = base_name * ".vtu"
    end

    # Gather the fields to serialize: the solution (u, p) plus any caller-supplied
    # diagnostics, each tagged with the name it will appear under in the VTK file.
    Ω = Triangulation(model)
    fields = Pair{String, Any}["u"=>u_h, "p"=>p_h]
    for f in extra_fields
        push!(fields, f)
    end

    writevtk(Ω, base_name, cellfields=fields)

    println("Results exported to: ", base_name)
end
