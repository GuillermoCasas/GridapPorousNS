using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using JSON
using Gridap

println("\n=======================================================")
println("--- Test 1: High-Re Linesearch & Cusp-Freezing ---")
println("=======================================================")
c1 = Dict(
    "physical_parameters" => Dict("Re" => 1e6, "Da" => 1.0, "f_x" => 1.0),
    "porosity_field" => Dict("alpha_0" => 0.5),
    "mesh" => Dict("element_type" => "QUAD", "partition" => [10, 10], "domain" => [0.0, 1.0, 0.0, 1.0]),
    "discretization" => Dict("k_velocity" => 2, "k_pressure" => 2),
    "solver" => Dict("method" => "ASGS", "xtol" => 1e-8, "stagnation_tol" => 1e-5, "ftol" => 1e-10, "newton_iterations" => 20)
)
mkpath("test/long/ManufacturedSolutions/data")
write("test/long/ManufacturedSolutions/data/test1.json", JSON.json(c1))
run(`julia --project=. test/long/ManufacturedSolutions/run_test.jl test1.json`)

println("\n=======================================================")
println("--- Test 2: Error Throw Test ---")
println("=======================================================")
c2 = copy(c1)
c2["solver"]["newton_iterations"] = 2
c2["solver"]["max_increases"] = 2
write("test/long/ManufacturedSolutions/data/test2.json", JSON.json(c2))
try
    run(`julia --project=. test/long/ManufacturedSolutions/run_test.jl test2.json`)
catch
    println("SUCCESS: Test 2 Caught exception!")
end

println("\n=======================================================")
println("--- Test 3: Pristine Convergence Test ---")
println("=======================================================")
c3 = Dict(
    "physical_parameters" => Dict("Re" => 1e-6, "Da" => 1e-6, "f_x" => 1.0),
    "porosity_field" => Dict("alpha_0" => 1.0),
    "mesh" => Dict("element_type" => "QUAD", "convergence_partitions" => [5, 10], "domain" => [0.0, 1.0, 0.0, 1.0]),
    "discretization" => Dict("k_velocity" => 2, "k_pressure" => 2),
    "solver" => Dict("method" => "ASGS", "xtol" => 1e-12, "stagnation_tol" => 1e-5, "ftol" => 1e-10)
)
write("test/long/ManufacturedSolutions/data/test3.json", JSON.json(c3))
run(`julia --project=. test/long/ManufacturedSolutions/run_test.jl test3.json`)

# Cleanup
rm("test/long/ManufacturedSolutions/data/test1.json", force=true)
rm("test/long/ManufacturedSolutions/data/test2.json", force=true)
rm("test/long/ManufacturedSolutions/data/test3.json", force=true)
