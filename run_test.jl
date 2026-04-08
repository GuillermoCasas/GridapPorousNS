# run_test.jl
# Top-level wrapper to run the Manufactured Solutions test.
# We change the directory to ensure it reads its local config data properly.

println("Redirecting to test/extended/ManufacturedSolutions/run_test.jl...")

# Execute the test script in its own directory
cd(joinpath(@__DIR__, "test", "extended", "ManufacturedSolutions")) do
    cmd = `$(Base.julia_cmd()) --project=../../.. run_test.jl $(ARGS)`
    run(cmd)
end
