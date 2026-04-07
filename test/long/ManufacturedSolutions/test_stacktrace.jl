open("run_test.jl") do f
    lines = readlines(f)
    for (i, line) in enumerate(lines)
        if occursin("println(e)", line)
            lines[i] = "                            println(e, \"\\n\", stacktrace(catch_backtrace()))"
        end
    end
    write("run_test2.jl", join(lines, "\n"))
end
