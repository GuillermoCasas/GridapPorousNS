content = read("src/formulation.jl", String)
# Perform multiple dispatch refactoring programmatically since multi_replace is too complex for this scale
# ... actually, I can just use write_to_file with the full repaired content!
