using Pkg
Pkg.activate("../../..")
using PorousNSSolver

deg1 = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, 1)
deg2 = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, 2)
println("deg1 = $deg1, deg2 = $deg2")
