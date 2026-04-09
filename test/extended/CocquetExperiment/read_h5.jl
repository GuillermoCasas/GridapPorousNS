using Pkg
Pkg.activate("../../..")
using HDF5
h5open("results/convergence_cocquet.h5", "r") do file
    N_list = read(file["N_list"])
    println("N_list: ", N_list)
    println("--- P1P1 ---")
    println("L2 u: ", read(file["ASGS/P1P1/errors_l2_u"]))
    println("H1 u: ", read(file["ASGS/P1P1/errors_h1_u"]))
    println("--- P2P2 ---")
    println("L2 u: ", read(file["ASGS/P2P2/errors_l2_u"]))
    println("H1 u: ", read(file["ASGS/P2P2/errors_h1_u"]))
end
