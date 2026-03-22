## write some tests for the tomography module
using SpinShuttling
using Test

@testset "test fidelities" begin
    #### Single qubit
    # Define Kraus operators for a simple depolarizing channel
    σx = [0 1; 1 0]
    σy = [0 -1im; 1im 0]
    σz = [1 0; 0 -1]
    σi = [1 0; 0 1]
    for p in [0.02, 0.1, 0.5]
        Us = [σi, σx, σy, σz]
        Ps = [1-p, p/3, p/3, p/3]

        E=MixingUnitaryChannel(Us,Ps)


        PTM1 = paulitransfermatrix(krausops(E); normalized=true)
        PTM2= paulitransfermatrix(σi; normalized=true)
        println("PTM1: ", PTM1)
        println("PTM2: ", PTM2)


        f_ent = processfidelity(PTM1, PTM2)

        @assert f_ent == processfidelity(E, σi)

        f_avg = averagegatefidelity(E, σi)

        KO = krausops(E)
        @assert sum([e'*e for e in KO]) ≈ σi

        println("f_ent: ", f_ent, " f_avg: ", f_avg)
        
        let d=2
            @assert f_avg ≈ (d * f_ent + 1) / (d + 1)
        end
    end
end

