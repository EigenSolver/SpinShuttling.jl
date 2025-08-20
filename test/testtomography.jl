## write some tests for the tomography module
using SpinShuttling
using Test

@testset "test paulitransfermatrix" begin
    #### Single qubit
    # Define Kraus operators for a simple depolarizing channel
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    σI = [1 0; 0 1]
    p=0.5 
    Us = [σI, σx, σy, σz]
    Ps = [1-p, p/3, p/3, p/3]
    KO = KrausOps(MixingUnitaryChannel(Us,Ps))
    
    # Compute the Pauli transfer matrix
    PTM = paulitransfermatrix(KO)
    
    # Check the size of the resulting matrix
    @test size(PTM) == (4, 4)
    
    # # Check if the matrix is Hermitian
    @test ishermitian(PTM)

    ### Two qubit case
    U2s = [kron(a,b) for a in Us for b in Us]
    P2s = zeros(length(U2s))
    P2s[1]=1-p
    P2s[2:end].=p/15
    KO2 = KrausOps(MixingUnitaryChannel(U2s,P2s))

    # Compute the Pauli transfer matrix
    PTM2 = paulitransfermatrix(KO2)
    
    # Check the size of the resulting matrix
    @test size(PTM2) == (16, 16)
    
    # # Check if the matrix is Hermitian
    @test ishermitian(PTM2)
end

