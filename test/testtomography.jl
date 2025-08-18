## write some tests for the tomography module
using SpinShuttling
using Test

@testset "test paulitransfermatrix" begin
    # Define Kraus operators for a simple depolarizing channel

    # Compute the Pauli transfer matrix
    PTM = paulitransfermatrix(KrausOps)
    
    # Check the size of the resulting matrix
    # @test size(PTM) == (16, 16)
    
    # # Check if the matrix is Hermitian
    # @test ishermitian(PTM)
end