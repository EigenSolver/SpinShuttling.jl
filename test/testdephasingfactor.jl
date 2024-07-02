## 
@testset begin "test dephasing matrix"
    T=400; L=10; σ = sqrt(2) / 20; M = 5000; N=601; κₜ=1/20;κₓ=1/0.1;

    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=OneSpinModel(T,L,N,B)
    # test customize println
    println(model)

    f=averagefidelity(model)
    w=dephasingmatrix(model)

    rho=model.Ψ*model.Ψ'
    @test rho[1,1]+rho[2,2]≈ 1
    @test w==w'

    println(w)
    f_c=(model.Ψ'*(w.*rho)*model.Ψ)

    @test f≈ f_c
end

## 
@testset begin "test two spin dephasing matrix"
    L=10; σ =sqrt(2)/20; M=5000; N=501; T1=200; T0=25*0.05; κₜ=1/20; κₓ=1/0.1;
    B=OrnsteinUhlenbeckField(0,[κₜ,κₓ],σ)
    model=TwoSpinModel(T0, T1, L, N, B)
    println(model)
    f=averagefidelity(model)
    w=dephasingmatrix(model)

    rho=model.Ψ*model.Ψ'
    @test sum([rho[i,i] for i in 1:4 ]) ≈ 1
    @test w==w'

    println(w)
    f_c=(model.Ψ'*(w.*rho)*model.Ψ)

    println(f)
    println(f_c)
    @test f ≈ f_c
end

