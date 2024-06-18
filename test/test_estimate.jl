@testset "error propagation" begin
    using Measurements

    M = 5
    if use_units
        a = randn(M)u"K" .± rand(M)u"K"
    else
        a = randn(M) .± rand(M)
    end

    E = randn(M,M)

    aerr = Measurements.uncertainty.(a);
    x = Estimate(Measurements.value.(a),
        Diagonal(aerr.^2))

    @test Measurements.value.(E*a) ≈ (E*x).v
    @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

    # combine two estimates
    xplus = combine(x,x,alg=:underdetermined)
    # error should decrease by 70%
    @test sum( xplus.σ./x.σ .< 0.8) == M
    # central estimate should not change
    @test isapprox( xplus.v, x.v )

        # combine two estimates
    xplus = combine(x,x,alg=:overdetermined)
    # error should decrease by 70%
    @test sum( xplus.σ./x.σ .< 0.8) == M
    # central estimate should not change
    @test isapprox( xplus.v, x.v )
    
end

@testset "mixed signals: dimensionless matrix" begin
    N = 2
    for M in 2:4

        if use_units
            σₓ = rand()
            # exact = false to work
            E = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E))
            x = UnitfulMatrix(randn(N)m)
        else
            σₓ = rand()
            E = randn(M,N)
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M))
            x = randn(N)
        end
        
        y = E*x
        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        x̃ = solve(problem,alg=:textbook)
        @test x ≈ x̃.v
        @test cost(x̃,problem) < 1e-5 # no noise in obs
        @test x ≈ pinv(problem) * y # inefficient way to solve problem
    end
end

@testset "trend analysis: left-uniform matrix" begin

    M = 10  # number of obs
    if use_units
        t = collect(0:1:M-1)s
        a = randn()m # intercept
        b = randn()m/s # slope
        y = UnitfulMatrix(a .+ b.*t .+ randn(M)m)
        E = UnitfulMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M))
    else
        t = collect(0:1:M-1)
        a = randn() # intercept
        b = randn() # slope
        y = a .+ b.*t .+ randn(M)
        E = hcat(ones(M),t)
        Cnn⁻¹ = Diagonal(fill(1.0,M))
    end
        
    problem = OverdeterminedProblem(y,E,Cnn⁻¹)
    x̃ = solve(problem,alg=:textbook)
    x̃1 = solve(problem,alg=:hessian)
    @test cost(x̃,problem) < 5M # rough guide, could get unlucky and not pass
    
end

@testset "left-uniform problem with prior info" begin
    if use_units
	y = UnitfulMatrix([-1.9permil])
	σₙ = 0.2permil
	a = -0.24permil*K^-1
	γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
	γT = 1.0K^-2
        E = UnitfulMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
        x₀ = UnitfulMatrix([-1.0permil, 4.0K])

        # "overdetermined, problem 1.4" 
        Cxx⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K])
	Cnn⁻¹  = Diagonal([σₙ.^-2],[permil^-1],[permil])

        # "underdetermined, problem 2.1" 
	Cnn  = Diagonal([σₙ.^2],[permil],[permil^-1])
        Cxx = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1])

    else
    	y = [-1.9]
	σₙ = 0.2
	a = -0.24
	γδ = 1.0 # to keep units correct, have two γ (tapering) variables
	γT = 1.0
        E = [1 a] 
        x₀ = [-1.0, 4.0]
        
        # "overdetermined, problem 1.4" 
        Cxx⁻¹ = Diagonal([γδ,γT])
	Cnn⁻¹  = Diagonal([σₙ.^-2])
            
        # "underdetermined, problem 2.1" 
	Cnn  = Diagonal([σₙ.^2])
        Cxx = Diagonal(ustrip.([γδ,γT].^-1))
    end

    oproblem = OverdeterminedProblem(y,E,Cnn⁻¹,Cxx⁻¹,x₀)
    x̃1 = solve(oproblem)
    @test cost(x̃1,oproblem) < 1 # rough guide, coul

    uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)
    x̃2 = solve(uproblem)
    @test cost(x̃2,uproblem) < 1 # rough guide, could ge

    # same answer both ways?
    @test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)
end

@testset "polynomial fitting, problem 2.3" begin

    M = 50
    N = 4 
    if use_units
        g = u"g"
        kg = u"kg"
        d = u"d"
        t = (1:M)d
        Eparent = ustrip.(hcat(t.^0, t, t.^2, t.^3))
        E = UnitfulMatrix(Eparent,fill(g/kg,M),[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)

        σₙ = 0.1g/kg
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M),fill(g/kg,M),fill(kg/g,M)) 
        Cₙₙ¹² = cholesky(Cₙₙ)
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M),fill(kg/g,M),fill(g/kg,M)) 
        Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)

        γ = [1.0e1kg^2/g^2, 1.0e2kg^2*d^2/g^2, 1.0e3kg^2*d^4/g^2, 1.0e4kg^2*d^6/g^2]

        Cxx⁻¹ = Diagonal(ustrip.(γ),[kg/g,kg*d/g,kg*d^2/g,kg*d^3/g],[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3])
        #x₀ = UnitfulMatrix(zeros(N).*unitdomain(Cxx⁻¹))
        x₀ = zeros(N).*unitdomain(Cxx⁻¹)
    else
        t = (1:M)
        E =hcat(t.^0, t, t.^2, t.^3)
        σₙ = 0.1
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M))
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M))
        γ = [1.0e1, 1.0e2, 1.0e3, 1.0e4]
        Cxx⁻¹ = Diagonal(γ)
        x₀ = zeros(N)
    end
    Cₙₙ¹² = cholesky(Cₙₙ)
    Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)
    Cxx = inv(Cxx⁻¹)
    Cxx¹² = cholesky(Cxx)
    N = size(Cxx⁻¹,1)
    x = Cxx¹².L*randn(N)
    y = E*x
    oproblem = OverdeterminedProblem(y,E,Cₙₙ⁻¹,Cxx⁻¹,x₀)

    # not perfect data fit b.c. of prior info
    x̃ = solve(oproblem,alg=:hessian)
    @test cost(x̃,oproblem) < 5M
    x̃ = solve(oproblem,alg=:textbook)
    @test cost(x̃,oproblem) < 5M
end

@testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

end

@testset "objective mapping, problem 4.3" begin

    M = 11
    if use_units
        yr = u"yr"; cm = u"cm"
        τ = range(0.0yr,5.0yr,step=0.1yr)
        ρ = exp.(-τ.^2/(1yr)^2)
        n = length(ρ)
        Cxx = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) + Diagonal(fill(1e-6,n),fill(cm,n),fill(cm^-1,n))
        σₙ = 0.1cm
        Cnn = Diagonal(fill(ustrip(σₙ),M),fill(cm,M),fill(cm^-1,M))
        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n))
        x₀ = zeros(n)cm
    else
        τ = range(0.0,5.0,step=0.1)
        ρ = exp.(-τ.^2/(1)^2)
        n = length(ρ)
        Cxx = SymmetricToeplitz(ρ) + Diagonal(fill(1e-6,n))
        σₙ = 0.1
        Cnn = Diagonal(fill(σₙ,M))
        E = sparse(1:M,1:5:n,fill(1.0,M))
        x₀ = zeros(n)
    end
    
    Cxx¹² = cholesky(Cxx)
    x = Cxx¹².L*randn(n)
    y = E*x
    uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)

    x̃ = solve(uproblem)
    @test cost(x̃,uproblem) < 5M 

end

# additional problem: 5.1 model of exponential decay
@testset "overdetermined named tuple E,y" begin
    N = 2
    M = 1
    σₓ = rand()
    if use_units

        # exact = false to work
        E1 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N))
        E2 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N))
        E = (one=E1,two=E2)
        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M),unitrange(E1).^-1,unitrange(E1))
        x = randn(N)m
    else
        E1 = randn(M,N)
        E2 = randn(M,N)
        E = (one=E1,two=E2)
        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M))
        x = randn(N)
    end

    Cnn⁻¹ = (one=Cnn⁻¹1, two =Cnn⁻¹1)
    # create perfect data
    y = E*x
    problem = OverdeterminedProblem(y,E,Cnn⁻¹)
    x̃ = solve(problem,alg=:hessian)
    @test x ≈ x̃.v # no noise in obs
    @test cost(x̃,problem) < 1e-5 # no noise in obs
end
