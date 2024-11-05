@testset "unitful linear algebra" begin

    @testset "trend analysis: left-uniform matrix" begin

        M = 10  # number of obs
        t = collect(0:1:M-1)s

        a = randn()m # intercept
        b = randn()m/s # slope

        #y = UnitfulMatrix(a .+ b.*t .+ randn(M)m)
        y = a .+ b.*t .+ randn(M)m

        E = UnitfulMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Py⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M))
        Py = Diagonal(fill(1.0,M),fill(m,M),fill(m^-1,M))

        y1 = Estimate(y,Py)
        x̃0 = E \ y1 # no information to combine, simply translate
        
        problem = OverdeterminedProblem(y,E,Py⁻¹)
        x̃ = solve(problem,alg=:textbook)
        x̃1 = solve(problem,alg=:hessian)
        @test cost(x̃0,problem) < 3M # rough guide, could get unlucky and not pass
        @test cost(x̃,problem) < 3M # rough guide, could get unlucky and not pass
        @test cost(x̃1,problem) < 3M # rough guide, could get unlucky and not pass
    end

    @testset "left-uniform problem with prior info" begin
	#y = UnitfulMatrix([-1.9permil])
	y = -1.9permil
	σy = 0.2permil
	a = -0.24permil*K^-1
	γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
	γT = 1.0K^-2
        E = UnitfulMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
        #x₀ = UnitfulMatrix([-1.0permil, 4.0K])
        x₀ = [-1.0permil, 4.0K]

        # "overdetermined, problem 1.4" 
        Px⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K])
	Py⁻¹  = Diagonal([σy.^-2],[permil^-1],[permil])

	Py  = Diagonal([σy.^2],[permil],[permil^-1])
        Px = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1])

        # Px = inv(Px⁻¹)
        # Py = inv(Py⁻¹)

        y1 = Estimate([y],Py)
        x0 = Estimate(x₀,Px)

        x1 = combine(x0,y1,E)
        
        oproblem = OverdeterminedProblem([y],E,Py⁻¹,Px⁻¹,x₀)
        xo = solve(oproblem)
        @test cost(xo,oproblem) < 1 # rough guide, coul
        @test cost(x1,oproblem) < 1 # rough guide, coul

        # "underdetermined, problem 2.1" 
        uproblem = UnderdeterminedProblem([y],E,Py,Px,x₀)
        xu = solve(uproblem)
        @test cost(xu,uproblem) < 1 # rough guide, could ge

        # same answer all three ways?
        @test cost(xu,uproblem) ≈ cost(xo,oproblem)
        @test cost(x1,uproblem) ≈ cost(xo,oproblem)
    end

    @testset "objective mapping, problem 4.3" begin
        M = 11
        if use_units
            yr = u"yr"; cm = u"cm"
            τ = range(0.0yr,5.0yr,step=0.1yr)
            ρ = exp.(-τ.^2/(1yr)^2)
            n = length(ρ)
            Px = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) +
                Diagonal(fill(1e-6,n),fill(cm,n),fill(cm^-1,n))
            σy = 0.1cm
            Py = Diagonal(fill(ustrip(σy),M),fill(cm,M),fill(cm^-1,M))
            Enm = sparse(1:M,1:5:n,fill(1.0,M))
            E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n))
            x₀ = zeros(n)cm
        else
            τ = range(0.0,5.0,step=0.1)
            ρ = exp.(-τ.^2/(1)^2)
            n = length(ρ)
            Px = SymmetricToeplitz(ρ) + Diagonal(fill(1e-6,n))
            σy = 0.1
            Py = Diagonal(fill(σy,M))
            E = sparse(1:M,1:5:n,fill(1.0,M))
            x₀ = zeros(n)
        end
    
        Px¹² = cholesky(Px)
        x = Px¹².L*randn(n)
        y = E*x
        x0 = Estimate(x₀, Px)
        y_estimate = Estimate(y, Py)
        x1 = combine(x0, y_estimate, E) # a BLUE
    
        uproblem = UnderdeterminedProblem(y,E,Py,Px,x₀)
        x̃ = solve(uproblem)
        @test cost(x̃,uproblem) < 5M 

        if use_units
            @test isapprox(ustrip.(x1.v), ustrip.(x̃.v))
            @test isapprox(ustrip.(x1.σ), ustrip.(x̃.σ))
        else
            @test isapprox(x1.v, x̃.v)
            @test isapprox(x1.σ, x̃.σ)
        end
    end 
end
