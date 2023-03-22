#using Revise
using BLUEs
using Test
using LinearAlgebra
using Statistics
using Unitful
using UnitfulLinearAlgebra
using Measurements
using ToeplitzMatrices
using SparseArrays
using DimensionalData
using DimensionalData:@dim
const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";
ENV["UNITFUL_FANCY_EXPONENTS"] = true

"""
    Are two matrices within a certain tolerance?
    Use to simplify tests.
    """
within(A,B,tol) =  maximum(abs.(ustrip.(A - B))) < tol
within(A::UnitfulLinearAlgebra.AbstractUnitfulType,B::UnitfulLinearAlgebra.AbstractUnitfulType,tol) =  maximum(abs.(parent(A - B))) < tol

"""
    function random_source_water_matrix_vector_pair(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
function random_source_water_matrix_vector_pair(M)

    #using DimensionalData
    #using DimensionalData: @dim
    #cm = u"cm"
    yr = u"yr"
    K = u"K"
    percent = u"percent"
    surfaceregions = [:NATL,:ANT,:SUBANT]
    interiorlocs = [Symbol("loc"*string(nloc)) for nloc = 1:M]
    N = length(surfaceregions)
    
    # observations have units of temperature
    urange = fill(K,M)
    # solution also has units of temperature
    udomain = fill(K,N)
    Eparent = rand(M,N)#*100percent

    # normalize to conserve mass
    for nrow = 1:size(Eparent,1)
        Eparent[nrow,:] ./= sum(Eparent[nrow,:])
    end
    
    E = UnitfulDimMatrix(ustrip.(Eparent),urange,udomain,dims=(InteriorLocation(interiorlocs),SurfaceRegion(surfaceregions)))

    x = UnitfulDimMatrix(randn(N),fill(K,N),dims=(SurfaceRegion(surfaceregions)))
    return E,x
end

"""
   Take the convolution of E and x

    Account for proper overlap of dimensions
    Sum and take into account units.
"""
function convolve(E::AbstractDimArray,x::AbstractDimArray)
    tnow = last(first(dims(x)))
    lags = first(dims(E))
    return sum([E[ii,:] ⋅ x[Near(tnow-ll),:] for (ii,ll) in enumerate(lags)])
end

@testset "BLUEs.jl" begin

    @testset "error propagation" begin

        a = randn(5) .± rand(5)
        E = randn(5,5)

        aerr = Measurements.uncertainty.(a);
        #x = Estimate(Measurements.value.(a),
        #             aerr*transpose(aerr));

        x = Estimate(Measurements.value.(a),
                     Diagonal(aerr.^2))

        @test Measurements.value.(E*a) ≈ (E*x).v
        @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

    end

    @testset "mixed signals: dimensionless matrix" begin
        N = 2
        for M in 2:4
            σₓ = rand()
            # exact = false to work
            E = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
            # NOTE: unitrange, unitdomain have different units. requires kludge below.
            Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,exact=true)
            x = UnitfulMatrix(randn(N)m)
            y = E*x

            problem = OverdeterminedProblem(y,E,Cnn⁻¹)
            #x̃ = solve(y,E,Cnn⁻¹)
            x̃ = solve(problem,alg=:hessian)
            x̃ = solve(problem,alg=:textbook)
            @test x ≈ x̃.v
            @test cost(x̃,problem) < 1e-5 # no noise in obs

            @test x ≈ pinv(problem) * y # inefficient way to solve problem
        end
    end

    @testset "trend analysis: left-uniform matrix" begin

        M = 10  # number of obs
        t = collect(0:1:M-1)s

        a = randn()m # intercept
        b = randn()m/s # slope

        y = UnitfulMatrix(a .+ b.*t .+ randn(M)m)

        E = UnitfulMatrix(hcat(ones(M),ustrip.(t)),fill(m,M),[m,m/s],exact=true)
        Cnn⁻¹ = Diagonal(fill(1.0,M),fill(m^-1,M),fill(m,M),exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:textbook)
        x̃1 = solve(problem,alg=:hessian)
        @test cost(x̃,problem) < 3M # rough guide, could get unlucky and not pass

    end

    @testset "left-uniform problem with prior info" begin
	y = UnitfulMatrix([-1.9permil])
	σₙ = 0.2permil
	a = -0.24permil*K^-1
	γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
	γT = 1.0K^-2
        E = UnitfulMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
        x₀ = UnitfulMatrix([-1.0permil, 4.0K])

        # "overdetermined, problem 1.4" 
        Cxx⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K],exact=true)
	Cnn⁻¹  = Diagonal([σₙ.^-2],[permil^-1],[permil])
        oproblem = OverdeterminedProblem(y,E,Cnn⁻¹,Cxx⁻¹,x₀)

        # "underdetermined, problem 2.1" 
	Cnn  = Diagonal([σₙ.^2],[permil],[permil^-1])
        Cxx = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1],exact=true)
        uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)

        x̃1 = solve(oproblem)
        @test cost(x̃1,oproblem) < 1 # rough guide, coul

        x̃2 = solve(uproblem)
        @test cost(x̃2,uproblem) < 1 # rough guide, could ge
        # same answer both ways?
        @test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)
    end

    @testset "polynomial fitting, problem 2.3" begin
        g = u"g"
        kg = u"kg"
        d = u"d"
	M = 50
        t = (1:M)d
        Eparent = ustrip.(hcat(t.^0, t, t.^2, t.^3))
        E = UnitfulMatrix(Eparent,fill(g/kg,M),[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)

        σₙ = 0.1g/kg
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M),fill(g/kg,M),fill(kg/g,M),exact=true) 
        Cₙₙ¹² = cholesky(Cₙₙ)
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M),fill(kg/g,M),fill(g/kg,M),exact=true) 
        Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)

	γ = [1.0e1kg^2/g^2, 1.0e2kg^2*d^2/g^2, 1.0e3kg^2*d^4/g^2, 1.0e4kg^2*d^6/g^2]

	Cxx⁻¹ = Diagonal(ustrip.(γ),[kg/g,kg*d/g,kg*d^2/g,kg*d^3/g],[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)
        Cxx = inv(Cxx⁻¹)
        Cxx¹² = cholesky(Cxx)

        N = size(Cxx⁻¹,1)
        x₀ = UnitfulMatrix(zeros(N).*unitdomain(Cxx⁻¹))
        x = Cxx¹².L*randn(N)
        y = E*x
        oproblem = OverdeterminedProblem(y,E,Cₙₙ⁻¹,Cxx⁻¹,x₀)

        # not perfect data fit b.c. of prior info
        x̃ = solve(oproblem,alg=:hessian)
        x̃ = solve(oproblem,alg=:textbook)
        @test cost(x̃,oproblem) < 3M
    end

    @testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

    end

    @testset "objective mapping, problem 4.3" begin
        
        yr = u"yr"; cm = u"cm"
        τ = range(0.0yr,5.0yr,step=0.1yr)
        ρ = exp.(-τ.^2/(1yr)^2)
        n = length(ρ)
        Cxx = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) + Diagonal(fill(1e-6,n),   fill(cm,n),fill(cm^-1,n),exact=true)

        M = 11
        σₙ = 0.1cm
        Cnn = Diagonal(fill(ustrip(σₙ),M),fill(cm,M),fill(cm^-1,M),exact=true)

        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n),exact=true)

        Cxx¹² = cholesky(Cxx)
        x₀ = UnitfulMatrix(zeros(n)cm)
        x = Cxx¹².L*randn(n)
        y = E*x

        uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,x₀)

        x̃ = solve(uproblem)
        @test cost(x̃,uproblem) < 3M 

    end

    # additional problem: 5.1 model of exponential decay 
    @testset "overdetermined named tuple E,y" begin
        N = 2
        M = 1
        σₓ = rand()
        # exact = false to work
        E1 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E2 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E = (one=E1,two=E2)

        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M),unitrange(E1).^-1,unitrange(E1).^1,exact=true)

        Cnn⁻¹ = (one=Cnn⁻¹1, two =Cnn⁻¹1)
        x = UnitfulMatrix(randn(N)m) 

        # create perfect data
        y = E*x

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)

        #x̃ = solve(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        
        @test x ≈ x̃.v # no noise in obs
        @test cost(x̃,problem) < 1e-5 # no noise in obs

        #@test x ≈ pinv(problem) * y # inefficient way to solve problem

        # contaminate observations, check if error bars are correct
    end

    @testset "source water inversion: obs at one time, no circulation lag" begin

        #using DimensionalData
        #using DimensionalData: @dim
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"

        M = 5
        E,x = random_source_water_matrix_vector_pair(M)

        # Run model to predict interior location temperature
        #y = uconvert.(K,E*x)
        y = E*x

        # do slices work?
        E[At(:loc1),At(:NATL)]
        E[:,At(:ANT)]

        # invert for x.  
        x̃ = E\y

        σₙ = 1.0
        Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn⁻¹ = UnitfulDimMatrix(Diagonal(fill(σₙ^-1,M)),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        x̃ = solve(problem,alg=:textbook)

        #@test x ≈ x̃.v
        @test cost(x̃,problem) < 1e-5 # no noise in ob
        @test within(x̃.v,x,1.0e-5)
    end

    @testset "source water inversion: obs at one time, many surface regions, with circulation lag" begin

        
        #using DimensionalData
        #using DimensionalData: @dim
        @dim YearCE "years Common Era"
        @dim SurfaceRegion "surface location"
        @dim InteriorLocation "interior location"

        """
    function random_source_water_matrix_vector_pair_with_lag(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
        function source_water_matrix_vector_pair_with_lag(M)

            #using DimensionalData
            #using DimensionalData: @dim
            #cm = u"cm"
            yr = u"yr"
            K = u"K"
            #percent = u"percent"
            surfaceregions = [:NATL,:ANT,:SUBANT]
            lags = (0:(M-1))yr
            years = (1990:2000)yr

            Myears = length(years)
            N = length(surfaceregions)
            
            # observations have units of temperature
            urange1 = fill(K,M)
            urange2 = fill(K,Myears)
            # solution also has units of temperature
            udomain = fill(K,N)
            Eparent = rand(M,N)#*100percent

            # normalize to conserve mass
            for nrow = 1:size(Eparent,1)
                Eparent /= sum(Eparent)
            end
            
            E = UnitfulDimMatrix(ustrip.(Eparent),urange1,udomain,dims=(Ti(lags),SurfaceRegion(surfaceregions)))
            x = UnitfulDimMatrix(randn(Myears,N),urange2,fill(unit(1.0),N),dims=(Ti(years),SurfaceRegion(surfaceregions)))
            return E,x
        end

"""
    function source_water_matrix_vector_pair_with_lag(M)

    M: number of interior locations, observations
    N=3: number of surface regions (fixed), solution or state

    return E,x
"""
        function source_water_DimArray_vector_pair_with_lag(M)

            #using DimensionalData
            #using DimensionalData: @dim
            #cm = u"cm"
            yr = u"yr"
            K = u"K"
            #percent = u"percent"
            surfaceregions = [:NATL,:ANT,:SUBANT]
            lags = (0:(M-1))yr
            years = (1990:2000)yr

            Myears = length(years)
            N = length(surfaceregions)
            
            Eparent = rand(M,N)#*100percent
            # normalize to conserve mass
            for nrow = 1:size(Eparent,1)
                Eparent /= sum(Eparent)
            end
            
            M = DimArray(Eparent,(Ti(lags),SurfaceRegion(surfaceregions)))
            x = DimArray(randn(Myears,N)K,(Ti(years),SurfaceRegion(surfaceregions)))
            return M,x
        end

        function addcontrol(x₀,u)

            x = deepcopy(x₀)
            ~isequal(length(x₀),length(u)) && error("x₀ and u different lengths")
            for ii in eachindex(x₀)
                # check units
                ~isequal(unit(x₀[ii]),unit(u[ii])) && error("x₀ and u different units")
                x[ii] += u[ii]
            end
            return x
        end

        function addcontrol!(x,u)

            ~isequal(length(x),length(u)) && error("x and u different lengths")
            for ii in eachindex(x)
                # check units
                ~isequal(unit(x[ii]),unit(u[ii])) && error("x and u different units")
                x[ii] += u[ii]
            end
            return x
        end

        m = 11
        M,x = source_water_DimArray_vector_pair_with_lag(m)
        #M,x = random_source_water_matrix_vector_pair_with_lag(m)

        # Run model to predict interior location temperature
        # convolve E and x
        # run through all lags
        y = convolve(M,x)

        ## invert for y for x̃

        # Given, M and y. Make first guess for x.
        n = size(M,2)
        # add adjustment
        yr = u"yr"
        years = (1990:2000)yr
        #urange = fill(K,length(years))

        # DimArray is good enough.
        # This is an array, not necessarily a matrix.
        x₀ = DimArray(zeros(size(x))K,(Ti(years),last(dims(M))))
        #x₀ = UnitfulDimMatrix(zeros(size(x)),urange,fill(K,n),dims=(Ti(years),last(dims(M))))

        # test that addcontrol works. It does.
        #u = randn(length(x₀))K
        #x = addcontrol(x₀,u) 

        function predictobs(u,M,x₀)
            x = addcontrol(x₀,u) 
            return y = convolve(M,x)
        end
            
        # probe to get E matrix.

        # what are the expected units of the E matrix?
        Eunits = Matrix{Unitful.FreeUnits}(undef,length(y),length(x₀))
        for ii in eachindex(y)
            for jj in eachindex(x₀)
                if length(y) == 1 && length(x) ==1
                    Eunits[ii,jj] = unit(y)/unit(x₀)
                elseif length(x) == 1
                    Eunits[ii,jj] = unit.(y)[ii]/unit(x₀)
                elseif length(y) ==1
                    Eunits[ii,jj] = unit(y)/unit.(x₀)[jj]
                else
                    Eunits[ii,jj] = unit.(y)[ii]/unit.(x₀)[jj]
                end
            end
        end
        
        Eu = zeros(1,length(x₀)).*Eunits
        u = zeros(length(x₀)).*unit.(x₀)[:]
        y₀ = predictobs(u,M,x₀)
        for rr in eachindex(x₀)
            u = zeros(length(x₀)).*unit.(x₀)[:]
            Δu = 1*unit.(x₀)[rr]
            u[rr] += Δu
            y = predictobs(u,M,x₀)
            Eu[rr] = (y - y₀)/Δu
        end
        E = UnitfulMatrix(Eu)

        # now in a position to use BLUEs to solve
        # should handle matrix left divide with
        # unitful scalars in UnitfulLinearAlgebra
        x̃ = E\UnitfulMatrix([y]) 

        σₙ = 1.0
        Cnndims = (first(dims(E)),first(dims(E)))
        #Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)
        Cnn⁻¹ = UnitfulDimMatrix(Diagonal(fill(σₙ^-1,M)),unitrange(E).^-1,unitrange(E).^1,dims=Cnndims,exact=true)

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        x̃ = solve(problem,alg=:textbook)

        #@test x ≈ x̃.v
        @test cost(x̃,problem) < 1e-5 # no noise in ob
        @test within(x̃.v,x,1.0e-5)

    end
end

