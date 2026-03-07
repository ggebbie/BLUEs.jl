@testset "error propagation" begin
    using Measurements

    # check what happens for a scalar
    Mlist = (1,5)
    for M in Mlist
        a = randn(M) .± rand(M)
        error_propagation(a)
    end
end

@testset "mixed signals: dimensionless matrix" begin
    N = 2
    for M in 1:4
        σₓ = rand()
        E = randn(M,N)
        Px⁻¹ = Diagonal(fill(σₓ^-1,N))
        v = randn(N)
        x = Estimate(v,inv(Px⁻¹))
        y = E*x

        if M == N
            xtilde1 = inv(E)*y # use inv for this special repeat, repeat with left divide below
            @test sum(isapprox.(x.v, xtilde1.v, atol=1e-8)) == N  
            @test sum(isapprox.(x.P, xtilde1.P, atol=1e-8)) == N^2
        end
        
        xtilde2 = E\y
        if  N ≤ M
            @test sum(isapprox.(x.v, xtilde2.v, atol=1e-8)) == N  
            @test sum(isapprox.(x.P, xtilde2.P, atol=1e-8)) == N^2
        end
    end
end

@testset "trend analysis: left-uniform matrix" begin

    M = 10  # number of obs
    t = collect(0:1:M-1)
    a = randn() # intercept
    b = randn() # slope

    ytrue = a .+ b.*t 
    ỹ = a .+ b.*t .+ randn(M)
    E = hcat(ones(M),t)
    Py⁻¹ = Diagonal(fill(1.0,M))
    y_estimate = Estimate(ytrue,inv(Py⁻¹))
    x = E\y_estimate # invert the observations to obtain solution
    @test isapprox(x.v[1],a)
    @test isapprox(x.v[2],b)
    
    # least squares solution method  
    problem = OverdeterminedProblem(ỹ,E,Py⁻¹)
    x̃ = solve(problem,alg=:textbook)
    x̃1 = solve(problem,alg=:hessian)
    @test cost(x̃,problem) < 5M # rough guide, could get unlucky and not pass
    @test cost(x̃1,problem) < 5M # rough guide, could get unlucky and not pass
end

@testset "trend analysis: custom type" begin

    import Base: vec, getindex, Matrix, size 
    M = 10  # number of obs
    t = collect(0.0:1.0:M-1)
    a = randn() # intercept
    b = randn() # slope

    struct Line{T} <: AbstractArray{T,1}
        intercept::T
        slope::T
    end
    Line(a::Vector) = Line(first(a),last(a))
     
    # make proper interface for Vector
    Base.vec(b::Line) = vcat(b.intercept,b.slope)
    Base.size(b::Line) = (2,)
    Base.getindex(b::Line, inds::Vararg) = getindex(vec(b), inds...)
    Base.getindex(b::Line; kw...) = getindex(vec(b); kw...)

    line_true = Line(a,b);

    struct LineObs{T} <: AbstractVector{T}
        time::Vector
        val::Vector{T}
    end

    vec(y::LineObs) = y.val
    size(y::LineObs) = (length(y.time),) 
    getindex(y::LineObs, inds) = getindex(y.val,inds...)
    function Matrix(A::LineObs{Line{T}}) where T
        B = Array{T,2}(undef, 2, length(A.val))
        for j in 1:length(A.val)
            B[:,j] = vec(A.val[j])
        end
        return B
    end

    function Matrix(A::Line{LineObs{T}}) where T
        B = Array{T,2}(undef, length(A.intercept), 2)
        for i in 1:length(A.intercept)
            B[i,1] = A.intercept[i]
            B[i,2] = A.slope[i]
        end
        return B
    end

    Base.:*(A::LineObs{Line{T}}, b::LineObs{T}) where T = Line(Base.:*(Matrix(A),vec(b)))
    
    function noise(t)
        return LineObs(t, randn(size(t)))
    end
    
    function obs(t, line::Line)
        return LineObs(t, line.intercept .+ line.slope.*t )
    end  

    obs_true(t) = obs(t, line_true)
    ytrue = obs_true(t)
    n = noise(t)
    function Base.:+(a::LineObs, b::LineObs)
        (a.time != b.time) && error("not at same time")
        return LineObs(a.time, a.val .+ b.val)
    end
    function Base.:-(a::LineObs, b::LineObs)
        (a.time != b.time) && error("not at same time")
        return LineObs(a.time, a.val .- b.val)
    end

    y0 = ytrue + n

    line0 = Line(0.,0.) # first guess of line
    line1 = Line(1.,0.) # line mode 1
    line2 = Line(0.,1.) # line mode 2

    # uncertainty of first guess
    struct LineUncertainty{T, L <: Line{T}} <: AbstractArray{T,2}
        intercept::L
        slope::L
    end 
    # make proper interface for Matrix
    Base.Matrix(b::LineUncertainty) = hcat(b.intercept,b.slope)
    Base.size(b::LineUncertainty) = (2,2)
    Base.getindex(b::LineUncertainty, inds::Vararg) = getindex(Matrix(b), inds...)
    Base.getindex(b::LineUncertainty; kw...) = getindex(Matrix(b); kw...)
    Base.transpose(A::Line{<:AbstractVector}) = vcat(vec(A.intercept), vec(A.slope))
    function Base.transpose(A::Line{LineObs{T}}) where T
        lines = Vector{Line{T}}(undef, length(A.intercept))
        for j = 1:length(A.intercept)
            lines[j] = Line(A.intercept[j], A.slope[j])
        end
        # vcat(vec(A.intercept), vec(A.slope))
        return LineObs(A.intercept.time, lines)
    end
    LineUncertainty(A::Matrix) = LineUncertainty(Line(A[:,1]),Line(A[:,2]))

    
    Base.:*(A::LineObs{<:Line}, B::Line{<:LineObs}) = LineUncertainty( Base.:*(Matrix(A), Matrix(B)))

    Base.:*(A::LineUncertainty, b::Line ) =  Line(Base.:*(Matrix(A), vec(b)))
    Base.:*(A::LineUncertainty, B::LineUncertainty) = LineUncertainty(Matrix(A) * Matrix(B))
    Base.:*(a::Number, b::Line) =  Line(a*b.intercept, a*b.slope)
    function obs(t, P::LineUncertainty)
        return Line( obs(t, P.intercept), obs(t, P.slope))
    end  
    function obs(t, P::LineObs{Line{T}}) where T
        # creates a LineObsUncertainty matrix
        columns = Vector{LineObs{T}}(undef, length(t))
        for j in 1:length(t)
            columns[j] = obs(t, P.val[j])
        end
        return LineObsUncertainty(t, columns)
    end  

    Px0 = LineUncertainty(line1,line2)
    x0 = Estimate(line0, Px0)

    struct LineObsUncertainty{T, L <: LineObs{T}} <: AbstractArray{T,2}
        time::AbstractVector{T}
        val::Vector{L}
    end
    
    function LineObsUncertainty(A::AbstractMatrix{T}, t::Vector{T}) where T
        columns = Vector{LineObs{T}}(undef, length(t))
        for j in 1:length(t)
            columns[j] = LineObs(t, A[:,j])
        end
        return LineObsUncertainty(t, columns)
        # LineObsUncertainty(Line(A[:,1]),Line(A[:,2]))
    end
    LinearAlgebra.diag(A::LineObsUncertainty) = diag(Matrix(A))

    LinearAlgebra.:(\)(E::AbstractMatrix, A::LineObsUncertainty) = E \ Matrix(A)
    LinearAlgebra.:(\)(A::LineObsUncertainty, b::LineObs) = LineObs(A.time, Matrix(A) \ vec(b))
    function LinearAlgebra.:(\)(A::LineObsUncertainty{T, LineObs{T}}, B::Line{LineObs{T}}) where T
        
        tmp = Matrix(A) \ Matrix(B)
        lineobs1 = LineObs(A.time, tmp[:,1])
        lineobs2 = LineObs(A.time, tmp[:,2])
        return Line(lineobs1, lineobs2)
    end
    Base.getindex(A::LineObsUncertainty, inds::Vararg) = (A.val[last(inds)]).val[first(inds)]

    size(A::LineObsUncertainty) = (length(A.time), length(A.time))
    Matrix(A::LineObsUncertainty) = Array(reshape(vec(A), size(A)))
    function Base.:+(a::LineObsUncertainty, b::LineObsUncertainty)
        (a.time != b.time) && error("not at same time")
        return LineObsUncertainty(a.time, a.val .+ b.val)
    end

    function Base.:-(a::LineObsUncertainty, b::LineObsUncertainty)
        (a.time != b.time) && error("not at same time")
        return LineObsUncertainty(a.time, a.val .- b.val)
    end

    Py = LineObsUncertainty(1.0*I(M), t)
    @test Matrix(Py) == 1.0*I(M)
    
    # Py⁻¹ = Diagonal(fill(1.0,M))
    y = Estimate(y0, Py)
    
    # impulse response method     #####
    obs1(t) = obs(t, line1)
    obs2(t) = obs(t, line2)
    E = hcat(obs1(t), obs2(t))

    # fine but x is not a `Line`
    x = E\y # invert the observations to obtain solution

    x2 = combine(x0,y,E) # also inverts the obs and combines with first guess
    
    @test all((x.v .- 4x.σ) .< [a,b] .< (x.v .+ 4x.σ))
    
    ######### try to `combine` with correct structs

    obs(x) = obs(t, x)
    # obs(x) = obs(t, x)
    E1 = obs
    y1 = y
    Pyx = E1(x0.P) 
    Pxy = transpose(Pyx)
    @test Matrix(Pyx) == transpose(Matrix(Pxy))
    EPxy = E1(Pxy)
    @test EPxy isa LineObsUncertainty
    Py = EPxy + y1.P
    y0 = E1(x0.v)
    n1 = y1.v - y0
    tmp = Py \ n1
    # v = Pxy * tmp # forgot to import, temporary problem
    v = Base.:*(Pxy, tmp)
    dP = Base.:*(Pxy, (Py \ Pyx))
    # dP = Pxy * (Py \ Pyx)
    P = x0.P - dP
    x̃ =  Estimate(v,P)
    
    # # also need to account for Lines that can be matrices
    # Base.Matrix(A::Line{<:AbstractVector}) = hcat(vec(A.intercept), vec(A.slope))
    # function Base.Matrix(A::AbstractVector{<:Line{T}}) where T
    #     ncol = length(A)
    #     Amat = Array{T,2}(undef,2,ncol) 
    #     for i in 1:ncol
    #         Amat[:,i] = vec(A[i])
    #     end
    # end
    # obs(x0::Line) = obs(t,x0)

    # EPx0 = obs(t, Px0)
    # EPx0 isa Line{<:AbstractVector}
    # Matrix(EPx0)

    # combine(x0,y,obs)

    
end

@testset "left-uniform problem with prior info" begin
    y = [-1.9]
    σn = 0.2
    a = -0.24
    γδ = 1.0 
    γT = 1.0
    E = [1 a] 
    x₀ = [-1.0, 4.0]
        
    # "overdetermined, problem 1.4" 
    Px⁻¹ = Diagonal([γδ,γT])
    Py⁻¹  = Diagonal([σn.^-2])
            
    # "underdetermined, problem 2.1" 
    Py  = Diagonal([σn.^2])
    Px = Diagonal([γδ,γT].^-1)
    x0 = Estimate(x₀, Px)
    y_estimate = Estimate(y, Py)
    x1 = combine(x0, y_estimate, E) # a BLUE
    
    # least squares methods
    oproblem = OverdeterminedProblem(y,E,Py⁻¹,Px⁻¹,x₀)
    x̃1 = solve(oproblem)
    @test cost(x̃1,oproblem) < 1 # rough guide, coul

    uproblem = UnderdeterminedProblem(y,E,Py,Px,x₀)
    x̃2 = solve(uproblem)
    @test cost(x̃2,uproblem) < 1 # rough guide, could ge

    # same answer both least-squares ways?
    @test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)
    @test isapprox(x1.v, x̃1.v)
    @test isapprox(x1.σ, x̃1.σ)
end

@testset "polynomial fitting, problem 2.3" begin
    M = 50
    N = 4 
    t = (1:M)
    E =hcat(t.^0, t, t.^2, t.^3)
    σy = 0.1
    Py = Diagonal(fill(σy^2,M))
    Py⁻¹ = Diagonal(fill(σy^-2,M))
    γ = [1.0e1, 1.0e2, 1.0e3, 1.0e4]
    Px⁻¹ = Diagonal(γ)
    x₀ = zeros(N)
    
    Py¹² = cholesky(Py)
    Py⁻¹² = cholesky(Py⁻¹)
    Px = inv(Px⁻¹)
    Px¹² = cholesky(Px)
    x = Px¹².L*randn(N)
    y = E*x

    x0 = Estimate(x₀, Px)
    y_estimate = Estimate(y, Py)
    x1 = combine(x0, y_estimate, E) # a BLUE

    oproblem = OverdeterminedProblem(y,E,Py⁻¹,Px⁻¹,x₀)
    
    # not perfect data fit b.c. of prior info
    x̃ = solve(oproblem,alg=:hessian)
    @test cost(x̃,oproblem) < 5M
    x̃ = solve(oproblem,alg=:textbook)
    @test cost(x̃,oproblem) < 5M
    @test isapprox(x1.v, x̃.v)
    @test isapprox(x1.σ, x̃.σ)
end

@testset "overdetermined problem for mean with autocovariance, problem 4.1" begin

end

# additional problem: 5.1 model of exponential decay
# Not implemented/tested for `combine` function of two Estimates.
@testset "overdetermined named tuple E,y" begin
    N = 2
    M = 1
    σₓ = rand()
    E1 = randn(M,N)
    E2 = randn(M,N)
    E = (one=E1,two=E2)
    Py⁻¹1 = Diagonal(fill(σₓ^-1,M))
    x = randn(N)
    Py⁻¹ = (one=Py⁻¹1, two =Py⁻¹1)
    # create perfect data
    y = E*x
    problem = OverdeterminedProblem(y,E,Py⁻¹)
    x̃ = solve(problem,alg=:hessian)
    @test x ≈ x̃.v # no noise in obs
    @test cost(x̃,problem) < 1e-5 # no noise in obs
end

@testset "objective mapping, problem 4.3" begin
    using ToeplitzMatrices
    using SparseArrays
    M = 11
    τ = range(0.0,5.0,step=0.1)
    ρ = exp.(-τ.^2/(1)^2)
    n = length(ρ)
    Px = SymmetricToeplitz(ρ) + Diagonal(fill(1e-6,n))
    σy = 0.1
    Py = Diagonal(fill(σy,M))
    E = sparse(1:M,1:5:n,fill(1.0,M))
    x₀ = zeros(n)
    Px¹² = cholesky(Px)
    x = Px¹².L*randn(n)
    y = E*x
    x0 = Estimate(x₀, Px)
    y1 = Estimate(y, Py)
    x1 = combine(x0, y1, E) # a BLUE
    
    uproblem = UnderdeterminedProblem(y,E,Py,Px,x₀)
    x̃ = solve(uproblem)
    @test cost(x̃,uproblem) < 5M 
    @test isapprox(x1.v, x̃.v)
    @test isapprox(x1.σ, x̃.σ)
end 
