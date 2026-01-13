@testset "unitful" begin

    @testset "error propagation" begin
        using Measurements

        # check what happens for a scalar as well
        Mlist = (1,5)
        for M in Mlist
            a = randn(M)u"K" .± rand(M)u"K"
            error_propagation(a)
        end
    end

    @testset "2d objective mapping (Unitful)" begin
        
        km = u"km" # for the grid size
        cm = u"cm" # for SSH (mapped variable)
        Nx = 20 # 50 # number of gridpoint in first (zonal) direction
        ΔX = 1000km # domain size in zonal direction
        Ny = 10 # 40 # number of gridpoints in second (meridional) direction
        ΔY = 500km
        # make grid axis number 1
        rx = range(0km,ΔX,length=Nx) # zonal distance
        ry = range(0km,ΔY,length=Ny)  # grid axis number 2: meridional distance

        # turn the 2D grid into a 1D bookkeeping system
        r = [(rx[i],ry[j])  for j in eachindex(ry) for i in eachindex(rx)] # a vector that gives x location as first element, y location as second element
        # set lengthscales
        Lx = 300km; Ly = 100km;
        Rρ = [exp( -((r[i][1]-r[j][1])/Lx)^2 - ((r[i][2] - r[j][2])/Ly)^2) for j in eachindex(r), i in eachindex(r)] # doesn't take advantage of symmetry

        # turn correlation matrix into autocovariance matrix: requires variance info
        σ² = (50cm)^2

        # Here I need to produce some "synthetic" data
        # these are the extra steps
        Rρ_posdef = Rρ + 1e-6I
        Rρ12 = cholesky(Rρ_posdef) # cholesky 
        xtrue = √σ²*Rρ12.L*randn(Nx*Ny) 

        Nobs = 20 # number of observations
        robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled
        # make E matrix for these observations, use bilinear interpolation
        # get bilinear interpolation coefficients
        E = zeros(Nobs,Nx*Ny)
        for oo in eachindex(robs)

	    # bilinear interpolation: put into a function 
	    ydist = [ry[ii] .- robs[oo][2] for ii in eachindex(ry)] # y distance between observation and y grid points
	    xdist = [rx[ii] .- robs[oo][1] for ii in eachindex(rx)] # x distance between observation and x grid points
	
	    xhi,ihi = findmin(x -> x > 0km ? x : 1e6km,xdist) # find minimum positive distance
	    δx,ilo = findmin(x -> x ≤ 0km ? abs(x) : 1e6km,xdist) # find minimum negative distance
	
	    yhi,jhi = findmin(y -> y > 0km ? y : 1e6km,ydist) # find minimum positive distance
	    δy,jlo = findmin(y -> y ≤ 0km ? abs(y) : 1e6km,ydist) # find minimum negative distance
	    Δx = δx + xhi # grid spacing
	    Δy = δy + yhi # grid spacing : y	
	
	    coeffs = zeros(Nx,Ny)
	    denom = Δx * Δy
	    coeffs[ilo,jlo] = ((Δx - δx)*(Δy - δy))/denom
	    coeffs[ilo,jhi] = ((Δx - δx)*(δy))/denom
	    coeffs[ihi,jlo] = ((δx)*(Δy - δy))/denom
	    coeffs[ihi,jhi] = (δx*δy)/denom
	    E[oo,:] = vec(coeffs)
        end
        # check that each row sums to one
        @test all(isapprox.(sum(E,dims=2),1.0))

        # how much observational noise
        σₙ = 1cm

        # get the noise covariance
        Cnn = Diagonal(fill(σₙ^2,Nobs))

        # Sample the true field
        yvals = E*xtrue + σₙ*randn(Nobs) 
        y = Estimate(yvals,Cnn)

        ## Now we have synthetic observations
        # Let's see if the true solution can be backed out from the sparse obs.
        Cxx = σ²*Rρ_posdef

        #use `BLUEs.jl` to package as an `Estimate`
        x0vals = zeros(eltype(xtrue),length(xtrue)) # first guess
        x0 = Estimate(x0vals, Cxx)
        x̃ = combine(x0, y, E)

        σcheck = √σ²
        @test maximum(x̃.σ[:])  ≤  σcheck
        @test minimum(x̃.σ[:])  ≥  0.0 * σcheck
    end 
    @testset "custom type with units" begin
        using Unitful

        import Base: vec, getindex, Matrix, size 
        M = 10  # number of obs
        s = u"s"
        K = u"K"
        t = (0:1:M-1)s
        a = randn()*K # intercept
        b = randn()*K/s # slope

        struct Line{T1, T2} <: AbstractVector{Any}
            intercept::T1
            slope::T2
        end
        Line(a::AbstractVector) = Line(first(a),last(a))
     
        # make proper interface for Vector
        Base.vec(b::Line) = vcat(b.intercept,b.slope)
        Base.size(b::Line) = (2,)
        Base.getindex(b::Line, inds::Vararg) = getindex(vec(b), inds...)
        Base.getindex(b::Line; kw...) = getindex(vec(b); kw...)

        line_true = Line(a,b)

        function obs(t, line::Line)
            return line.intercept + line.slope*t 
        end  

        obs_true(t) = obs(t, line_true)
        ytrue = obs_true.(t) 
        ỹcontaminated = ytrue .+ randn(M)*K

        line0 = Line(0.0*K,0.0*K/s) # first guess of line
        line1 = Line(1.0*K,0.0*K/s) # first guess of line
        line2 = Line(0.0*K,1.0*K/s) # first guess of line
        line1 = Line(1.0,0.0) # first guess of line
        line2 = Line(0.0,1.0) # first guess of line

        # uncertainty of first guess
        struct LineUncertainty <: AbstractMatrix{Any}
            intercept::Line
            slope::Line
        end 
        # struct LineUncertainty{T1, T2, L1 <: Line{T1}, L2 <: Line{T2}} <: AbstractMatrix{Any,2}
        #     intercept::L1
        #     slope::L2
        # end 
        # make proper interface for Matrix
        Base.Matrix(b::LineUncertainty) = hcat(b.intercept,b.slope)
        Base.size(b::LineUncertainty) = (2,2)
        Base.getindex(b::LineUncertainty, inds::Vararg) = getindex(Matrix(b), inds...)
        Base.getindex(b::LineUncertainty; kw...) = getindex(Matrix(b); kw...)

        LineUncertainty(A::Matrix) = LineUncertainty(Line(A[:,1]),Line(A[:,2]))

        Base.:*(A::LineUncertainty, b::Line ) =  Line(Matrix(A) * vec(b))
        Base.:*(A::LineUncertainty, B::LineUncertainty) = LineUncertainty(Matrix(A) * Matrix(B))
        Base.:*(a::Number, b::Line) =  Line(a*b.intercept, a*b.slope)
        Px0 = LineUncertainty(
            Line(1.0*K^2,0.0*K^2/s),
            Line(0.0*K^2/s,1.0*(K/s)^2) )

        x0 = Estimate(line0, Px0)
        Py⁻¹ = Diagonal(fill(1.0*K^-2,M))
        y = Estimate(ỹcontaminated, inv(Py⁻¹))
    
        # impulse response method
        obs1(t) = obs(t, line1)
        obs2(t) = obs(t, line2)
        E = hcat(obs1.(t),obs2.(t))

        x = E\y # invert the observations to obtain solution

        x2 = combine(x0,y,E) # also inverts the obs and combines with first guess
        @test all((x.v .- 4x.σ) .< [a,b] .< (x.v .+ 4x.σ))
    end
end 
