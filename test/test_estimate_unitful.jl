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

    @testset "trend analysis: custom type" begin

        import Base: vec, getindex, Matrix, size 
        M = 10  # number of obs
        yr = u"yr"
        t = collect(0.0:1.0:M-1)yr
        a = randn()*K # intercept
        b = randn()*K/yr # slope

        ##################
        # define `UnitfulLine` (i.e., `ULine`)
        # make proper interface for Vector
        struct ULine{T <: Number} <: AbstractArray{T,1}
            intercept
            slope
            ULine(a::Number, b::Number) = new{eltype(vcat(a, b))}(a, b)
        end
        ULine(l::Vector) = length(l) == 2 ?
                           ULine(l[1], l[2]) :
                           error("too many vals for a line")
        Base.vec(b::ULine) = vcat(b.intercept,b.slope)
        Base.size(b::ULine) = (2,)
        Base.getindex(b::ULine, inds::Vararg) = getindex(vec(b), inds...)
        Base.getindex(b::ULine; kw...) = getindex(vec(b); kw...)

        ######### 
        # construct some interesting and useful `ULine`s
        line_true = ULine(a,b)
        a0 = zero(typeof(a))
        a1 = oneunit(typeof(a))
        b0 = zero(typeof(b))
        b1 = oneunit(typeof(b))
        line0 = ULine(a0, b0) # first guess of line
        line1 = ULine(a1, b0) # line mode 1
        line2 = ULine(a0, b1) # line mode 2
        line3 = ULine(a1, b1) # line mode 2

        ######### 
        # define ULineObs
        struct ULineObs{T <: Number} <: AbstractVector{T}
            time::Vector
            val::Vector{T}
        end
        Base.vec(y::ULineObs) = y.val
        Base.size(y::ULineObs) = (length(y.time),) 
        Base.getindex(y::ULineObs, inds) = getindex(y.val,inds...)
        function noise(t::Vector{<:Quantity})
            return ULineObs(t, randn(size(t))*K)
        end
        function obs(t, line::ULine)
            return ULineObs(t, line.intercept .+ line.slope.*t )
        end  
        function Base.:+(a::ULineObs, b::ULineObs)
            (a.time != b.time) && error("not at same time")
            return ULineObs(a.time, a.val .+ b.val)
        end
        function Base.:-(a::ULineObs, b::ULineObs)
            (a.time != b.time) && error("not at same time")
            return ULineObs(a.time, a.val .- b.val)
        end

        # construct some `ULineObs`s
        obs_true(t) = obs(t, line_true)
        ytrue = obs_true(t)
        n = noise(t)
        y0 = ytrue + n

        #########
        # struct ULineMatrix{T} <: AbstractArray{T,2}
        # uncertainty of first guess
        struct ULineMatrix{T, D} <: AbstractMatrix{T}
            intercept
            slope
            function ULineMatrix(a, b)
                # println("ulm constructor")
                if (a isa Number || b isa  Number)
                    error("try ULine instead")
                else
                    mat_test = hcat(vec(a),vec(b))
                    # typeof(a) !== typeof(b) && error("input types need to be the same")
                    return new{eltype(mat_test), typeof(a)}(a,b)
                end
            end
        end 
        # function Base.Matrix(A::ULine{ULineObs{T}}) where T
        function Base.Matrix(A::ULineMatrix{T}) where T
            B = Array{T,2}(undef, length(A.intercept), 2)
            for i in 1:length(A.intercept)
                B[i,1] = A.intercept[i]
                B[i,2] = A.slope[i]
            end
            return B
        end
        Base.Matrix(b::ULineMatrix) = hcat(b.intercept,b.slope)
        Base.size(b::ULineMatrix{T, D}) where T <: Number where D <: ULine = (2,2)
        Base.size(b::ULineMatrix{T, D}) where T <: Number where D <: ULineObs = (length(b.slope),2)
        Base.getindex(b::ULineMatrix, inds::Vararg) = getindex(Matrix(b), inds...)
        Base.getindex(b::ULineMatrix; kw...) = getindex(Matrix(b); kw...)
        # Base.transpose(A::ULine{<:AbstractVector}) = vcat(vec(A.intercept), vec(A.slope))
        # Base.transpose(A::ULineMatrix) = vcat(vec(A.intercept), vec(A.slope))
        function Base.transpose(P::ULineMatrix{T, D}) where T <: Number where D <: ULine
            col1 = ULine(P.intercept[1], P.slope[1])
            col2 = ULine(P.intercept[2], P.slope[2])
            return ULineMatrix(col1, col2)
        end
        function Base.transpose(P::ULineMatrix{T, D}) where T <: Number where D <: ULineObs
            cols = Vector{ULine{T}}(undef, length(P.intercept))
            for j = 1:length(P.intercept)
                cols[j] = ULine(P.intercept[j], P.slope[j])
            end
            return ULineObsMatrix(P.intercept.time, cols)
        end
        # this is unclear (next line)
        # ULineMatrix(A::Matrix) = ULineMatrix(ULine(A[:,1]), ULine(A[:,2]))
        # Base.:*(A::ULineObsMatrix, b::ULine) =  ULine(Base.:*(Matrix(A), vec(b)))
        Base.:*(A::ULineObsMatrix, b::ULineObs) =  ULine(Matrix(A) * vec(b))
        # Base.:*(A::ULineMatrix, B::ULineMatrix) = LineUncertainty(Matrix(A) * Matrix(B))
        # Base.:*(a::Number, b::ULine) =  ULine(a*b.intercept, a*b.slope)
        function obs(t, P::ULineMatrix{T, D}) where T <: Number where D <: ULine
            return ULineMatrix( obs(t, P.intercept), obs(t, P.slope))
        end

        #########
        # make ULineMatrix concrete
        # to get proper units on uncertainty matrix, need an outer product
        α = 1_000 # α -> bigger, more similar to direct inversion
        Punits = line3 * transpose(line3)
        Pmatrix = α * I(2) .* Punits
        Pcol1 = ULine(Pmatrix[:,1])
        Pcol2 = ULine(Pmatrix[:,2])
        Px0 = ULineMatrix(Pcol1,Pcol2)
        x0 = Estimate(line0, Px0)

        #########
        # maybe not necessary: define ULineObsMatrix
        struct ULineObsMatrix{T, D} <: AbstractMatrix{T}
            time
            val
            # inner constructor
            function ULineObsMatrix(time, val::AbstractVector{D}) where D
                new{eltype(D), D}(time, val)
            end
        end
        # outer constructor
        function ULineObsMatrix(t::AbstractVector, A::AbstractMatrix{T}) where {T, D}
            columns = Vector{ULineObs{T}}(undef, length(t))
            for j in 1:length(t)
                columns[j] = ULineObs(t, A[:,j])
            end
            return ULineObsMatrix(t, columns)
        end
        function ULineObsMatrix{D}(t::AbstractVector, A::AbstractMatrix{T}) where {T, D}
            columns = Vector{D}(undef, length(t))
            for j in 1:length(t)
                columns[j] = D(t, A[:,j])
            end
            return ULineObsMatrix(t, columns)
        end
        Base.getindex(A::ULineObsMatrix{T, D}, inds::Vararg) where T <: Number where D <: ULineObs = (A.val[last(inds)]).val[first(inds)]
        function Base.getindex(A::ULineObsMatrix{T, D}, inds::Vararg) where T <: Number where D <: ULine
            first(inds) == 1 && return (A.val[last(inds)]).intercept
            first(inds) == 2 && return (A.val[last(inds)]).slope
        end
        Base.size(A::ULineObsMatrix{T, D}) where T <: Number where D <: ULineObs = (length(A.time), length(A.time))
        Base.size(A::ULineObsMatrix{T, D}) where T <: Number where D <: ULine = (2, length(A.time))
        # Base.Matrix(A::ULineObsMatrix) = Array(reshape(vec(A), size(A)))
        # function Base.Matrix(A::ULineObsMatrix{T, D}) where T <: Number where D <: ULineObs
        function Base.Matrix(A::ULineObsMatrix{T}) where T <: Number
            mat = Array{T, 2}(undef, size(A))
            for j in 1:size(A,2)
                mat[:,j] = vec(A.val[j])
            end
            return mat
        end
        Base.Matrix(A::ULineObsMatrix) = Array(reshape(vec(A), size(A)))
        LinearAlgebra.diag(A::ULineObsMatrix) = diag(Matrix(A))    
        function obs(t, P::ULineObsMatrix{T, D}) where T <: Number where D <: ULine
            # could do a better job figuring out output type, but are all columns the same?
            columns = Vector{ULineObs{eltype(first(P.val).intercept)}}(undef, length(t))
            for j in 1:length(t)
                columns[j] = obs(t, P.val[j])
            end
            return ULineObsMatrix(t, columns)
        end
        function Base.:+(a::ULineObsMatrix, b::ULineObsMatrix)
            (a.time != b.time) && error("not at same time")
            return ULineObsMatrix(a.time, a.val .+ b.val)
        end
        function Base.:-(a::ULineObsMatrix, b::ULineObsMatrix)
            (a.time != b.time) && error("not at same time")
            return ULineObsMatrix(a.time, a.val .- b.val)
        end
        # broadcast not working on ULineObsMatrix
        function LinearAlgebra.:(\)(P::ULineObsMatrix, y::ULineObs)
            # test if uniform here, I assume it    
            ULineObs(y.time, unit(first(y))/ unit(first(P)) * (ustrip.(Matrix(P)) \ ustrip.(vec(y))))
        end
        function LinearAlgebra.:(\)(A::ULineObsMatrix, B::ULineMatrix)
            tmp = ustrip.(Matrix(A)) \ ustrip.(Matrix(B))
            lineobs1 = ULineObs(A.time, tmp[:,1]*unit(first(B))/unit(first(A)))
            lineobs2 = ULineObs(A.time, tmp[:,2]*unit(last(B))/unit(first(A)))

            # ULineObs(y.time, unit(first(y))/ unit(first(P)) * (ustrip.(Matrix(P)) \ ustrip.(vec(y))))
            # tmp[:,1] .*= unit(first(B))
            # tmp[:,2] .*= unit(first(B))/unit(A[1,2])
            # return ULineMatrix(tmp)
            # return ULineMatrix(unit(first(B)) * tmp .* [NoUnits, 1/unit(A[1,2])])
            
            # fudge the units for now, PLEASE FIX
            # lineobs1 = ULineObs(A.time, tmp[:,1])
            # lineobs2 = ULineObs(A.time, tmp[:,2]/yr)
            return ULineMatrix(lineobs1, lineobs2)
        end


        
        # LinearAlgebra.:(\)(E::Matrix{<:Quantity}, y::ULineObs{<:Quantity}) =
        #     unit(first(y.val)) * (ustrip.(E) \ ustrip.(y)) .* [NoUnits, 1/unit(E[1,2])]



        
        # Base.:*(A::ULineObsMatrix, b::ULineObs) =
        #     ULine(Base.:*(Matrix(A),vec(b)))
        # Base.:*(A::ULineObs{ULine{T}}, b::ULineObs{T}) where T = Line(Base.:*(Matrix(A),vec(b)))

    
        Base.:*(A::ULineObsMatrix{<: ULine}, B::ULineMatrix{<:ULineObs}) = ULineMatrix( Base.:*(Matrix(A), Matrix(B)))
        # Base.:*(A::ULineObs{<:ULine}, B::ULine{<:ULineObs}) = LineUncertainty( Base.:*(Matrix(A), Matrix(B)))
        # assume obs are dimensionally uniform
        
        # function LineObsUncertainty(A::AbstractMatrix{T}, t::AbstractVector) where T
        #     columns = Vector{ULineObs{T}}(undef, length(t))
        #     for j in 1:length(t)
        #         columns[j] = ULineObs(t, A[:,j])
        #     end
        #     return LineObsUncertainty(t, columns)
        #     # LineObsUncertainty(ULine(A[:,1]),ULine(A[:,2]))
        # end

        ########## make ULineObsMatrix concrete

        # test inner constructor
        ytrue_test = Vector{typeof(ytrue)}(undef,length(ytrue))
        for i in 1:length(ytrue)
            ytrue_test[i] = ytrue
        end
        sss = ULineObsMatrix(ytrue.time, ytrue_test)
        @test sss isa ULineObsMatrix{T, D} where T <: Number where D <: ULineObs
        
        # a more realistic example
        Pymatrix = 1.0*I(M)*K^2
        Py = ULineObsMatrix(t, Pymatrix)
        # more explicit form might be useful for covariance matrices (i.e., Pyx, Pxy)
        Py2 = ULineObsMatrix{ULineObs{eltype(Pymatrix)}}(t, Pymatrix)
        @test Matrix(Py) == Matrix(Py2)
        @test ustrip.(Matrix(Py)) == 1.0*I(M)
        @test Matrix(Py) == Pymatrix
        y = Estimate(y0, Py)

        #####################################
        ######### methods that work on combinations of custom types        
        ## necessary for `combine` step
        # LinearAlgebra.:(\)(E::Matrix{<:Quantity}, y::ULineObs{<:Quantity}) =
        #     ULine( unit(first(y.val)) * (ustrip.(E) \ ustrip.(y)) .* [NoUnits, 1/unit(E[1,2])])  
        LinearAlgebra.:(\)(E::Matrix{<:Quantity}, y::ULineObs{<:Quantity}) =
            unit(first(y.val)) * (ustrip.(E) \ ustrip.(y)) .* [NoUnits, 1/unit(E[1,2])]

        function LinearAlgebra.:(\)(E::Matrix{<:Quantity}, P::LineObsUncertainty{<:Quantity})
            tmp = unit(first(Matrix(P))) * (ustrip.(Matrix(E)) \ ustrip.(Matrix(P)));# .* [NoUnits, 1/unit(E[1,2])]
            # tmp[2,:] ./= unit(E[1,2])
            return vcat(transpose(tmp[1,:]),transpose(tmp[2,:]/unit(E[1,2])))
        end
        # ULine( unit(first(Matrix(P))) * (ustrip.(Matrix(E)) \ ustrip.(Matrix(P))) .* [NoUnits, 1/unit(E[1,2])])  
        
        # LinearAlgebra.:(\)(E::AbstractMatrix, A::LineObsUncertainty) = E \ Matrix(A)
        LinearAlgebra.:(\)(A::LineObsUncertainty, b::ULineObs) = ULineObs(A.time, Matrix(A) \ vec(b))
        function LinearAlgebra.:(\)(A::LineObsUncertainty{T, ULineObs{T}}, B::ULine{ULineObs{T}}) where T
            tmp = Matrix(A) \ Matrix(B)
            lineobs1 = ULineObs(A.time, tmp[:,1])
            lineobs2 = ULineObs(A.time, tmp[:,2])
            return ULine(lineobs1, lineobs2)
        end

        ######### implementation 1 of methods that work on combinations
        ######### of custom types        
        # impulse response method     #####
        obs1(t) = (obs(t, line1) - obs(t, line0))/first(line1)
        obs2(t) = (obs(t, line2) - obs(t, line0))/last(line2)
        E = hcat(obs1(t), obs2(t))

        # time consuming to track all units due to need to use UnitfulLinearAlgebra
        # (but don't want to add the dependency)
        yla = Estimate(ustrip.(y.v), ustrip.(Matrix(y.P)))
        # invert the observations to obtain solution, this keeps right types
        xla = ustrip.(E)\yla 
        
        # just add units at the end (no checks)
        x = Estimate(xla.v.*[K,K/yr], xla.P .* unit.(Punits))
        @test all((x.v .- 4x.σ) .< [a,b] .< (x.v .+ 4x.σ))
    
        ######### implementation 2 of methods that work on combinations
        ######### of custom types        
        ######### try to `combine` with correct structs
        obs(x) = obs(t, x)
        E1 = obs
        y1 = y
        y0 = E1(x0.v)
        n1 = y1.v - y0
        Pyx = E1(x0.P) 

        Pxy = transpose(Pyx)
        @test Matrix(Pyx) == transpose(Matrix(Pxy))
        EPxy = E1(Pxy)
        @test EPxy isa ULineObsMatrix
        Pyy = EPxy + y1.P

        tmp = Pyy \ n1
        # v = Pxy * tmp # forgot to import, temporary problem
        v = Pxy * tmp
        dP = Pxy * (Pyy \ Pyx)
        P = x0.P - dP
        x̃ =  Estimate(v,P)

        @test isapprox( xla.v, x̃.v)
        @test isapprox( xla.P, x̃.P)

        # wishful thinking
        x_default = combine(x0,y,obs)
        x_nocheck = combine(x0,y,E,check=false)
        x_check = combine(x0,y,E,check=true)    
        @test isapprox( x_check.v, x_nocheck.v)
        @test isapprox( x_check.P, x_nocheck.P)

    end

    @testset "custom type with units original" begin
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
