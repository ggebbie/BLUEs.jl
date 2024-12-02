Nx = 20 # 50 # number of gridpoint in first (zonal) direction
Ny = 10 # 40 # number of gridpoints in second (meridional) direction
Nobs = 20 # number of observations
@dim Obs "observations"

cm = u"cm" # for SSH (mapped variable)
# turn correlation matrix into autocovariance matrix: requires variance info
σx = 50.0cm # created a difficult-to-find error if this is an integer (!!)
σy = 1cm # how much observational noise?
km = u"km" # for the grid size
Lx = 300km
Ly = 100km
ΔX = 1000km # domain size in zonal direction
ΔY = 500km
robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled
SSH_obs = Obs(robs)
rxobs = first.(robs) #[robs[oo][1] for oo in eachindex(robs)]
ryobs = last.(robs) # [robs[oo][2] for oo in eachindex(robs)]
# make grid axis number 1
rx = range(0km,ΔX,length=Nx) # zonal distance
ry = range(0km,ΔY,length=Ny)  # grid axis number 2: meridional distance
Grid = (X(rx),Y(ry))
# diagonal elements of first-guess uncertainty matrix
# nonzero for stability of Cholesky decomposition
Rρ_diag = fill(1e-6, Grid, :VectorArray)

# diagonal `MatrixArray` to precondition matrix for stability
Rρ = Diagonal(Rρ_diag)

# first two loops for the grid location referenced in the columns of Rρ
for (icol,rxcol) in enumerate(first(Grid))
    for (jcol,rycol) in enumerate(last(Grid))
	# second two loops for the grid location referenced in the rows of Rρ
	for (irow,rxrow) in enumerate(first(Grid))
	    for (jrow,ryrow) in enumerate(last(Grid))
		Rρ[icol,jcol][irow,jrow] += exp( -((rxcol-rxrow)/Lx)^2 - 
					         ((rycol - ryrow)/Ly)^2) 
		#+ 1.0e-6 * (icol == irow) * (jcol == jrow)
	    end
	end
    end	
end

# `cholesky` not yet implemented in AlgebraicArrays
# here's a workaround
Rρ12 = AlgebraicArray(cholesky(Matrix(Rρ)).U, Grid, Grid) # cholesky 

"""
function interpolate_bilinearly(x::VectorArray, location::Tuple)

Given gridded information `x`, predict an observation at `location`

function interpolate_bilinearly(x::VectorArray, locations::DimensionalData.Dimension)

Given gridded information `x`, predict many observations at `locations`

function interpolate_bilinearly(P::MatrixArray, locations::DimensionalData.Dimension)

Given many gridded fields in the columns of `P`, predict many observations at `locations`
"""
function interpolate_bilinearly(x::VectorArray, location::Tuple)
	
    xdist = val(first(dims(x))) .- first(location) 
    ydist = val(last(dims(x))) .- last(location) 
    largeval = 1e6km
    xhi,ihi = findmin(x -> x > 0km ? x : largeval,xdist) # find minimum positive distance
    δx,ilo = findmin(x -> x ≤ 0km ? abs(x) : largeval,xdist) # find minimum negative distance
	
    yhi,jhi = findmin(y -> y > 0km ? y : largeval,ydist) # find minimum positive distance
    δy,jlo = findmin(y -> y ≤ 0km ? abs(y) : largeval,ydist) # find minimum negative distance
    Δx = δx + xhi # grid spacing
    Δy = δy + yhi # grid spacing : y	
    denom = Δx * Δy

    obs = zero(eltype(x))
    obs += x[ilo, jlo] * ((Δx - δx)*(Δy - δy))/denom
    obs += x[ilo, jhi] * ((Δx - δx)*(δy))/denom
    obs += x[ihi, jlo] * ((δx)*(Δy - δy))/denom
    obs += x[ihi, jhi] * (δx*δy)/denom
    return obs
end

function interpolate_bilinearly(x::VectorArray, locations::DimensionalData.Dimension)
    y = zeros(eltype(x), locations, :VectorArray)
    for (i, o) in enumerate(locations)
	y[i] += interpolate_bilinearly(x, o)	
    end
    return y
end
# boilerplate function to extend from `VectorArray`s to `MatrixArray`s
function interpolate_bilinearly(P::MatrixArray, locations::DimensionalData.Dimension) 
    T = typeof(parent(interpolate_bilinearly(first(P), locations)))
    Pyx = Array{T}(undef,size(P))
    for i in eachindex(P)
        Pyx[i] = parent(interpolate_bilinearly(P[i], locations))
    end
    return  MatrixArray(DimArray(Pyx, domaindims(P)))
end

# check that each observation is one when truth is one (bilinear interpolation is a true average)
@test all(isapprox.(interpolate_bilinearly( ones(Grid, :VectorArray), SSH_obs), 1.0))

# solve with `combine`; it expects a function with just one argument
predict_obs(x0) = interpolate_bilinearly(x0, SSH_obs)

xtrue = σx*transpose(Rρ12)*randn(Grid, :VectorArray)
ytrue = interpolate_bilinearly(xtrue, SSH_obs) # perfect observations

#use `BLUEs.jl` to package as an `Estimate`
x0vals = zeros(eltype(xtrue), Grid, :VectorArray) # first guess
x0 = Estimate(x0vals, σx^2 * Rρ)

# standard error on the grid
yerr = fill(σy, SSH_obs, :VectorArray)
n = σy * randn(SSH_obs, :VectorArray)
ycontaminated = ytrue + n
y = Estimate(ycontaminated, yerr) # bundle observations and uncertainty
x̃ = combine(x0, y, predict_obs)
@test maximum(x̃.σ[:])  ≤  σx
@test minimum(x̃.σ[:])  ≥  0.0 * σx
