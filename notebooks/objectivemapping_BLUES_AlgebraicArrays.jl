### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 84a3f486-6566-4dad-99f0-944d8b67f6a0
import Pkg; Pkg.activate("."); Pkg.instantiate()

# ╔═╡ 82de0964-3a0a-4580-bb83-b8d43a9e7d1b
using PlutoUI

# ╔═╡ 71178a40-391d-4abc-8a7a-d03c90c28952
using Unitful

# ╔═╡ 793180e7-d32d-4b09-8a04-1861b3306a46
using LinearAlgebra

# ╔═╡ 7bcfbc9d-cc6b-49f3-a2af-dc9356afceb1
using Plots

# ╔═╡ 391f6b72-d8ba-46d1-8adb-d03e8516d0cd
using BLUEs

# ╔═╡ 0b40d2be-55da-4909-a8b8-446f653d3e5c
using DimensionalData

# ╔═╡ bcaec4ee-efe8-4594-b0a1-3c910b957dd1
using DimensionalData:@dim

# ╔═╡ 402a8bd1-bf35-4558-b389-806aaf9c35cc
using AlgebraicArrays

# ╔═╡ a72352ab-cdfb-42db-85de-ea88132ba9dd
md""" # Objective mapping with `BLUEs.jl` and  `AlgebraicArrays.jl`"""

# ╔═╡ de4be099-b717-48c1-a713-4e0bb785953e
# BLUES.jl: handles linear algebra of Gauss-Markov method
# AlgebraicArrays.jl: handles translation between 2D grid and linear algebra matrices and vectors

# ╔═╡ 7d13a231-8ee5-459b-9c0e-dd81ab87bda1
md""" ## 2D objective map of sea surface height """

# ╔═╡ b3392f99-d4a9-407a-a347-94b60cc9ae6e
md"""### Parameters of the problem """

# ╔═╡ 892694bf-e8ef-4a34-9239-394754b75a16
Nx = 50 # number of gridpoint in first (zonal) direction

# ╔═╡ 5d0dc6b4-24e1-413f-8790-33d6610e09e6
Ny = 40 # number of gridpoints in second (meridional) direction

# ╔═╡ 6510d727-5612-4605-9f35-9ce8eda98f02
Nobs = 20 # number of observations

# ╔═╡ ae71056a-3520-415a-9962-9e953a32780b
md""" ### jumping right into it, the results """

# ╔═╡ 545a3bc2-f399-41eb-99fc-1ea40239d856
md""" ## how well is the data fit? """

# ╔═╡ bbcbe01c-d852-49e3-a8ee-b5b2e26d798d
# other diagnostics: z-score, actual/true error, accuracy of error bars

# ╔═╡ 7e677e5a-7b3e-4d9a-8bf4-e8bbd986fc6b
md""" ### make the grid """

# ╔═╡ d1a99581-7505-49f3-9184-db5278de12a4
md""" ### create autocovariance matrix """

# ╔═╡ 9adb250f-3b02-4db7-a31d-9f7e79cb1910
md""" ### produce synthetic data """

# ╔═╡ 767418fa-ed77-4835-8c5c-0cc89d8d71bc
# Here I need to produce some "synthetic" data
# these are the extra steps

# ╔═╡ df917296-aab3-424a-9371-0ba849c51189
@dim Obs "observations"

# ╔═╡ b917adee-6145-4766-bc5d-ca53d31da582
function interpolate_bilinearly(x::VectorArray, locations::DimensionalData.Dimension)
	y = zeros(eltype(x), locations, :VectorArray)
 	for (i, o) in enumerate(locations)
		y[i] += interpolate_bilinearly(x, o)	
	end
	return y
end

# ╔═╡ 137af3c0-d802-4752-897c-f800fe635a4d
# boilerplate function to extend from `VectorArray`s to `MatrixArray`s
function interpolate_bilinearly(P::MatrixArray, locations::DimensionalData.Dimension) 
    T = typeof(parent(interpolate_bilinearly(first(P), locations)))
    Pyx = Array{T}(undef,size(P))
    for i in eachindex(P)
        Pyx[i] = parent(interpolate_bilinearly(P[i], locations))
    end
    return  MatrixArray(DimArray(Pyx, domaindims(P)))
end

# ╔═╡ affa28b6-8efe-41e3-a588-38851aa7903d
md""" ### make first guess """

# ╔═╡ a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
## Now we have synthetic observations
# Let's see if the true solution can be backed out from the sparse obs.

# ╔═╡ cb54ec24-8a02-4f3a-99fd-9ae1a98d73e0
md""" ### combine first-guess and observational information """

# ╔═╡ 180b7edb-e029-49f7-b301-787b2a71a138
md""" ### set up the physical units in the problem"""

# ╔═╡ 672f4800-bada-484b-b6f7-43c38106cec0
km = u"km" # for the grid size

# ╔═╡ 0828e154-50ba-4dee-be43-df880e34ab1b
# set correlation lengthscales
Lx = 300km 

# ╔═╡ 14b2c84e-78c1-459b-8c50-e0b54e4e533c
Ly = 100km

# ╔═╡ 11f890ef-3091-4886-834c-51e2e619228a
ΔX = 1000km # domain size in zonal direction

# ╔═╡ f9bda6f5-97e4-4ea5-9e7a-8826b528dd41
ΔY = 500km

# ╔═╡ bdafc085-e15e-4130-bca1-7abf5e724025
robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled

# ╔═╡ 6c97f8e0-e814-4e6f-9421-62b546de821e
SSH_obs = Obs(robs)

# ╔═╡ 1538327f-dc2a-494c-a146-d725f91c0a34
# add the data to the plot
rxobs = first.(robs) #[robs[oo][1] for oo in eachindex(robs)]

# ╔═╡ 74413262-c13e-42c7-be41-6b8b829e9b5e
ryobs = last.(robs) # [robs[oo][2] for oo in eachindex(robs)]

# ╔═╡ ae3a2af0-8fae-11ee-288d-732288c2bc04
# make grid axis number 1
rx = range(0km,ΔX,length=Nx) # zonal distance

# ╔═╡ 28a84a7a-36b8-4319-9eed-693491723f6d
ry = range(0km,ΔY,length=Ny)  # grid axis number 2: meridional distance

# ╔═╡ 444a8791-93f8-4ba1-906f-858d365092f7
Grid = (X(rx),Y(ry))

# ╔═╡ 7dc54541-0664-464b-92fa-901e7ed512b6
# diagonal elements of first-guess uncertainty matrix
# nonzero for stability of Cholesky decomposition
Rρ_diag = fill(1e-6, Grid, :VectorArray)

# ╔═╡ f8b54566-c037-4c9d-b151-a4fd2ac2b76e
# diagonal `MatrixArray` to precondition matrix for stability
Rρ = Diagonal(Rρ_diag)

# ╔═╡ d022a766-657e-4566-a3de-7d25a4810d74
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

# ╔═╡ f5a4bdbe-863c-4397-a163-c547b397c46c
begin
	# `cholesky` not yet implemented in AlgebraicArrays
	# here's a workaround
	Rρ12 = AlgebraicArray(cholesky(Matrix(Rρ)).U, Grid, Grid) # cholesky 
end

# ╔═╡ a42f66cb-3dc1-46cb-be94-38979d815d78
begin
	# correlation map
	ival = 13
	jval = 6
	iprint = string(round(typeof(1km),rx[ival]))
	jprint = string(round(typeof(1km),ry[jval]))
	contourf(rx,ry,transpose(parent(Rρ[ival,jval])),
		title="grid point ("*iprint*","*jprint*")",
		titlefontsize=8,
		ylabel="Y",
		xlabel="X",
		color=:amp) 
end

# ╔═╡ 9cc0fcc8-c92f-411b-9ab4-cccfbe73be61
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

# ╔═╡ dcd92fb1-33aa-43f7-b0aa-396df9faf4de
# check that each observation is one when truth is one (bilinear interpolation is a true average)
all(isapprox.(interpolate_bilinearly( ones(Grid, :VectorArray), SSH_obs), 1.0))

# ╔═╡ bfea9c06-0ac7-4810-8a3b-9819c9f019e9
# solve with `combine`; it expects a function with just one argument
predict_obs(x0) = interpolate_bilinearly(x0, SSH_obs)

# ╔═╡ b67f393d-2cd4-4f97-88d3-accf7da3d79a
cm = u"cm" # for SSH (mapped variable)

# ╔═╡ 12028e54-7bda-41ab-b1a6-916f1e2619ca
# turn correlation matrix into autocovariance matrix: requires variance info
σx = 50.0cm # created a difficult-to-find error if this is an integer (!!)

# ╔═╡ 38b8e924-6ec1-41f0-a972-756ce1c5cd4d
# xtrue = √σ²*Rρ12.L*randn(Grid) # this would be proper if Cholesky factorization existed for `MatrixArray`
xtrue = σx*transpose(Rρ12)*randn(Grid, :VectorArray)

# ╔═╡ 0f7fbda2-cb2a-42ed-a345-0686486fe4c1
contourf(rx,ry,transpose(parent(xtrue)),xlabel="zonal distance",ylabel="meridional distance",clabels=true,levels=-100:10:100,cbar=false,title="true SSH")

# ╔═╡ 52d1e4d2-86b0-4aa6-8a9c-532e1cafa3c1
zeros(eltype(xtrue),dims(SSH_obs),:VectorArray)

# ╔═╡ 54228f3e-1291-4d8e-8fc9-f3a8a44085de
ytrue = interpolate_bilinearly(xtrue, SSH_obs) # perfect observations

# ╔═╡ a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
#use `BLUEs.jl` to package as an `Estimate`
x0vals = zeros(eltype(xtrue), Grid, :VectorArray) # first guess

# ╔═╡ 299b6f3b-1181-4059-ba2f-c53a6b6477fa
x0 = Estimate(x0vals, σx^2 * Rρ)

# ╔═╡ 93168c75-eeba-4b64-9054-35981c93701b
# how much observational noise?
σy = 1cm

# ╔═╡ 180bce23-cc3c-4869-ae30-2cd0d61aa34f
# standard error on the grid
yerr = fill(σy, SSH_obs, :VectorArray)

# ╔═╡ 63eb3d93-e49e-477d-bbf2-02c9b59123f3
n = σy * randn(SSH_obs, :VectorArray)

# ╔═╡ aeebaa1f-f83b-45d2-8e57-7385e5641bcb
ycontaminated = ytrue + n

# ╔═╡ 8e6032a0-0de4-4fef-9849-60e963daf952
y = Estimate(ycontaminated, yerr) # bundle observations and uncertainty

# ╔═╡ 007e1141-4a33-4538-a089-b3238117af3d
begin
	contour(rx,ry,transpose(parent(xtrue)),
		xlabel="zonal distance",
		ylabel="meridional distance",
		clabels=true,
		cbar=false,
		title="true SSH with $Nobs obs",
		fill=true,
		levels=-100:10:100)
	scatter!(rxobs,ryobs,zcolor=y.v,cbar=false,label="y",markersize=7) 
end

# ╔═╡ 1be2b0ac-39c0-467f-8a27-eae0b176c753
x̃ = combine(x0, y, predict_obs) 

# ╔═╡ 371c28b5-90e0-4f69-ac9b-e706fe960d07
begin
	contourf(rx,ry,transpose(parent(x̃.v)),
		xlabel="zonal distance",
		ylabel="meridional distance",
		title="SSH objective map",
		levels=-100:10:100,
		clabels=true)
	scatter!(rxobs,
		ryobs,
		zcolor=y.v,
		label="y",
		ms=6,
		cbar=false) 
end

# ╔═╡ 573c07ba-2c7a-4b95-9ed6-dd0563c257e5
begin
	contourf(rx,ry,transpose(parent(x̃.σ)),
		xlabel="zonal distance",
		ylabel="meridional distance",
		title="SSH uncertainty",
		clabels=true,
		cbar=false,
		level=0:10:100)
	scatter!(rxobs,ryobs,color=:white,label="y",ms=6) 
end

# ╔═╡ 4f814626-c20c-47e7-93b5-623c90afcc1b
begin
	contourf(rx,ry,transpose(parent(x̃.v-xtrue)),
		xlabel="zonal distance",
		ylabel="meridional distance",
		title="SSH objective map error",
		levels=-100:10:100,
		clabels=true)
	scatter!(rxobs,
		ryobs,
		zcolor=y.v-ytrue,
		label="y",
		ms=6,
		cbar=false) 
end

# ╔═╡ 387f5555-c3f3-4b80-849d-8d65491fc6f8
scatter(y.v,predict_obs(x̃.v),xlabel="y",ylabel="ỹ")

# ╔═╡ 935faa9d-990d-4bfa-b70c-b11bc824e20b
scatter(y.v,y.v-predict_obs(x̃.v),xlabel="y",ylabel="ñ")

# ╔═╡ c406a8d6-674e-4f38-9944-186664a3de8b
md""" ### package management """

# ╔═╡ f9975f99-3795-4f7e-95ea-fb0ce894d4ec
plotly()

# ╔═╡ Cell order:
# ╟─a72352ab-cdfb-42db-85de-ea88132ba9dd
# ╠═de4be099-b717-48c1-a713-4e0bb785953e
# ╟─7d13a231-8ee5-459b-9c0e-dd81ab87bda1
# ╟─b3392f99-d4a9-407a-a347-94b60cc9ae6e
# ╠═0828e154-50ba-4dee-be43-df880e34ab1b
# ╠═14b2c84e-78c1-459b-8c50-e0b54e4e533c
# ╠═892694bf-e8ef-4a34-9239-394754b75a16
# ╠═11f890ef-3091-4886-834c-51e2e619228a
# ╠═5d0dc6b4-24e1-413f-8790-33d6610e09e6
# ╠═f9bda6f5-97e4-4ea5-9e7a-8826b528dd41
# ╠═6510d727-5612-4605-9f35-9ce8eda98f02
# ╠═12028e54-7bda-41ab-b1a6-916f1e2619ca
# ╟─ae71056a-3520-415a-9962-9e953a32780b
# ╟─0f7fbda2-cb2a-42ed-a345-0686486fe4c1
# ╟─007e1141-4a33-4538-a089-b3238117af3d
# ╟─371c28b5-90e0-4f69-ac9b-e706fe960d07
# ╟─573c07ba-2c7a-4b95-9ed6-dd0563c257e5
# ╟─4f814626-c20c-47e7-93b5-623c90afcc1b
# ╟─545a3bc2-f399-41eb-99fc-1ea40239d856
# ╟─387f5555-c3f3-4b80-849d-8d65491fc6f8
# ╟─935faa9d-990d-4bfa-b70c-b11bc824e20b
# ╠═bbcbe01c-d852-49e3-a8ee-b5b2e26d798d
# ╟─7e677e5a-7b3e-4d9a-8bf4-e8bbd986fc6b
# ╠═ae3a2af0-8fae-11ee-288d-732288c2bc04
# ╠═28a84a7a-36b8-4319-9eed-693491723f6d
# ╠═444a8791-93f8-4ba1-906f-858d365092f7
# ╟─d1a99581-7505-49f3-9184-db5278de12a4
# ╠═7dc54541-0664-464b-92fa-901e7ed512b6
# ╠═f8b54566-c037-4c9d-b151-a4fd2ac2b76e
# ╠═d022a766-657e-4566-a3de-7d25a4810d74
# ╟─a42f66cb-3dc1-46cb-be94-38979d815d78
# ╟─9adb250f-3b02-4db7-a31d-9f7e79cb1910
# ╠═767418fa-ed77-4835-8c5c-0cc89d8d71bc
# ╠═f5a4bdbe-863c-4397-a163-c547b397c46c
# ╠═38b8e924-6ec1-41f0-a972-756ce1c5cd4d
# ╠═bdafc085-e15e-4130-bca1-7abf5e724025
# ╠═df917296-aab3-424a-9371-0ba849c51189
# ╠═6c97f8e0-e814-4e6f-9421-62b546de821e
# ╠═9cc0fcc8-c92f-411b-9ab4-cccfbe73be61
# ╠═b917adee-6145-4766-bc5d-ca53d31da582
# ╠═52d1e4d2-86b0-4aa6-8a9c-532e1cafa3c1
# ╠═137af3c0-d802-4752-897c-f800fe635a4d
# ╠═54228f3e-1291-4d8e-8fc9-f3a8a44085de
# ╠═dcd92fb1-33aa-43f7-b0aa-396df9faf4de
# ╠═93168c75-eeba-4b64-9054-35981c93701b
# ╠═180bce23-cc3c-4869-ae30-2cd0d61aa34f
# ╠═63eb3d93-e49e-477d-bbf2-02c9b59123f3
# ╠═aeebaa1f-f83b-45d2-8e57-7385e5641bcb
# ╠═8e6032a0-0de4-4fef-9849-60e963daf952
# ╠═1538327f-dc2a-494c-a146-d725f91c0a34
# ╠═74413262-c13e-42c7-be41-6b8b829e9b5e
# ╟─affa28b6-8efe-41e3-a588-38851aa7903d
# ╠═a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
# ╠═a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
# ╠═299b6f3b-1181-4059-ba2f-c53a6b6477fa
# ╠═bfea9c06-0ac7-4810-8a3b-9819c9f019e9
# ╟─cb54ec24-8a02-4f3a-99fd-9ae1a98d73e0
# ╠═1be2b0ac-39c0-467f-8a27-eae0b176c753
# ╟─180b7edb-e029-49f7-b301-787b2a71a138
# ╠═672f4800-bada-484b-b6f7-43c38106cec0
# ╠═b67f393d-2cd4-4f97-88d3-accf7da3d79a
# ╟─c406a8d6-674e-4f38-9944-186664a3de8b
# ╠═84a3f486-6566-4dad-99f0-944d8b67f6a0
# ╠═82de0964-3a0a-4580-bb83-b8d43a9e7d1b
# ╠═71178a40-391d-4abc-8a7a-d03c90c28952
# ╠═793180e7-d32d-4b09-8a04-1861b3306a46
# ╠═7bcfbc9d-cc6b-49f3-a2af-dc9356afceb1
# ╠═391f6b72-d8ba-46d1-8adb-d03e8516d0cd
# ╠═0b40d2be-55da-4909-a8b8-446f653d3e5c
# ╠═bcaec4ee-efe8-4594-b0a1-3c910b957dd1
# ╠═402a8bd1-bf35-4558-b389-806aaf9c35cc
# ╠═f9975f99-3795-4f7e-95ea-fb0ce894d4ec
