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
md""" # Objective mapping with `BLUEs.jl`"""

# ╔═╡ 7d13a231-8ee5-459b-9c0e-dd81ab87bda1
md""" ## 2D objective map of sea surface height """

# ╔═╡ e3b69a7f-cad0-4535-b691-a2d53c586581
km = u"km" # for the grid size

# ╔═╡ 8d49fedf-1bf6-45cb-b85b-8b438663452e
cm = u"cm" # for SSH (mapped variable)

# ╔═╡ 892694bf-e8ef-4a34-9239-394754b75a16
Nx = 50 # number of gridpoint in first (zonal) direction

# ╔═╡ 11f890ef-3091-4886-834c-51e2e619228a
ΔX = 1000km # domain size in zonal direction

# ╔═╡ 5d0dc6b4-24e1-413f-8790-33d6610e09e6
Ny = 40 # number of gridpoints in second (meridional) direction

# ╔═╡ f9bda6f5-97e4-4ea5-9e7a-8826b528dd41
ΔY = 500km

# ╔═╡ ae3a2af0-8fae-11ee-288d-732288c2bc04
# make grid axis number 1
rx = range(0km,ΔX,length=Nx) # zonal distance

# ╔═╡ 28a84a7a-36b8-4319-9eed-693491723f6d
ry = range(0km,ΔY,length=Ny)  # grid axis number 2: meridional distance

# ╔═╡ 444a8791-93f8-4ba1-906f-858d365092f7
Grid = (X(rx),Y(ry))

# ╔═╡ cd4671f0-cdbe-4bd7-a04e-d51cbbfeb5d4
# ╠═╡ disabled = true
#=╠═╡
Rρ = MatrixArray(zeros(prod(size(Grid)),prod(size(Grid))), Grid, Grid); # long-winded way to make a `MatrixArray`
  ╠═╡ =#

# ╔═╡ 7dc54541-0664-464b-92fa-901e7ed512b6
# diagonal elements of uncertainty matrix
# nonzero for stability of Cholesky decomposition
Rρ_diag = fill(1e-6,Grid, :VectorArray)

# ╔═╡ f8b54566-c037-4c9d-b151-a4fd2ac2b76e
# diagonal `MatrixArray` to precondition matrix for stability
Rρ = Diagonal(Rρ_diag)

# ╔═╡ 03509c0a-6122-4005-9686-8aca4bd38f99
Rρ

# ╔═╡ 0828e154-50ba-4dee-be43-df880e34ab1b
# set lengthscales
Lx = 300km; Ly = 100km;

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

# ╔═╡ a42f66cb-3dc1-46cb-be94-38979d815d78
begin
	# correlation map
	ival = 13
	jval = 6
	iprint = string(round(typeof(1km),rx[ival]))
	jprint = string(round(typeof(1km),ry[jval]))
	contourf(rx,ry,parent(Rρ[ival,jval])',
		title="grid point ("*iprint*","*jprint*")",
		titlefontsize=8,
		ylabel="Y",
		xlabel="X") 
end

# ╔═╡ 12028e54-7bda-41ab-b1a6-916f1e2619ca
# turn correlation matrix into autocovariance matrix: requires variance info
σx = 50cm

# ╔═╡ 767418fa-ed77-4835-8c5c-0cc89d8d71bc
# Here I need to produce some "synthetic" data
# these are the extra steps

# ╔═╡ f5a4bdbe-863c-4397-a163-c547b397c46c
Rρ12 = cholesky(Rρ) # cholesky 

# ╔═╡ b1027abd-b4af-42d2-8fe3-32a5c39b5b77
MatrixArray(cholesky(Matrix(Rρ)),(dims(Rρ),dims(Rρ)))


# ╔═╡ 38b8e924-6ec1-41f0-a972-756ce1c5cd4d
xtrue = √σ²*Rρ12.L*randn(Nx*Ny) 

# ╔═╡ 1632e60f-5568-4873-8900-84a0608c9dab
xtruefield = reshape(xtrue,Nx,Ny)

# ╔═╡ 3fb8271b-bcbd-4a38-9589-8723c490fbba
contourf(rx,ry,ustrip.(xtruefield'),xlabel="zonal distance",ylabel="meridional distance",clabels=true,cbar=false,title="true SSH")

# ╔═╡ 6510d727-5612-4605-9f35-9ce8eda98f02
Nobs = 20 # number of observations

# ╔═╡ bdafc085-e15e-4130-bca1-7abf5e724025
robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled

# ╔═╡ 37c5091f-3cf5-46b0-8874-59a047a191e6
# make E matrix for these observations, use bilinear interpolation
# get bilinear interpolation coefficients

# ╔═╡ 3dba0928-09f3-4108-9d55-5124f3ec59c1
E = zeros(Nobs,Nx*Ny)

# ╔═╡ 64f8cb6c-9c8a-4f08-b970-697708da228b
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
	

# ╔═╡ dcd92fb1-33aa-43f7-b0aa-396df9faf4de
# check that each row sums to one
sum(E,dims=2)

# ╔═╡ df917296-aab3-424a-9371-0ba849c51189
@dim Obs "observations"

# ╔═╡ 1538327f-dc2a-494c-a146-d725f91c0a34
# add the data to the plot
rxobs = [robs[oo][1] for oo in eachindex(robs)]

# ╔═╡ 74413262-c13e-42c7-be41-6b8b829e9b5e
ryobs = [robs[oo][2] for oo in eachindex(robs)]

# ╔═╡ 3a5b4094-c465-4041-8486-f5c2574980b9
robs

# ╔═╡ bcd788ba-c369-4993-ae62-d6075dda282b
# noise on observations
n = randn(Obs(robs),:VectorArray)*σy

# ╔═╡ fe154fd1-e45a-46ac-afa1-e008abf6e62d
# better output format (need to update AlgebraicArraysDimensionalDataExt to make this automatic)
parent(n) 

# ╔═╡ 7276ef68-34d6-4dd8-9a25-95e86e6b6183
# Sample the true field
yvals = E*xtrue + σₙ*randn(Nobs) 

# ╔═╡ 8e6032a0-0de4-4fef-9849-60e963daf952
y = Estimate(yvals,Cnn)

# ╔═╡ d8ac83da-1829-4064-a08f-89041e60bdc2


# ╔═╡ 007e1141-4a33-4538-a089-b3238117af3d
begin
	contour(rx,ry,ustrip.(xtruefield'),xlabel="zonal distance",ylabel="meridional distance",clabels=true,title="true SSH with $Nobs obs",fill=true)
	scatter!(rxobs,ryobs,zcolor=ustrip.(y.v),label="y",cbar=false,markersize=6) 
end

# ╔═╡ a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
## Now we have synthetic observations
# Let's see if the true solution can be backed out from the sparse obs.


# ╔═╡ 3792cee9-3788-40f9-acf4-66f7b6f5cfcf
Px = σ²*Rρ

# ╔═╡ a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
#use `BLUEs.jl` to package as an `Estimate`
x0vals = zeros(Grid, :VectorArray)*unit(√σ²) # first guess

# ╔═╡ 9f01b5b4-d086-44d7-83d1-90f9d8ed21b6
x0 = Estimate(x0vals, Px)

# ╔═╡ 1be2b0ac-39c0-467f-8a27-eae0b176c753
x̃ = combine(x0, y, E)

# ╔═╡ 61fddba8-ec75-44c0-ab1e-a91967843075
x̃field = reshape(x̃.v,Nx,Ny) # turn it back into 2D

# ╔═╡ 371c28b5-90e0-4f69-ac9b-e706fe960d07
begin
	contourf(rx,ry,x̃field',xlabel="zonal distance",ylabel="meridional distance",title="SSH objective map")
	scatter!(rxobs,ryobs,zcolor=ustrip.(y.v),label="y",ms=6,cbar=false) 
end

# ╔═╡ e155b5ea-da51-45d4-a951-1b3aa1f28b63
σx̃field = reshape(x̃.σ,Nx,Ny) # turn it back into 2D

# ╔═╡ 573c07ba-2c7a-4b95-9ed6-dd0563c257e5
begin
	contourf(rx,ry,σx̃field',xlabel="zonal distance",ylabel="meridional distance",title="SSH uncertainty",clabels=true,cbar=false)
	scatter!(rxobs,ryobs,color=:white,label="y",ms=6) 
end

# ╔═╡ 545a3bc2-f399-41eb-99fc-1ea40239d856
md""" ## how well is the data fit? """

# ╔═╡ 387f5555-c3f3-4b80-849d-8d65491fc6f8
scatter(y.v,(E*x̃).v,xlabel="y",ylabel="ỹ")

# ╔═╡ 935faa9d-990d-4bfa-b70c-b11bc824e20b
scatter(y.v,y.v-(E*x̃).v,xlabel="y",ylabel="ñ")

# ╔═╡ f9975f99-3795-4f7e-95ea-fb0ce894d4ec
plotly()

# ╔═╡ 180bce23-cc3c-4869-ae30-2cd0d61aa34f
# ╠═╡ disabled = true
#=╠═╡
# get the noise covariance
σy = Diagonal(fill(σₙ^2,Nobs))
  ╠═╡ =#

# ╔═╡ 93168c75-eeba-4b64-9054-35981c93701b
# how much observational noise
σy = 1cm

# ╔═╡ Cell order:
# ╠═a72352ab-cdfb-42db-85de-ea88132ba9dd
# ╟─7d13a231-8ee5-459b-9c0e-dd81ab87bda1
# ╠═e3b69a7f-cad0-4535-b691-a2d53c586581
# ╠═8d49fedf-1bf6-45cb-b85b-8b438663452e
# ╠═892694bf-e8ef-4a34-9239-394754b75a16
# ╠═11f890ef-3091-4886-834c-51e2e619228a
# ╠═5d0dc6b4-24e1-413f-8790-33d6610e09e6
# ╠═f9bda6f5-97e4-4ea5-9e7a-8826b528dd41
# ╠═ae3a2af0-8fae-11ee-288d-732288c2bc04
# ╠═28a84a7a-36b8-4319-9eed-693491723f6d
# ╠═444a8791-93f8-4ba1-906f-858d365092f7
# ╠═cd4671f0-cdbe-4bd7-a04e-d51cbbfeb5d4
# ╠═7dc54541-0664-464b-92fa-901e7ed512b6
# ╠═f8b54566-c037-4c9d-b151-a4fd2ac2b76e
# ╠═d022a766-657e-4566-a3de-7d25a4810d74
# ╠═03509c0a-6122-4005-9686-8aca4bd38f99
# ╠═0828e154-50ba-4dee-be43-df880e34ab1b
# ╠═a42f66cb-3dc1-46cb-be94-38979d815d78
# ╠═12028e54-7bda-41ab-b1a6-916f1e2619ca
# ╠═767418fa-ed77-4835-8c5c-0cc89d8d71bc
# ╠═f5a4bdbe-863c-4397-a163-c547b397c46c
# ╠═b1027abd-b4af-42d2-8fe3-32a5c39b5b77
# ╠═38b8e924-6ec1-41f0-a972-756ce1c5cd4d
# ╠═1632e60f-5568-4873-8900-84a0608c9dab
# ╠═3fb8271b-bcbd-4a38-9589-8723c490fbba
# ╠═6510d727-5612-4605-9f35-9ce8eda98f02
# ╠═bdafc085-e15e-4130-bca1-7abf5e724025
# ╠═37c5091f-3cf5-46b0-8874-59a047a191e6
# ╠═3dba0928-09f3-4108-9d55-5124f3ec59c1
# ╠═64f8cb6c-9c8a-4f08-b970-697708da228b
# ╠═dcd92fb1-33aa-43f7-b0aa-396df9faf4de
# ╠═93168c75-eeba-4b64-9054-35981c93701b
# ╠═180bce23-cc3c-4869-ae30-2cd0d61aa34f
# ╠═df917296-aab3-424a-9371-0ba849c51189
# ╠═8e6032a0-0de4-4fef-9849-60e963daf952
# ╠═1538327f-dc2a-494c-a146-d725f91c0a34
# ╠═74413262-c13e-42c7-be41-6b8b829e9b5e
# ╠═3a5b4094-c465-4041-8486-f5c2574980b9
# ╠═bcd788ba-c369-4993-ae62-d6075dda282b
# ╠═fe154fd1-e45a-46ac-afa1-e008abf6e62d
# ╠═7276ef68-34d6-4dd8-9a25-95e86e6b6183
# ╠═d8ac83da-1829-4064-a08f-89041e60bdc2
# ╠═007e1141-4a33-4538-a089-b3238117af3d
# ╠═a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
# ╠═3792cee9-3788-40f9-acf4-66f7b6f5cfcf
# ╠═a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
# ╠═9f01b5b4-d086-44d7-83d1-90f9d8ed21b6
# ╠═1be2b0ac-39c0-467f-8a27-eae0b176c753
# ╠═61fddba8-ec75-44c0-ab1e-a91967843075
# ╠═371c28b5-90e0-4f69-ac9b-e706fe960d07
# ╠═e155b5ea-da51-45d4-a951-1b3aa1f28b63
# ╠═573c07ba-2c7a-4b95-9ed6-dd0563c257e5
# ╠═545a3bc2-f399-41eb-99fc-1ea40239d856
# ╠═387f5555-c3f3-4b80-849d-8d65491fc6f8
# ╠═935faa9d-990d-4bfa-b70c-b11bc824e20b
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
