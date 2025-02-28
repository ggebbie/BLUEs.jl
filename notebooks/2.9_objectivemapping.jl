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

# ╔═╡ a72352ab-cdfb-42db-85de-ea88132ba9dd
md""" # 2.9 Objective mapping """

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

# ╔═╡ e7ceeb0e-155d-40b2-ada0-c60306f1cbb9
# turn the 2D grid into a 1D bookkeeping system

# ╔═╡ 5952da64-d65c-47fc-8071-9a9d8b14bece
r = [(rx[i],ry[j])  for j in eachindex(ry) for i in eachindex(rx)] # a vector that gives x location as first element, y location as second element

# ╔═╡ 4854bf8b-21d4-42a3-bfe9-3cc77d790544
# location of first gridpoint
r[1]

# ╔═╡ e42ecaa5-052d-47e3-aaea-9905f6c4b508
# location of second gridpoint
r[2]

# ╔═╡ 96ad4635-70a1-4e41-912e-28cca0b33e75
#location of 51st gridpoint
r[51]

# ╔═╡ 0828e154-50ba-4dee-be43-df880e34ab1b
# set lengthscales
Lx = 300km; Ly = 100km;

# ╔═╡ 16a42d32-853b-417c-845b-98da700e1be0
Rρ = [exp( -((r[i][1]-r[j][1])/Lx)^2 - ((r[i][2] - r[j][2])/Ly)^2) for j in eachindex(r), i in eachindex(r)] # doesn't take advantage of symmetry

# ╔═╡ 8cd47f91-c3b5-4522-8ac3-323e22601633
# show a slice

# ╔═╡ 412e835f-d518-45c1-b6f0-38e58a3d8bfc
scatter(Rρ[20,:],xlabel="grid point (1D index)",ylabel="ρ") # not simply monotonic

# ╔═╡ 12028e54-7bda-41ab-b1a6-916f1e2619ca
# turn correlation matrix into autocovariance matrix: requires variance info
σ² = (50cm)^2

# ╔═╡ 767418fa-ed77-4835-8c5c-0cc89d8d71bc
# Here I need to produce some "synthetic" data
# these are the extra steps

# ╔═╡ ee023446-b4df-4be6-abcb-694f848d4c27
Rρ_posdef = Rρ + 1e-6I

# ╔═╡ f5a4bdbe-863c-4397-a163-c547b397c46c
Rρ12 = cholesky(Rρ_posdef) # cholesky 

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

# ╔═╡ 93168c75-eeba-4b64-9054-35981c93701b
# how much observational noise
σₙ = 1cm

# ╔═╡ 180bce23-cc3c-4869-ae30-2cd0d61aa34f
# get the noise covariance
Cnn = Diagonal(fill(σₙ^2,Nobs))

# ╔═╡ 7276ef68-34d6-4dd8-9a25-95e86e6b6183
# Sample the true field
y = E*xtrue + σₙ*randn(Nobs) 

# ╔═╡ 1538327f-dc2a-494c-a146-d725f91c0a34
# add the data to the plot
rxobs = [robs[oo][1] for oo in eachindex(robs)]

# ╔═╡ 74413262-c13e-42c7-be41-6b8b829e9b5e
ryobs = [robs[oo][2] for oo in eachindex(robs)]

# ╔═╡ 007e1141-4a33-4538-a089-b3238117af3d
begin
	contour(rx,ry,ustrip.(xtruefield'),xlabel="zonal distance",ylabel="meridional distance",clabels=true,title="true SSH with $Nobs obs",fill=true)
	scatter!(rxobs,ryobs,zcolor=ustrip.(y),label="y",cbar=false,markersize=6) 
end

# ╔═╡ a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
## Now we have synthetic observations
# Let's see if the true solution can be backed out from the sparse obs.


# ╔═╡ 3792cee9-3788-40f9-acf4-66f7b6f5cfcf
Cxx = σ²*Rρ_posdef

# ╔═╡ 32a29c48-1dc1-485f-a343-71a8949aef3d
Cxy = Cxx*transpose(E)

# ╔═╡ 5ded26d7-1871-4aa5-b8f2-e6905cb38e81
Cyy = E*Cxx*transpose(E) + Cnn

# ╔═╡ 7be7ebe9-4429-4c73-8a64-4bc868b7e442
# x̃ = vec(UnitfulMatrix(Cxy)*(UnitfulMatrix(Cyy)\y)) # would require UnitfulLinearAlgebra
#x̃ = Cxy*(Cyy\y) # not handled by Unitful.jl
x̃ = Cxy*inv(Cyy)*y # a slow-ish workaround

# ╔═╡ d3dbbaa7-7b72-49fa-8aac-fa4d02259806
inv(Cyy)*y

# ╔═╡ 61fddba8-ec75-44c0-ab1e-a91967843075
x̃field = reshape(x̃,Nx,Ny) # turn it back into 2D

# ╔═╡ 371c28b5-90e0-4f69-ac9b-e706fe960d07
begin
	contourf(rx,ry,x̃field',xlabel="zonal distance",ylabel="meridional distance",title="SSH objective map")
	scatter!(rxobs,ryobs,zcolor=ustrip.(y),label="y",ms=6,cbar=false) 
end

# ╔═╡ 3dce7547-db3f-48b5-aff4-fce02bbb012f
## next, calculate the map uncertainty
# P = Cxx - Matrix(Cxx * transpose(E) * (UnitfulMatrix(Cyy) \ UnitfulMatrix(E*Cxx)))
P = Cxx - Cxx * transpose(E) * inv(Cyy) * E*Cxx

# ╔═╡ 7b59fc48-dd30-4f84-8f7c-6d00b2ab85b7
σₓ̃ = .√(reshape(diag(P),Nx,Ny)) # turn uncertainty back into 2D

# ╔═╡ 573c07ba-2c7a-4b95-9ed6-dd0563c257e5
begin
	contourf(rx,ry,σₓ̃',xlabel="zonal distance",ylabel="meridional distance",title="SSH uncertainty",clabels=true,cbar=false)
	scatter!(rxobs,ryobs,color=:white,label="y",ms=6) 
end

# ╔═╡ c3b5327b-0cbc-4ae4-873c-ccc2f2d47123
contourf(rx,ry,σₓ̃',xlabel="zonal distance",ylabel="meridional distance",title="SSH uncertainty",clabels=true)

# ╔═╡ 545a3bc2-f399-41eb-99fc-1ea40239d856
md""" ## how well is the data fit? """

# ╔═╡ 387f5555-c3f3-4b80-849d-8d65491fc6f8
scatter(y,E*x̃,xlabel="y",ylabel="ỹ")

# ╔═╡ 935faa9d-990d-4bfa-b70c-b11bc824e20b
scatter(y,y-E*x̃,xlabel="y",ylabel="ñ")

# ╔═╡ f9975f99-3795-4f7e-95ea-fb0ce894d4ec
plotly()

# ╔═╡ Cell order:
# ╟─a72352ab-cdfb-42db-85de-ea88132ba9dd
# ╟─7d13a231-8ee5-459b-9c0e-dd81ab87bda1
# ╠═e3b69a7f-cad0-4535-b691-a2d53c586581
# ╠═8d49fedf-1bf6-45cb-b85b-8b438663452e
# ╠═892694bf-e8ef-4a34-9239-394754b75a16
# ╠═11f890ef-3091-4886-834c-51e2e619228a
# ╠═5d0dc6b4-24e1-413f-8790-33d6610e09e6
# ╠═f9bda6f5-97e4-4ea5-9e7a-8826b528dd41
# ╠═ae3a2af0-8fae-11ee-288d-732288c2bc04
# ╠═28a84a7a-36b8-4319-9eed-693491723f6d
# ╠═e7ceeb0e-155d-40b2-ada0-c60306f1cbb9
# ╠═5952da64-d65c-47fc-8071-9a9d8b14bece
# ╠═4854bf8b-21d4-42a3-bfe9-3cc77d790544
# ╠═e42ecaa5-052d-47e3-aaea-9905f6c4b508
# ╠═96ad4635-70a1-4e41-912e-28cca0b33e75
# ╠═0828e154-50ba-4dee-be43-df880e34ab1b
# ╠═16a42d32-853b-417c-845b-98da700e1be0
# ╠═8cd47f91-c3b5-4522-8ac3-323e22601633
# ╠═412e835f-d518-45c1-b6f0-38e58a3d8bfc
# ╠═12028e54-7bda-41ab-b1a6-916f1e2619ca
# ╠═767418fa-ed77-4835-8c5c-0cc89d8d71bc
# ╠═ee023446-b4df-4be6-abcb-694f848d4c27
# ╠═f5a4bdbe-863c-4397-a163-c547b397c46c
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
# ╠═7276ef68-34d6-4dd8-9a25-95e86e6b6183
# ╠═1538327f-dc2a-494c-a146-d725f91c0a34
# ╠═74413262-c13e-42c7-be41-6b8b829e9b5e
# ╠═007e1141-4a33-4538-a089-b3238117af3d
# ╠═a00b4df5-e23c-4f1e-9188-c7305ca3bf1d
# ╠═3792cee9-3788-40f9-acf4-66f7b6f5cfcf
# ╠═32a29c48-1dc1-485f-a343-71a8949aef3d
# ╠═5ded26d7-1871-4aa5-b8f2-e6905cb38e81
# ╠═7be7ebe9-4429-4c73-8a64-4bc868b7e442
# ╠═d3dbbaa7-7b72-49fa-8aac-fa4d02259806
# ╠═61fddba8-ec75-44c0-ab1e-a91967843075
# ╠═371c28b5-90e0-4f69-ac9b-e706fe960d07
# ╠═3dce7547-db3f-48b5-aff4-fce02bbb012f
# ╠═7b59fc48-dd30-4f84-8f7c-6d00b2ab85b7
# ╠═573c07ba-2c7a-4b95-9ed6-dd0563c257e5
# ╠═c3b5327b-0cbc-4ae4-873c-ccc2f2d47123
# ╠═545a3bc2-f399-41eb-99fc-1ea40239d856
# ╠═387f5555-c3f3-4b80-849d-8d65491fc6f8
# ╠═935faa9d-990d-4bfa-b70c-b11bc824e20b
# ╠═84a3f486-6566-4dad-99f0-944d8b67f6a0
# ╠═82de0964-3a0a-4580-bb83-b8d43a9e7d1b
# ╠═71178a40-391d-4abc-8a7a-d03c90c28952
# ╠═793180e7-d32d-4b09-8a04-1861b3306a46
# ╠═7bcfbc9d-cc6b-49f3-a2af-dc9356afceb1
# ╠═f9975f99-3795-4f7e-95ea-fb0ce894d4ec
