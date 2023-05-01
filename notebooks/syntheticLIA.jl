### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 31029ab4-e82b-11ed-0571-2fbfeebb813b
#activate package one level up from /notebooks
import Pkg; Pkg.activate(abspath(joinpath(pwd(), "..")))

# ╔═╡ c2d82ba9-624a-46f6-a53d-97e3f00db657
begin
	using BLUEs, Unitful, DimensionalData, ToeplitzMatrices, Plots, UnitfulLinearAlgebra, LinearAlgebra, SparseArrays, Statistics, Measurements
	using DimensionalData:@dim
	using InteractiveUtils, PlutoUI
end


# ╔═╡ c2b6940f-6eb3-4aa9-9b3f-58f02d8270cb
begin
	include("../test/test_functions.jl")
	const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";yr = u"yr"; °C = u"°C"
	ENV["UNITFUL_FANCY_EXPONENTS"] = true

	@dim YearCE "years Common Era"
	@dim SurfaceRegion "surface location"
	@dim InteriorLocation "interior location"

end


# ╔═╡ 780e8541-b757-4f28-bccf-eda77481e2d1
md"""
# One core, two regions, no lagged circulation 
"""

# ╔═╡ 803e0cf4-3ad5-4e0c-8b91-d74aeb61bd5e
begin
	NATL = "NATL"
	ANT = "ANT"
	surfaceregions = [:NATL, :ANT]
	years = collect((1000:50:2000)yr)
	n = length(surfaceregions)
	locs = [:loc1]
	M = DimArray([0.6 0.4], (InteriorLocation(locs), SurfaceRegion(surfaceregions)))
end

# ╔═╡ cff90ba8-9abc-43fc-befd-c33ce85f36b7
begin
	obstimes = [1000yr, 1300yr, 1600yr, 1800yr, 2000yr]
	obs_NATL = UnitfulMatrix(uconvert.(K, [1.5; 1.7; 1.; 1.5; 1.8]°C))
	obs_ANT =  UnitfulMatrix(uconvert.(K, [0.5; 0.6; 0.4; 0.5; 0.6]°C))
	σ_obs = 0.1K
	ρ = exp.(-(years.-first(obstimes)).^2/(200yr)^2) #starts at 1, decreases to 0.36
	x_NATL = synthetic_timeseries(obs_NATL, obstimes, σ_obs, years, ρ, K)
	x_ANT = synthetic_timeseries(obs_ANT, obstimes, σ_obs, years, ρ, K)
	x = DimArray(transpose(hcat(x_NATL.v, x_ANT.v)K), (SurfaceRegion(surfaceregions), Ti(years)))
end

# ╔═╡ d7c9ed1b-2557-4439-abbb-d7534a7917fa
begin
	y = M*x
	x₀ = copy(x) .* 0 .+ mean(y) #first guess - mean of y
	flipped_mult(a,b) = b*a
	E = impulseresponse(flipped_mult, x₀, M)
end

# ╔═╡ 02c5d686-6600-47fc-ba7b-6b5dec5bd701
nslider = @bind σₙ Slider(0.001:0.05:3, show_value = true, default = 0.1)

# ╔═╡ 41ea9b57-3901-4581-b9b4-e2f9afb16ce7
xslider = @bind σₓ Slider(0.0001:0.1:5, show_value = true, default = 0.1)

# ╔═╡ fe12f17f-f461-40c1-b0cd-26641f318e51
begin
	Cnn = Diagonal(fill(σₙ^2, length(y)), fill(K, length(y)), fill(K^-1, length(y)))
	Cxx = Diagonal(fill(σₓ^2, length(x)), fill(K, length(x)), fill(K^-1, length(x)))
	problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)), E, Cnn, Cxx, x₀)
	x̃ = solve(problem)
	#make first guess into dimensional estimate so we can use same plotting syntax
	x₀DE = DimEstimate(vec(x₀), Cxx, dims(x₀))
end

# ╔═╡ 77b2f73b-5c96-49f0-9c93-9846c6579759
begin
	darkcolors = NamedTuple{Tuple(surfaceregions)}([:orange, :lightskyblue])
	lightcolors = NamedTuple{Tuple(surfaceregions)}([:lightgoldenrod1, :cyan])
end

# ╔═╡ 72ce8f5d-7f67-4c07-a9c5-d323239c2ba7
md"""
# Two cores, two regions, no lagged circulation 
"""

# ╔═╡ 750159f4-7206-4ed6-9e48-f33b5ea87d22
begin
	locs2 = [:loc1, :loc2]
	M2 = DimArray([0.6 0.4; 0.3 0.7], (InteriorLocation(locs2), SurfaceRegion(surfaceregions)))
	y2 = M2*x
	x₀2 = copy(x) .* 0 .+ mean(y2) #first guess - mean of y
	E2 = impulseresponse(flipped_mult,x₀2, M2)
	Cnn2 = Diagonal(fill(σₙ.^2,length(y2)),vec(unit.(y2)),vec(unit.(y2)).^-1)
    Cxx2 = Diagonal(fill(σₓ.^2,length(x₀2)),vec(unit.(x₀2)),vec(unit.(x₀2)).^-1)
	problem2 = UnderdeterminedProblem(UnitfulMatrix(vec(y2)),E2,Cnn2,Cxx2,x₀2)
    x̃2 = solve(problem2)
	y2DE = DimEstimate(vec(y2), Cnn2, dims(y2))
	@show datacost(x̃2, problem2)
	@show controlcost(x̃2, problem2)
end

# ╔═╡ 1dfba472-cd25-4641-888f-51d6368fa7b1
begin
		@show datacost(x̃, problem)
		@show controlcost(x̃, problem)
		@show datacost(x̃2, problem2)
		@show controlcost(x̃2, problem2)
end

# ╔═╡ 73ece91f-7309-43e1-8832-ecfcfe32a5bf
begin
	l = @layout [a b]
	lw = 5
	p = scatter(years, vec(y), yerror = sqrt.(diag(Cnn)), label = "y", linewidth = 3, color = :black)
	for region in surfaceregions
		plot!(years, vec(Measurements.value.(x̃.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x̃.x[At(region), :])), label = "x̃:" * eval(region), linewidth = lw, color = darkcolors[region]) 
		#plot!(years, vec(Measurements.value.(x₀DE.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x₀DE.x[At(region), :])), label = "x₀:" * eval(region))
	end
	#true values 
	plot!(years, vec(x_NATL.v), label = "xₙₐₜₗ", legend = :outertopright, linewidth = lw, color = lightcolors[:NATL])
	plot!(years, vec(x_ANT.v), label = "xₐₙₜ", linewidth = lw, color = lightcolors[:ANT])

	p2 = plot()
	for loc in locs2 
		scatter!(years, vec(Measurements.value.(y2DE.x[At(loc), :])), yerror = vec(Measurements.uncertainty.(y2DE.x[At(loc), :])), label = "", linewidth = 3, color = :black)
	end
	for region in surfaceregions
		plot!(years, vec(Measurements.value.(x̃2.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x̃2.x[At(region), :])), label = "", linewidth = lw, color = darkcolors[region]) 
		#plot!(years, vec(Measurements.value.(x₀DE.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x₀DE.x[At(region), :])), label = "x₀:" * eval(region))
	end
	#true values 
	plot!(years, vec(x_NATL.v), label = "", legend = :outertopright, linewidth = lw, color = lightcolors[:NATL])
	plot!(years, vec(x_ANT.v), label = "", linewidth = lw, color = lightcolors[:ANT])

	plot(p, p2, layout = l)
end

# ╔═╡ 9c40e36c-37ee-46d4-b9cd-90edcc7f7f04
randn((1, 42)) * x̃2 

# ╔═╡ 38cf6662-f3bf-4f08-bc91-3df16ac77052
x̃2.dims

# ╔═╡ Cell order:
# ╠═31029ab4-e82b-11ed-0571-2fbfeebb813b
# ╠═c2d82ba9-624a-46f6-a53d-97e3f00db657
# ╠═c2b6940f-6eb3-4aa9-9b3f-58f02d8270cb
# ╟─780e8541-b757-4f28-bccf-eda77481e2d1
# ╠═803e0cf4-3ad5-4e0c-8b91-d74aeb61bd5e
# ╠═cff90ba8-9abc-43fc-befd-c33ce85f36b7
# ╠═d7c9ed1b-2557-4439-abbb-d7534a7917fa
# ╠═fe12f17f-f461-40c1-b0cd-26641f318e51
# ╠═02c5d686-6600-47fc-ba7b-6b5dec5bd701
# ╠═41ea9b57-3901-4581-b9b4-e2f9afb16ce7
# ╠═1dfba472-cd25-4641-888f-51d6368fa7b1
# ╠═73ece91f-7309-43e1-8832-ecfcfe32a5bf
# ╠═77b2f73b-5c96-49f0-9c93-9846c6579759
# ╠═72ce8f5d-7f67-4c07-a9c5-d323239c2ba7
# ╠═750159f4-7206-4ed6-9e48-f33b5ea87d22
# ╠═9c40e36c-37ee-46d4-b9cd-90edcc7f7f04
# ╠═38cf6662-f3bf-4f08-bc91-3df16ac77052
