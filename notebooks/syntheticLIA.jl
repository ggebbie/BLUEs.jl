### A Pluto.jl notebook ###
# v0.19.26

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
	using Revise
	using BLUEs, Unitful, DimensionalData, ToeplitzMatrices, Plots, UnitfulLinearAlgebra, LinearAlgebra, SparseArrays, Statistics, Measurements
	using DimensionalData:@dim
	using InteractiveUtils, PlutoUI
	plotlyjs()
end


# ╔═╡ c2b6940f-6eb3-4aa9-9b3f-58f02d8270cb
begin
	include("../test/test_functions.jl")
	const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";yr = u"yr"; °C = u"°C"
	ENV["UNITFUL_FANCY_EXPONENTS"] = true

	#define our dimensions 
	@dim YearCE "years Common Era"
	@dim SurfaceRegion "surface location"
	@dim InteriorLocation "interior location"
	@dim StateVar "state variable"
end


# ╔═╡ a29f7c8e-b86f-4ae2-ac0c-b549b9b54da2
md"""
# Synthetic LIA 
This notebook demonstrates how to solve a toy problem version of the OPT2k problem, in three different cases 
- **one sediment core (recording temperature)** affected by **temperature at two different surface regions** each exhibiting a 'Common Era-esque' temperature pattern (warm during the MCA, cool during the LIA, warming after) of different magnitudes. One region will be 'N.Atl.' and will have a slightly earlier MCA and LIA, the other region will be 'Antarctica' and will have a slightly later MCA and LIA. 
- **two sediment cores (recording temperature)** affected by the **temperature at two surface regions**
- **two sediment cores (recording d18Oc)** affected by the **temperature and d18Ow at two surface regions** (assume warm + fresh during MCA, cool + salty during LIA) 
"""

# ╔═╡ 780e8541-b757-4f28-bccf-eda77481e2d1
md"""
# One core, two regions, no lagged circulation 
For this example, 60% of the water at the sediment core site will be sourced from N.Atl., and 40% will be sourced from Antarctica. 
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

# ╔═╡ 2f44ad50-7cb7-43e0-9d4b-e2e1af919e70
md"""
We'll use objective mapping to generate two time series (one for Antarctica, one for the North Atlantic) that fits some points selected to give us a MCA-LIA transition signal with a defined autocovariance function. 

We'll propagate these two signals to the sediment core site and then add some noise. 
"""

# ╔═╡ cff90ba8-9abc-43fc-befd-c33ce85f36b7
begin
	#generate data for the pattern described above 
	obstimes_NATL = [1000yr, 1300yr, 1600yr, 1800yr, 2000yr] #earlier
	obstimes_ANT = [1000yr, 1400yr, 1700yr,1900yr,2000yr] #later
	obs_NATL = UnitfulMatrix(uconvert.(K, [1.5; 1.7; 1.; 1.5; 1.8]°C)) #warmer, greater magnitude signal
	obs_ANT =  UnitfulMatrix(uconvert.(K, [0.5; 0.6; 0.4; 0.5; 0.6]°C)) #colder, smaller signal magnitude
	σ_obs = 0.1K

	#use objective mapping to generate a signal with covariance defined by the following function (decays ~200yrs)
	ρ(t) = exp.(-(years.-first(t)).^2/(200yr)^2) 
	x_NATL = synthetic_timeseries(obs_NATL, obstimes_NATL, σ_obs, years, ρ(obstimes_NATL), K)
	x_ANT = synthetic_timeseries(obs_ANT, obstimes_ANT, σ_obs, years, ρ(obstimes_ANT), K)
	#x: TRUE surface conditions 
	x = DimArray(transpose(hcat(x_NATL.v, x_ANT.v)K), (SurfaceRegion(surfaceregions), Ti(years)))
end

# ╔═╡ d7c9ed1b-2557-4439-abbb-d7534a7917fa
begin
	y = M*x #TRUE surface conditions propagated to depth 
	x₀ = copy(x) .* 0 .+ y[end] #first guess - mean of y
	flipped_mult(a,b) = b*a #impulseresponse needs it to be defined so x is the first arg. 
	#generate E, the matrix that maps between vectorized x and the output of M*x 
	E = impulseresponse(flipped_mult, x₀, M) 

	#sprinkle in a little noise, but just a little! we only have one core, after all 
	y_contam = y .+ DimArray(randn(size(y))K ./ 50, y.dims)
	plot(years, vec(y[At(:loc1), :]), label = "yₜᵣᵤₑ")
	plot!(years, vec(y_contam[At(:loc1), :]), ylabel = "Temperature", title = "Timeseries at Core Site", label = "y", xlabel = "Time")
	#plot(ustrip.(years), ustrip.(vec(y[At(:loc1), :])), label = "yₜᵣᵤₑ")
	#plot!(ustrip.(years), ustrip.(vec(y_contam[At(:loc1), :])), ylabel = "Temperature [K]", title = "Timeseries at Core Site", label = "y", xlabel = "Time [years]")
end

# ╔═╡ f54a5e6c-6ded-4ae5-8213-834975c38db7
md"""
Then define a noise covariance matrix and a first guess covariance matrix. Both will be diagonal with some defined constant variance. 
"""

# ╔═╡ b8e76a08-c431-494d-bd82-1e5034b1bddf
md"""
# Two cores, two regions, no lagged circulation
Repeat the above process, but now we have two sites, one with the same relationship to ANT and NATL as above, and one that is 70% NATL, 30% ANT. 
"""

# ╔═╡ 6fd318f0-e747-4f51-994c-45a331810740
md"""
# Plot the solutions of these tests 
The one core case is very underdetermined, and changing our confidence in our observations does very little to change the result. 

The two core case is less underdetermined, and by toggling the σₙ slider, we can show that when our confidence in our data is greater (σₙ small), we introduce more noise into our reconstruction, but our solution approaches the truth. Vice versa, when our confidence in our data is less (σₙ big), we reconstruct a smoother solution, but it's biased from the truth. 
"""

# ╔═╡ 02c5d686-6600-47fc-ba7b-6b5dec5bd701
begin
	σₓ = 5
	nslider = @bind σₙ Slider(0.001:0.05:3, show_value = true, default = 0.1)
end

# ╔═╡ fe12f17f-f461-40c1-b0cd-26641f318e51
begin
	#generate noise covariance and first guess covariance (tapering) matrices 
	Cnn = Diagonal(fill(σₙ^2, length(y)), fill(K, length(y)), fill(K^-1, length(y)))
	Cxx = Diagonal(fill(σₓ^2, length(x)), fill(K, length(x)), fill(K^-1, length(x)))
	#solve problem 
	problem = UnderdeterminedProblem(UnitfulMatrix(vec(y_contam)), E, Cnn, Cxx, x₀)
	x̃ = solve(problem)
	#make first guess into dimensional estimate so we can use same plotting syntax
	x₀DE = DimEstimate(vec(x₀), Cxx, dims(x₀))
	#we will plot this solution with the next solution 
end

# ╔═╡ 6b4a307b-0cc4-4e8f-8aa7-bcba8ac07265
begin
	locs2 = [:loc1, :loc2]
	M2 = DimArray([0.6 0.4; 0.3 0.7], (InteriorLocation(locs2), SurfaceRegion(surfaceregions)))
	y2 = M2*x
	y2_contam = y2 .+ DimArray(randn(size(y2))K ./ 10, y2.dims)
	x₀2 = copy(x) .* 0 .+ y2[end] #first guess - mean of y
	E2 = impulseresponse(flipped_mult,x₀2, M2)
	Cnn2 = Diagonal(fill(σₙ.^2,length(y2)),vec(unit.(y2)),vec(unit.(y2)).^-1)
    Cxx2 = Diagonal(fill(σₓ.^2,length(x₀2)),vec(unit.(x₀2)),vec(unit.(x₀2)).^-1)
	problem2 = UnderdeterminedProblem(UnitfulMatrix(vec(y2_contam)),E2,Cnn2,Cxx2,x₀2)
    x̃2 = solve(problem2)
	y2DE = DimEstimate(vec(y2), Cnn2, dims(y2))
end

# ╔═╡ 41ea9b57-3901-4581-b9b4-e2f9afb16ce7
#xslider = @bind σₓ Slider(0.0001:0.1:5, show_value = true, default = 0.1)

# ╔═╡ 1dfba472-cd25-4641-888f-51d6368fa7b1
begin
		@show datacost(x̃, problem)
		@show controlcost(x̃, problem)
		@show datacost(x̃2, problem2)
		@show controlcost(x̃2, problem2)
end

# ╔═╡ 77b2f73b-5c96-49f0-9c93-9846c6579759
begin
	darkcolors = NamedTuple{Tuple(surfaceregions)}([:orange, :lightskyblue])
	lightcolors = NamedTuple{Tuple(surfaceregions)}([:lightgoldenrod1, :cyan])
end

# ╔═╡ 73ece91f-7309-43e1-8832-ecfcfe32a5bf
begin
	l = @layout [a b]
	lw = 5
	p = scatter(ustrip.(years), ustrip.(vec(y)), yerror = ustrip.(sqrt.(diag(Cnn))), label = "y", linewidth = 3, color = :black)
	for region in surfaceregions
		plot!(ustrip.(years), ustrip.(vec(Measurements.value.(x̃.x[At(region), :]))), ribbon = ustrip.(vec(Measurements.uncertainty.(x̃.x[At(region), :]))), label = "x̃:" * eval(region), linewidth = lw, color = darkcolors[region]) 
		#plot!(years, vec(Measurements.value.(x₀DE.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x₀DE.x[At(region), :])), label = "x₀:" * eval(region))
	end
	#true values 
	plot!(ustrip.(years), ustrip.(vec(x_NATL.v)), label = "xₙₐₜₗ", legend = :outertopright, linewidth = lw, color = lightcolors[:NATL])
	plot!(ustrip.(years), ustrip.(vec(x_ANT.v)), label = "xₐₙₜ", linewidth = lw, color = lightcolors[:ANT])

	p2 = plot()
	for loc in locs2 
		scatter!(ustrip.(years), ustrip.(vec(Measurements.value.(y2DE.x[At(loc), :]))), yerror = ustrip.(vec(Measurements.uncertainty.(y2DE.x[At(loc), :]))), label = "", linewidth = 3, color = :black)
	end
	for region in surfaceregions
		plot!(ustrip.(years), ustrip.(vec(Measurements.value.(x̃2.x[At(region), :]))), ribbon = ustrip.(vec(Measurements.uncertainty.(x̃2.x[At(region), :]))), label = "", linewidth = lw, color = darkcolors[region]) 
		#plot!(years, vec(Measurements.value.(x₀DE.x[At(region), :])), ribbon = vec(Measurements.uncertainty.(x₀DE.x[At(region), :])), label = "x₀:" * eval(region))
	end
	#true values 
	plot!(ustrip.(years), ustrip.(vec(x_NATL.v)), label = "", legend = :outertopright, linewidth = lw, color = lightcolors[:NATL])
	plot!(ustrip.(years), ustrip.(vec(x_ANT.v)), label = "", linewidth = lw, color = lightcolors[:ANT])

	plot(p, p2, layout = l)
end

# ╔═╡ 535bde0c-dd75-42f2-bcc8-e3830699a7de
md"""
# Two state variables
Now, repeat the above test, but with two state variables. Assume both oceans increase in d18O (get saliter) and then decrease in d18O (get fresher)
"""

# ╔═╡ a76301d4-2831-4902-8b5f-2148451d5eef
begin
	statevars = [:θ, :d18O]
	x1 = x
	#obstimes = [1000yr, 1300yr, 1600yr, 1800yr, 2000yr]
	#obs_NATL = UnitfulMatrix(uconvert.(K, [1.5; 1.7; 1.; 1.5; 1.8]°C))
	#obs_ANT =  UnitfulMatrix(uconvert.(K, [0.5; 0.6; 0.4; 0.5; 0.6]°C))
	obs_NATL2 = UnitfulMatrix([-0.1; -0.15; -0.05; -0.1; -0.2]permil)
	obs_ANT2 = UnitfulMatrix([0.5; 0.45; 0.55; 0.5; 0.3]permil)
	σ_obs2 = 0.05permil
	#ρ = exp.(-(years.-first(obstimes)).^2/(200yr)^2) #starts at 1, decreases to 0.36
	x_NATL2 = synthetic_timeseries(obs_NATL2, obstimes_NATL, σ_obs2, years, ρ(obstimes_NATL), permil)
	x_ANT2 = synthetic_timeseries(obs_ANT2, obstimes_ANT, σ_obs2, years, ρ(obstimes_ANT), permil)
	x2 = DimArray(transpose(hcat(x_NATL2.v, x_ANT2.v)permil), (SurfaceRegion(surfaceregions), Ti(years))) 
	end


# ╔═╡ 168ce97e-9e70-46d5-963b-1429b627c224
begin
	α = -0.2(permil/K)
	β = 3.5permil
	offset = 0.27permil
end

# ╔═╡ 7913c5e7-4232-44c7-bc1b-85e8395ec848
begin 
	x3 = DimArray(cat(hcat(x_NATL.v, x_ANT.v)K .- 273.15K, hcat(x_NATL2.v, x_ANT2.v)permil, dims = 3), (Ti(years),SurfaceRegion(surfaceregions), StateVar(statevars)))
	propagate(x, M, α) = α * ((x[:, :, At(:θ)]) * transpose(M)) .+ x[:, :, At(:d18O)] * transpose(M)
	y3 = propagate(x3, M2, α) .+ β .- offset
	#y3test = α * (M2 * x1 .- 273.15K) .+ β .+ M2 * x2 .- offset #same as above
	x₀3 =  DimArray(cat((vec(x3[At(2000yr), :,  At(:θ)]) .* ones((2, length(years))))', (vec(x3[At(2000yr), : , At(:d18O)]) .* ones((2, length(years))))', dims =3), (Ti(years),SurfaceRegion(surfaceregions), StateVar(statevars)))
end

# ╔═╡ fb7988aa-b5a4-4fe0-b44e-86d291117eaf
begin
	l5 = @layout [a b]
	y3_contam = y3 + DimArray(randn(size(y3))permil ./ 10, y3.dims)
	p5 = Vector{Any}(undef, 2)
	for (i, l) in enumerate(locs2)
		p5[i] = plot(ustrip.(vec(y3[:, At(l)])), label = i == 1 ? "yₜᵣᵤₑ" : "")
		plot!(ustrip.(vec(y3_contam[:, At(l)])), label = i == 1 ? "y, contaminated" : "", ylabel = "d18Oc", title = "Location " * string(
			i), legend = :outertopright)
	end
	plot(p5..., layout = l5)
end

# ╔═╡ cefdf603-64b7-4588-bf54-cb043a4616c9
begin
	E3 = impulseresponse(propagate, x₀3, M2, α)
	#demonstrate that this is equivalent 
	#display(E3 * UnitfulMatrix(vec(x₀3)))
	#display(vec(propagate(x₀3, M2, α)))
end

# ╔═╡ b474cdc2-f007-42b0-970d-692d5ba45776
@bind σₙ3 Slider(0.001:0.05:1, show_value = true, default = 0.1)

# ╔═╡ 3049a85e-fb0e-4808-99bf-1118b34eae92
begin
	#σₙ3 = 0.07
	σₓ3_T = 1
	σₓ3_d = 0.1
	L = length(x3[:, :, At(:θ)])
	cxx_diagonal = vcat(fill(σₓ3_T, L), fill(σₓ3_d, L))
	Cnn3 = Diagonal(fill(σₙ3^2, length(y3)), unit.(vec(y3)), unit.(vec(y3_contam)).^-1)
	Cxx3 = Diagonal(cxx_diagonal, unit.(vec(x3)), unit.(vec(x3)).^-1)
	problem3 = UnderdeterminedProblem(UnitfulMatrix(vec(y3) .- β .+ offset), E3, Cnn3, Cxx3, x₀3)
	x̃3 = solve(problem3)
end

# ╔═╡ b265984e-320a-45c1-9c25-da5044a43ffe
begin
	l3 = @layout [a b;c d]
	
	selectdims = [[:, At(:NATL), At(:θ)], [:, At(:NATL), At(:d18O)], [:, At(:ANT), At(:θ)], [:, At(:ANT), At(:d18O)]]
	titles = ["NATL:θ", "NATL:d18O", "ANT:θ", "ANT:d18O"]
	p3 = Vector{Any}(undef, 4)
	for (i, s) in enumerate(selectdims)
		p3[i] = plot(Measurements.measurement.(vec(ustrip.(x̃3.x[s...]))), label = i == 1 ? "x̃" : "", ylabel = "Temperature [K]")
		plot!(ustrip.(vec(x3[s...])), label = i == 1 ? "xₜᵣᵤₑ" : "")
		plot!(ustrip.(vec(x₀3[s...])), label = i == 1 ? "x₀" : "", title = titles[i], legend = :outertopright)
	end
	plot(p3..., layout = l3)
end

# ╔═╡ ce96eddb-cbef-4a94-82ca-a5a67ea7107a
begin 
	ỹ3 = propagate(x̃3.x, M2, α) .+ β .- offset
	l4 = @layout [a b]
	p4 = Vector{Any}(undef, 2)
	titles4 = ["Location 1", "Location 2"]
	for (i, l) in enumerate(locs2) 
		p4[i] = plot(vec(ỹ3[:, At(l)]), label = i == 1 ? "ỹ" : "")
		plot!(vec(y3[:, At(l)]), label = i == 1 ? "yₜᵣᵤₑ" : "", title = titles4[i])
		plot!(vec(y3_contam[:, At(l)]), label = i == 1 ? "y, contam." : "", title = titles4[i], legend = :outertopright)
		
		#p4[i] = plot(Measurements.measurement.(ustrip.(vec(ỹ3[:, At(l)]))), label = i == 1 ? "ỹ" : "")
		#plot!(ustrip.(vec(y3[:, At(l)])), label = i == 1 ? "yₜᵣᵤₑ" : "", title = titles4[i])
		#plot!(ustrip.(vec(y3_contam[:, At(l)])), label = i == 1 ? "y, contam." : "", title = titles4[i], legend = :outertopright)
	end 
	plot(p4..., layout = l4)
end

# ╔═╡ Cell order:
# ╠═31029ab4-e82b-11ed-0571-2fbfeebb813b
# ╠═c2d82ba9-624a-46f6-a53d-97e3f00db657
# ╟─a29f7c8e-b86f-4ae2-ac0c-b549b9b54da2
# ╠═c2b6940f-6eb3-4aa9-9b3f-58f02d8270cb
# ╟─780e8541-b757-4f28-bccf-eda77481e2d1
# ╠═803e0cf4-3ad5-4e0c-8b91-d74aeb61bd5e
# ╟─2f44ad50-7cb7-43e0-9d4b-e2e1af919e70
# ╠═cff90ba8-9abc-43fc-befd-c33ce85f36b7
# ╠═d7c9ed1b-2557-4439-abbb-d7534a7917fa
# ╟─f54a5e6c-6ded-4ae5-8213-834975c38db7
# ╠═fe12f17f-f461-40c1-b0cd-26641f318e51
# ╠═b8e76a08-c431-494d-bd82-1e5034b1bddf
# ╠═6b4a307b-0cc4-4e8f-8aa7-bcba8ac07265
# ╟─6fd318f0-e747-4f51-994c-45a331810740
# ╠═02c5d686-6600-47fc-ba7b-6b5dec5bd701
# ╟─41ea9b57-3901-4581-b9b4-e2f9afb16ce7
# ╠═1dfba472-cd25-4641-888f-51d6368fa7b1
# ╠═73ece91f-7309-43e1-8832-ecfcfe32a5bf
# ╟─77b2f73b-5c96-49f0-9c93-9846c6579759
# ╟─535bde0c-dd75-42f2-bcc8-e3830699a7de
# ╠═a76301d4-2831-4902-8b5f-2148451d5eef
# ╠═7913c5e7-4232-44c7-bc1b-85e8395ec848
# ╠═fb7988aa-b5a4-4fe0-b44e-86d291117eaf
# ╠═168ce97e-9e70-46d5-963b-1429b627c224
# ╠═cefdf603-64b7-4588-bf54-cb043a4616c9
# ╠═3049a85e-fb0e-4808-99bf-1118b34eae92
# ╠═b474cdc2-f007-42b0-970d-692d5ba45776
# ╠═b265984e-320a-45c1-9c25-da5044a43ffe
# ╠═ce96eddb-cbef-4a94-82ca-a5a67ea7107a
