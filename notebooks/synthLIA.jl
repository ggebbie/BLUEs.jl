using DrWatson
import Pkg; Pkg.activate("/home/brynn/Code/BLUEs.jl")
using BLUEs, Unitful, DimensionalData, ToeplitzMatrices, Plots, UnitfulLinearAlgebra, LinearAlgebra, SparseArrays, Statistics
using DimensionalData:@dim

include("../test/test_functions.jl")
const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";yr = u"yr"; °C = u"°C"
ENV["UNITFUL_FANCY_EXPONENTS"] = true

#following "source water inversion: one obs TIMESERIES, many surface regions, with NO circulation lag" on branch associated with issue 27
@dim YearCE "years Common Era"
@dim SurfaceRegion "surface location"
@dim InteriorLocation "interior location"

surfaceregions = [:NATL,:ANT]
years = collect((1000:50:2000)yr)
n = length(surfaceregions)
locs = [:loc1]

M = DimArray([0.6 0.4], (InteriorLocation(locs), SurfaceRegion(surfaceregions)))

#generate a timeseries based on a few fixed values
#on mean, assume NADW = 1.5°C, AABW = 0.5°C
#NADW will experience a stronger LIA signal than AABW
#signal - warm to MCA peak (1300yr), cool to LIA peak (1800yr), industrial warming
obstimes = [1000yr, 1300yr, 1600yr, 1800yr, 2000yr]
obs_NATL = UnitfulMatrix(uconvert.(K, [1.5; 1.7; 1.; 1.5; 1.8]°C))
obs_ANT =  UnitfulMatrix(uconvert.(K, [0.5; 0.6; 0.4; 0.5; 0.6]°C))
σ_obs = 0.1K
ρ = exp.(-(years.-first(obstimes)).^2/(200yr)^2) #starts at 1, decreases to 0.36
x_NATL = synthetic_timeseries(obs_NATL, obstimes, σ_obs, years, ρ, K)
x_ANT = synthetic_timeseries(obs_ANT, obstimes, σ_obs, years, ρ, K)

x = DimArray(transpose(hcat(x_NATL.v, x_ANT.v)K), (SurfaceRegion(surfaceregions), Ti(years)))
y = M*x

x₀ = copy(x) .* 0 .+ mean(y) #first guess - mean of y
flipped_mult(a,b) = b*a
E = impulseresponse(flipped_mult, x₀, M)

σₙ = 0.01
σₓ = 10
Cnn = UnitfulMatrix(Diagonal(fill(σₙ^2, length(y))), fill(K, length(y)), fill(K^-1, length(y)))
Cxx = UnitfulMatrix(Diagonal(fill(σₓ^2, length(x))), fill(K, length(x)), fill(K^-1, length(x)))
problem = UnderdeterminedProblem(UnitfulMatrix(vec(y)), E, Cnn, Cxx, x₀)
x̃ = solve(problem)

s1 = scatter(years, vec(x_NATL.v), label = "xₙₐₜₗ", legend = :outertopright)
scatter!(years, vec(x_ANT.v), label = "xₐₙₜ")
plot!(years, vec(y), ribbon = sqrt.(diag(Cnn)), label = "y")
plot!(years, x̃.x[At(:NATL), :], label = "x̃ₙₐₜₗ")
plot!(years, x̃.x[At(:ANT), :], label = "x̃ₐₙₜ")
#png(s1, plotsdir("lia"))
