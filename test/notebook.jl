### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ 1ccfc590-90f7-11ed-1e7c-dfaec0816597
begin
	ENV["UNITFUL_FANCY_EXPONENTS"] = true
	import Pkg; Pkg.activate("../")

	using BLUEs
	using Test
	using LinearAlgebra
	using Statistics
	using Unitful
	using UnitfulLinearAlgebra
	using Measurements
	using ToeplitzMatrices
	using SparseArrays
end


# ╔═╡ 1aa4af78-f0ee-4d5a-916b-c6e0decdd300
md""" 
# Examples from runtests
"""

# ╔═╡ 970a7544-c5f2-4f2e-ba52-316f67554482
	const permil = u"permille"; const K = u"K"; const	K² = u"K^2"; m = u"m"; s = u"s"; MMatrix = BestMultipliableMatrix;

# ╔═╡ f862f30e-ea14-41a1-93ee-63d92380cd8d
md""" 
## Example 1: Error propagation

Generate an array of Gaussian distributed numbers $\mathbf{a}$, with uncertainty between 0 and 1.

Generate a matrix $\mathbf{E}$ that will propagate our error
"""

# ╔═╡ 7dd4252c-f2d0-4839-9052-981b86e83804
let
	begin 
		@show a = randn(5) .± rand(5)
		E = randn(5,5)
		println("E = ")
		display(E)
		println()
		
		@show aerr = Measurements.uncertainty.(a);
		@show x = Estimate(Measurements.value.(a),
				 Diagonal(aerr.^2))
		println()
		
		@show E*a
		@show E*x
		println()

		@show Measurements.value.(E*a) ≈ (E*x).v
		@show Measurements.uncertainty.(E*a) ≈ (E*x).σ
		
	end
end


# ╔═╡ 6b9668b2-dcdc-4ac9-b29c-feeb155ec42f
md"""
## Example 2: mixed signals, dimensionless matrix 

Solve $\mathbf{y = Ex}$ with uncertainty/weighting $\mathbf{C_nn}$ given some observations $\mathbf{y}$ with units and uncertainty and given some dimensionless matrix $\mathbf{E}$

Populates an OverdeterminedProblem object and solves according to textbook solution 1.208/1.209 and Hessian solution (see documentation for equations)

Note: each of these methodologies can take in prior ($\mathbf{C_{xx}}^{-1}$ and $\mathbf{x_0}$) information (but doesn't for this example)
"""

# ╔═╡ 98024f12-fc38-40e3-a4a4-c7678940f5ee
let
	begin
	N = 2;
	M = 2;
	@show σₓ = rand()
	println()
	E = MMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
	println("E = ")
	display(E)
	println()
	
	Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E),exact=true)
	println("Cnn⁻¹ = ")
	display(Cnn⁻¹)
	println()
	
	@show x = randn(N)m
	@show y = E*x
	println() 
	
	@show problem = OverdeterminedProblem(y,E,Cnn⁻¹)
	@time @show x̃ = solve(problem,alg=:hessian)
	@time @show x̃ = solve(problem,alg=:textbook)
	@show x ≈ x̃.v
	@show cost(x̃,problem) < 1e-5 # no noise in obs
	@show x ≈ pinv(problem) * y # inefficient way to solve problem
	end
end


# ╔═╡ 2263f882-a64d-4d18-bff5-61abac8199a2


# ╔═╡ a820781d-2e6a-4b4d-a1cb-f824c293de57
md"""
## Example 3: trend analysis, left-uniform matrix

For a temporal array $\mathbf{t}$ that ranges from 0:9 [s], generate an array of "observations", in [m] with a random slope [m/s] and y-intercept [m]. Add some randomly distributed noise. 
Then, construct the model matrix $\mathbf{E}$, of the following form 

$\mathbf{E} = \begin{bmatrix}
1 && t_1 [s]\\ 
1 && t_2 [s]\\ 
\vdots && \vdots \\ 
1 && t_N [s]\\ 
\end{bmatrix}$
"""

# ╔═╡ f3ddcd82-b84c-4504-bbd5-309ec62a10c3
let 
	begin
		    numobs = 10  # number of obs
	        @show t = collect(0:1:numobs-1)s
	
	        @show a = randn()m # intercept
	        @show b = randn()m/s # slope
	
	        @show y = a .+ b.*t .+ randn(numobs)m
	
	        @show E = MMatrix(hcat(ones(numobs),ustrip.(t)),fill(m,numobs),[m,m/s],exact=true)
	        @show Cnn⁻¹ = Diagonal(fill(1.0,numobs),fill(m^-1,numobs),fill(m,numobs),exact=true)
	
	        problem = OverdeterminedProblem(y,E,Cnn⁻¹)
	        @show x̃ = solve(problem,alg=:textbook)
	        x̃1 = solve(problem,alg=:hessian) #solutions will be identical 
	        @show cost(x̃,problem) < 3numobs; # rough guide, could get unlucky and not pass
	end
end


# ╔═╡ Cell order:
# ╟─1aa4af78-f0ee-4d5a-916b-c6e0decdd300
# ╠═1ccfc590-90f7-11ed-1e7c-dfaec0816597
# ╠═970a7544-c5f2-4f2e-ba52-316f67554482
# ╠═f862f30e-ea14-41a1-93ee-63d92380cd8d
# ╠═7dd4252c-f2d0-4839-9052-981b86e83804
# ╠═6b9668b2-dcdc-4ac9-b29c-feeb155ec42f
# ╠═98024f12-fc38-40e3-a4a4-c7678940f5ee
# ╠═2263f882-a64d-4d18-bff5-61abac8199a2
# ╠═a820781d-2e6a-4b4d-a1cb-f824c293de57
# ╠═f3ddcd82-b84c-4504-bbd5-309ec62a10c3
