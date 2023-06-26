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

# ╔═╡ 1ccfc590-90f7-11ed-1e7c-dfaec0816597
begin
	ENV["UNITFUL_FANCY_EXPONENTS"] = true
	import Pkg; Pkg.activate(".")
	using BLUEs
	using Test
	using LinearAlgebra
	using Statistics
	using Unitful
	using UnitfulLinearAlgebra
	using Measurements
	using ToeplitzMatrices
	using SparseArrays
	using Plots
	using Random
	using InteractiveUtils
	using PlutoUI
	using Pluto
	plotlyjs()
	#gr()
end


# ╔═╡ 1aa4af78-f0ee-4d5a-916b-c6e0decdd300
md""" 
# Examples from runtests
"""

# ╔═╡ 970a7544-c5f2-4f2e-ba52-316f67554482
	const permil = u"permille"; const K = u"K"; const	K² = u"K^2"; m = u"m"; s = u"s"; 

# ╔═╡ f862f30e-ea14-41a1-93ee-63d92380cd8d
md""" 
## Example 1: Error propagation

Generate an array of Gaussian distributed numbers $\mathbf{a}$, with uncertainty between 0 and 1.

Generate a matrix $\mathbf{E}$ that will propagate our error
"""

# ╔═╡ 7dd4252c-f2d0-4839-9052-981b86e83804
let
	begin 
		Random.seed!(1234)
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

		#one advantage of Measurements is that it has built-in Plots.jl functionality
		plot(E*a, xlabel ="Index", ylabel ="value", label = "Ea")
		#but we can reproduce the same behavior easily 
		plot!((E*x).v; yerror = (E*x).σ, label = "Ex")
		
	end
end


# ╔═╡ 6b9668b2-dcdc-4ac9-b29c-feeb155ec42f
md"""
## Example 2: mixed signals, dimensionless matrix 

Solve $\mathbf{y = Ex}$ where $\mathbf{y}$ is our measurement with uncertainty/weighting $\mathbf{C_nn}$ and $\mathbf{E}$ is some dimensionless matrix. 

Populates an OverdeterminedProblem object and solves according to textbook solution 1.208/1.209 and Hessian solution (see documentation for equations)

Note: each of these methodologies can take in prior ($\mathbf{C_{xx}}^{-1}$ and $\mathbf{x_0}$) information (but doesn't for this example)
"""

# ╔═╡ 98024f12-fc38-40e3-a4a4-c7678940f5ee
let
	begin
	Random.seed!(1234)
	N = 2;
	M = 2;
	@show σₓ = rand()
	println()
	E = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
	println("E = ")
	display(E)
	println()
	
	Cnn⁻¹ = Diagonal(fill(σₓ^-1,M),unitrange(E).^-1,unitrange(E))
	println("Cnn⁻¹ = ")
	display(Cnn⁻¹)
	println()
	
	@show x = UnitfulMatrix(randn(N)m)
	@show y = E*x
	println()
	
	@show problem = OverdeterminedProblem(y,E,Cnn⁻¹)
	println()	
	@time @show x̃ = solve(problem,alg=:hessian)
	@time @show x̃ = solve(problem,alg=:textbook) # Unitful error
	
	println()
	@show x ≈ x̃.v
	@show cost(x̃,problem) < 1e-5 # no noise in obs
	@show x ≈ pinv(problem) * y # inefficient way to solve problem

		
	l = @layout [
		a{0.5w} b{0.5w}
	]
	    println(diag(Cnn⁻¹))
	p1 = scatter(vec(y), yerror = sqrt.(1 ./ diag(Cnn⁻¹)), label = "y", xlabel = "Index", ylabel = "y", title = "Obs. and Recon. Obs")
	scatter!(vec((E*x̃).v); yerror = (E*x̃).σ, label = "Ex̃")
	p2 = scatter(vec(x); label = "xₜₕₑₒ", xlabel = "Index", ylabel = "x", title = "True Par. and Est. Par.")
	scatter!(vec(x̃.v);yerror = x̃.σ, label = "x̃")
	plot(p1, p2, layout = l)
	end
end


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
	Random.seed!(1234)
	numobs = 10  # number of obs
	@show t = collect(0:1:numobs-1)s
	
	@show a = randn()m # intercept
	@show b = randn()m/s # slope
	
	@show y = a .+ b.*t .+ randn(numobs)m
	
	E = UnitfulMatrix(hcat(ones(numobs),ustrip.(t)),fill(m,numobs),[m,m/s],exact=true)
	println()
	display(E)
	println()
	Cnn⁻¹ = Diagonal(fill(1.0,numobs),fill(m^-1,numobs),fill(m,numobs),exact=true)
	display(Cnn⁻¹)
	
	problem = OverdeterminedProblem(UnitfulMatrix(y),E,Cnn⁻¹)
	@show x̃ = solve(problem,alg=:textbook)
	x̃1 = solve(problem,alg=:hessian) #solutions will be identical 
	@show cost(x̃,problem) < 3numobs; # rough guide, could get unlucky and not pass
	scatter(t, y; yerror = sqrt.(1 ./ diag(Cnn⁻¹)), xlabel = "time", ylabel = "distance", label = "y(t)")
	plot!(t, vec((E*x̃).v), ribbon = (E*x̃).σ, label = "ỹ(t)")
	
    end
end


# ╔═╡ 1be54490-8984-42da-b26c-f3eb39b507f4
md""" 
## Example 4: left-uniform problem with prior info
If we have one observation of $\delta^{18}\mathrm{O}_{\text{calcite}} = (-1.9 \pm 0.2)‰$, and we want to obtain values of $\delta^{18}\mathrm{O}_{\text{water}}, T$ from it, we can use the following relation 

$$\delta^{18}\mathrm{O}_{\text{calcite}} = -0.24 [\text{‰/K}] T +  \delta^{18}\mathrm{O}_{\text{water}}$$

(above equation is rough estimate, and assumes we've already subtracted off the calcite-water $\delta^{18}\mathrm{O}$ offset)

As this is an underdetermined problem, we will define some first guess $\mathbf{x_0} = (1‰, 4K)$ with some tapering value $\mathbf{\gamma} = (1 ‰^{-2}, 1K^{-2})$ and solve for $\mathbf{\tilde{x}}$ 
"""

# ╔═╡ d56d533f-e2ae-4528-897e-cdc726616db1
γδ_slider = @bind γδ Slider(0.1permil^-2:0.1permil^-2:10permil^-2,show_value=true,default=1permil^-2)

# ╔═╡ 51eb9f4a-8c8d-4488-a499-9ff9555ac992
γT_slider = @bind γT Slider(0.1K^-2:0.1K^-2:10K^-2,show_value=true,default=1K^-2)

# ╔═╡ 6338513d-6adf-4ba1-8d78-cce3d2178f51
begin
	let
		y = [-1.9permil]
		σₙ = 0.2permil
				
		a = -0.24permil*K^-1
		#γδ = 1.0permil^-2 # to keep units correct, have two γ (tapering) variables
		#γT = 1.0K^-2
		E = UnitfulMatrix(ustrip.([1 a]),[permil],[permil,K],exact=true) # problem with exact E and error propagation
		x₀ = [-1.0permil, 4.0K]
		
		# "overdetermined, problem 1.4" 
		Cxx⁻¹ = Diagonal(ustrip.([γδ,γT]),[permil^-1,K^-1],[permil,K],exact=true)
		Cnn⁻¹  = Diagonal([σₙ.^-2],[permil^-1],[permil])
		oproblem = OverdeterminedProblem(UnitfulMatrix(y),E,Cnn⁻¹,Cxx⁻¹,UnitfulMatrix(x₀))
		
		# "underdetermined, problem 2.1" 
		Cnn  = Diagonal([σₙ.^2],[permil],[permil^-1])
		Cxx = Diagonal(ustrip.([γδ,γT].^-1),[permil,K],[permil^-1,K^-1],exact=true)
		uproblem = UnderdeterminedProblem(UnitfulMatrix(y),E,Cnn,Cxx,UnitfulMatrix(x₀))
		
		@time x̃1 = solve(oproblem)
		@test cost(x̃1,oproblem) < 1 # rough guide, coul
		
		@time x̃2 = solve(uproblem)
		@test cost(x̃2,uproblem) < 1 # rough guide, could ge
		# same answer both ways?
		@test cost(x̃2,uproblem) ≈ cost(x̃1,oproblem)
	
		l = @layout [
		a{1.0w} 
		b{0.5w} c{0.5w}
		]
		p1 = scatter(["y"], y; yerror = σₙ, legend = false, ylabel = "δ¹⁸O")
		scatter!(["Ex̃₁"], vec((E*x̃1).v), yerror = (E*x̃1).σ)
		scatter!(["Ex̃₂"], vec((E*x̃2).v), yerror = (E*x̃2).σ)
		scatter!(["Exₒ"], vec(E*UnitfulMatrix(x₀)))

		p2 = scatter(["x̃1"], [vec(x̃1.v)[1]], yerror = [x̃1.σ[1]], legend = false, ylims = (-1.5, -0.3))
		scatter!(["x̃2"], [vec(x̃2.v)[1]], yerror = x̃2.σ[1])
		scatter!(["x₀"],[x₀[1]])

		p3 = scatter(["x̃1"], [vec(x̃1.v)[2]], yerror = x̃1.σ[2], legend = false, ylims = (2, 6))
		scatter!(["x̃2"], [vec(x̃2.v)[2]], yerror = x̃2.σ[2])
		scatter!(["x₀"], [x₀[2]])
		plot(p1,p2,p3, layout = l)	
end
end


# ╔═╡ 17b6c57a-0b3e-4aa2-bffd-df97bbd5a1c8
md"""
## Example 5: Polynomial fitting, problem 2.3
For a function 

$$y(t) [\text{g/kg}] = a [\text{g/kg}] + b [\text{g/(kg day)}] t + c [\text{g/(kg day}^2)] t^2 + d[\text{g/(kg day}^3)]t^3$$

Assume we can make measurements of some quantity with $\sigma = 0.1$ g/kg. We will make 50 observations, making this an overdetermined system, and we will solve it using a prior $\mathbf{x_0} = \vec{0}$

We will define a tapering matrix 
$\mathbf{C_xx} = \begin{bmatrix}
0.1 && 0 \text{day}^{-1} && 0 \text{day}^{-2} && 0 \text{day}^{-3}\\
0 \text{day}^{-1} && 0.01 \text{day}^{-2} && 0 \text{day}^{-3} && 0 \text{day}^{-4} \\ 
0 \text{day}^{-2}&& 0 \text{day}^{-3}&& 0.001 \text{day}^{-4}  && 0 \text{day}^{-5} \\
0 \text{day}^{-3}&& 0\text{day}^{-4} && 0 \text{day}^{-5}&& 0.0001 \text{day}^{-6}\\
\end{bmatrix} [\text{g}^2 \text{kg}^{-2}]$

We will generate an observational vector $\mathbf{x}$ from the Cholesky decomposition of the $\mathbf{C_xx}$ matrix 
$$\mathbf{x}  = \mathbf{Cxx}^{T/2} \mathbf{r}$$
where $\mathbf{r}$ is a vector of normally distributed values, meaning that $\mathbf{x}$ will have the covariance of $\mathbf{C_xx}$ 

By using the M slider, which controls the number of points, we can see that the more points we have, the closer $\tilde{x}$ gets to $x_{\text{true}}$

Note that in the M = 1:11 range, $\tilde{x} > x_0, x_{\text{true}}$ for the third (or $c$ constant in the polynomial value)
"""

# ╔═╡ 7c39ff4c-8ceb-4af8-9189-0ead81d6388a
M_slider = @bind M Slider(2:1:100,show_value=true,default=10)

# ╔═╡ 838d2317-455a-4502-97cb-9a5d68eba1a2
let
	begin
		Random.seed!(1234)
		g = u"g"
        kg = u"kg"
        d = u"d"
		#M = 50
        t = (1:M)d

        E = UnitfulMatrix(ustrip.(hcat(t.^0, t, t.^2, t.^3)),fill(g/kg,M),[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)

        σₙ = 0.1g/kg
        Cₙₙ = Diagonal(fill(ustrip(σₙ^2),M),fill(g/kg,M),fill(kg/g,M),exact=true)
		display(Cₙₙ)
        #Cₙₙ¹² = cholesky(Cₙₙ)
        Cₙₙ⁻¹ = Diagonal(fill(ustrip(σₙ^-2),M),fill(kg/g,M),fill(g/kg,M),exact=true) 
        #Cₙₙ⁻¹² = cholesky(Cₙₙ⁻¹)

		γ = [1.0e1kg^2/g^2, 1.0e2kg^2*d^2/g^2, 1.0e3kg^2*d^4/g^2, 1.0e4kg^2*d^6/g^2]

		Cxx⁻¹ = Diagonal(ustrip.(γ),[kg/g,kg*d/g,kg*d^2/g,kg*d^3/g],[g/kg,g/kg/d,g/kg/d^2,g/kg/d^3],exact=true)
        Cxx = inv(Cxx⁻¹)
		display(Cxx)
        @show Cxx¹² = cholesky(Cxx)

        N = size(Cxx⁻¹,1)
        @show x₀ = zeros(N).*unitdomain(Cxx⁻¹)
        x = Cxx¹².L*randn(N)
        @show y = E*x
        oproblem = OverdeterminedProblem(y,E,Cₙₙ⁻¹,Cxx⁻¹,UnitfulMatrix(x₀))

        # not perfect data fit b.c. of prior info
        x̃ = solve(oproblem,alg=:hessian)
        @show x̃ = solve(oproblem,alg=:textbook)
		@show cost(x̃,oproblem)
        @show cost(x̃,oproblem) < 3M
		l = @layout[
			a{1.0w}
			b{0.25w} c{0.25w} d{0.25w} e{0.25w}
			f{0.25w}
		]
		p1 = scatter(t, vec(y); yerror = sqrt.(diag(Cₙₙ)), label = "y", xlabel = "time", ylabel = "y")
		scatter!(t, vec((E*x̃).v); yerror = (E*x̃).σ, label = "Ex̃")
		p2 = scatter(["true", "x̃", "x₀"], [vec(x)[1], vec(x̃.v)[1], x₀[1]], ylabel = "", legend = false)
		p3 = scatter(["true", "x̃", "x₀"], [vec(x)[2], vec(x̃.v)[2], x₀[2]], ylabel = "", legend = false)
		p4 = scatter(["true", "x̃", "x₀"], [vec(x)[3], vec(x̃.v)[3], x₀[3]], ylabel = "", legend = false)
		p5 = scatter(["true", "x̃", "x₀"], [vec(x)[4], vec(x̃.v)[4], x₀[4]], ylabel = "", legend = false)
		plot(p1,p2,p3, p4, p5, layout = l)
	end
end


# ╔═╡ c2ebefbb-2d78-40de-809e-8207caa9cc6d
md"""
## Example 6: Overdetermined problem for mean with autocovariance, problem 4.1 
"""

# ╔═╡ 9fea9058-0220-41e3-9254-c42b13dee248
md"""
## Example 7: Objective mapping, problem 4.3
"""

# ╔═╡ 8fafebcd-7df4-4ff1-905c-4a14c71a5290
let
	begin
	   	yr = u"yr"; cm = u"cm"
        τ = range(0.0yr,5.0yr,step=0.1yr)
        ρ = exp.(-τ.^2/(1yr)^2)
        n = length(ρ)
        Cxx = UnitfulMatrix(SymmetricToeplitz(ρ),fill(cm,n),fill(cm^-1,n),exact=true) + Diagonal(fill(1e-6,n),   fill(cm,n),fill(cm^-1,n),exact=true)

		display(Cxx)
		
        M = 11
        σₙ = 0.1cm
        Cnn = Diagonal(fill(ustrip(σₙ),M),fill(cm,M),fill(cm^-1,M),exact=true)

        Enm = sparse(1:M,1:5:n,fill(1.0,M))
        E = UnitfulMatrix(Enm,fill(cm,M),fill(cm,n),exact=true)

        Cxx¹² = cholesky(Cxx)
        x₀ = zeros(n)cm
        x = Cxx¹².L*randn(n)
        @show y = E*x

        uproblem = UnderdeterminedProblem(y,E,Cnn,Cxx,UnitfulMatrix(x₀))

           x̃ = solve(uproblem)
                #    @show x̃ = solve(uproblem) # show method missing
        @test cost(x̃,uproblem) < 3M 
		t = range(0.0yr, 5.0yr, length = length(y))
		p1 = scatter(t, vec(y), yerror = σₙ, label = "y", xlabel = "time", ylabel = "sea level")
		scatter!(t, vec((E*x̃).v), yerror = (E*x̃).σ, label = "ỹ")
		plot!(τ, vec(x̃.v), ribbon = x̃.σ, label = "x̃")
	end
end


# ╔═╡ 293a74b4-f862-45e6-a27e-8dda5d9264f9
md"""
## Example 8: overdetermined named tuple E, y
For some system 

$\mathbf{y_1} = \mathbf{E_1x}$
$\mathbf{y_2} = \mathbf{E_2x}$
solve for x
"""

# ╔═╡ cbc01f32-76bb-478c-be4e-969542ef32e0
let
	begin
		N = 2
        M = 1
        σₓ = rand()
        # exact = false to work
        E1 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E2 = UnitfulMatrix(randn(M,N),fill(m,M),fill(m,N),exact=true)
        E = (one=E1,two=E2)

        Cnn⁻¹1 = Diagonal(fill(σₓ^-1,M),unitrange(E1).^-1,unitrange(E1),exact=true)

        Cnn⁻¹ = (one=Cnn⁻¹1, two =Cnn⁻¹1)
        x = UnitfulMatrix(randn(N)m)

        # create perfect data
        @show y = E*x

        problem = OverdeterminedProblem(y,E,Cnn⁻¹)

        #x̃ = solve(y,E,Cnn⁻¹)
        x̃ = solve(problem,alg=:hessian)
        
        @show x ≈ x̃.v # no noise in obs
        @show cost(x̃,problem) #< 1e-5 # no noise in obs
	end
end


# ╔═╡ Cell order:
# ╟─1aa4af78-f0ee-4d5a-916b-c6e0decdd300
# ╠═1ccfc590-90f7-11ed-1e7c-dfaec0816597
# ╟─970a7544-c5f2-4f2e-ba52-316f67554482
# ╟─f862f30e-ea14-41a1-93ee-63d92380cd8d
# ╠═7dd4252c-f2d0-4839-9052-981b86e83804
# ╟─6b9668b2-dcdc-4ac9-b29c-feeb155ec42f
# ╠═98024f12-fc38-40e3-a4a4-c7678940f5ee
# ╟─a820781d-2e6a-4b4d-a1cb-f824c293de57
# ╠═f3ddcd82-b84c-4504-bbd5-309ec62a10c3
# ╠═1be54490-8984-42da-b26c-f3eb39b507f4
# ╠═d56d533f-e2ae-4528-897e-cdc726616db1
# ╠═51eb9f4a-8c8d-4488-a499-9ff9555ac992
# ╠═6338513d-6adf-4ba1-8d78-cce3d2178f51
# ╟─17b6c57a-0b3e-4aa2-bffd-df97bbd5a1c8
# ╠═7c39ff4c-8ceb-4af8-9189-0ead81d6388a
# ╠═838d2317-455a-4502-97cb-9a5d68eba1a2
# ╠═c2ebefbb-2d78-40de-809e-8207caa9cc6d
# ╠═9fea9058-0220-41e3-9254-c42b13dee248
# ╠═8fafebcd-7df4-4ff1-905c-4a14c71a5290
# ╟─293a74b4-f862-45e6-a27e-8dda5d9264f9
# ╠═cbc01f32-76bb-478c-be4e-969542ef32e0
