@testset "blues_unitful" begin

using Unitful

    @testset "error propagation" begin
        using Measurements

        # check what happens for a scalar
        Mlist = (1,5)
        for M in Mlist
            a = randn(M)u"K" .± rand(M)u"K"
            E = randn(M,M)
            aval = Measurements.value.(a)
            aerr = Measurements.uncertainty.(a);

            # allow scalar input to Estimate constructor 
            x9 = Estimate(first(aval), first(aerr))

            x0 = Estimate(Measurements.value.(a),
                Diagonal(aerr.^2))
            x  = Estimate(aval, aerr) # just provide standard error
            x1  = Estimate(a) # just provide Vector{Measurement}

            @test   isequal(x.v, x0.v)
            @test   isequal(x.P, x0.P)
            @test   isequal(x1.v, x0.v)
            @test   isequal(x1.P, x0.P)
            @test Measurements.value.(E*a) ≈ (E*x).v
            @test Measurements.uncertainty.(E*a) ≈ (E*x).σ

            # combine two estimates
            xplus = combine(x,x,alg=:underdetermined)
            # error should decrease by 70%
            @test sum( xplus.σ./x.σ .< 0.8) == M
            # central estimate should not change
            @test isapprox( xplus.v, x.v )

            # combine two estimates another way
            xplus2 = combine(x,x,alg=:overdetermined)
            # error should decrease by 70%
            @test sum( xplus2.σ./x.σ .< 0.8) == M
            # central estimate should not change
            @test isapprox( xplus2.v, x.v )
        end
    end
end 
