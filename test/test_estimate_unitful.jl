@testset "blues_unitful" begin

    @testset "error propagation" begin
        using Measurements

        # check what happens for a scalar as well
        Mlist = (1,5)
        for M in Mlist
            a = randn(M)u"K" .± rand(M)u"K"
            error_propagation(a)
        end
    end
end 
