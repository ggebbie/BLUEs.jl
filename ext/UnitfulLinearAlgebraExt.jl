module UnitfulLinearAlgebraExt

using LinearAlgebra
using BLUEs
using UnitfulLinearAlgebra

function BLUEs.standard_error(P::UnitfulMatrix)
   sigma = .√diag(P)
   (sigma isa Vector) ? (return sigma) : (return [sigma])
end

# function BLUEs.UnitfulMatrix_from_input_output(Eu,y,x)
#     if length(x) == 1 && length(y) == 1
#         return UnitfulMatrix(ustrip.(Eu),[unit(y)],[unit(x)])
#     elseif length(y) == 1
#         return UnitfulMatrix(ustrip.(Eu),[unit.(y)],vec(unit.(x)))
#     elseif length(x) == 1
#         return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),[unit(x)])
#     else
#         return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),vec(unit.(x)))
#     end
# end

end 
