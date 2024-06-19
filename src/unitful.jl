standard_error(P::UnitfulMatrix) = .√diag(P)

function UnitfulMatrix_from_input_output(Eu,y,x)
    if length(x) == 1 && length(y) == 1
        return UnitfulMatrix(ustrip.(Eu),[unit(y)],[unit(x)])
    elseif length(y) == 1
        return UnitfulMatrix(ustrip.(Eu),[unit.(y)],vec(unit.(x)))
    elseif length(x) == 1
        return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),[unit(x)])
    else
        return UnitfulMatrix(ustrip.(Eu),vec(unit.(y)),vec(unit.(x)))
    end
end

function expectedunits(y,x)
    Eunits = Matrix{Unitful.FreeUnits}(undef,length(y),length(x))
    for ii in eachindex(y)
        for jj in eachindex(x)
            if length(y) == 1 && length(x) ==1
                Eunits[ii,jj] = unit(y)/unit(x)
            elseif length(x) == 1
                Eunits[ii,jj] = unit.(y)[ii]/unit(x)
            elseif length(y) ==1
                Eunits[ii,jj] = unit(y)/unit.(x)[jj]
            else
                Eunits[ii,jj] = unit.(y)[ii]/unit.(x)[jj]
            end
        end
    end
    return Eunits
end
