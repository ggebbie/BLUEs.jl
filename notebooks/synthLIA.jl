using DrWatson
@quickactivate "../BLUEs"
using BLUEs, Unitful, DimensionalData
using DimensionalData:@dim

const permil = u"permille"; const K = u"K"; const K² = u"K^2"; m = u"m"; s = u"s";yr = u"yr"
ENV["UNITFUL_FANCY_EXPONENTS"] = true

#following "source water inversion: one obs TIMESERIES, many surface regions, with NO circulation lag" on branch associated with issue 27
@dim YearCE "years Common Era"
@dim SurfaceRegion "surface location"
@dim InteriorLocation "interior location"

surfaceregions = [:NATL,:ANT]
years = (1000:50:2000)yr
n = length(surfaceregions)

M = DimArray
