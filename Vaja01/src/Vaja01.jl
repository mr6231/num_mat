module Vaja01
"""Izračunaj `x` kordinato Geronove lemniskate."""
lemniskata_x(t) = (t^2 - 1) / (t^2 + 1)
"""Izračunaj `y` kordinato Geronove lemniskate."""
lemniskata_y(t) = 2t * (t^2 - 1) / (t^2 + 1)^2
# izvozimo imeni funkcij, da sta dostopni brez predpone `Vaja01`
export lemniskata_x, lemniskata_y
end # module Vaja01
