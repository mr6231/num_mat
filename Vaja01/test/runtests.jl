using Vaja01, Test
@testset "Koordinata x" begin
 @test lemniskata_x(1.0) ≈ 0.0
 @test lemniskata_x(2.0) ≈ 3 / 5
end
@testset "Koordinata y" begin
 @test lemniskata_y(1.0) ≈ 0.0
 @test lemniskata_y(2.0) ≈ 12 / 25
end
