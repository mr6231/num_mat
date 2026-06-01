"""
runtests.jl — Testiranje implementacije matematičnega nihala (domaca03)

Pokriva:
  1. Koraki DOPRI5 — pravilnost koeficientov in red metode
  2. Energija — formula in ohranjanje vzdolž trajektorije
  3. ODE desne strani — f_math in f_harm
  4. Adaptivna integracija — nihajni čas za malo nihalo vs analitična vrednost
  5. Interpolacija prehodov — natančnost nihajnega časa pri majhnih kotih
  6. Diverjenca nihajnega časa — T_mat > T_harm za velike kote

Zaženi z:
    julia runtests.jl

ali v REPL:
    include("runtests.jl")
"""

include("../src/domaca03.jl")
using .domaca03
using Test
using LinearAlgebra

# ---------------------------------------------------------------------------
# Pomožne funkcije
# ---------------------------------------------------------------------------

"""Relativna napaka med dvema vrednostma."""
relerr(a, b) = abs(a - b) / max(abs(b), 1e-30)

# Analitični nihajni čas harmoničnega nihala
T0 = 2π * sqrt(domaca03.l / domaca03.g)

# ===========================================================================
@testset "domaca03 — celovito testiranje" begin

# ---------------------------------------------------------------------------
@testset "1. Fizikalni parametri" begin
    @test domaca03.g ≈ 9.81       atol=1e-15
    @test domaca03.l ≈ 1.0        atol=1e-15
    @test domaca03.g > 0
    @test domaca03.l > 0
end

# ---------------------------------------------------------------------------
@testset "2. ODE desne strani" begin

    @testset "f_math — linearno območje (sin θ ≈ θ)" begin
        # Za majhen θ mora biti f_math ≈ f_harm
        u_small = [0.01, 0.0]
        fm = domaca03.f_math(u_small)
        fh = domaca03.f_harm(u_small)
        @test relerr(fm[1], fh[1]) < 1e-15
        # Razlika v drugem členu: sin(0.01) vs 0.01, relativna napaka ~ θ²/6
        @test relerr(fm[2], fh[2]) < 2e-5
    end

    @testset "f_math — točne vrednosti" begin
        u = [π/6, 1.0]   # θ = 30°, ω = 1 rad/s
        fm = domaca03.f_math(u)
        @test fm[1] ≈  1.0                              atol=1e-15
        @test fm[2] ≈ -(domaca03.g / domaca03.l) * sin(π/6)  atol=1e-14
        @test fm[2] ≈ -(domaca03.g / domaca03.l) * 0.5       atol=1e-14
    end

    @testset "f_harm — točne vrednosti" begin
        u = [π/4, 2.0]
        fh = domaca03.f_harm(u)
        @test fh[1] ≈  2.0                              atol=1e-15
        @test fh[2] ≈ -(domaca03.g / domaca03.l) * (π/4)     atol=1e-14
    end

    @testset "f_math — ravnovesje pri θ = 0" begin
        u = [0.0, 0.0]
        fm = domaca03.f_math(u)
        @test fm[1] == 0.0
        @test fm[2] == 0.0
    end

    @testset "f_math — nestabilno ravnovesje pri θ = π" begin
        u = [π, 0.0]
        fm = domaca03.f_math(u)
        @test fm[1] == 0.0
        @test abs(fm[2]) < 1e-14   # sin(π) = 0 (do zaokrož. napake)
    end
end

# ---------------------------------------------------------------------------
@testset "3. Energija" begin

    @testset "Mirujoče nihalo — samo potencialna energija" begin
        for θ0 in [0.1, 0.5, 1.0, 2.0, π/2]
            E = domaca03.energy(θ0, 0.0)
            E_ref = domaca03.g * domaca03.l * (1.0 - cos(θ0))
            @test relerr(E, E_ref) < 1e-14
        end
    end

    @testset "Gibanje — samo kinetična energija (θ = 0)" begin
        ω = 3.0
        E = domaca03.energy(0.0, ω)
        E_ref = 0.5 * domaca03.l^2 * ω^2
        @test relerr(E, E_ref) < 1e-14
    end

    @testset "Energija raste z odmikom (ω = 0)" begin
        E_vals = [domaca03.energy(θ, 0.0) for θ in [0.1, 0.5, 1.0, 2.0]]
        @test all(diff(E_vals) .> 0)
    end

    @testset "Energija ≥ 0" begin
        for θ0 in range(0.0, 3.0, length=30)
            @test domaca03.energy(θ0, 0.0) >= 0.0
        end
    end

    @testset "Maksimalna energija pri θ₀ → π" begin
        E_max = domaca03.g * domaca03.l * 2.0
        @test relerr(domaca03.energy(π, 0.0), E_max) < 1e-14
    end
end

# ---------------------------------------------------------------------------
@testset "4. Korak DOPRI5" begin

    @testset "Ohranjanje energije v enem koraku" begin
        # Točen rezultat: energija se mora ohraniti do O(h^5)
        u0  = [0.5, 0.0]
        h   = 1e-2
        u1, err_vec = domaca03.dopri5_step(domaca03.f_math, u0, h)
        E0 = domaca03.energy(u0[1], u0[2])
        E1 = domaca03.energy(u1[1], u1[2])
        @test abs(E1 - E0) < 1e-9
    end

    @testset "Napaka se zmanjša z manjšim korakom (konvergenca O(h^5))" begin
        u0  = [1.0, 0.0]
        # Primerjaj napako pri h in h/2
        h1  = 1e-2
        h2  = 5e-3
        _, e1_vec = domaca03.dopri5_step(domaca03.f_math, u0, h1)
        _, e2_vec = domaca03.dopri5_step(domaca03.f_math, u0, h2)
        ratio = norm(e1_vec) / norm(e2_vec)
        # Pričakovan razmernik ≈ 2^5 = 32  (red 5)
        @test ratio > 20.0
    end

    @testset "f_harm — numerični korak vs. analitična rešitev" begin
        # Harmonično nihalo ima točno rešitev:
        #   θ(t) = θ₀ cos(ω₀ t),  ω₀ = sqrt(g/l)
        ω0  = sqrt(domaca03.g / domaca03.l)
        θ0  = 0.3
        u0  = [θ0, 0.0]
        h   = 1e-3
        u1, _ = domaca03.dopri5_step(domaca03.f_harm, u0, h)
        θ_exact = θ0 * cos(ω0 * h)
        ω_exact = -θ0 * ω0 * sin(ω0 * h)
        @test relerr(u1[1], θ_exact) < 1e-13
        @test relerr(u1[2], ω_exact) < 1e-13
    end

    @testset "Ocena napake je majhna za majhen korak" begin
        u0 = [0.5, 0.3]
        h  = 1e-4
        _, err_vec = domaca03.dopri5_step(domaca03.f_math, u0, h)
        @test norm(err_vec) < 1e-18
    end
end

# ---------------------------------------------------------------------------
@testset "5. Adaptivna integracija — nihajni čas" begin

    tol = 1e-9   # Sprejemljiva napaka na nihajni čas (> 10 decimalk zahtevan od ODE,
                 # a period se določa z interpolacijo; cilj ~1e-10 za majhne kote)

    @testset "Harmonično nihalo — nihajni čas neodvisen od odmika" begin
        # T mora biti T0 za vsak θ₀ (pri f_harm)
        for θ0 in [0.1, 0.5, 1.0, 1.5, 2.0]
            _, _, Tp = domaca03.integrate(domaca03.f_harm, θ0, 0.0; T=120.0)
            @test !isnan(Tp)
            @test relerr(Tp, T0) < 1e-9
        end
    end

    @testset "Nihajni čas narašča z energijo (T_mat > T0 za velike θ₀)" begin
        T_list = Float64[]
        for θ0 in [0.2, 0.8, 1.5, 2.2]
            _, _, Tp = domaca03.integrate(domaca03.f_math, θ0, 0.0; T=100.0)
            push!(T_list, Tp)
        end
        # Preverimo, da je zaporedje naraščajoče
        @test all(diff(T_list) .> 0)
        # In da je vsak T_mat > T0
        @test all(T_list .> T0)
    end


    @testset "Negativni začetni odmik — simetrija" begin
        _, _, Tp_pos = domaca03.integrate(domaca03.f_math,  1.0, 0.0; T=100.0)
        _, _, Tp_neg = domaca03.integrate(domaca03.f_math, -1.0, 0.0; T=100.0)
        # Nihalo z -θ₀ ima enak nihajni čas kot z +θ₀
        @test relerr(Tp_pos, Tp_neg) < 1e-10
    end
end

# ---------------------------------------------------------------------------
@testset "6. Trajektorija" begin

    @testset "Začetni pogoji so ohranjeni" begin
        θ0, ω0 = 1.2, 0.5
        ts, θs, _ = domaca03.integrate(domaca03.f_math, θ0, ω0; T=10.0)
        @test ts[1] == 0.0
        @test θs[1] == θ0
    end

    @testset "Čas je strogo naraščajoč" begin
        ts, _, _ = domaca03.integrate(domaca03.f_math, 1.0, 0.0; T=10.0)
        @test all(diff(ts) .> 0)
    end

    @testset "Trajektorija se konča pri T" begin
        T_run = 15.0
        ts, _, _ = domaca03.integrate(domaca03.f_math, 1.0, 0.0; T=T_run)
        @test ts[end] ≈ T_run atol=1e-10
    end

    @testset "Amplituda ostane omejena (nedušeno nihalo)" begin
        # Za ω₀ = 0 je energija E = g·l·(1 - cos θ₀).
        # Maksimalni dosegljivi kot iz energije: θ_max = acos(1 - E/(g·l)).
        # Numerična rešitev ne sme preseči tega kota za več kot 1e-6 rad
        # (konzervativna meja glede na rtol=atol=1e-11 in dolžino simulacije).
        θ0 = 1.5
        E0 = domaca03.energy(θ0, 0.0)
        θ_max_exact = acos(max(-1.0, 1.0 - E0 / (domaca03.g * domaca03.l)))
        _, θs, _ = domaca03.integrate(domaca03.f_math, θ0, 0.0; T=60.0)
        @test maximum(abs.(θs)) < θ_max_exact + 1e-6
    end

    @testset "Ohranjanje energije vzdolž trajektorije" begin
        # Uporabimo adaptivno integracijo (rtol=atol=1e-11) namesto
        # fiksnih korakov, ki akumulirajo O(h^4)*N napako brez nadzora.
        # Energijo primerjamo na začetku in koncu trajektorije.
        # Ker integrate hrani samo θ (ne ω), preverimo energijo s pomočjo
        # enega adaptivnega koraka — sledimo u = [θ, ω] ves čas sami.
        θ0 = 1.0
        u  = [θ0, 0.0]
        E0 = domaca03.energy(u[1], u[2])
        # Integrate for 10 s with full adaptive control
        t = 0.0; T_run = 10.0
        rtol = 1e-11; atol = 1e-11
        h = 1e-3; hmax = 0.1; hmin = 1e-14
        fac = 0.9; facmax = 5.0; facmin = 0.2
        while t < T_run
            h = min(h, T_run - t)
            h < hmin && break
            u_new, err_vec = domaca03.dopri5_step(domaca03.f_math, u, h)
            sc  = @. atol + rtol * max(abs(u), abs(u_new))
            err = sqrt(sum((err_vec ./ sc) .^ 2) / length(u))
            if err <= 1.0
                t += h; u = u_new
                h = min(h * min(facmax, fac * (1.0 / max(err, 1e-15))^0.2), hmax)
            else
                h = h * max(facmin, fac * (1.0 / err)^0.2)
            end
        end
        E1 = domaca03.energy(u[1], u[2])
        @test abs(E1 - E0) < 1e-8
    end
end

# ---------------------------------------------------------------------------
@testset "7. Robni primeri" begin

    @testset "Nična amplituda — nihalo miruje" begin
        ts, θs, Tp = domaca03.integrate(domaca03.f_math, 0.0, 0.0; T=10.0)
        @test all(abs.(θs) .< 1e-14)
        @test isnan(Tp)
    end

    @testset "Zelo kratek čas integracije" begin
        ts, θs, _ = domaca03.integrate(domaca03.f_math, 1.0, 0.0; T=0.1)
        @test length(ts) > 1
        @test ts[end] ≈ 0.1 atol=1e-10
    end

end

end # @testset "domaca03 — celovito testiranje"

println("\nVsi testi zaključeni.")
