# Matematično nihalo v Julii

## Avtor
Matej Rupnik

---

## Opis projekta

Projekt implementira numerično rešitev matematičnega nihala z uporabo
adaptivnega integratorja Dormand–Prince RK45 (DOPRI5) z vgrajenim nadzorom napake.

Diferencialna enačba drugega reda se prevede na sistem prvega reda, ki se numerično
integrira z metodo DOPRI5. Rešitev se primerja z rešitvijo harmoničnega nihala ter
prikaže odvisnost nihajnega časa od energije nihala.

Implementirane so naslednje metode:

- **Adaptivni DOPRI5 integrator** z oceno lokalne napake (4. vs. 5. red) in PI-krmilnikom koraka
- **Linearna interpolacija ničelnih prehodov** za natančno določitev nihajnega časa
- **Primerjava matematičnega in harmoničnega nihala** pri različnih začetnih pogojih
- **Graf odvisnosti nihajnega časa od energije** matematičnega nihala

Vse numerične metode so implementirane ročno. Uporabljena je le knjižnica `Plots.jl`
za risanje grafov.

---

## Implementirane funkcije

- `dopri5_step(f, u, h)`
  En adaptivni korak Dormand–Prince RK45. Vrne rešitev 5. reda, vgrajeno oceno napake (4. red) in zadnji k-vektor (FSAL).

- `integrate(f, θ0, ω0; T, h0, rtol, atol)`
  Adaptivna integracija sistema prvega reda od `t=0` do `t=T`. Zaznava ničelne prehode z linearno interpolacijo in vrne časovni niz, trajektorijo ter seznam nihajnih časov.

- `energy(θ0, ω0)`
  Izračun skupne mehanske energije nihala:
  $$E = \tfrac{1}{2}\,l^2\,\omega^2 + g\,l\,(1 - \cos\theta)$$

- `f_math(u)`
  Desna stran sistema za matematično nihalo:
  $$\dot{u} = \begin{bmatrix} \omega \\ -\tfrac{g}{l}\sin(\theta) \end{bmatrix}$$

- `f_harm(u)`
  Desna stran sistema za harmonično nihalo:
  $$\dot{u} = \begin{bmatrix} \omega \\ -\tfrac{g}{l}\,\theta \end{bmatrix}$$

---

## Uporaba kode

1. Odpri Julio v mapi projekta.
2. Naloži skripto:

```julia
include("pendulum.jl")
```

3. Primer izračuna odmika ob določenem času:

```julia
# Začetni pogoji: θ₀ = 0.5 rad, ω₀ = 0.0 rad/s
ts, us, periods = integrate(f_math, 0.5, 0.0; T=20.0)

# Odmik ob t ≈ 5 s
idx = argmin(abs.(ts .- 5.0))
println("θ(5 s) = ", us[idx][1], " rad")
```

4. Primer izračuna nihajnega časa:

```julia
T_math = mean(diff(periods))
T_harm = 2π * sqrt(l / g)
println("Nihajni čas (matematično): ", T_math)
println("Nihajni čas (harmonično):  ", T_harm)
```

---

## Zagon skripte

Za generiranje vseh grafov zaženi skripto neposredno:

```bash
julia script/demo.jl
```

Skripta ustvari naslednje izhodne datoteke:

```
plot1_primerjava_mali_odmik.png
plot2_primerjava_veliki_odmik.png
plot3_fazni_portret.png
plot4_nihajni_cas_energija.png
plot_skupni.pdf
```

---

## Zahteve in natančnost

Zahtevana relativna natančnost je $10^{-10}$ (10 decimalk). To je doseženo z:

- Adaptivnim DOPRI5 z `rtol = atol = 1e-11`
- Linearno interpolacijo pri zaznavi ničelnih prehodov, kar zmanjša napako ocene
  nihajnega časa iz $O(h)$ na $O(h^2)$

Fizikalni parametri: $g = 9.81\ \mathrm{m/s^2}$, $l = 1.0\ \mathrm{m}$.

---

## Generiranje poročila

Poročilo je napisano v markdown in se nahaja v doc/domaca-03_report.md.
Že prevedena verzija se nahaja zraven doc/domaca-03_report.pdf

Za ročno prevajanje:

pandoc domaca-03_report.md -o domaca-03_report.pdf

To ustvari:

   doc/domaca-03_report.pdf
