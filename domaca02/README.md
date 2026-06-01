# Numerično računanje normalne porazdelitvene funkcije (CDF) v Julii

## Avtor
Matej Rupnik

---

## Opis projekta

Projekt implementira več numeričnih metod za računanje standardne normalne kumulativne porazdelitvene funkcije (CDF):

$$
\Phi(x) = P(Z \le x), \quad Z \sim \mathcal{N}(0,1).
$$

Implementirane so tri metode:

- **Asimptotska aproksimacija** za repne vrednosti porazdelitve
- **Gauss–Legendre kvadraturna metoda** za numerično integracijo
- **Hibridna metoda (CDF_ND)**, ki kombinira obe pristopa za večjo stabilnost in natančnost

Projekt uporablja paket `FastGaussQuadrature` za učinkovito generiranje Gauss–Legendre točk in uteži.


---

## Implementirane funkcije

- `asymptotic_cdf(x)`  
  Izračun CDF z uporabo asimptotske razširitve (primerna za velike |x|).

- `gauss_legendre_cdf(x)`  
  Numerična integracija z Gauss–Legendre kvadraturo.

- `cdf_ND(x)`  
  Hibridna metoda za celotni CDF:
  - za `x ≤ 5` uporablja Gauss–Legendre metodo
  - za `x > 5` uporablja asimptotsko aproksimacijo

---

## Uporaba kode

1. Odpri Julio v mapi projekta.
2. Aktiviraj okolje:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Naloži modul:

```julia
include("src/domaca02.jl")
using .domaca02
```

4. Primer uporabe:
```julia
x = 1.5

println("Asimptotska CDF: ", asymptotic_cdf(x))
println("Gauss–Legendre CDF: ", gauss_legendre_cdf(x))
println("Hibridna CDF: ", cdf_ND(x))
```

## Zagon testov

Za zagon vseh testov:

   include("test/runtests.jl")

ali preko upravljalnika paketov Julia:

   using Pkg  
   Pkg.test()

---

---

## Generiranje poročila

Poročilo je napisano v markdown in se nahaja v doc/domaca-02_report.md.
Že prevedena verzija se nahaja zraven doc/domaca-02_report.pdf

Za ročno prevajanje:

pandoc domaca-02_report.md -o domaca-02_report.pdf

To ustvari:

   doc/domaca-02_report.pdf