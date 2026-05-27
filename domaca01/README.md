# Razpršene matrike in SOR metoda v Julii

## Avtor
Matej Rupnik

---

## Opis projekta

Projekt implementira razpršeno (sparse) matriko ter iterativno metodo Successive Over-Relaxation (SOR) za reševanje linearnih sistemov oblike

$$
A x = b.
$$

Matrika je shranjena v učinkoviti razpršeni predstavitvi tipa List-of-Lists, kjer hranimo samo neničelne elemente in njihove stolpčne indekse. Na tej strukturi so implementirane osnovne operacije linearne algebre, vključno z množenjem matrike z vektorjem ter osnovnimi aritmetičnimi operacijami.

Metoda SOR je iterativni postopek za približno reševanje linearnih sistemov. Uporablja se za razpršene sisteme in tudi pri fizikalni vložitvi grafov, kjer problem temelji na modelu vzmeti in Laplaceovi matriki grafa.

---

## Uporaba kode

1. Odpri Julio v mapi projekta.
2. Aktiviraj okolje:

   using Pkg  
   Pkg.activate(".")  
   Pkg.instantiate()

3. Naloži paket:

   include("src/domaca01.jl")  
   using .domaca01

4. Primer uporabe:

   A = toRedka([4.0 1.0; 1.0 3.0])  
   b = [1.0, 2.0]

   x, iter = sor(A, b, 1.0, 1e-10)

   println("Rešitev: ", x)  
   println("Število iteracij: ", iter)

---

## Zagon testov

Za zagon vseh testov:

   include("test/runtests.jl")

ali preko upravljalnika paketov Julia:

   using Pkg  
   Pkg.test()

---

## Generiranje poročila

Poročilo je napisano v LaTeXu in se nahaja v mapi doc/.

Za ročno prevajanje:

   cd doc  
   pdflatex -shell-escape NM_dn1.tex

To ustvari:

   doc/NM_dn1.pdf
