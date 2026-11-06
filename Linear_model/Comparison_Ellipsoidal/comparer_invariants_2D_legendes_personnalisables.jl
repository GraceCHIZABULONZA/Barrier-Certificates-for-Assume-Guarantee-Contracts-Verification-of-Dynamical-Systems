# FICHIER UNIQUE, REEXECUTABLE DANS LE REPL SANS REDEFINITION DE CONSTANTES.

# IMPORTANT : conserver H/P/Q2 et Q4 en pleine precision pour tracer les certificats.

# Installation : import Pkg; Pkg.add(["Plots", "LaTeXStrings"])

using LinearAlgebra

using DelimitedFiles

using Printf

using Plots

using LaTeXStrings



let

# =============================================================================

# SUPERPOSITION DE DEUX CERTIFICATS INVARIANTS ET DE L'ENSEMBLE ADMISSIBLE

# Modele LINEAIRE du robot, semantique weak, delta = Inf.

#

# Deux projections (axes dans l'ordre demande) :

#   1) abscisse v, ordonnee rho = x-w

#   2) abscisse w, ordonnee rho = x-w

#

# Entrees (generees par les DEUX programmes d'origine) :

#   barriere quadratique : barrier_H.csv, barrier_Q_degree2.csv, ou barrier_P.csv

#   barriere quartique  : barrier_Q_degree4.csv

#

# Execution : julia comparer_invariants_v_xw_et_w_xw.jl

#              julia comparer_invariants_v_xw_et_w_xw.jl --H="C:/.../barrier_H.csv" --Q="C:/.../barrier_Q_degree4.csv"

# Necessite : import Pkg; Pkg.add(["Plots", "LaTeXStrings"])

# NE necessite PAS MOSEK lorsque les matrices ont deja ete calculees.

# Les figures ne constituent pas une nouvelle verification SOS.

# =============================================================================





gr()



# ---------------------- REGLAGES DE L'AFFICHAGE -----------------------------

FONT_AXES = 30

FONT_TICKS = 20

FONT_LEGEND = 30  # parametre historique, conserve

# === MODIFICATION LEGENDE 2D (1/3) ===
# Taille des noms des ensembles dans les deux projections 2D.
# Augmenter a 32, 36 ou 40 pour une ecriture plus grande.
FONT_LEGEND_2D = 28
# Taille des carres de couleur dans la legende (independante du texte).
LEGEND_MARKER_SIZE_2D = 15
# === FIN MODIFICATION LEGENDE 2D ===

LINE_WIDTH = 3.3

SIZE_EACH = (1150, 920)

SIZE_COMBINED = (2200, 940)

NTHETA = 240

MARGIN = 1.18

SAVE_PNG = true

ALPHA_ADMISSIBLE = 0.25

ALPHA_QUADRATIC = 0.20

ALPHA_QUARTIC = 0.31



# ---------------------- PARAMETRES COMMUNS ----------------------------------

ALPHA = 0.8

BETA = 0.15

EPS_SYS = 5.0e-5

D1 = 1.0

D2 = 1.0

D3 = 2.9

# r = N*z = [(x-w)/3, v/0.85, w/21]

N = Diagonal([1.0/3.0, 1.0/0.85, 1.0/21.0]) *

          [1.0 0.0 -1.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

Ni = inv(N)



# ---------------------- CHARGEMENT DES MATRICES -----------------------------

function get_options(args)

    opts = Dict(

        "H"=>"", "P"=>"", "Q2"=>"", "Q"=>"",

        "search-dir"=>"",

        "outdir"=>joinpath(@__DIR__, "comparaison_projections")

    )

    for arg in args

        startswith(arg, "--") || error("Argument invalide : $arg")

        parts = split(arg[3:end], "="; limit=2)

        length(parts) == 2 || error("Attendu : --H=CHEMIN, --Q=CHEMIN ou --search-dir=DOSSIER")

        haskey(opts, parts[1]) || error("Option inconnue : $(parts[1])")

        opts[parts[1]] = strip(parts[2], ['"', '\''])

    end

    return opts

end



function load_matrix(path, n)

    isfile(path) || error("Matrice introuvable : $path")

    M = Matrix{Float64}(readdlm(path, ',', Float64))

    size(M) == (n,n) || error("Matrice de dimension incorrecte : $path ; attendu $(n)x$(n), obtenu $(size(M))")

    all(isfinite, M) || error("Coefficients non finis dans $path")

    maximum(abs.(M .- M')) <= 1e-7*max(1.0, maximum(abs.(M))) ||

        error("Matrice non symetrique numeriquement : $path")

    M = (M + M') / 2

    eigmin = minimum(eigvals(Symmetric(M)))

    eigmin > 0 || error("Matrice non definie positive : $path ; valeur propre minimale=$eigmin")

    @printf("Matrice chargee : %s ; lambda_min = %.8g\n", abspath(path), eigmin)

    return M

end



# Recherche ciblee des CSV generes par les deux scripts d'origine.

# Lancement depuis "Downloads/comparaison_invariants_2D_julia" : explorer aussi

# Downloads et ses sous-dossiers (niveau limite a 3 par securite/performance).

function search_roots(opts)

    roots = String[@__DIR__, dirname(@__DIR__)]

    if !isempty(opts["search-dir"])

        extra = abspath(expanduser(opts["search-dir"]))

        isdir(extra) || error("--search-dir n'est pas un dossier existant : $extra")

        pushfirst!(roots, extra)

    end

    return unique(filter(isdir, roots))

end



function candidates_by_name(fname, roots; maxdepth=3)

    found = String[]

    skipnames = Set([".git", ".julia", "node_modules", ".venv", "venv", "__pycache__"])

    for root in roots

        for (dir, dirs, files) in walkdir(root; follow_symlinks=false, onerror=e->nothing)

            filter!(d -> !(d in skipnames), dirs)

            depthname = relpath(dir,root)

            depth = depthname == "." ? 0 : length(splitpath(depthname))

            if depth >= maxdepth

                empty!(dirs)

            end

            if fname in files

                push!(found, normpath(joinpath(dir,fname)))

            end

        end

    end

    return unique(found)

end



function explicit_file(value, roots, optname)

    isempty(value) && error("Chemin manquant pour $optname")

    tries = isabspath(value) ? [value] : vcat([value], [joinpath(r,value) for r in roots])

    for candidate in tries

        if isfile(candidate)

            return abspath(candidate)

        end

    end

    error("Fichier introuvable pour $optname : $value\nDossiers explores :\n  " * join(roots, "\n  "))

end



# Priorite : chemin explicite > fichier voisin du script > recherche.

# Si plusieurs fichiers DIFFERENTS existent, choix numerote dans un REPL.

# Dans un script sans terminal, on exige --H=... ou --Q=... .

function choose_ambiguous_file(fname, matches; flag="--H")

    paths = sort(unique(abspath.(matches)))

    println("\nPlusieurs fichiers $(fname) ont ete trouves :")

    for (i, path) in enumerate(paths)

        println("  [$(i)] $(path)")

    end



    # Un doublon bit-a-bit identique est sans ambiguite sur les coefficients.

    if all(read(path) == read(first(paths)) for path in paths[2:end])

        println("Les fichiers sont identiques : utilisation de $(first(paths))")

        return first(paths)

    end



    if isinteractive() || isatty(stdin)

        print("Choisir le NUMERO du certificat pour $(fname) [1-$(length(paths))] : ")

        flush(stdout)

        response = try

            strip(readline(stdin))

        catch

            ""

        end

        index = tryparse(Int, response)

        (index !== nothing && 1 <= index <= length(paths)) ||

            error("Selection invalide. Relancer avec $(flag)=CHEMIN_COMPLET.")

        println("Certificat selectionne : $(paths[index])")

        return paths[index]

    end



    error("Fichier ambigu dans cette execution non interactive. " *

          "Relancer avec $(flag)=CHEMIN_COMPLET (voir les chemins ci-dessus).")

end



function auto_file(fname, roots; flag="--H")

    neighboring = joinpath(@__DIR__, fname)

    if isfile(neighboring)

        return neighboring

    end

    matches = candidates_by_name(fname, roots)

    if length(matches) == 1

        println("Trouve automatiquement $fname : $(only(matches))")

        return only(matches)

    elseif length(matches) > 1

        return choose_ambiguous_file(fname, matches; flag=flag)

    end

    return nothing

end



function quadratic_matrix(opts, roots, quartic_path)

    selected = [k for k in ("H", "Q2", "P") if !isempty(opts[k])]

    length(selected) <= 1 || error("Utilise une seule option de certificat quadratique : --H, --Q2 ou --P")

    if !isempty(selected)

        key = only(selected)

        file = explicit_file(opts[key], roots, "--$key")

        M = load_matrix(file, 3)

        return key == "P" ? Matrix(Symmetric(Ni' * M * Ni)) : M

    end



    # Pour comparer TES DEUX scripts, preferer le fichier H du programme

    # robot_weak_infty_IEEE_figures_package quand il existe. Si ce chemin

    # n'existe pas, la selection standard reste disponible.

    for root in roots

        preferred = joinpath(root, "robot_weak_infty_IEEE_figures_package", "barrier_H.csv")

        if isfile(preferred)

            println("Certificat quadratique (script IEEE) choisi : ", preferred)

            return load_matrix(preferred, 3)

        end

    end



    # Sinon, un H place a cote du Q quartique est un bon choix.

    siblingH = joinpath(dirname(quartic_path), "barrier_H.csv")

    if isfile(siblingH)

        println("Certificat quadratique voisin du quartique : ", siblingH)

        return load_matrix(siblingH, 3)

    end



    # Utiliser les certificats du premier script en priorite. En presence de

    # plusieurs fichiers differents, le code demande explicitement un choix.

    for (key, name) in (("H", "barrier_H.csv"),

                        ("Q2", "barrier_Q_degree2.csv"),

                        ("P", "barrier_P.csv"))

        file = auto_file(name, roots; flag="--$key")

        if file !== nothing

            M = load_matrix(file, 3)

            return key == "P" ? Matrix(Symmetric(Ni' * M * Ni)) : M

        end

    end



    error("""

    CERTIFICAT QUADRATIQUE ABSENT.

    Aucun des fichiers suivants n'a ete trouve :

      barrier_H.csv, barrier_Q_degree2.csv, barrier_P.csv

    Dossiers recherches :

      $(join(roots, "\n      "))



    Il faut d'abord executer le script quadratique pour enregistrer barrier_H.csv,

    ou le script quartique pour produire barrier_Q_degree2.csv (si faisable).



    Ensuite, par exemple sous Windows :

      julia comparer_invariants_v_xw_et_w_xw.jl --H="C:/Users/Grace/Downloads/PROJET_1/barrier_H.csv" --Q="C:/Users/Grace/Downloads/PROJET_2/barrier_Q_degree4.csv"

    IMPORTANT : Ne pas inventer de matrices et ne pas utiliser un ancien

    certificat de reference signale comme invalide par ton autre script.

    """)

end



function quartic_matrix(opts, roots)

    file = isempty(opts["Q"]) ? auto_file("barrier_Q_degree4.csv", roots; flag="--Q") :

            explicit_file(opts["Q"], roots, "--Q")

    file === nothing && error("""

        CERTIFICAT QUARTIQUE ABSENT : barrier_Q_degree4.csv introuvable.

        Tu dois d'abord lancer le programme SOS quartique, puis fournir --Q=CHEMIN.

        Dossiers recherches : $(join(roots, ", "))

        """)

    return load_matrix(file, 9), file

end



# ---------------------- ENSEMBLE ADMISSIBLE ---------------------------------

# rho = x-w. Conditions robustes :

#   |v| <= D1 - eps, |BETA*rho + ALPHA*v| <= D2-eps,

#   |rho+v| <= D3-eps.

# Le polygone ci-dessous est EXACT dans le plan (rho,v).

function admissible_polygon()

    a, b, c = D1-EPS_SYS, D2-EPS_SYS, D3-EPS_SYS

    cuts = [(0.0,1.0,a), (0.0,-1.0,a),

            (BETA,ALPHA,b), (-BETA,-ALPHA,b),

            (1.0,1.0,c), (-1.0,-1.0,c)]

    vertices = Vector{Float64}[]

    for i in 1:(length(cuts)-1), j in (i+1):length(cuts)

        ai,bi,ci = cuts[i]

        aj,bj,cj = cuts[j]

        C = [ai bi; aj bj]

        abs(det(C)) < 1e-12 && continue

        pt = C \ [ci,cj]

        if all(a1*pt[1]+a2*pt[2] <= bb + 1e-9 for (a1,a2,bb) in cuts) &&

           !any(norm(pt-q)<1e-8 for q in vertices)

            push!(vertices, pt)

        end

    end

    length(vertices)>=3 || error("Section admissible non bornee ou vide")

    ctr = sum(vertices)/length(vertices)

    sort!(vertices; by=q->atan(q[2]-ctr[2],q[1]-ctr[1]))

    return reduce(vcat, [reshape(q,1,2) for q in vertices])  # col 1: rho, col 2: v

end



# ---------------------- PROJECTION ELLIPSOIDALE EXACTE ----------------------

# K_2 = {r : r' H r <= 1}. Projection sur y=T*r :

# y' (T*inv(H)*T')^{-1} y <= 1.

function boundary_quadratic(H, plane; ntheta=NTHETA)

    T = plane == :rho_v ? [3.0 0.0 0.0; 0.0 0.85 0.0] :

                          [3.0 0.0 0.0; 0.0 0.0 21.0]

    C = Symmetric(T * (H \ T'))

    L = cholesky(C).L

    ts = range(0.0, 2pi; length=ntheta)

    pts = [L*[cos(t),sin(t)] for t in ts]

    return [p[1] for p in pts], [p[2] for p in pts]

end



# ---------------------- PROJECTION QUARTIQUE --------------------------------

# B4(r) = m(r)' Q m(r) - 1, m=[r1,r2,r3,r1^2,r1*r2,r1*r3,r2^2,r2*r3,r3^2].

# Pour y=(rho,v) ou y=(rho,w), on MINIMISE B4 sur la coordonnee eliminee.

# Cette minimisation est globale : les candidats sont toutes les racines

# reelles du polynome derive (degre <= 3). Aucun calcul ellipsoidal pour B4.

function basis9(r)

    a,b,c = r

    return [a,b,c,a*a,a*b,a*c,b*b,b*c,c*c]

end



function coeffs_in_free_coordinate(Q, base::Vector{Float64}, free::Int)

    e = zeros(3); e[free]=1.0

    a = basis9(base)

    p = basis9(base + e)

    n = basis9(base - e)

    b = (p - n) / 2.0

    c = (p + n) / 2.0 - a

    Qa = Q*a; Qb = Q*b; Qc = Q*c

    return [dot(a,Qa)-1.0,

            2.0*dot(a,Qb),

            dot(b,Qb)+2.0*dot(a,Qc),

            2.0*dot(b,Qc),

            dot(c,Qc)]

end



function roots_real(coeff::Vector{Float64})

    deg=length(coeff)-1

    scale=maximum(abs.(coeff))

    scale==0 && return Float64[]

    while deg>0 && abs(coeff[deg+1]) <= 1e-13*scale

        deg -= 1

    end

    deg==0 && return Float64[]

    deg==1 && return [-coeff[1]/coeff[2]]

    C=zeros(deg,deg)

    for k in 1:(deg-1)

        C[k+1,k]=1.0

    end

    C[:,deg] .= -coeff[1:deg] ./ coeff[deg+1]

    values=eigvals(C)

    return [real(z) for z in values if abs(imag(z)) <= 1e-7*(1.0+abs(real(z)))]

end



function polyval(c, t)

    val=0.0

    for k in reverse(eachindex(c))

        val=val*t+c[k]

    end

    return val

end



function projected_min(Q, rho::Float64, other::Float64, plane)

    if plane==:rho_v

        base=[rho/3.0,other/0.85,0.0]  # eliminate w/21

        free=3

    else

        base=[rho/3.0,0.0,other/21.0] # eliminate v/0.85

        free=2

    end

    co=coeffs_in_free_coordinate(Q,base,free)

    dco=[k*co[k+1] for k in 1:(length(co)-1)]

    roots=roots_real(dco)

    isempty(roots) && error("Minimisation quartique : aucune racine stationnaire reelle")

    return minimum(polyval(co,t) for t in roots)

end



# Les fonctions des deux sources utilisent une frontiere radiale.

# On conserve cette representation et controlons numeriquement qu'il n'y a

# pas d'autre composante sur les rayons testes. En cas d'echec, on arrete.

function boundary_quartic(Q, plane; ntheta=NTHETA)

    sx,sy = plane==:rho_v ? (3.0,0.85) : (3.0,21.0)

    bx=Float64[]; by=Float64[]

    for theta in range(0.0,2pi;length=ntheta)

        cx=cos(theta); cy=sin(theta)

        f(s)=projected_min(Q, Float64(s*sx*cx), Float64(s*sy*cy),plane)

        f(0.0)<0.0 || error("L'origine n'est pas interieure a K4")

        lo=0.0; hi=1.0

        for _ in 1:24

            f(hi)>0.0 && break

            hi *= 2.0

        end

        f(hi)>0.0 || error("La frontiere de K4 n'a pas pu etre encadree")

        for _ in 1:43

            mid=(lo+hi)/2

            if f(mid)<=0.0

                lo=mid

            else

                hi=mid

            end

        end

        rad=(hi+lo)/2

        # Precaution : detecter d'eventuelles reinclusions au-dela du bord.

        # Ceci ne constitue pas une preuve de simplicite radiale universelle.

        for test_rad in (1.15*rad,1.5*rad,2.0*rad,3.0*rad)

            f(test_rad)>-1e-6 || error("Projection quartique non radialement simple detectee. Utiliser une isocourbe sur grille.")

        end

        push!(bx,rad*sx*cx)

        push!(by,rad*sy*cy)

    end

    return bx,by

end



function polygon_area(x,y)

    s=0.0

    for i in 1:(length(x)-1)

        s += x[i]*y[i+1]-x[i+1]*y[i]

    end

    return abs(s)/2

end



# ---------------------- DEUX FIGURES : (v,rho) ET (w,rho) ---------------------

# rho = x-w. Les deux fonctions boundary_... retournent des points (rho,autre).

# Pour respecter les axes demandes, le traceur inverse systematiquement x et y.

function projection_plot(H,Q,section,plane)

    # Coordonnees des frontieres dans l'ordre interne (rho,v) ou (rho,w)

    rho2,other2 = boundary_quadratic(H,plane)

    rho4,other4 = boundary_quartic(Q,plane)



    rho_lim=MARGIN*max(maximum(abs.(section[:,1])),

                       maximum(abs.(rho2)),maximum(abs.(rho4)))



    if plane==:rho_v

        # Axe horizontal v, axe vertical rho=x-w

        admissible_x=section[:,2]

        admissible_y=section[:,1]

        other_lim=MARGIN*max(maximum(abs.(section[:,2])),

                             maximum(abs.(other2)),maximum(abs.(other4)))

        xlabel=L"v"

        plane_name="(v,x-w)"

        title_tag="v_x_minus_w"

    elseif plane==:rho_w

        # Bande admissible illimitee en w : restriction graphique explicite

        other_lim=1.40*max(maximum(abs.(other2)),maximum(abs.(other4)),1.0)

        rho_min,rho_max=extrema(section[:,1])

        admissible_x=[-other_lim,other_lim,other_lim,-other_lim]

        admissible_y=[rho_min,rho_min,rho_max,rho_max]

        xlabel=L"w"

        plane_name="(w,x-w)"

        title_tag="w_x_minus_w"

    else

        error("Projection inconnue : $plane")

    end



    p=plot(Shape(admissible_x,admissible_y);

        fillcolor=:sandybrown,fillalpha=ALPHA_ADMISSIBLE,

        linecolor=:sienna,linewidth=2.0,

        # === MODIFICATION LEGENDE 2D (2/3) ===
        # Le polygone est trace sans legende automatique :
        # on ajoutera un carre colore identique a la figure 3D.
        label="",

        xlabel=xlabel,ylabel=L"x-w",

        xlims=(-other_lim,other_lim),ylims=(-rho_lim,rho_lim),

        legend=:outertop,legend_column=1,

        legendfontsize=FONT_LEGEND_2D,guidefontsize=FONT_AXES,

        tickfontsize=FONT_TICKS,

        foreground_color_legend=:black,background_color_legend=:white,

        size=SIZE_EACH,grid=false,framestyle=:box,

        title="",background_color=:white)



    # Inversion effective des abscisses et ordonnees : Shape(other,rho)

    plot!(p,Shape(other2,rho2);

        fillcolor=:red,fillalpha=ALPHA_QUADRATIC,

        linecolor=:firebrick,linewidth=LINE_WIDTH,linestyle=:dash,

        # MODIFICATION LEGENDE 2D : etiquette geree plus bas
        label="")



    plot!(p,Shape(other4,rho4);

        fillcolor=:steelblue,fillalpha=ALPHA_QUARTIC,

        linecolor=:navy,linewidth=LINE_WIDTH,

        # MODIFICATION LEGENDE 2D : etiquette geree plus bas
        label="")



    # === MODIFICATION LEGENDE 2D (3/3) ===
    # Les coordonnees NaN ne dessinent aucun point dans le plan.
    # Les carres apparaissent toutefois dans la legende, comme dans la 3D.
    # ^(2) : certificat quadratique/ellipsoidal ; ^(4) : certificat quartique.
    # Les noms affiches correspondent aux PROJECTIONS des ensembles.
    scatter!(p, [NaN], [NaN];
        markershape=:rect, markersize=LEGEND_MARKER_SIZE_2D,
        markercolor=:sandybrown, markerstrokecolor=:sienna,
        label=L"\mathcal{Z}_{a}^{\phi}")

    scatter!(p, [NaN], [NaN];
        markershape=:rect, markersize=LEGEND_MARKER_SIZE_2D,
        markercolor=:red, markerstrokecolor=:firebrick,
        label=L"\mathcal{Z}^{\phi}: \mathrm{ellipsoidal}")

    scatter!(p, [NaN], [NaN];
        markershape=:rect, markersize=LEGEND_MARKER_SIZE_2D,
        markercolor=:steelblue, markerstrokecolor=:navy,
        label=L"\mathcal{Z}^{\phi}")
    # === FIN MODIFICATION LEGENDE 2D ===

    @printf("Projection %s : aire K2 = %.6f ; aire K4 = %.6f\n",

        plane_name,polygon_area(rho2,other2),polygon_area(rho4,other4))

    return p,title_tag

end



function main()

    opts=get_options(Base.ARGS)

    outdir=opts["outdir"]

    mkpath(outdir)

    roots=search_roots(opts)

    println("Dossiers inspectes : ", join(roots, " ; "))

    Q, quartic_path = quartic_matrix(opts, roots)

    H = quadratic_matrix(opts, roots, quartic_path)

    section=admissible_polygon()



    p1,_=projection_plot(H,Q,section,:rho_v)

    p2,_=projection_plot(H,Q,section,:rho_w)



    f1=joinpath(outdir,"superposition_v_x_moins_w.pdf")

    f2=joinpath(outdir,"superposition_w_x_moins_w.pdf")

    f3=joinpath(outdir,"superposition_deux_projections.pdf")

    savefig(p1,f1)

    savefig(p2,f2)

    both=plot(p1,p2;layout=(1,2),size=SIZE_COMBINED,

              background_color=:white)

    savefig(both,f3)

    if SAVE_PNG

        savefig(p1,replace(f1,".pdf"=>".png"))

        savefig(p2,replace(f2,".pdf"=>".png"))

        savefig(both,replace(f3,".pdf"=>".png"))

    end

    try

        display(both)

    catch err

        @warn "Impossible d'ouvrir la fenetre interactive, les PDF sont deja enregistres" exception=(err, catch_backtrace())

    end

    println("\nFigures enregistrees dans : ",outdir)

    println("1) ",f1)

    println("2) ",f2)

    println("3) ",f3)

    println("IMPORTANT : une comparaison graphique n'est pas une validation SOS des matrices rechargees.")

end



main()



end # let : variables et fonctions locales ; pas de constantes Main.N/Main.Ni
