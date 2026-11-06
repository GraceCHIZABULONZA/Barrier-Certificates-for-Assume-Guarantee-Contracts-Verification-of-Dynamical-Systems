# =============================================================================

# WEAK assume-guarantee robot certificate, delta = infinity.

# Non-ellipsoidal quartic SOS barrier with a genuinely nonzero CUBIC term.

#

# Requires Julia packages: JuMP, MosekTools, SumOfSquares, DynamicPolynomials,

#                          Plots, LaTeXStrings

# Installation:

#   import Pkg

#   Pkg.add(["JuMP", "MosekTools", "SumOfSquares", "DynamicPolynomials",

#            "Plots", "LaTeXStrings"])

# Run: julia robot_weak_infty_quartic_SOS_LaTeX_3D.jl

# Replot using certified Q CSV without re-solving:

#      julia robot_weak_infty_quartic_SOS_LaTeX_3D.jl --figures-only

# A valid MOSEK license is needed. Nothing is saved as "certified" on failure.

#

# IMPORTANT: this is the LINEAR robot of the uploaded source code, not the

# nonlinear rational model discussed earlier.

# Assumption UNCHANGED: |w+ - sigma*w| <= D0, with D0 = 0.2.

# =============================================================================



module RobotQuarticSOS



using LinearAlgebra

using Random

using DelimitedFiles

using Printf

using JuMP

using SumOfSquares

using DynamicPolynomials

using Plots

using LaTeXStrings

import MosekTools



const MOI = JuMP.MOI



# ============================================================================

# CONFIGURATION GRAPHIQUE - modifier les valeurs ici uniquement

# ============================================================================

const FONT_AXES        =30  # labels des axes 2D et 3D

const FONT_TICKS       = 21   # nombres sur les axes

const FONT_LEGEND      = 40   # texte LaTeX des legendes

# MODIFICATION LEGENDE 2D : taille des noms des ensembles dans les figures 2D.
# Augmenter cette valeur (ex. 30, 34, 40) sans modifier la legende 3D.
const FONT_LEGEND_2D = 30
# MODIFICATION LEGENDE 2D : taille des carres colores devant les noms.
const LEGEND_MARKER_SIZE_2D = 14

const FONT_PANELS_AXES = 24   # labels de la figure a quatre panneaux

const FONT_PANELS_TICKS= 19

const FONT_PANELS_LEGEND=18

const FIGURE_2D_SIZE   = (1300, 950)

const FIGURE_3D_SIZE   = (1500, 1150)

const FIGURE_4P_SIZE   = (1300, 1850)

const LINE_WIDTH       = 3.2

const INVARIANT_ALPHA  = 0.68

const ADMISSIBLE_ALPHA = 0.14

const CAMERA_3D        = (48, 23)    # (azimut, elevation), en degres

const SURFACE_NTHETA   = 96         # resolution de la surface 3D

const SURFACE_NPHI     = 49

const W_CROP_FACTOR    = 1.18       # coupe de Z_a, non bornee dans w

const DRAW_3D_FIGURE   = true



# Configuration generale ; les etiquettes mathematiques sont LaTeXStrings.

gr()

default(

    guidefontsize=FONT_AXES,

    tickfontsize=FONT_TICKS,

    legendfontsize=FONT_LEGEND,

    fontfamily="Computer Modern",

    background_color=:white,

    background_color_legend=:white,

    foreground_color_legend=:black,

    grid=false,

)



const alpha = 0.8

const beta = 0.15

const sigma = 0.99

const eps_sys = 5.0e-5

const D0 = 0.2

const D1 = 1.0

const D2 = 1.0

const D3 = 2.9

const DSCALE = 1.015



const A = [1.0 1.0 0.0;

           -beta 1.0-alpha beta;

           0.0 0.0 sigma]

const E = Matrix(Diagonal([eps_sys, eps_sys, D0]))

const N = Diagonal([1/3.0, 1/0.85, 1/21.0]) *

          [1.0 0.0 -1.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

const Ni = inv(N)

const Ar = N*A*Ni

const Wr = N*E



const AZ = [[0.0, 1.0, 0.0],

            [-beta, -alpha, beta],

            [1.0, 1.0, -1.0]]

const AR = [Ni' * a for a in AZ]

const DB = [D1, D2, D3]



# A symmetric quadratic form in this 9-dimensional vector contains

# quadratic, cubic and quartic terms in r.  The first three entries are

# linear monomials and the last six are quadratic monomials.

function monomial_basis(r, degree::Int)

    if degree == 2

        return [r[1], r[2], r[3]]

    elseif degree == 4

        a,b,c = r

        return [a,b,c, a^2,a*b,a*c,b^2,b*c,c^2]

    else

        error("Only degree 2 and degree 4 are implemented")

    end

end



function poly_quadratic_form(Q, mons)

    n = length(mons)

    return sum(Q[i,j]*mons[i]*mons[j] for i in 1:n, j in 1:n)

end



# """

#     synthesize(degree; asym=0.0, kappa=[0.45,0.45,2.0])



# Find B(r) = m(r)' Q m(r)-1 and SOS multipliers for the fixed-scaling

# condition

#     B(r)-DSCALE*B(Ar*r+Wr*u)-sum_i lambda_i(r,u)*(1-u_i^2) >= 0,

# for all r,u.  Each lambda_i is an SOS polynomial of degree <= 2.



# For degree=4, Q[1,4]=asym fixes a nonzero coefficient 2*asym of r1^3,

# which makes the BARRIER genuinely quartic and asymmetric. Q positive definite ensures a compact sublevel set.



# The guarantee checks are q_j(r) + kappa_j*B(r) SOS, where

# q_j = (D_j-eps_sys)^2 - (a_j' Ni r)^2.



# Fixed multipliers kappa_j and fixed DSCALE make this a convex SDP.

# """



function synthesize(degree::Int; asym=0.0, kappa=(0.45, 0.45, 2.0))

    @assert degree == 2 || degree == 4

    @polyvar r[1:3] u[1:3]

    model = Model(MosekTools.Optimizer)

    set_silent(model)



    m = monomial_basis(r, degree)

    n = length(m)

    @variable(model, Q[1:n,1:n], Symmetric)

    qfloor = degree == 4 ? 1.0e-7 : 1.0e-6

    @constraint(model, Symmetric([Q[i,j] - (i == j ? qfloor : 0.0)

                                   for i=1:n,j=1:n]) in PSDCone())

    if degree == 4

        # A nonzero cubic term is deliberately enforced.

        @constraint(model, Q[1,4] == asym)

        @constraint(model, Q[4,4] >= 2.0e-4)

        @constraint(model, Q[7,7] >= 2.0e-4)

        @constraint(model, Q[9,9] >= 2.0e-4)

    end



    B = poly_quadratic_form(Q, m) - 1.0

    rn = [sum(Ar[i,j]*r[j] for j in 1:3) +

          sum(Wr[i,j]*u[j] for j in 1:3) for i in 1:3]

    Bn = poly_quadratic_form(Q, monomial_basis(rn, degree)) - 1.0



    # SOS S-procedure on the compact noise cube |u_i| <= 1.

    linmons = [one(r[1]), r..., u...]

    L = [@variable(model, [1:7,1:7], Symmetric) for k in 1:3]

    lambda = Vector{Any}(undef,3)

    for k in 1:3

        @constraint(model, Symmetric(L[k]) in PSDCone())

        lambda[k] = poly_quadratic_form(L[k],linmons)

    end

    invexpr = B - DSCALE*Bn - sum(lambda[k]*(1-u[k]^2) for k in 1:3)

    @constraint(model, invexpr in SOSCone())



    for j in 1:3

        ajr = sum(AR[j][k]*r[k] for k in 1:3)

        gj = (DB[j]-eps_sys)^2 - ajr^2

        @constraint(model, gj + kappa[j]*B in SOSCone())

    end



    # Minimizing trace(Q) is a heuristic to make the sublevel set large.

    # It is NOT an exact area/volume-maximization objective.

    @objective(model, Min, sum(Q[i,i] for i in 1:n))

    optimize!(model)

    st = termination_status(model)

    ps = primal_status(model)

    println("Degree $degree, cubic half-coefficient=$asym : status=$st, primal=$ps")

    if !(st in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)) ||

       ps != MOI.FEASIBLE_POINT || !has_values(model)

        return nothing

    end

    Qval = Matrix(value.(Q))

    mineig = minimum(eigvals(Symmetric(Qval)))

    if mineig <= 0 || !all(isfinite, Qval)

        @warn "Nonpositive or invalid Gram matrix; rejecting solution" mineig

        return nothing

    end

    if degree == 4 && abs(Qval[1,4]) < 1e-8

        @warn "The nonzero cubic coefficient was lost numerically"

        return nothing

    end

    return (; Q=Qval, degree, asym, st, ps, mineig, objective=objective_value(model))

end



# ----------------- numerical barrier and projections -------------------------



function basis_num(r, degree::Int)

    a,b,c = r

    return degree == 2 ? [a,b,c] : [a,b,c,a*a,a*b,a*c,b*b,b*c,c*c]

end



function bnorm(cert,r)

    m = basis_num(r,cert.degree)

    return dot(m, cert.Q*m) - 1.0

end

barrier(cert,z) = bnorm(cert,N*z)



# Polynomial in the eliminated normalized variable t.  The remaining two

# normalized variables are fixed in "base"; t replaces coordinate "free".

function mulpoly(a,b)

    c=zeros(length(a)+length(b)-1)

    for i in eachindex(a), j in eachindex(b)

        c[i+j-1] += a[i]*b[j]

    end

    return c

end



function polycoef_free(cert, base::Vector{Float64}, free::Int)

    p = [[base[i], i == free ? 1.0 : 0.0] for i in 1:3]

    m = if cert.degree == 2

        p

    else

        [p[1],p[2],p[3], mulpoly(p[1],p[1]),

         mulpoly(p[1],p[2]),mulpoly(p[1],p[3]),

         mulpoly(p[2],p[2]),mulpoly(p[2],p[3]),

         mulpoly(p[3],p[3])]

    end

    co = zeros(5)

    for i in eachindex(m), j in eachindex(m)

        pr = mulpoly(m[i],m[j])

        for k in eachindex(pr)

            co[k] += cert.Q[i,j]*pr[k]

        end

    end

    co[1] -= 1.0

    return co

end



function all_real_roots(co::Vector{Float64})

    # Ascending coefficients, strip small leading coefficients.

    degree = length(co)-1

    scale = max(1.0, maximum(abs.(co)))

    while degree > 0 && abs(co[degree+1]) <= 1e-13*scale

        degree -= 1

    end

    degree == 0 && return Float64[]

    degree == 1 && return [-co[1]/co[2]]

    C=zeros(degree,degree)

    for j in 1:degree-1

        C[j+1,j]=1.0

    end

    C[:,degree] .= -co[1:degree]/co[degree+1]

    eig = eigvals(C)

    return [real(t) for t in eig if abs(imag(t)) <= 1e-7*(1+abs(real(t)))]

end



function polyval(co,t)

    v=0.0

    for k in reverse(eachindex(co))

        v=v*t+co[k]

    end

    return v

end



function min_projected_barrier(cert, horizontal::Float64, vertical::Float64,

                               is_vr::Bool)

    if is_vr

        # (v, x-w) : eliminate w/21, i.e. normalized r3.

        base=[vertical/3.0,horizontal/0.85,0.0]

        free=3

    else

        # (w, x-w) : eliminate v/0.85, i.e. normalized r2.

        base=[vertical/3.0,0.0,horizontal/21.0]

        free=2

    end

    co=polycoef_free(cert,base,free)

    dc=[k*co[k+1] for k in 1:4]

    roots=all_real_roots(dc)

    isempty(roots) && error("Projection minimization has no real stationary point")

    return minimum(polyval(co,t) for t in roots)

end



function boundary_of_projection(cert,is_vr::Bool; ntheta=181)

    scalex = is_vr ? 0.85 : 21.0

    scaley = 3.0

    X = Float64[]; Y = Float64[]

    for theta in range(0.0,2pi,length=ntheta)

        f(s) = min_projected_barrier(cert,

                   Float64(s*scalex*cos(theta)),

                   Float64(s*scaley*sin(theta)),is_vr)

        @assert f(0.0) < 0.0

        lo=0.0; hi=1.0

        for _ in 1:20

            f(hi) > 0 && break

            hi *= 2

        end

        f(hi) > 0 || error("Failed to bracket the boundary")

        for _ in 1:46

            mid = (lo+hi)/2

            if f(mid) <= 0

                lo=mid

            else

                hi=mid

            end

        end

        push!(X,lo*scalex*cos(theta))

        push!(Y,lo*scaley*sin(theta))

    end

    return X,Y

end



# --- Physical admissible set, unchanged from the uploaded quadratic script. ---

function admissible_vertices()

    a,b,c=D1-eps_sys,D2-eps_sys,D3-eps_sys

    cuts=[(0.0,1.0,a),(0.0,-1.0,a),

          (beta,alpha,b),(-beta,-alpha,b),

          (1.0,1.0,c),(-1.0,-1.0,c)]

    verts=Vector{Float64}[]

    for i in 1:length(cuts)-1, j in i+1:length(cuts)

        aa,bb,cc=cuts[i]; dd,ee,ff=cuts[j]

        mat=[aa bb;dd ee]

        abs(det(mat)) < 1e-12 && continue

        q=mat\[cc,ff]

        if all(h1*q[1]+h2*q[2] <= h3+1e-9 for (h1,h2,h3) in cuts) &&

           !any(norm(q-v0)<1e-8 for v0 in verts)

            push!(verts,q)

        end

    end

    ctr=[sum(q[1] for q in verts),sum(q[2] for q in verts)]/length(verts)

    sort!(verts,by=q->atan(q[2]-ctr[2],q[1]-ctr[1]))

    return reduce(vcat,[reshape(q,1,2) for q in verts]) # (x-w,v)

end



# MODIFICATION LEGENDE 2D : meme methode que dans save_3d_sets().
# On ajoute un carre colore a une position NaN : invisible sur les axes,
# mais visible dans la legende avec le nom LaTeX de l'ensemble.
function ajouter_legende_2d!(p, couleur, bordure, description)
    scatter!(p, [NaN], [NaN];
             markershape=:rect,
             markersize=LEGEND_MARKER_SIZE_2D,
             markercolor=couleur,
             markerstrokecolor=bordure,
             label=description)
    return p
end

# MODIFICATION LEGENDE 2D : les Shapes servent au remplissage,
# les carres NaN servent exclusivement aux entrees de legende.

function save_projection(cert,is_vr; outdir=joinpath(@__DIR__,"figures_quartic"))

    mkpath(outdir)

    sx,sy=boundary_of_projection(cert,is_vr)

    section=admissible_vertices()

    if is_vr

        ax,ay = section[:,2],section[:,1]

        xlab=L"v"; ylab=L"x-w"; basename="projection_v_x_minus_w"

        xlim=1.15*max(maximum(abs.(sx)),maximum(abs.(ax)))

        ylim=1.15*max(maximum(abs.(sy)),maximum(abs.(ay)))

    else

        rmin,rmax=extrema(section[:,1])

        xlim=1.17*max(maximum(abs.(sx)), 1.0)

        ylim=1.15*max(maximum(abs.(sy)),abs(rmin),abs(rmax))

        ax=[-xlim,xlim,xlim,-xlim]

        ay=[rmin,rmin,rmax,rmax]

        xlab=L"w"; ylab=L"x-w"; basename="projection_w_x_minus_w"

    end

    p=plot(Shape(ax,ay);

        fillcolor=:sandybrown, fillalpha=0.28,

        linecolor=:sienna, linewidth=LINE_WIDTH,

        # MODIFICATION LEGENDE 2D : desactiver la legende automatique du Shape.
        label=false,

        xlabel=xlab,ylabel=ylab,

        xlims=(-xlim,xlim),ylims=(-ylim,ylim),

        size=FIGURE_2D_SIZE,

        legend=:outertop,legend_column=1,

        guidefontsize=FONT_AXES,tickfontsize=FONT_TICKS,

        legendfontsize=FONT_LEGEND_2D, # MODIFICATION LEGENDE 2D : police independante.

        framestyle=:box,grid=false,title="",background_color=:white)

    plot!(p, Shape(sx,sy);

        fillcolor=:steelblue,fillalpha=0.50,

        linecolor=:navy,linewidth=LINE_WIDTH,

        # MODIFICATION LEGENDE 2D : desactiver aussi la legende du second Shape.
        label=false)

    # MODIFICATION LEGENDE 2D : memes noms mathematiques et couleurs que la 3D.
    # La figure montre les projections de ces deux ensembles.
    ajouter_legende_2d!(p, :sandybrown, :sienna,
                        L"\mathcal{Z}_{a}^{\phi}")
    ajouter_legende_2d!(p, :steelblue, :navy,
                        L"\mathcal{Z}^{\phi}=\{z:B_{\infty}(z)\leq 0\}")

    path=joinpath(outdir,basename*"_quartic.pdf")

    savefig(p,path)

    println("Saved: ",path)

    return (sx,sy)

end



# =============================================================================

# FIGURE 3D : ensemble admissible Z_a^w ET ensemble invariant quartique K_B4

# Coordonnees de la figure : (x-w, v, w).

# Z_a^w est un cylindre NON BORNE suivant w. Pour l'affichage seulement,

# on represente sa portion |w| <= w_crop, sans la presenter comme l'ensemble total.

# La surface de K_B4 est obtenue par resolution du polynome B4(t*d) = 0

# sur chaque direction d dans les coordonnees normalisees r = N*z.

# Aucune formule de projection ellipsoidale n'est utilisee pour K_B4.

# =============================================================================



function ray_polynomial(cert, direction::Vector{Float64})

    a, b, c = direction

    if cert.degree == 2

        quad = dot(direction, cert.Q * direction)

        return [-1.0, 0.0, quad, 0.0, 0.0]

    end

    m2 = [a*a, a*b, a*c, b*b, b*c, c*c]

    q2 = dot(direction, cert.Q[1:3,1:3] * direction)

    q3 = 2.0 * dot(direction, cert.Q[1:3,4:9] * m2)

    q4 = dot(m2, cert.Q[4:9,4:9] * m2)

    return [-1.0, 0.0, q2, q3, q4]

end



function positive_ray_radius(cert, direction::Vector{Float64})

    co = ray_polynomial(cert, direction)

    roots = sort([t for t in all_real_roots(co) if t > 1e-9])

    isempty(roots) && error("B4(t*d)=0 : aucune racine positive (verifier Q)")

    # Le sous-niveau d'un quartique general peut ne pas etre radialement simple.

    # En presence de plusieurs composantes le trace spheroidal ne serait pas exact.

    if length(roots) != 1

        error("La surface B4=0 n'est pas radialement simple dans une direction. " *

              "Ne pas utiliser une surface spheroidale pour ce certificat. " *

              "Employer une extraction d'isosurface 3D (marching cubes).")

    end

    return roots[1]

end



function invariant_surface_3d(cert;

        ntheta::Int=SURFACE_NTHETA, nphi::Int=SURFACE_NPHI)

    X = zeros(nphi, ntheta) # x-w

    Y = zeros(nphi, ntheta) # v

    Z = zeros(nphi, ntheta) # w

    for (j,phi) in enumerate(range(0.0, pi; length=nphi))

        for (i,theta) in enumerate(range(0.0, 2pi; length=ntheta))

            direction = Float64[

                sin(phi)*cos(theta),

                sin(phi)*sin(theta),

                cos(phi),

            ]

            radius = positive_ray_radius(cert, direction)

            r = radius .* direction

            # r = [(x-w)/3, v/0.85, w/21]

            X[j,i] = 3.0*r[1]

            Y[j,i] = 0.85*r[2]

            Z[j,i] = 21.0*r[3]

        end

    end

    return X, Y, Z

end



function save_3d_sets(cert; outdir=joinpath(@__DIR__,"figures_quartic"))

    mkpath(outdir)

    X,Y,Z = invariant_surface_3d(cert)

    section = admissible_vertices()   # ordered vertices: (x-w, v)

    n = size(section,1)

    w_crop = W_CROP_FACTOR * maximum(abs.(Z))

    @assert w_crop > 0.0



    # Draw the bounded invariant's boundary surface B4(z)=0.

    p = plot(; xlabel=L"x-w", ylabel=L"v", zlabel=L"w",

        size=FIGURE_3D_SIZE, camera=CAMERA_3D,

        guidefontsize=FONT_AXES, tickfontsize=FONT_TICKS,

        legendfontsize=FONT_LEGEND,

        legend=:outertop, legend_column=1,

        title="", framestyle=:box, grid=false,

        background_color=:white, colorbar=false)



    # The *admissible* set is an unbounded polyhedral cylinder.  Display the

    # clipped lateral facets and the upper/lower crop edges in transparent orange.

    for k in 1:n

        l = mod1(k+1, n)

        x1,v1 = section[k,1], section[k,2]

        x2,v2 = section[l,1], section[l,2]

        xx = [x1 x2; x1 x2]

        yy = [v1 v2; v1 v2]

        zz = [-w_crop -w_crop; w_crop w_crop]

        surface!(p,xx,yy,zz;

            c=:sandybrown, alpha=ADMISSIBLE_ALPHA,

            linealpha=0.0, colorbar=false, label=false)

        for wfix in (-w_crop, w_crop)

            plot!(p,[x1,x2],[v1,v2],[wfix,wfix];

                color=:sienna, linewidth=1.5, label=false)

        end

        plot!(p,[x1,x1],[v1,v1],[-w_crop,w_crop];

            color=:sienna, linewidth=1.0, alpha=0.6, label=false)

    end



    # Boundary of the quartic invariant; PDF can contain rasterized 3D shading

    # depending on the GR backend, but the resulting file is a standard PDF.

    surface!(p,X,Y,Z;

        c=:steelblue, alpha=INVARIANT_ALPHA,

        linealpha=0.0, colorbar=false, label=false)



    # Explicit legend entries with LATEX labels and square-colored swatches.

    # NaN dummy points are not displayed in the 3D space.

    scatter!(p,[NaN],[NaN],[NaN]; markershape=:rect, markersize=12,

        markercolor=:sandybrown, markerstrokecolor=:sienna,

        label=L"\mathcal{Z}_{a}^{\phi}\ ")

    scatter!(p,[NaN],[NaN],[NaN]; markershape=:rect, markersize=12,

        markercolor=:steelblue, markerstrokecolor=:navy,

        label=L"\mathcal{Z}^{\phi}=\{z:B_{\infty}(z)\leq 0\}")



    xmax = max(maximum(abs.(X)), maximum(abs.(section[:,1]))) * 1.10

    ymax = max(maximum(abs.(Y)), maximum(abs.(section[:,2]))) * 1.10

    plot!(p; xlims=(-xmax,xmax), ylims=(-ymax,ymax),

        zlims=(-w_crop,w_crop))



    file = joinpath(outdir,"invariant_et_admissible_3D_quartique.pdf")

    savefig(p,file)

    println("Saved 3D: ",file)

    println("Admissible cylinder has only been clipped for DISPLAY: |w| <= ",w_crop)

    return file

end



function save_comparison(c2,c4; outdir=joinpath(@__DIR__,"figures_quartic"))

    mkpath(outdir)

    for (is_vr,basename,xlab) in

        ((true,"compare_v_x_minus_w",L"v"),(false,"compare_w_x_minus_w",L"w"))

        px,py=boundary_of_projection(c2,is_vr)

        qx,qy=boundary_of_projection(c4,is_vr)

        p=plot(Shape(px,py);fillcolor=:darkorange,fillalpha=0.25,

               linecolor=:sienna,linewidth=2,

               # MODIFICATION LEGENDE 2D : label du Shape desactive.
                label=false,

               xlabel=xlab,ylabel=L"x-w",

               legend=:outertop,legend_column=2,

               guidefontsize=FONT_AXES,tickfontsize=FONT_TICKS,

               legendfontsize=FONT_LEGEND_2D, # MODIFICATION LEGENDE 2D.

               size=FIGURE_2D_SIZE,framestyle=:box,grid=false,title="")

        plot!(p,Shape(qx,qy);fillcolor=:steelblue,fillalpha=0.40,

              linecolor=:navy,linewidth=LINE_WIDTH,

              # MODIFICATION LEGENDE 2D : label du Shape desactive.
                label=false)

        # MODIFICATION LEGENDE 2D : la comparaison montre deux ensembles
        # a horizon infini : l'un quadratique (ellipsoidal), l'autre quartique.
        ajouter_legende_2d!(p, :darkorange, :sienna,
                            L"\mathcal{K}_{B_{\infty}}^{(2)}\;\mathrm{(ellipsoidal)}")
        ajouter_legende_2d!(p, :steelblue, :navy,
                            L"\mathcal{K}_{B_{\infty}}^{(4)}\;\mathrm{(quartic)}")

        savefig(p,joinpath(outdir,basename*".pdf"))

    end

end



# --- Original 60-transition illustrative simulation, with original assumption.

function simulate_switching(cert; ns=60)

    zz=zeros(ns+1,3)

    g=zeros(ns,3)

    residual=zeros(ns)

    bvalues=zeros(ns+1)

    bvalues[1]=barrier(cert,zeros(3))

    for t in 1:ns

        z=Vector(zz[t,:])

        dw = t<=20 ? 0.18 : t<=40 ? (iseven(div(t-21,4)) ? 4.0 : -4.0) : -0.18

        ex=eps_sys*sin(0.37*t)

        ev=eps_sys*cos(0.61*t)

        zn=A*z+[ex,ev,dw]

        zz[t+1,:] .= zn

        residual[t]=abs(zn[3]-sigma*z[3])

        g[t,1]=abs(zn[1]-z[1])

        g[t,2]=abs(zn[2]-z[2])

        g[t,3]=abs(zn[1]-z[3])

        bvalues[t+1]=barrier(cert,zn)

    end

    @assert all(residual[1:20] .<= D0+1e-10)

    @assert all(residual[21:40] .> D0)

    @assert all(residual[41:60] .<= D0+1e-10)

    @assert maximum(bvalues[1:21]) <= 1e-5 "Initial compliant prefix must remain inside K"

    @assert all(maximum(g[1:20,j]) <= DB[j]+1e-7 for j in 1:3)

    return (;zz,g,residual,bvalues)

end



function save_four_panels(sim;outdir=joinpath(@__DIR__,"figures_quartic"))

    mkpath(outdir)

    t=1:60

    vals=[sim.residual,sim.g[:,1],sim.g[:,2],sim.g[:,3]]

    ylabs=[L"|w_{t+1}-\sigma w_t|",L"|x_{t+1}-x_t|",

           L"|v_{t+1}-v_t|",L"|x_{t+1}-w_t|"]

    bs=[D0,D1,D2,D3]

    blabs=[L"D_0",L"D_1",L"D_2",L"D_3"]

    ps=Plots.Plot[]

    for j in 1:4

        yy=vals[j]

        p=plot(;xlabel=j==4 ? L"t" : "",ylabel=ylabs[j],

               xlims=(0.5,60.5),ylims=(0.0,1.15*max(maximum(yy),bs[j])),

               size=FIGURE_4P_SIZE,title="",legend=:topright,

               guidefontsize=FONT_PANELS_AXES,

               tickfontsize=FONT_PANELS_TICKS,

               legendfontsize=FONT_PANELS_LEGEND,

               framestyle=:box,grid=false)

        vspan!(p,[20.5,40.5];color=:gray,alpha=0.15,label=false)

        plot!(p,t,yy;color=:steelblue,linewidth=2.7,label=ylabs[j])

        hline!(p,[bs[j]];color=:firebrick,linestyle=:dash,linewidth=2.2,label=blabs[j])

        push!(ps,p)

    end

    fig=plot(ps...;layout=(4,1),link=:x,size=FIGURE_4P_SIZE)

    path=joinpath(outdir,"assumption_guarantees_4panels.pdf")

    savefig(fig,path)

    println("Saved: ",path)

end



function save_barrier_coefficients(cert,path)

    mkpath(dirname(path))

    writedlm(path,cert.Q,',')

    println("Saved: ",path)

end



function read_saved_certificate(degree::Int, path::String)

    isfile(path) || error("Missing saved Gram matrix: " * path)

    Q = Matrix{Float64}(readdlm(path, ',', Float64))

    n = degree == 2 ? 3 : 9

    size(Q) == (n,n) || error("Gram matrix must be $(n)x$(n): " * path)

    Q = (Q + Q') / 2

    mineig = minimum(eigvals(Symmetric(Q)))

    mineig > 0 || error("Saved Q is not positive definite: " * path)

    # Reusing CSV coefficients does not itself recheck the SOS constraints.

    return (; Q, degree, asym=degree == 4 ? Q[1,4] : 0.0,

            st=:LOADED_FROM_CSV, ps=:NOT_RECHECKED, mineig, objective=NaN)

end



function main()

    println("Linear AG robot: SOS quadratic baseline vs NON-ELLIPSOIDAL quartic")

    println("Assumption unchanged |w+ - 0.99*w| <= 0.2")

    # Figures-only mode to change fonts/camera without re-running expensive SOS.

    # This requires previously saved Gram matrices from a successful SOS run.

    figures_only = "--figures-only" in ARGS

    base = nothing

    cert = nothing

    if figures_only

        println("FIGURES ONLY: reading saved Q; SOS proof is NOT rechecked")

        cert = read_saved_certificate(4, joinpath(@__DIR__,"barrier_Q_degree4.csv"))

        basefile = joinpath(@__DIR__,"barrier_Q_degree2.csv")

        if isfile(basefile)

            base = read_saved_certificate(2, basefile)

        end

    else

        # Solve baseline for an actual comparison, if possible.

        base=synthesize(2)

        if base===nothing

            println("Quadratic SOS baseline not certified for fixed kappa values.")

        else

            save_barrier_coefficients(base,joinpath(@__DIR__,"barrier_Q_degree2.csv"))

        end



        # Nonzero fixed cubic coefficients are enforced for the quartic.

        for gamma in (0.02,0.01,0.003,0.001)

            cert=synthesize(4;asym=gamma)

            cert!==nothing && break

        end

        if cert===nothing

            error("No certified quartic was found with the selected SOS relaxation. " *

                  "Try modifying the fixed kappa values, dscale or multiplier " *

                  "degrees; this is NOT evidence of contract failure.")

        end

        save_barrier_coefficients(cert,joinpath(@__DIR__,"barrier_Q_degree4.csv"))

    end

    @printf("Quartic Gram smallest eigenvalue: %.4g\n",cert.mineig)

    @printf("Coefficient of r1^3 in B4: %.6g\n",2*cert.asym)

    outdir=joinpath(@__DIR__,"figures_quartic")

    save_projection(cert,true;outdir=outdir)

    save_projection(cert,false;outdir=outdir)

    if base!==nothing

        save_comparison(base,cert;outdir=outdir)

    end

    sim=simulate_switching(cert)

    save_four_panels(sim;outdir=outdir)

    if DRAW_3D_FIGURE

        save_3d_sets(cert;outdir=outdir)

    end

    println("Quartic candidate + projection and 3D plots exported.")

    println("No conclusions about improvement are made without comparing the projections.")

end



end # module



RobotQuarticSOS.main()
