# =============================================================================
# HSCC: exact rational nonlinear robot — quartic SOS synthesis
# Weak AG semantics: finite horizon delta=5 (B0 ... B5 only).
# All B_0,...,B_5 use the SAME quartic structural template:
#                 B_i(r) = m(r)' * Q_i * m(r) - 1
# m=[r1,r2,r3,r1²,r1*r2,r1*r3,r2²,r2*r3,r3²], Q_i PSD (9 x 9).
#
# Required Julia packages:
#   import Pkg
#   Pkg.add(["JuMP","SumOfSquares","DynamicPolynomials", "MosekTools",
#            "Plots","LaTeXStrings"])
# Usage:
#   julia robot_nonlineaire_delta5_B0_B5_SOS.jl
#   julia robot_nonlineaire_delta5_B0_B5_SOS.jl --scale 1.005 --beta 0.25 --zeta 0.21
#   julia robot_nonlineaire_delta5_B0_B5_SOS.jl --figures-only
#
# Notes:
# - This code is NOT an assertion of numerical feasibility: MOSEK must run.
# - beta and zeta are NOT numerically specified in the provided paper.
#   beta=0.25 and zeta=0.21 below are EXPLICIT TEST CHOICES.
# - The rational drag is represented EXACTLY by an algebraic lifting p,
#     h(r,p)=(1+v²)*p - v³ = 0.  No truncation or cubic substitution.
# - For invariance of B0 use physical zeta (full system); for assumptions
#   use D0 (restricted system).  Guarantee does not depend on w+.
# - Sigma[r,u,p] is enforced via SOSCone. Numerical SDP certificates need
#   residual checks for a rigorous paper.
# =============================================================================
module RobotNonlinearDelta5

using LinearAlgebra
using DelimitedFiles
using Printf
using Random
using TOML
using JuMP
using SumOfSquares
using DynamicPolynomials
using Plots
using Plots.PlotMeasures: mm  # MODIFICATION FIGURES 2D : unites des marges
using LaTeXStrings
import MosekTools
const MOI = JuMP.MOI

const ALPHA1 = 0.2
const ALPHA2 = 0.15
const SIGMA  = 0.99
const ETA    = 5.0e-5
const D0     = 0.20
const D1     = 1.0
const D2     = 1.0
const D3     = 2.9
const KAPPA  = (0.45, 0.45, 2.0)
const QFLOOR = 1.0e-8
const QUARTIC_FLOOR = 1.0e-5
# ============================================================================
# MODIFICATION FIGURES 2D : tous les reglages de lecture et de mise en page.
# Pour changer UNIQUEMENT la police des noms des ensembles : FONT_LEGEND_2D.
# Ces reglages ne changent ni les certificats ni les calculs SOS.
# ============================================================================
# MODIFICATION LEGENDE : taille de police des noms des ensembles.
# Modifier ces deux valeurs, puis relancer avec --figures-only.
const FONT_LEGEND_2D = 50          # police des 3 noms dans les figures 2D separees
const FONT_LEGEND_COMBINED_2D = 24 # police des 3 noms dans la figure a 2 panneaux
const LEGEND_MARKER_2D = 40        # taille des carres colores dans la legende

# MODIFICATION LEGENDE : noms mathematiques des TROIS ensembles REPRESENTES.
# B0 : invariant pour F_S (systeme complet).
# B5 : invariant pour F_S ∩ F_A (dynamique sous hypothese).
# Il s'agit des PROJECTIONS 2D des ensembles 3D.
# Modifier ces lignes si vous souhaitez changer le texte affiche.
const LABEL_ADMISSIBLE_2D = L"\mathcal{Z}_a^{\phi}"
const LABEL_B0_2D = L"\mathcal{Z}'^{\phi}=\{z\in \mathcal{Z}\mid \mathcal{K}_{B_0}(z)\leq 0\}"
const LABEL_B5_2D = L"\mathcal{Z}^{\phi}=\{z\in \mathcal{Z}\mid \mathcal{K}_{B_5}(z)\leq 0\}"
const FONT_AXES_2D = 40           # labels v, w et x-w
const FONT_TICKS_2D = 30          # valeurs des graduations
const FIG_SIZE_2D = (2100, 1550) # taille de chaque projection (pixels)
const FIG_SIZE_COMBINED = (4400, 1850) # deux projections cote a cote
const LEFT_MARGIN_2D = 30mm       # espace pour l'axe x-w a gauche
const RIGHT_MARGIN_2D = 12mm      # marge droite
const BOTTOM_MARGIN_2D = 26mm     # espace pour l'axe v ou w en bas
const TOP_MARGIN_2D = 8mm         # espace supplementaire au-dessus
const LEGEND_POSITION_2D = :outertop # legende hors de la zone graphique
const LEGEND_COLUMNS_2D = 1       # trois lignes pour eviter les chevauchements
const PROJ_NTHETA = 181        # precision du contour radial
const PROJ_GRID = 161         # grille de secours si la frontiere n'est pas radiale
const ALPHA_ADMISSIBLE = 0.15
const ALPHA_B0 = 0.18
const ALPHA_B5 = 0.25
const OUT = joinpath(@__DIR__, "resultats_delta5_B0_B5")

function argfloat(name, default)
    i = findfirst(==(name), ARGS)
    if i === nothing
        return default
    end
    i < length(ARGS) || error("Argument manquant apres $name")
    return parse(Float64, ARGS[i+1])
end

function config()
    beta = argfloat("--beta", 0.25)  # TEST VALUE only
    zeta = argfloat("--zeta", 0.21)  # TEST VALUE only
    epsilon = argfloat("--epsilon", 5.0e-5)
    scale = argfloat("--scale", 1.015)
    a = argfloat("--a", 1.0)
    b = argfloat("--b", scale)
    d = argfloat("--d", scale)
    c = ntuple(i -> argfloat("--c$(i-1)", scale), 5)
    beta >= 0 || error("beta >= 0 requis")
    zeta >= D0 || error("Le modele du papier exige zeta >= D0")
    epsilon >= 0 || error("epsilon doit etre positif ou nul")
    all(t -> t > 0, (a,b,d,c...)) || error("Tous les coefficients a,b,c_i,d doivent etre > 0")
    a >= 1 || @warn "Avec B_i(0)=-1, a<1 rend l'inclusion a l'origine impossible."
    return (;beta,zeta,epsilon,a,b,c,d)
end

# r=((x-w)/3,v/0.85,w/21), rho=x-w
function quartic_basis(r)
    x,y,z=r
    return [x,y,z,x^2,x*y,x*z,y^2,y*z,z^2]
end

function qform(Q,m)
    return sum(Q[i,j]*m[i]*m[j] for i in eachindex(m), j in eachindex(m))
end

# Algebraic lifting for EXACT nonlinear rational dynamics.
# p=v^3/(1+v²), encoded by h=0, and 1+v²>0 for all real v.
function lifted_dynamics(r,u,p,physical_w_bound,cfg)
    rho=3*r[1]
    v=0.85*r[2]
    w=21*r[3]
    rho_next=rho+v+(1-SIGMA)*w+ETA*u[1]-physical_w_bound*u[3]
    v_next=ALPHA1*v-ALPHA2*rho-cfg.beta*p+ETA*u[2]
    w_next=SIGMA*w+physical_w_bound*u[3]
    return [rho_next/3, v_next/0.85, w_next/21]
end

function equality_drag(r,p)
    v=0.85*r[2]
    return (1+v^2)*p-v^3
end

# h = [1, r1,r2,r3, u1,u2,u3,p].
# Each lambda_j=h' L_j h is SOS of degree <=2, L_j PSD (8x8).
# An affine unrestricted multiplier nu(r,u,p) cancels the equality h_drag=0.
function add_transition_sos!(model, tag, Bleft, Qright, scale,
                              r,u,p, drag_eq,physical_w_bound,cfg, savedL)
    rnext=lifted_dynamics(r,u,p,physical_w_bound,cfg)
    Bnext=qform(Qright,quartic_basis(rnext))-1
    lin=[one(r[1]);r;u;p]
    @assert length(lin)==8
    # 3 noise bounds + 1 implied nonlinear-drag bound |p| <= |v|.
    # The fourth multiplier avoids a spurious -p^4 leading term off h=0.
    L=[@variable(model, [1:8,1:8], Symmetric) for _ in 1:4]
    for k in 1:4
        @constraint(model, Symmetric(L[k]) in PSDCone())
    end
    lambdas=[qform(L[k],lin) for k in 1:4]
    v=0.85*r[2]
    g_drag=v^2-p^2  # valid for p=v^3/(1+v^2), all real v
    nucoef=@variable(model, [1:8])
    nu=sum(nucoef[j]*lin[j] for j in 1:8)
    residual=Bleft-scale*Bnext-
             sum(lambdas[k]*(1-u[k]^2) for k in 1:3)-
             lambdas[4]*g_drag-nu*drag_eq
    @constraint(model, residual in SOSCone())
    savedL[tag]=L
    return nothing
end

# For every rational p satisfying h(r,p)=0, guarantee constraints are
# g_j >= 0, where the physical noise eta is already taken worst-case.
# g_j + kappa_j*(B-epsilon) - nu_j*h is SOS in (r,p).
# This entails B > epsilon at every inadmissible state.
function add_admissibility_sos!(model,tag,B,r,p,drag_eq,cfg)
    rho=3*r[1]
    v=0.85*r[2]
    g=[(D1-ETA)^2-v^2,
       (D2-ETA)^2-((ALPHA1-1)*v-ALPHA2*rho-cfg.beta*p)^2,
       (D3-ETA)^2-(rho+v)^2]
    lin=[one(r[1]);r;p]  # 5 monomials, degree <=1
    for j in 1:3
        mult=@variable(model, [1:5])
        nu=sum(mult[i]*lin[i] for i in 1:5)
        tau=@variable(model, lower_bound=0.0)
        residual=g[j]+KAPPA[j]*(B-cfg.epsilon)-
                 tau*(v^2-p^2)-nu*drag_eq
        @constraint(model, residual in SOSCone())
    end
    return nothing
end

function synthesize(cfg)
    @polyvar r[1:3] u[1:3] p
    drag_eq=equality_drag(r,p)
    model=Model(MosekTools.Optimizer)
    set_silent(model)
    # Six INDEPENDENT quartic barriers: B0,...,B5.
    # A common barrier is permitted but not artificially imposed.
    names=["B0","B1","B2","B3","B4","B5"]
    Qs=[@variable(model, [1:9,1:9], Symmetric) for _ in names]
    for Q in Qs
        @constraint(model,
            Symmetric([Q[i,j]-(i==j ? QFLOOR : 0.0)
                       for i in 1:9,j in 1:9]) in PSDCone())
        # Guarantee an actual quartic term in each barrier.
        @constraint(model, Q[4,4]>=QUARTIC_FLOOR)
        @constraint(model, Q[7,7]>=QUARTIC_FLOOR)
        @constraint(model, Q[9,9]>=QUARTIC_FLOOR)
    end
    Bs=[qform(Q,quartic_basis(r))-1 for Q in Qs]

    # Initialization B5(0)=-1 is automatically satisfied.
    # K_B5 subset K_B0, a>0:
    @constraint(model, Bs[6]-cfg.a*Bs[1] in SOSCone())

    savedL=Dict{String,Any}()
    # Full-system invariance of K_B0 — MUST use zeta, not D0.
    add_transition_sos!(model,"B0_full",Bs[1],Qs[1],cfg.b,
                        r,u,p,drag_eq,cfg.zeta,cfg,savedL)
    # R-recovery B_i --> B_{i+1}, i=0,...,4 — uses D0.
    for i in 1:5
        tag="B$(i-1)_to_B$i"
        add_transition_sos!(model,tag,Bs[i],Qs[i+1],cfg.c[i],
                            r,u,p,drag_eq,D0,cfg,savedL)
    end
    # Restricted invariance of K_B5 under F_S intersection F_A (uses D0).
    add_transition_sos!(model,"B5_restricted",Bs[6],Qs[6],cfg.d,
                        r,u,p,drag_eq,D0,cfg,savedL)

    # B5 sublevel set must lie in the admissible-state set.
    add_admissibility_sos!(model,"B5",Bs[6],r,p,drag_eq,cfg)

    # Linear convex surrogate for enlarging sets. Not volume maximization!
    @objective(model, Min, sum(Qs[k][j,j] for k in eachindex(Qs), j in 1:9))
    println("\nSOS quartique NON LINEAIRE — horizon delta=5 ; B0 sous F_S, B5 sous F_S inter F_A")
    println("Exact drag: p=v^3/(1+v^2), avec h=(1+v^2)p-v^3=0")
    println("beta=$(cfg.beta), zeta=$(cfg.zeta), D0=$D0")
    println("epsilon=$(cfg.epsilon), a=$(cfg.a), b=$(cfg.b), c=$(cfg.c), d=$(cfg.d)")
    println("Solveur : MOSEK ; calcul des Gram SOS...")
    flush(stdout)
    try
        optimize!(model)
    catch err
        @warn "Echec du solveur SOS/MOSEK, aucun certificat exporte" exception=(err,catch_backtrace())
        return nothing
    end
    status=termination_status(model)
    primal=primal_status(model)
    println("Statut : $status | primal=$primal")
    if !(status==MOI.OPTIMAL && primal==MOI.FEASIBLE_POINT && has_values(model))
        println("ECHEC ou statut non certifiant : aucun fichier Q/L n'est produit.")
        println("Essayer --scale 1.005, verifier beta/zeta, ou augmenter les degres SOS.")
        return nothing
    end
    Qnum=Dict(names[k]=>Matrix{Float64}(value.(Qs[k])) for k in eachindex(names))
    Lnum=Dict{String,Vector{Matrix{Float64}}}()
    for (tag,matrices) in savedL
        Lnum[tag]=[Matrix{Float64}(value.(M)) for M in matrices]
    end
    for (name,Q) in Qnum
        ev=eigmin(Symmetric((Q+Q')/2))
        @printf("min eig Q_%-5s = %.8e\n",name,ev)
        if ev < -5e-7
            @warn "PSD de Q numeriquement violee; verification necessaire" name ev
            return nothing
        end
    end
    for (tag,matrices) in Lnum, (j,L) in enumerate(matrices)
        ev=eigmin(Symmetric((L+L')/2))
        if ev < -5e-7
            @warn "PSD d'un multiplicateur violee" tag j ev
            return nothing
        end
    end
    return (;Q=Qnum,L=Lnum,config=cfg,status,primal,objective=objective_value(model))
end


# --------------------- EXPORT DES CERTIFICATS -------------------------------
# Les Q en CSV ont la pleine precision. Les matrices L des multiplicateurs
# sont aussi exportees pour faciliter le controle numerique des SOS.
function export_result(result)
    mkpath(OUT)
    cfg=result.config
    open(joinpath(OUT,"parametres.toml"),"w") do io
        TOML.print(io, Dict(
            "alpha1"=>ALPHA1,"alpha2"=>ALPHA2,"sigma"=>SIGMA,"eta"=>ETA,
            "D0"=>D0,"D1"=>D1,"D2"=>D2,"D3"=>D3,
            "beta"=>cfg.beta,"zeta"=>cfg.zeta,"epsilon"=>cfg.epsilon,
            "a"=>cfg.a,"b"=>cfg.b,"d"=>cfg.d,"c"=>collect(cfg.c)))
    end
    open(joinpath(OUT,"diagnostics_SOS.txt"),"w") do io
        println(io,"NONLINEAR ROBOT / WEAK AG / DELTA = 5")
        println(io,"Statut de resolution : $(result.status); primal=$(result.primal)")
        println(io,"beta=$(cfg.beta) et zeta=$(cfg.zeta) : CHOIX D'ESSAI, non fixes dans le manuscrit")
        println(io,"rho=x-w, r=(rho/3,v/0.85,w/21)")
        println(io,"B_i(r)=m(r)'Q_i*m(r)-1, i=0,...,5")
        println(io,"Invariance B0 : systeme complet, |w+ - sigma*w| <= zeta")
        println(io,"Invariance B5 et transitions intermediaires : |w+ - sigma*w| <= D0")
        println(io,"Objectif trace (pas maximisation exacte du volume) = $(result.objective)")
        println(io,"Verifier residus SOS et tolérances pour toute revendication de certification.")
        for i in 0:5
            key="B$i"
            Q=result.Q[key]
            writedlm(joinpath(OUT,"Q_$(key)_full.csv"),Q,',')
            @printf(io,"min eigenvalue Q_%s = %.12e | symerr = %.3e\n",
                    key,eigmin(Symmetric(Q)),norm(Q-Q',Inf))
        end
        for tag in sort(collect(keys(result.L)))
            for (i,L) in enumerate(result.L[tag])
                writedlm(joinpath(OUT,"L$(i)_$(tag)_full.csv"),L,',')
                @printf(io,"min eigenvalue L%d %s = %.12e\n",i,tag,eigmin(Symmetric(L)))
            end
        end
    end
    println("Gram Q0,...,Q5 et L exportes (pleine precision) : ",OUT)
end


# ------------------ CONTROLE NUMERIQUE PAR ECHANTILLONS ---------------------
# Ce test ne remplace PAS le certificat SOS : il sert a detecter un probleme
# numerique manifeste apres la resolution SDP.
function numbar(Q,r)
    m=Float64.(quartic_basis(r))
    return dot(m,Q*m)-1.0
end

function random_in_sublevel(Q,rng)
    direction=randn(rng,3)
    direction ./= norm(direction)
    hi=1.0
    while numbar(Q,hi.*direction)<=0.0
        hi*=2
        hi<1.0e8 || error("B semble non coercive sur un rayon")
    end
    lo=0.0
    for _ in 1:44
        mid=(lo+hi)/2
        if numbar(Q,mid.*direction)<=0.0
            lo=mid
        else
            hi=mid
        end
    end
    return (rand(rng)^(1/3)*lo).*direction
end

function numerical_spot_checks(result; n=350)
    cfg=result.config
    rng=MersenneTwister(2027)
    records=Tuple{String,Float64}[]
    # Chaque verification maximale porte sur B_destination(r_next).
    for (source,dest,bound,name) in
        vcat([("B0","B0",cfg.zeta,"F_S : B0 -> B0")],
             [("B$i","B$(i+1)",D0,"R : B$i -> B$(i+1)") for i in 0:4],
             [("B5","B5",D0,"R : B5 -> B5")])
        err=-Inf
        for _ in 1:n
            r=random_in_sublevel(result.Q[source],rng)
            v=0.85*r[2]
            p=v^3/(1+v^2)
            for _ in 1:8
                u=2.0.*rand(rng,3).-1.0
                rn=Float64.(lifted_dynamics(r,u,p,bound,cfg))
                err=max(err,numbar(result.Q[dest],rn))
            end
        end
        push!(records,(name,err))
    end
    incl=-Inf
    for _ in 1:n
        r=random_in_sublevel(result.Q["B5"],rng)
        incl=max(incl,numbar(result.Q["B0"],r))
    end
    push!(records,("inclusion B5 -> B0",incl))
    open(joinpath(OUT,"controle_echantillons.txt"),"w") do io
        println(io,"Controle aleatoire indicatif ; NE PROUVE PAS les SOS ni l'invariance globale.")
        println(io,"Points par ensemble : $n, echantillons de u par point : 8")
        for (name,maxval) in records
            @printf(io,"%-28s max B_destination = %+14.6e\n",name,maxval)
        end
    end
    for (name,maxval) in records
        maxval <= 1.0e-5 || @warn "Controle numerique echantillonne suspect" name maxval
    end
    println("Controle d'echantillons : ",joinpath(OUT,"controle_echantillons.txt"))
end

# ------------- TRUE 2D projections, minimizing over hidden variable ---------
function poly_mul(a,b)
    c=zeros(Float64,length(a)+length(b)-1)
    for i in eachindex(a),j in eachindex(b)
        c[i+j-1]+=a[i]*b[j]
    end
    return c
end

function coefficients_one_free(Q,base,free)
    z=[[base[i], i==free ? 1.0 : 0.0] for i in 1:3]
    m=[z[1],z[2],z[3],poly_mul(z[1],z[1]),poly_mul(z[1],z[2]),
       poly_mul(z[1],z[3]),poly_mul(z[2],z[2]),poly_mul(z[2],z[3]),poly_mul(z[3],z[3])]
    co=zeros(Float64,5)
    for i in 1:9,j in 1:9
        pr=poly_mul(m[i],m[j])
        for k in eachindex(pr)
            co[k]+=Q[i,j]*pr[k]
        end
    end
    co[1]-=1
    return co
end

function roots_real_ascending(co)
    degree=length(co)-1
    scale=max(1.0,maximum(abs.(co)))
    while degree>0 && abs(co[degree+1])<1e-13*scale
        degree-=1
    end
    degree==0 && return Float64[]
    degree==1 && return [-co[1]/co[2]]
    companion=zeros(Float64,degree,degree)
    for j in 1:degree-1
        companion[j+1,j]=1.0
    end
    companion[:,degree].=-co[1:degree]./co[degree+1]
    roots=eigvals(companion)
    return [real(z) for z in roots if abs(imag(z))<1e-7*(1+abs(real(z)))]
end

function horner(co,x)
    ans=0.0
    for k in reverse(eachindex(co))
        ans=ans*x+co[k]
    end
    return ans
end

function min_projected_B(Q,horizontal,vertical,is_vrho)
    base=is_vrho ? [vertical/3.0,horizontal/0.85,0.0] :
                    [vertical/3.0,0.0,horizontal/21.0]
    free=is_vrho ? 3 : 2
    co=coefficients_one_free(Q,base,free)
    dc=[k*co[k+1] for k in 1:4]
    roots=roots_real_ascending(dc)
    isempty(roots) && error("Projection non coercive: verifier Q (valeur propre minimale)")
    return minimum(horner(co,t) for t in roots)
end

# Admissible set in (v,rho) EXACT for the rational robot.
# Independent of w, i.e. unbounded in the waypoint direction.
function admissible_polygon(cfg;n=600)
    vlimit=D1-ETA
    hi_v=Float64[]; hi_rho=Float64[]
    lo_v=Float64[]; lo_rho=Float64[]
    for v in range(-vlimit,vlimit,length=n)
        drag=cfg.beta*v^3/(1+v^2)
        center=(ALPHA1-1)*v-drag
        low=max(-D3+ETA-v,(center-(D2-ETA))/ALPHA2)
        high=min(D3-ETA-v,(center+(D2-ETA))/ALPHA2)
        if low<=high
            push!(hi_v,v);push!(hi_rho,high)
            push!(lo_v,v);push!(lo_rho,low)
        end
    end
    isempty(hi_v) && error("Ensemble admissible vide")
    return (; v=[hi_v;reverse(lo_v)],rho=[hi_rho;reverse(lo_rho)],
            rhomin=minimum(lo_rho),rhomax=maximum(hi_rho))
end


# -------------------- PROJECTIONS 2D ----------------------------------------
# La projection est calculee par MINIMISATION en la coordonnee eliminee :
#    (v,rho) : min_{w} B_i(rho,v,w) <= 0
#    (w,rho) : min_{v} B_i(rho,v,w) <= 0.
# On calcule la minimisation exacte du quartique unidimensionnel en
# comparant ses points stationnaires reels. La frontiere est numerique.

# Contour radial. Une verification numerique cherche des reinclusions
# eventuelles. Si une direction pose probleme, on trace une isocourbe
# sur grille au lieu de supposer faussement que l'ensemble est etoile.
function projected_boundary(Q,is_vrho; ntheta=PROJ_NTHETA)
    sx=is_vrho ? 0.85 : 21.0
    sy=3.0
    xb=Float64[]
    yb=Float64[]
    for theta in range(0,2pi;length=ntheta)
        c=cos(theta); s=sin(theta)
        f(rad)=min_projected_B(Q,Float64(rad*sx*c),Float64(rad*sy*s),is_vrho)
        f(0.0)<0 || error("B(0) ne definit pas un interieur negatif")
        lo=0.0; hi=1.0
        for _ in 1:24
            f(hi)>0 && break
            hi*=2
        end
        f(hi)>0 || error("Impossible d'encadrer le niveau zero")
        for _ in 1:42
            mid=(lo+hi)/2
            if f(mid)<=0
                lo=mid
            else
                hi=mid
            end
        end
        rad=(lo+hi)/2
        # Verification heuristique, pas preuve globale de radialite.
        for t in (1.15,1.5,2.0,3.0)
            f(t*rad)>-1e-6 || error("Reinclusion detectee apres la frontiere")
        end
        push!(xb,rad*sx*c)
        push!(yb,rad*sy*s)
    end
    return xb,yb
end

function try_boundary(Q,is_vrho,name)
    try
        return projected_boundary(Q,is_vrho)
    catch err
        @warn "Projection $name non radialement simple; passage a la grille" exception=(err,catch_backtrace())
        return nothing
    end
end

# MODIFICATION LEGENDE 2D : meme style de carres colores que la figure 3D.
# Chacun des trois noms provient des constantes LABEL_*_2D en haut du code.
# Les points NaN servent uniquement a la legende et ne sont pas traces.
# IMPORTANT : les trois Shape representant les ensembles ont label=false,
# pour eviter une seconde legende automatique (ou des doublons).
function add_projection_legend!(fig)
    scatter!(fig, [NaN], [NaN];
             markershape=:rect, markersize=LEGEND_MARKER_2D,
             markercolor=:sandybrown, markerstrokecolor=:sienna,
             markerstrokewidth=1.2, markeralpha=1.0,
             label=LABEL_ADMISSIBLE_2D)
    scatter!(fig, [NaN], [NaN];
             markershape=:rect, markersize=LEGEND_MARKER_2D,
             markercolor=:tomato, markerstrokecolor=:darkred,
             markerstrokewidth=1.2, markeralpha=1.0,
             label=LABEL_B0_2D)
    scatter!(fig, [NaN], [NaN];

             markershape=:rect, markersize=LEGEND_MARKER_2D,
             markercolor=:steelblue, markerstrokecolor=:navy,
             markerstrokewidth=1.2, markeralpha=1.0,    
             label=LABEL_B5_2D)
    return fig
end

# Le polygone Z_a est exact en (v,rho) sous approximation du paramétrage
# numerique de son bord. Dans (w,rho), il est une bande infinie en w :
# les deux limites verticales du dessin sont seulement un recadrage.
function plot_projection(Qs,cfg,is_vrho)
    P=admissible_polygon(cfg)
    b0=try_boundary(Qs["B0"],is_vrho,"B0")
    b5=try_boundary(Qs["B5"],is_vrho,"B5")
    bnds=[b for b in (b0,b5) if b !== nothing]
    if is_vrho
        max_x=max(maximum(abs.(P.v)), (isempty(bnds) ? 1.1 : maximum(maximum(abs.(b[1])) for b in bnds)))
    else
        max_x=max(21.0, (isempty(bnds) ? 21.0 : maximum(maximum(abs.(b[1])) for b in bnds)))
    end
    max_y=max(maximum(abs.(P.rho)), (isempty(bnds) ? 3.0 : maximum(maximum(abs.(b[2])) for b in bnds)))
    xmax=1.18*max_x
    ymax=1.18*max_y
    gr()
    # MODIFICATION FIGURES 2D : marges explicites pour les labels et legendes.
    # La legende est placee EN DEHORS de l'aire de trace, sur trois lignes.
    fig=plot(; xlabel=is_vrho ? L"v" : L"w", ylabel=L"x-w",
       xlims=(-xmax,xmax),ylims=(-ymax,ymax),
       size=FIG_SIZE_2D,
       legend=LEGEND_POSITION_2D, legend_column=LEGEND_COLUMNS_2D,
       guidefontsize=FONT_AXES_2D,tickfontsize=FONT_TICKS_2D,
       legendfontsize=FONT_LEGEND_2D, # MODIFICATION : taille des noms des ensembles
       left_margin=LEFT_MARGIN_2D, right_margin=RIGHT_MARGIN_2D,
       bottom_margin=BOTTOM_MARGIN_2D, top_margin=TOP_MARGIN_2D,
       background_color=:white,foreground_color_legend=:black,
       background_color_legend=:white,framestyle=:box,grid=false,title="")

    # 1) Ensemble admissible Z_a^w (orange, en arriere-plan)
    if is_vrho
        plot!(fig,Shape(P.v,P.rho);fillcolor=:sandybrown,
            fillalpha=ALPHA_ADMISSIBLE,linecolor=:sienna,
            linewidth=2,label=false)
    else
        plot!(fig,Shape([-xmax,xmax,xmax,-xmax],
                        [P.rhomin,P.rhomin,P.rhomax,P.rhomax]);
              fillcolor=:sandybrown,fillalpha=ALPHA_ADMISSIBLE,
              linealpha=0,label=false)
        hline!(fig,[P.rhomin,P.rhomax];color=:sienna,
               linewidth=2,label=false)
    end

    # 2) K_B0 : invariant pour F_S entier (bleu)
    # 3) K_B5 : invariant pour F_S inter F_A (rouge)
    for (name,bnd,fc,ec,alpha_fill) in
        (("B0",b0,:tomato,:darkred,ALPHA_B0),
         ("B5",b5,:steelblue,:navy,ALPHA_B5))
        if bnd !== nothing
            xx,yy=bnd
            plot!(fig,Shape(xx,yy);fillcolor=fc,fillalpha=alpha_fill,
                  linecolor=ec,linewidth=3.4,label=false)
        else
            # Contour direct du niveau 0 : reste correct si la forme
            # n'est pas radialement simple. Aucun faux remplissage.
            xs=collect(range(-xmax,xmax,length=PROJ_GRID))
            ys=collect(range(-ymax,ymax,length=PROJ_GRID))
            zz=[min_projected_B(Qs[name],Float64(x),Float64(y),is_vrho)
                for y in ys,x in xs]
            contour!(fig,xs,ys,zz;levels=[0.0],linewidth=3.4,
                     color=ec,label=false)
        end
    end

    # MODIFICATION LEGENDE 2D : ajout des trois noms et carres colores.
    # FONT_LEGEND_2D et LEGEND_MARKER_2D peuvent etre modifies en tete du code.
    add_projection_legend!(fig)

    file=is_vrho ? "projection_v_x_moins_w_B0_B5" : "projection_w_x_moins_w_B0_B5"
    savefig(fig,joinpath(OUT,file*".pdf"))
    savefig(fig,joinpath(OUT,file*".png"))
    println("Projections ecrites : ",joinpath(OUT,file*".pdf"))
    return fig
end

function plot_all(Qs,cfg)
    p1=plot_projection(Qs,cfg,true)
    p2=plot_projection(Qs,cfg,false)
    # MODIFICATION LEGENDE COMBINEE : deux COPIES avec une police plus petite.
    # Cela ne change pas la police des PDF individuels exportes plus haut.
    p1_combined=deepcopy(p1)
    p2_combined=deepcopy(p2)
    plot!(p1_combined;legendfontsize=FONT_LEGEND_COMBINED_2D)
    plot!(p2_combined;legendfontsize=FONT_LEGEND_COMBINED_2D)
    both=plot(p1_combined,p2_combined;layout=(1,2),size=FIG_SIZE_COMBINED,
              left_margin=LEFT_MARGIN_2D, right_margin=RIGHT_MARGIN_2D,
              bottom_margin=BOTTOM_MARGIN_2D, top_margin=TOP_MARGIN_2D,
              background_color=:white)
    savefig(both,joinpath(OUT,"projections_B0_B5_combinees.pdf"))
    savefig(both,joinpath(OUT,"projections_B0_B5_combinees.png"))
end

function read_certificates_and_params()
    Qs=Dict{String,Matrix{Float64}}()
    for key in ("B0","B5")
        path=joinpath(OUT,"Q_$(key)_full.csv")
        isfile(path) || error("Matrice absente: $path. Lancer d'abord le SOS.")
        Q=Matrix{Float64}(readdlm(path,',',Float64))
        size(Q)==(9,9) || error("Dimension incorrecte: $path")
        norm(Q-Q',Inf)<1e-6 || error("Matrice non symetrique: $path")
        eigmin(Symmetric(Q))>0 || error("Matrice non definie positive: $path")
        Qs[key]=Q
    end
    cfg=config()
    filename=joinpath(OUT,"parametres.toml")
    if isfile(filename)
        saved=TOML.parsefile(filename)
        cfg=merge(cfg,(beta=Float64(saved["beta"]),zeta=Float64(saved["zeta"])))
        println("Parametres du calcul original recuperes: beta=$(cfg.beta), zeta=$(cfg.zeta)")
    else
        @warn "parametres.toml absent: verifier que --beta correspond au calcul des Q."
    end
    return Qs,cfg
end

function main()
    cfg=config()
    mkpath(OUT)
    if "--figures-only" in ARGS
        Qs,original_cfg=read_certificates_and_params()
        plot_all(Qs,original_cfg)
        println("Figures recreees SANS resoudre le SDP. Les contraintes SOS ne sont pas reverifiees.")
        return
    end
    result=nothing
    if "--search-scales" in ARGS
        # Chaque essai est un SDP convex avec multiplicateurs fixes.
        for scale in (1.015,1.010,1.005,1.001)
            trial=merge(cfg,(b=scale,c=ntuple(_->scale,5),d=scale))
            println("ESSAI avec b=c_i=d=$scale")
            result=synthesize(trial)
            result!==nothing && break
        end
    else
        result=synthesize(cfg)
    end
    if result===nothing
        println("Aucun certificat verifie au statut OPTIMAL; aucune figure calculee.")
        println("Ce n'est pas une preuve que le contrat est impossible.")
        return
    end
    export_result(result)
    numerical_spot_checks(result)
    plot_all(result.Q,result.config)
    println("Fin : K_B0 pour F_S et K_B5 pour F_S inter F_A, delta=5.")
end

main()
end # module RobotNonlinearDelta5
