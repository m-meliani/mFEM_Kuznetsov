# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:26:49 2021
@author: Mostafa Meliani <m.meliani@math.ru.nl>

This sample coded is based on the example with unknown solution from  
Meliani M and Nikolic V, "A priori error analysis of mixed finite element formulations for the Kuznetsov equation"

In particular, it solves the Kuznetsov equation with the mixed potential-velocity FEM formulation:
(1-2k psi_t)psi_tt - c^2 div(v) - b div(v_t) - 2s v.v_t = f
v = grad(psi)
"""

import numpy as np
from dolfin import *

b = 6e-9
c = 1500  
k = 15.  # small enough for 1-2k phi_t positive (real wave)  (Kuznetsov)
s = 1. # 0 for westervelt, 1 for Kuznetsov

# Source term parameters
amp = 400
std = 0.03
var = std**2
alpha = 5e4

  
## paraview-like output file
outfile = File("unknown_psi_dot.pvd") 

# model parameter options
Hdivname = "RT"    # Element type used to approximate Hdiv space, e.g., "RT" or "BDM"
L2name = "DG"       # Element type used to approximate L2 space: "DG"
p_ord = 1           # Polynomial order of element for "DG", Hdiv space polynomial order is p_ord+1

## FEnics expressions' degree, controls quadrature rules, chosen p_ord+2 so that the approximation error 
## from the expressions does not interfere with convergence rate of the scheme 
degree = p_ord+2    


## Solver parameters chosen for solving the Kuznetsov equation 
prm = LinearVariationalSolver.default_parameters()
if True:
    prm["linear_solver"] = 'gmres'
    prm["preconditioner"] = 'jacobi'
    prm["krylov_solver"]["absolute_tolerance"] = 1e-8
    prm["krylov_solver"]["relative_tolerance"] = 1e-8
    prm["krylov_solver"]["maximum_iterations"] = 2500
    prm["krylov_solver"]["error_on_nonconvergence"] = True
    prm["krylov_solver"]["nonzero_initial_guess"] = True
    prm["krylov_solver"]["monitor_convergence"] = False


# mesh creation with MPI to allow for parallelization of computations 
n_el =  64
mesh = UnitSquareMesh(MPI.comm_world, n_el, n_el)

rank = MPI.comm_world.rank      # rank variable necessary to remove double-printing


# creation of mixed space 
Hdiv_el = FiniteElement(Hdivname, mesh.ufl_cell(), p_ord+1) 
L2_el = FiniteElement(L2name, mesh.ufl_cell(), p_ord) 
W_elem = MixedElement([Hdiv_el,L2_el])
W = FunctionSpace(mesh,W_elem)


# creation of Hdiv and L2 spaces separately for the needs of creating psi and v functions independently
Hdiv = FunctionSpace(mesh, Hdiv_el) 
L2 = FunctionSpace(mesh, L2_el)


sol_w = Function(W)             # Function to store solution of linear solution


(u_vec, u) = TrialFunctions(W)  # Trial function for which the linear system is solved
(w, phi) = TestFunctions(W)     # Mixed test functions for approximate weak formulation


# Initialization of functions of interest
psi_dot_dot = Function(L2)      
psi_dot_dot_k = Function(L2)    # contains information of last iteration solution to judge convergence of fixed-point

psi_dot = Function(L2, name="psi_dot")
psi = Function(L2, name="psi")

#Newmark pred function (corrector-predictor-type implementation)
psi_dot_pred = Function(L2)
psi_pred = Function(L2)


v_dot = Function(Hdiv)
v = Function(Hdiv, name="v")
v_pred = Function(Hdiv)

set_log_level(30) 

outfile.write(psi_dot)

# we set the timestepping variables::
dt = 1e-7
T = 1000*dt
t=0
step=0


# Newmark scheme parameters
gamma =0.85
beta = 0.45


# Fixed-point parameters
max_iter_k = 100    # Maximum number of linear iterations for each time step
delta_k = 1e-6      # Relative error allowed to approve convergence of a step



if rank == 0:
    CFLm = 0.5
    lc = 1/n_el
    CFL = dt*c/lc
    if  CFL> CFLm:
        print("please, choose a coarser mesh size or smaller time step\n CFL=", CFL)
        exit()


f = Expression('exp(-alpha*t)*amp/sqrt(std)*exp((-pow(x[0]-0.5,2)-pow(x[1]-0.5,2))/2/var)', amp=amp, var=var, std=std, alpha = alpha,t=t, degree=degree)

## Bilinear form independent of t
a = u * phi * dx - c**2 *dt* div(u_vec) * phi *dx - b * div(u_vec) * phi *dx  + dot(u_vec, w)*dx 

while t<=T:
    step += 1
    t += dt
    
    ## Source term update
    f.t = t
    
    psi_dot_pred.vector()[:] = psi_dot.vector() + (1-gamma)* dt * psi_dot_dot.vector()
    psi_pred.vector()[:]  = psi.vector() + dt * psi_dot.vector() + (dt**2)/2 * (1-2*beta) * psi_dot_dot.vector()    
    v_pred.vector()[:] = v.vector()
  
    psi_dot.vector()[:] = psi_dot_pred.vector()
    
    # Reset counter and cvg_condition for fixed-point iterations
    counter = 0
    cvg_condition = delta_k + 1.
    while cvg_condition > delta_k and counter < max_iter_k: 
        counter += 1
        L = c**2* div(v_pred)* phi * dx - psi_dot * div(w) *dx + f * phi * dx
    #     Add nonlinear terms
        L += 2*k * psi_dot * psi_dot_dot * phi * dx
        L += 2*s * dot(v,v_dot)* phi * dx

        solve(a==L,sol_w,solver_parameters=prm)
        (v_dot, psi_dot_dot) = sol_w.split(deepcopy=True)
    
        psi_dot.vector()[:] = psi_dot_pred.vector() + gamma * dt * psi_dot_dot.vector()
        v.vector()[:] = v_pred.vector() +  dt* v_dot.vector() 
         # check convergence conditions
        norm_u = norm(psi_dot_dot) + DOLFIN_EPS     # add "epsilon" to avoid norm_u = 0 singularity
        errnorm = sqrt(assemble((psi_dot_dot_k-psi_dot_dot)**2*dx))
        
        cvg_condition = errnorm / norm_u     
        # copy psi_dot_dot in psi_dot_dot_k to calculate convergence condition for next iteration    
        if rank == 0:
            print("step: ", step, flush = True) 
            if counter == max_iter_k:
                print("step was terminated because limit iters reached ", flush = True) 
            
        psi_dot_dot_k.vector()[:] = psi_dot_dot.vector()
    
        
    psi.vector()[:] = psi_pred.vector() +  dt**2 * beta * psi_dot_dot.vector()
    if step % 5 == 0:
        outfile.write(psi_dot)

                



