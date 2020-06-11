""" Mechanisms behind perivascular fluid flow (C. Daversin-Catty, V. Vinje, K.-A. Mardal, and M.E. Rognes) - 2Daxi model """

import os
import sys
import math
import pylab as plt
import numpy as np
import scipy.interpolate as sp_interpolate

from dolfin import *

microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

def default_params():
    params = dict()
    params["Length"] = 5 #[mm]
    params["R1"] = 20 #[µm]
    params["R2"] = 60 #[µm]
    params["mesh_refinement"] = 1
    params["type_s"] = "axi"
    params["c_vel"] = 1e3  #[mm/s]
    params["frequency"] = 10 #[Hz]
    params["nb_cycles"] = 1
    params["traveling_wave"] = False
    params["rigid_motion"] = False
    params["rigid_motion_X0"] = 0
    params["rigid_motion_dir"] = [0,1]
    params["rigid_motion_amplitude"] = 0
    params["origin"] = [0,0,0]
    params["p_static_gradient"] = 0
    params["dt"] = 0.001
    params["p_oscillation_phi"] = 0

    return params

def pvs_model(params = default_params()):

    ### -------- Setup - Mesh, Markers, Parameters -------- ###

    # Collect params from dictionnary
    Length = params["Length"]
    R1 = params["R1"]
    R2 = params["R2"]
    mesh_refinement = params["mesh_refinement"]
    type_s = params["type_s"]
    c_vel = params["c_vel"]
    frequency = params["frequency"]
    nb_cycles = params["nb_cycles"]
    traveling_wave = params["traveling_wave"]
    rigid_motion = params["rigid_motion"]
    rigid_motion_X0 = params["rigid_motion_X0"]
    rigid_motion_dir = params["rigid_motion_dir"]
    rigid_motion_amplitude = params["rigid_motion_amplitude"]
    x0,y0,z0 = params["origin"][0], params["origin"][1], params["origin"][2]
    p_static_gradient = params["p_static_gradient"]
    dt = params["dt"]
    p_oscillation_phi = params["p_oscillation_phi"]
    
    if type_s == "axi":
        print("Model Axi 2D")
        axi = True
    elif type_s == "cart":
        print("Model Cartesian 2D")
        axi = False
    else:
        print("Options : axi - cart (default : axi)")

    ### -------- Setup - Mesh, Markers, Parameters -------- ###
    # 2D PVS with internal radius r1, outer radius r2, length L
    
    # Radius
    r1 = R1*microm #[mm]
    r2 = R2*microm #[mm]
    # Diffusion coefficient D = 6.55e-13 [m^2/s]
    D_PVS = 6.55e-7 #[mm^2/s]
    # Kynematic viscosity of (CSF) water at 36.8 celsius degrees (Mestre et al.) : 0.697e-6 [m^2/s]
    nu = Constant(0.697) # Mesh is in mm : 0.697e-6 [m^2/s] -> 0.697 [mm^2/s]
    # Density of (CSF) water : 1000 [kg/m^3] -> 1e-3 [g/mm^3]
    # Note : We use [g/mm^3] to make sur we obtain the pressure in [Pa]
    rho = Constant(1e-3)
    # Source function f
    f = Constant((0.0, 0.0))
    # Cycle duration
    cycle_duration = 1/frequency # [s]

    # Time
    time = 0.0
    T_final = cycle_duration*nb_cycles
    nb_time_steps = int(T_final/dt)
    nb_time_steps_per_cycle = int(cycle_duration/dt)

    ### --------------------------------------------------- ###

    ### -------- PVS mesh ---------------------------- ###
    mesh_file = "2D%s_L%.1f"%(type_s, Length)

    m_ = 5 # Mesh resoution - PVS width
    n_ = int(10*Length)
    # Mesh refinement
    m_ = m_*mesh_refinement
    n_ = n_*mesh_refinement
    print("Mesh size = ", Length, " x ", r2-r1, " [mm]")
    print("Mesh resolution = ", n_, " x ", m_)
    mesh = RectangleMesh(Point(0.0, r1), Point(Length, r2), n_, m_)
    # PVS width
    L_PVS = r2 - r1

    # Store the initial mesh as mesh0
    mesh0 = Mesh(mesh)
    mf = MeshFunction('size_t',mesh, mesh.topology().dim()-1) # Moving mesh
    mf0 = MeshFunction('size_t',mesh0, mesh0.topology().dim()-1) # Reference mesh

    ### --------------------------------------------------- ###

    ### -------- Markers and Measures --------- ###
    inl = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    outl = CompiledSubDomain("near(x[0], %g) && on_boundary" % Length)
    bottom = CompiledSubDomain("near(x[1], %g) && on_boundary" % r1)
    top = CompiledSubDomain("near(x[1], %g) && on_boundary" % r2)
    
    # PVS - inflow : 1
    # PVS - outflow : 2
    # PVS - bottom wall : 3
    # PVS - top wall : 4

    # mesh markers
    mf.set_all(0)
    inl.mark(mf, 1)
    outl.mark(mf, 2)
    bottom.mark(mf, 3)
    top.mark(mf, 4)
    # mesh0 markers
    mf0.set_all(0)
    inl.mark(mf0, 1)
    outl.mark(mf0, 2)
    bottom.mark(mf0, 3)
    top.mark(mf0, 4)
    
    ds0 = Measure("ds", domain = mesh0, subdomain_data=mf0) # External facets - ref mesh
    ds = Measure("ds", domain = mesh, subdomain_data=mf) # External facets - moving mesh
    ### --------------------------------------------------- ###

    ### -------- Stokes formulation ----------------------- ###
    # Define Taylor-Hood function spaces for Stokes
    V = VectorElement("CG", mesh.ufl_cell(), 2)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    VQ = FunctionSpace(mesh, MixedElement(V, Q))
    # Normal
    n = FacetNormal(mesh)

    # Function spaces for mesh displacement
    W0 = VectorFunctionSpace(mesh0, "CG", 1) #mesh0
    W = VectorFunctionSpace(mesh, "CG", 1) #mesh

    # Velocity-pressure at time n on Omega_t_n
    up_ = Function(VQ)
    (u_, p_) = split(up_)

    # Velocity-pressure test functions  
    (v, q) = TestFunctions(VQ)
    # Velocity-pressure trial functions  
    (u, p) = TrialFunctions(VQ)

    # Mesh velocity on Omega_0 and Omega_t
    w = Function(W)

    # Bilinear terms on Omega_t_n+1
    z, r  = SpatialCoordinate(mesh)
    z0, r0  = SpatialCoordinate(mesh0)

    # Operators in cylindrical coordinates
    def grad_cyl(u,r):
        return as_matrix([[Dx(u[1],1), 0, Dx(u[1],0)],
                          [0, (Constant(1.)/r)*u[1], 0],
                          [Dx(u[0],1), 0, Dx(u[0],0)]])
    def div_cyl(u):
        return Dx(u[1],1) + (Constant(1.)/r)*u[1] + Dx(u[0],0)

    if axi:
        a = (rho*dot(u, v)*r*dx
             + nu*rho*dt*inner(grad_cyl(u,r), grad_cyl(v,r))*r*dx
             - rho*dt*dot(as_vector([div_cyl(w[0]*u), div_cyl(w[1]*u)]), v)*r*dx
             - dt*p*div_cyl(v)*r*dx
             + dt*div_cyl(u)*q*r*dx)
        # Linear terms on Omega_t_n+1
        L = dt*dot(f, v)*r*dx
        # Linear term on Omega_t_n
        L_ = rho*dot(u_, v)*r*dx
    else:
        a = (rho*dot(u, v)*dx
             + nu*rho*dt*inner(grad(u), grad(v))*dx
             - rho*dt*dot(div(outer(w, u)), v)*dx
             - dt*p*div(v)*dx
             + dt*div(u)*q*dx)
        # Linear terms on Omega_t_n+1
        L = dt*dot(f, v)*dx
        # Linear term on Omega_t_n
        L_ = rho*dot(u_, v)*dx

    # Add pressure drop (static pressure gradient)
    if p_static_gradient:
        p_inlet = Length*p_static_gradient
        if axi:
            L += -dt*p_inlet*dot(v,n)*r*ds(1)
        else:
            L += -dt*p_inlet*dot(v,n)*ds(1)
        # p_outlet is zero

    # Add pressure oscillation (systemic)
    p_grad = 0
    if p_oscillation_phi:
        dp = 1.5*133.33    # Amplitude of pressure gradient in Pa/m
        p_grad = Expression('A*sin(2*pi*t/T + 2*pi*phi)',
                            A = dp*Length*1e-3, t = 0, T = cycle_duration, phi = p_oscillation_phi,
                            degree=2)
        if axi:
            L += -dt*p_grad*dot(v,n)*r*ds(2)
        else:
            L += -dt*p_grad*dot(v,n)*ds(2)

    bcs = [DirichletBC(VQ.sub(0), w, mf, 3), # pvs outer wall
           DirichletBC(VQ.sub(0), w, mf, 4)]  # pvs inner wall

    # Velocity-pressure at time n+1 on Omega_t_n+1 
    up = Function(VQ)

    ### --------------------------------------------------- ###

    ### -------- Mesh displacement ------------------------ ###
    # Input data from Mestre paper [(\Delta diam)/(diam)]
    # Dataset obtained from Fig 3e, using WebPlotDigitizer [https://automeris.io/WebPlotDigitizer]
    # Post-processed data using spline approximation
    data_csv = plt.loadtxt('../mestre_spline_refined_data.dat')
    x_data_refined, y_data_refined = data_csv

    # Fitting dataset using scipy interpolate (piecewise linear)
    fdata = sp_interpolate.interp1d(x_data_refined, y_data_refined)

    def RelDeltaD(_t):
        # Data is given for one cardiac cycle , x \in [0,1]
        # Compute value at given time _t
        val = fdata((_t/cycle_duration)%1)
        # The data are given in percents
        val = val*1e-2
        return val

    ## -- Rigid Motion -- ##
    def RigidMotion(_X, _theta, time):
        shift = min(RelDeltaD(x_data_refined))
        scale = max(RelDeltaD(x_data_refined)) - shift
        func = RelDeltaD(time)/scale
        return tan(_theta)*(rigid_motion_X0-_X)*func

    theta = np.arctan(rigid_motion_amplitude/rigid_motion_X0) # rigid motion

    # We impose d = -d0_expr*n0 along the normal direction n0, so ||d|| = d0_expr.
    # Diameter change : the change of radius is radius <- radius + d0_expr,
    # So : diameter <- diameter + 2*d0_expr (diameter = 2*radius)
    # Then : (\Delta diam)/(diam) [ = our dataset RelDeltaD ] = 2*d0_expr/(diam)
    # gives d0_expr defined as d0_expr = (1/2)*RelDeltaD*diam
    class NormalDisp(UserExpression):
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh
            super().__init__(**kwargs)
        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            n = cell.normal(ufc_cell.local_facet)
            X = x[0] # Distance to origin -- Only in (x = z in axi) direction
        
            if traveling_wave:
                d0_expr = 0.5*RelDeltaD(time - X/c_vel)*L_PVS
            else:
                d0_expr = 0.5*RelDeltaD(time)*L_PVS

            values[0] = -d0_expr*n[0]
            values[1] = -d0_expr*n[1]

            if rigid_motion:
                values[0] += rigid_motion_dir[0]*RigidMotion(X, theta, time)
                values[1] += rigid_motion_dir[1]*RigidMotion(X, theta, time)

        def value_shape(self):
            return (self.mesh.topology().dim(),)

    # Define vector Laplacian type problem (harmonic smoothing?) for mesh movement
    d = TrialFunction(W0)
    e = TestFunction(W0)
    zero = Function(W0)
    mu = 1.0
    if axi:
        m = mu*inner(grad_cyl(d,r0), grad_cyl(e,r0))*r0*dx
        dummy_ = dot(zero, e)*r0*dx
    else:
        m = mu*inner(grad(d), grad(e))*dx
        dummy_ = dot(zero, e)*dx

    # Boundary condition
    bcs_mesh = [DirichletBC(W0, (0.0, 0.0), mf0, 4)]        
    ND = NormalDisp(mesh0, degree=2)
    bcs_mesh.append(DirichletBC(W0, ND, mf0, 3))

    M = assemble(m)

    phi = Function(W0) # Current mesh displacement Omega_t_n (relative to Omega_0)
    phi.rename("d", "deformation")
    phi_ = Function(W0) # Previous mesh displacement Omga_t_{n-1} (rel to Omega_0)
    dphi = Function(W0)
    ### --------------------------------------------------- ###


    ### -------- MAIN time loop --------------------------- ###
    
    reponame, file_extension = os.path.splitext(mesh_file)
    reponame = reponame.split('/')[-1]

    # XDMF files (visualization)
    dfile = XDMFFile(MPI.comm_world, reponame + "/XDMF/d.xdmf")
    ufile = XDMFFile(MPI.comm_world, reponame + "/XDMF/u.xdmf")
    pfile = XDMFFile(MPI.comm_world, reponame + "/XDMF/p.xdmf")
    # HDF5 files (post-processing)
    dhfile = HDF5File(MPI.comm_world, reponame + "/HDF5/d.h5", "w")
    uhfile = HDF5File(MPI.comm_world, reponame + "/HDF5/u.h5", "w")
    phfile = HDF5File(MPI.comm_world, reponame + "/HDF5/p.h5", "w")
    mhfile = HDF5File(MPI.comm_world, reponame + "/HDF5/mesh.h5", "w")

    # Store initial conditions
    (u0, p0) = up.split()
    u0.rename("u", "velocity")
    p0.rename("p", "pressure")
    ufile.write(u0,0)
    pfile.write(p0,0)
    # Write initial (u,p) to HDF5 file
    uhfile.write(u0, "/function", 0)
    phfile.write(p0, "/function", 0)

    time = dt
    u_avg = 0
    inflow = 0
    outflow = 0

    infile = open(reponame + '/inflow.txt','w')
    outfile = open(reponame + '/outflow.txt','w')
    uavgfile = open(reponame + '/uavg.txt','w')

    time_counter = 0
    for i in range(nb_cycles):
        cycle = i + 1
        if MPI.rank(MPI.comm_world) == 0:
            print("-- Start cycle ", cycle, " -- [", time, ", ", cycle*cycle_duration, "]")
        while(time <= cycle*cycle_duration):
            if MPI.rank(MPI.comm_world) == 0:
                print("Solving for t = %g" % time)

            # Update oscillating boundary terms
            if p_oscillation_phi:
                p_grad.t = time
    
            # Assemble contributions on previous domain Omega_t_{n-1}
            b1 = assemble(L_)

            # Compute current mesh displacement: Solve a vector-Laplacian type
            # problem for the deformation on and of mesh0
            # Need to reassemble because the vf depends on time (not only the bc)
            dummy = assemble(dummy_)
            for bc in bcs_mesh:
                bc.apply(M, dummy)
            solve(M, phi.vector(), dummy)
            dfile.write(phi, time)
            dhfile.write(phi, "/function", time) # HDF5

            dphi.assign(phi)
            dphi.vector().axpy(-1.0, phi_.vector())
            dphi_t = Function(W)
            dphi_t.vector()[:] = dphi.vector()

            ALE.move(mesh, dphi_t)

            # Export updated mesh to file
            mhfile.write(mesh, "/mesh%d" % time_counter)

            # Compute mesh velocity w0 in Omega_t
            w.assign(dphi_t)
            w.vector()[:] /= dt

            # Assemble contributions on current domain Omega_t
            A = assemble(a)
            b = assemble(L)

            # Combine right-hand side contributions from previous/current domains
            b.axpy(1.0, b1)

            # Apply boundary conditions
            for bc in bcs:
                bc.apply(A, b)

            # Solve system
            solve(A, up.vector(), b, "mumps")
            
            # Update up_
            up_.assign(up)
    
            (u0, p0) = up.split()
            # XDMF
            u0.rename("u", "velocity")
            p0.rename("p", "pressure")
            ufile.write(u0,time)
            pfile.write(p0,time)
            # HDF5
            uhfile.write(u0, "/function", time)
            phfile.write(p0, "/function", time)
            
            if axi:
                infl = 2*pi*assemble(dot(u0,n)*r*ds(1))
                outfl = 2*pi*assemble(dot(u0,n)*r*ds(2))
                pressure_drop = 2*pi*assemble(p0*r*ds(1) - p0*r*ds(2))
            else:
                infl = assemble(dot(u0,n)*ds(1))
                outfl = assemble(dot(u0,n)*ds(2))
                pressure_drop = assemble(p0*ds(1) - p0*ds(2))

            inflow += infl
            outflow += outfl

            if MPI.rank(MPI.comm_world) == 0:
                infile.write('%g %g\n'%(time, infl))
                outfile.write('%g %g\n'%(time, outfl))
        
            # Update phi_
            phi_.assign(phi)
            if MPI.rank(MPI.comm_world) == 0:
                print("")

            if axi:
                #Volume = pi*(r2*r2 - r1*r1)*Length
                Volume = 2*pi*assemble(Constant(1)*r*dx(domain = mesh))
            else:
                #Volume = (r2 - r1)*Length
                Volume = assemble(Constant(1)*dx(domain = mesh))

            # To compute Reynolds/Peclet
            if axi:
                u_avg_ = sqrt(Volume**-1*2*pi*assemble(dot(u0,u0)*r*dx))
            else:
                u_avg_ = sqrt(Volume**-1*assemble(dot(u0,u0)*dx))
            u_avg += u_avg_

            if MPI.rank(MPI.comm_world) == 0:
                uavgfile.write('%g %g\n'%(time, u_avg))

            # Update time
            time = time + dt
            time_counter = time_counter + 1

        ### -------- Compute some Quantities of Interest - Per cycle ------ ###
        # U : RMS velocity - time averaged over a cardiac cycle
        Vrms = u_avg/nb_time_steps_per_cycle

        # Reynolds number Re = UL/nu
        Re = Vrms*L_PVS/float(nu)
        # Peclet number Pe = UL/D
        Pe = Vrms*L_PVS/D_PVS
        if MPI.rank(MPI.comm_world) == 0:
            print("nb dt per cycle = ", nb_time_steps_per_cycle)
            print("Root Mean Square velocity (v_rms) = ", Vrms*1e3, " [µm/s]")
            print("Reynolds number = ", Re)
            print("Peclet number = ", Pe)
            print('-dot(u,n) at inlet : ', -inflow)
            print('dot(u,n) at outlet : ', outflow)
        
        u_avg = 0

    infile.close()
    outfile.close()
    uavgfile.close()

    # Closing XDMF
    dfile.close()
    ufile.close()
    pfile.close()
    # Closing HDF5
    dhfile.close()
    uhfile.close()
    phfile.close()
    mhfile.close()

    # Compute inflow area
    if axi:
        inflow_area = 2*pi*assemble(Constant(1)*r*ds(1))
    else:
        inflow_area = assemble(Constant(1)*ds(1))

    return inflow_area

### --------------------------------------------------- ###

### -------------------- MAIN ------------------------- ###
if __name__ == "__main__":
    print("Please use dedicated script")
    inflow_area = pvs_model()
### --------------------------------------------------- ###
