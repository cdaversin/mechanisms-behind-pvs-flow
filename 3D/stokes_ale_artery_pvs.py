""" Mechanisms behind perivascular fluid flow (C. Daversin-Catty, V. Vinje, K.-A. Mardal, and M.E. Rognes) - 3D model """
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
    params["mesh_file"] = "default.xdmf"
    params["markers_file"] = "default_mf.xdmf"
    params["inlet_markers"] = []
    params["outlet_markers"] = []
    params["c_vel"] = 1e3 #[mm/s]
    params["frequency"] = 10 #[Hz]
    params["nb_cycles"] = 1
    params["traveling_wave"] = False
    params["rigid_motion"] = False
    params["rigid_motion_X0"] = 0
    params["rigid_motion_dir"] = []
    params["rigid_motion_amplitude"] = 0
    params["coord_factor"] = 1
    params["origin"] = [0,0,0]
    params["center"] = [0,0,0]
    params["p_static_gradient"] = 0
    params["dt"] = 0.001
    params["p_oscillation_L"] = []
    params["p_oscillation_phi"] = 0

    return params

def pvs_model(params = default_params()):

    ### -------- Setup - Mesh, Markers, Parameters -------- ###

    # Collect params from dictionnary
    mesh_file = params["mesh_file"]
    markers_file = params["markers_file"]
    inlet_markers = params["inlet_markers"]
    outlet_markers = params["outlet_markers"]
    c_vel = params["c_vel"]
    frequency = params["frequency"]
    nb_cycles = params["nb_cycles"]
    traveling_wave = params["traveling_wave"]
    rigid_motion = params["rigid_motion"]
    rigid_motion_X0 = params["rigid_motion_X0"]
    rigid_motion_dir = params["rigid_motion_dir"]
    rigid_motion_amplitude = params["rigid_motion_amplitude"]
    x0,y0,z0 = params["origin"][0], params["origin"][1], params["origin"][2]
    coord_factor  =params["coord_factor"]
    p_static_gradient = params["p_static_gradient"]
    dt = params["dt"]
    p_oscillation_L = params["p_oscillation_L"]
    p_oscillation_phi = params["p_oscillation_phi"]

    # Average PVS width (Mestre, fig 2d) : L = 4.4e-5 [m]
    L_PVS = 44e-3 # [mm]
    # Diffusion coefficient D = 6.55e-13 [m^2/s]
    D_PVS = 6.55e-7 #[mm^2/s]
    # Kynematic viscosity of (CSF) water at 36.8 celsius degrees (Mestre et al.) : 0.697e-6 [m^2/s]
    nu = Constant(0.697) # Mesh is in mm : 0.697e-6 [m^2/s] -> 0.697 [mm^2/s]
    # Density of (CSF) water : 1000 [kg/m^3] -> 1e-3 [g/mm^3]
    # Note : We use [g/mm^3] to make sur we obtain the pressure in [Pa]
    rho = Constant(1e-3)
    # Source function f
    f = Constant((0.0, 0.0, 0.0))
    # Cycle duration
    cycle_duration = 1/frequency # [s]

    # Time
    time = 0.0
    T_final = cycle_duration*nb_cycles
    nb_time_steps = int(T_final/dt)
    nb_time_steps_per_cycle = int(cycle_duration/dt)

    ### --------------------------------------------------- ###

    ### -------- Read PVS mesh ---------------------------- ###

    mesh = Mesh()
    
    with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
        xdmf.read(mesh)
    # Scale coordinate to match Mestre (fig 2d)
    mesh.coordinates()[:]/=coord_factor

    # Store the initial mesh as mesh0
    mesh0 = Mesh(mesh)

    # Define boundaries from physical markers
    mf_tmp = MeshFunction('size_t',mesh, mesh.topology().dim()-1)
    with XDMFFile(MPI.comm_world, markers_file) as xdmf:
        xdmf.read(mf_tmp)

    mf = MeshFunction('size_t',mesh, mesh.topology().dim()-1) # Moving mesh
    mf0 = MeshFunction('size_t',mesh0, mesh0.topology().dim()-1) # Reference mesh

    ### --------------------------------------------------- ###

    ### -------- Read Markers and define Measures --------- ###
    # Renumbering
    for facet in facets(mesh):
        # PVS inlets - marked as 10, 11, ...
        if mf_tmp[facet] in inlet_markers:
            for inlet_idx, inlet_tag in enumerate(inlet_markers):
                if mf_tmp[facet] == inlet_tag:
                    mf[facet] = 10 + inlet_idx
                    mf0[facet] = 10 + inlet_idx
        # PVS outlets - marked as 20, 21, ...
        elif mf_tmp[facet] in outlet_markers:
            for outlet_idx, outlet_tag in enumerate(outlet_markers):
                if mf_tmp[facet] == outlet_tag:
                    mf[facet] = 20 + outlet_idx
                    mf0[facet] = 20 + outlet_idx
        # PVS outer wall - marked as 40
        elif mf_tmp[facet] == 40:
            mf[facet] = 3
            mf0[facet] = 3
        # PVS inner wall - marked as 30
        elif mf_tmp[facet] == 30:
            mf[facet] = 4
            mf0[facet] = 4

    # Only one volume here, no need to define dx
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
    a = (rho*dot(u, v)*dx
         + nu*rho*dt*inner(grad(u), grad(v))*dx
         - rho*dt*dot(div(outer(w, u)), v)*dx
         - dt*p*div(v)*dx
         + dt*div(u)*q*dx)

    # Linear terms on Omega_t_n+1
    L = dt*dot(f, v)*dx

    # Add pressure drop (static pressure gradient)
    if p_static_gradient:
        L_max = max(p_oscillation_L) # p_oscillation_L is th list of lengths (branches)
        p_inlet = L_max*p_static_gradient
        for inlet_idx, inlet_tag in enumerate(inlet_markers):
            tag = 10 + inlet_idx
            L += -dt*p_inlet*dot(v,n)*ds(tag) # Impose L1*dp/dx at inlet

        for idx, length in enumerate(p_oscillation_L):
            # Impose (L1-L2)*dp/dx at inlet
            if length < L_max: # impose 0 otherwise
                p_outlet = p_static_gradient*(L_max - length)
                tag = 20 + idx
                L += -dt*p_outlet*dot(v,n)*ds(tag)

    # Add pressure oscillation (systemic)
    p_grads = []
    if p_oscillation_phi:
        dp = 1.5*133.33    # Amplitude of pressure gradient in Pa/m
        assert( len(p_oscillation_L) == len(outlet_markers) )
        for idx, length in enumerate(p_oscillation_L):
            p_grad = Expression('A*sin(2*pi*t/T + 2*pi*phi)',
                                A = dp*length*1e-3, t = 0, T = cycle_duration, phi = p_oscillation_phi,
                                degree=2)
            p_grads.append(p_grad)
            tag = 20 + idx
            L += -dt*p_grad*dot(v,n)*ds(tag)

    # Linear term on Omega_t_n   
    L_ = rho*dot(u_, v)*dx

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
            X = sqrt( (x[0] - x0)**2 + (x[1] - y0)**2 + (x[2] - z0)**2 )

            if traveling_wave:
                d0_expr = 0.5*RelDeltaD(time - X/c_vel)*L_PVS
            else:
                d0_expr = 0.5*RelDeltaD(time)*L_PVS

            values[0] = -d0_expr*n[0]
            values[1] = -d0_expr*n[1]
            values[2] = -d0_expr*n[2]
            if rigid_motion:
                values[0] += rigid_motion_dir[0]*RigidMotion(X, theta, time)
                values[1] += rigid_motion_dir[1]*RigidMotion(X, theta, time)
                values[2] += rigid_motion_dir[2]*RigidMotion(X, theta, time)
        def value_shape(self):
            return (3,)

    # Define vector Laplacian type problem (harmonic smoothing?) for mesh movement
    d = TrialFunction(W0)
    e = TestFunction(W0)
    mu = 1.0
    m = mu*inner(grad(d), grad(e))*dx

    zero = Function(W0)
    dummy_ = dot(zero, e)*dx

    bcs_mesh = [DirichletBC(W0, (0.0, 0.0, 0.0), mf0, 3)]
    ND = NormalDisp(mesh0, degree=2)
    bcs_mesh.append(DirichletBC(W0, ND, mf0, 4))

    M = assemble(m)

    phi = Function(W0) # Current mesh displacement Omega_t_n (relative to Omega_0)
    phi.rename("d", "deformation")
    phi_ = Function(W0) # Previous mesh displacement Omga_t_{n-1} (rel to Omega_0)
    dphi = Function(W0)
    ### --------------------------------------------------- ###


    ### -------- MAIN time loop --------------------------- ###
    
    reponame, file_extension = os.path.splitext(mesh_file)
    reponame = reponame.split('/')[-2] + "_results"

    parameters = "c%.2f_dt%.3f_dp%.2f"%(c_vel, dt, p_static_gradient)
    if p_oscillation_phi:
        parameters += "_Bilston_phi_%.2f"%(p_oscillation_phi)

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
    # Write initial (u,p) to XDMF files
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
            for p_grad in p_grads:
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

            ALE.move(mesh, dphi_t) # NB: Check floating of the mesh with this!!

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
            
            infl = assemble(dot(u0,n)*ds(10))
            #outfl = assemble(dot(u0,n)*ds(20) + dot(u0,n)*ds(21)) # 2 outlets
            outfl = 0
            for outlet_idx, outlet_tag in enumerate(outlet_markers):
                outfl += assemble(dot(u0,n)*ds(20 + outlet_idx))
            inflow += infl
            outflow += outfl

            if MPI.rank(MPI.comm_world) == 0:
                infile.write('%g %g\n'%(time, infl))
                outfile.write('%g %g\n'%(time, outfl))
        
            # Update phi_
            phi_.assign(phi)
            if MPI.rank(MPI.comm_world) == 0:
                print("")

            # To compute Reynolds/Peclet
            Volume = assemble(Constant(1)*dx(domain = mesh))
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
    inflow_area = assemble(Constant(1)*ds(10))

    return inflow_area

### --------------------------------------------------- ###

### -------------------- MAIN ------------------------- ###
if __name__ == "__main__":
    print("Please use dedicated script")
    inflow_area = pvs_model()
### --------------------------------------------------- ###
