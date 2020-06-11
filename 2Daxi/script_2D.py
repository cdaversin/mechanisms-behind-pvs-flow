""" Mechanisms behind perivascular fluid flow (C. Daversin-Catty, V. Vinje, K.-A. Mardal, and M.E. Rognes) - 2Daxi script """

from math import *
from stokes_ale_artery_pvs import *
from scipy.signal import find_peaks
import shutil

microm = 1e-3 # Convert [µm] to [mm]
meter = 1e3 # Convert [m] to [mm]

params = dict()
params["type_s"] = "axi"
params["R1"] = 20 #[µm]
params["R2"] = 60 #[µm]

params["nb_cycles"] = 5
params["frequency"] = 10 #[Hz]

params["origin"] = [0,0,0]
params["center"] = []

params["traveling_wave"] = True

# Rigid motion
params["rigid_motion"] = False
params["rigid_motion_dir"] =  [0,1]
params["rigid_motion_amplitude"] = 6*microm

# Pressure gradient
p_static_gradient = 0
# - 3rd circulation : 0.01 mmHg/m = 0.00133 Pa/mm
# p_static_gradient = 0.00133
# - Respiration : 0.5 mmHg/m = 0.0665 Pa/mm
# p_static_gradient = 0.0665
params["p_static_gradient"] = p_static_gradient

# Pulsatile systemic pressure gradient
phis = [0] # No systemic pressure gradient

# List of lengths
Lengths = [1,5,10,50,100,200] #[mm]
# List of wave speed
c_vels = [1e3] #[mm/s]
# List of time steps
dts = [3.125e-4]
# Mesh resolution
mesh_refinement = [2]

for i_l, Length in enumerate(Lengths):
    params["Length"] = Length
    params["rigid_motion_X0"] = params["Length"]*0.5

    mesh_file = "2D%s_L%.1f"%(params["type_s"], Length)
    reponame, file_extension = os.path.splitext(mesh_file)
    reponame = reponame.split('/')[-1]

    for i_c, c_vel in enumerate(c_vels):
        params["c_vel"] = c_vel

        for refinement in mesh_refinement:
            params["mesh_refinement"] = refinement

            for dt in dts:
                params["dt"] = dt

                for phi in phis:
                    params["p_oscillation_phi"] = phi

                    parameters = "dp%.2f"%(p_static_gradient)
                    if phi:
                        parameters += "_Bilston_phi_%.2f"%(phi)

                    fig, figs = plt.subplots(1, 3)
                    fig.set_size_inches(18.5, 10.5)
                    fig.suptitle(parameters)
                    figs[0].set_title('u avg [mm/s]')
                    figs[1].set_title('inflow [mm^3/s]')
                    figs[2].set_title('Position [mm]')
                    
                    inflow_area = pvs_model(params)
                
                    ##  -- Plot u_avg
                    t,uavg = plt.loadtxt(reponame + '/uavg.txt').transpose()
                    ## Save plot to png
                    figs[0].plot(t, uavg, label="uavg")
                    figs[0].legend()
                    figs[0].set_xlabel("Time [s]")

                    ## -- Plot inflow
                    t,v = plt.loadtxt(reponame + '/inflow.txt').transpose()
                    header_v_avg = "T \t Vavg"
                    data_v_avg = plt.array([t,v/inflow_area]).transpose()
                    plt.savetxt(reponame + '/vavg.dat', data_v_avg, delimiter="\t", header=header_v_avg)
                    ## Save plot to png
                    figs[1].plot(t,v, label="vavg")
                    figs[1].legend()
                    figs[1].set_xlabel("Time [s]")

                    ## -- Plot Q
                    Q = plt.array([plt.trapz(v[:i],t[:i]) for i in range(1,len(t))])
                    position = Q/inflow_area
                    ## Save data to file with suitable formating
                    header_Q = "T \t Q"
                    header_position = "T \t Position"
                    data_Q = plt.array([t[1:],Q]).transpose()
                    data_position = plt.array([t[1:],position]).transpose()
                    plt.savetxt(reponame + '/Q.dat', data_Q, delimiter="\t", header=header_Q)
                    plt.savetxt(reponame + '/position.dat', data_position, delimiter="\t", header=header_position)
                    ## Save plot to png
                    figs[2].plot(t[1:], position, label="position")
                    figs[2].legend()
                    figs[2].set_xlabel("Time [s]")
                    
                    ## Export figure with the three subplots
                    plt.savefig(reponame + "/" + parameters + ".png")

                    # Note : Find peaks doesn't work when peaks are not acute enough
                    # If so, compute slope finding two peaks manually
                    ppfile_peak = open(reponame + "/peaks.dat", 'a+')
                    try:
                        idx = find_peaks(position)[0]
                        v0 = position[idx[-2]]
                        v1 = position[idx[-1]]
                        t0 = t[idx[-2]]
                        t1 = t[idx[-1]]
                        v_avg = (v1-v0)/(t1-t0)
                        for peak in idx:
                            if MPI.rank(MPI.comm_world) == 0:
                                ppfile_peak.write("%g \t %g \n" %(position[peak], t[peak]))
                    except:
                        shift = len(position)/params["nb_cycles"]
                        idx1 = np.argmax(position)
                        idx0 = np.argmax(position[:idx1 - int(shift)])
                        v0 = position[idx0]
                        v1 = position[idx1]
                        t0 = t[idx0]
                        t1 = t[idx1]
                        v_avg = (v1-v0)/(t1-t0)

                    ppfile_peak.close()

                    # Renaming directory
                    current_reponame = os.getcwd() + "/" + reponame
                    new_reponame = current_reponame \
                                   + "_mesh%i"%(refinement) \
                                   + "_dt_%.4f"%(dt) \
                                   + "_phi_%.2f"%(phi)
                    if MPI.rank(MPI.comm_world) == 0:
                        print("Moving ", current_reponame, " to ", new_reponame)
                        for content in os.listdir(current_reponame):
                            if not os.path.exists(new_reponame):
                                os.mkdir(new_reponame)
                            current_content = current_reponame + "/" + content
                            new_content = new_reponame + "/" + content
                            if os.path.isfile(current_content):
                                shutil.move(current_content, new_content)
                            elif os.path.isdir(current_content):
                                if not os.path.exists(new_content):
                                    os.mkdir(new_content)
                                for f in os.listdir(current_content):
                                    shutil.move(current_content + "/" + f, new_content + "/" + f)
