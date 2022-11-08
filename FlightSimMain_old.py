from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import FlightSimTVC
import FlightSimAero
import FlightSimAtt

##### Simulation Guide Document will be released soon #####

# Define main simulation function
def simulate(t_ignite,Kp,Kd,drop_height,fin_area,lever_c,lever_f,wind,omega_init):

    dt = 0.01                       # time step
    t_sim = np.arange(0,8,dt)       # time range for simulation
    loops = len(t_sim)              # number of steps

    # arrays for collecting data
    acc = np.zeros((loops,3))                   # acceleration vector of rocket
    v = np.zeros((loops,3))                     # velocity vector of rocket      
    p = np.zeros((loops,3))                     # position vector of rocket
    q = np.zeros((loops,4))                     # quaternion of rocket attitude
    omega = np.zeros((loops,3))                 # angular velocity of rocket
    aero_torque = np.zeros((loops,3))
    aero_force = np.zeros((loops,3))
    thrust_force = np.zeros((loops,3))          # body frame
    thrust_torque = np.zeros((loops,3))         # body frame
    torque = np.zeros((loops,3))
    z_i = np.zeros((loops,3))                   # unit vector of z axis of rocket in fixed frame
    z_angle = np.zeros(loops)                   # rocket angle from vertical

    # Initial Conditions
    m = 0.95                                                # rocket initial mass (kg)
    g = np.array([ 0, 0,-9.81])                             # gravitational acceleration (m/s^2), Z axis is up
    acc[0,:] = g                                            # acceleration (m/s^2)
    v[0,:] = np.array([0.00001,0.0000001,0.000001])         # velocity (m/s)
    p[0,:] = np.array([0,0,drop_height])                    # position (m)
    omega[0,:] = omega_init                                 # angular rate (rad/sec)
    torque[0,:] = np.zeros(3)                               # (N-m)
    

    z_i[0,:] = np.zeros(3)

    r = R.from_euler('ZYX',np.array([0, 0, 0]))
    q0 = r.as_quat().T
    q[0,:] = q0    

    # Rocket model parameters
    feet_area_v = (np.pi*0.025**2) * 4
    legs_area_v = (0.03*0.14) *4
    tube_circle = np.pi*0.04**2
    base_area = feet_area_v+legs_area_v+tube_circle         # vertically projected area of rocket
    legs_area_s = (0.01*0.160+0.05*0.06) *2
    tube_diameter = 0.07874
    length = 0.3302
    tube_diameter = 0.07874
    tube_side_area = length * tube_diameter
    side_area = tube_side_area + 2*fin_area  + legs_area_s  # horizontally projected area of rocket


    cd_base = 1.15                                          # drag coeff of rocket in pure vertical descent (experimentally determined)
    cd_side = 1.28                                          # drag coeff of rocket in pure horizantal descent with 2 fins perpendicular to flow (approximated)

    # Engine parameters for mass calculation
    net_impulse = 13.8                      # net impulse from ThrustCurve.com (Ns)
    m_propellant = 0.009                    # propellant of mass from ThrustCurve.com (kg)
    Isp = net_impulse/(m_propellant*9.81)   # specific impulse (s)

    # initialize servo servo angles
    phi_save = np.zeros(loops)
    theta_save = np.zeros(loops)

    # create attitude_simulator object 
    att_sim_obj = FlightSimAtt.attitude_simulator( q[0,:], omega[0,:], dt )

    # create aero_simulator object
    aero_sim_obj = FlightSimAero.aero_simulator( lever_f, lever_c, fin_area, side_area)

    # create tvc_simulator object
    tvc_sim_obj = FlightSimTVC.tvc_simulator(dt,Kp,Kd)
    burn_time = tvc_sim_obj.t_thrust[-1]


    # loop through time step and simulate dynamics
    i = 0     # index i tracks simulation time step
    j = 0     # index j tracks time step within thrust curve
    while i < loops-1:

        #------- run attitude simulator --------%
        # this updates the rocket orientation for the next time step
        att_sim_obj.att_RK4( torque[i,:] )
        omega[i+1,:] = att_sim_obj.w
        q[i+1,:] = att_sim_obj.q
        rot = R.as_matrix(R.from_quat([q[i+1,1],q[i+1,2],q[i+1,3],q[i+1,0]]))
    
        #-------- run aero simulator --------%
        aero_torque[i,:], aero_force[i,:], z_i[i,:], z_angle[i] = aero_sim_obj.aero_3d( rot, v[i,:], omega[i,:], q[i,:],cd_base, cd_side, base_area, side_area, wind )

        #-------- free fall condition --------%
        # note: used both before and after burn
        if t_sim[i] < t_ignite or t_sim[i] >= t_ignite + burn_time:
            # no thrust so...
            thrust_force[i,:] = np.zeros(3)
            thrust_torque[i,:] = np.zeros(3)
            phi_save[i] = 0
            theta_save[i] = 0     # Define as zero after engine stops thrusting

            # translation motion
            # update rocket position for next time step based on current position and applicable forces
            acc[i+1,:] = g + aero_force[i,:] / m
            p[i+1,:] = p[i,:] + v[i,:] * dt + 0.5 * acc[i+1,:] * dt**2
            v[i+1,:] = v[i,:] + acc[i+1,:] * dt

        #-------- power descent --------%
        else:
            # Control in general is calculated about every 0.04 seconds
            if j == 0 or j%4 == 0:
                thrust_torque[i,:], thrust_force[i,:], phi_save[i], theta_save[i] = tvc_sim_obj.tvc_control( q[i-3,:], omega[i- 3,:], j, phi_save[i-1],theta_save[i-1] )
        
            # between these larger control steps, the control output is maintained at the most recent values returned by tvc_control
            else:                   
                thrust_force[i,:] = thrust_force[i-1,:]
                thrust_torque[i,:] = thrust_torque[i-1,:]
                theta_save[i] = theta_save[i-1]
                phi_save[i] = phi_save[i-1]

            # translation motion
            # update rocket position for next time step based on current position and applicable forces
            acc[i+1,:] = g + rot @ thrust_force[i,:]/m + aero_force[i,:] / m    # rotate thrust force into inertial frame
            p[i+1,:] = p[i,:] + v[i,:] * dt + 0.5 * acc[i,:] * dt**2
            v[i+1,:] = v[i,:] + acc[i+1,:] * dt
            # update mass
            mdot = -tvc_sim_obj.thrust_curve[j]/(9.81*Isp)
            m = m + mdot * dt
            # update thrust time step
            j += 1
    

        # update total torque acting on rocket 
        torque[i+1,:] = thrust_torque[i,:] + aero_torque[i,:]

        # update simulation time step
        i += 1

        #-------- landed --------%
        if  p[i,2] <= 0.:
            #trim remaining zeros...
            p = p[:i,:]
            v = v[:i,:]
            acc = acc[:i,:]
            z_i = z_i[:i,:]
            z_angle = z_angle[:i]
            q = q[:i,:]
            phi_save = phi_save[:i]
            theta_save = theta_save[:i]
            #...and exit the while loop
            break 
    return p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim

def plotOneSim(p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim):
    ignite_index = t_ignite * 100
    end_prop = int(ignite_index + burn_time*100)
    t = t_sim[:len(p)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    

    ax1.plot(t,p[:,2],label='Altitude [m]')
    ax1.plot(t,v[:,2],label='Vertical Velocity [m/s]')
    ax1.plot(t,acc[:,2],label='Vertical Acceleration [m/s^2]')
    ax1.plot(t,np.zeros(len(t)),'k--',label='Ground')
    ax1.legend()
    ax1.set_title('Simulated Flight')
    ax1.set_xlabel('Time [s]')

    ax2.plot(t,v[:,2],label='Vertical Velocity')
    ax2.plot(t,v[:,0],label='X Velocity')
    ax2.plot(t,v[:,1],label='Y Velocity')
    ax2.legend()
    ax2.set_title('Velocity Components During Descent')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')

    ax3.plot(t[1:],np.rad2deg(phi_save[1:]), label='phi')
    ax3.plot(t[1:],np.rad2deg(theta_save[1:]), label='theta')
    ax3.legend()
    ax3.set_ylabel('Angle [degrees]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Gimbal Angles During Descent')

    ax4.plot(t,np.rad2deg(z_angle[:]))
    ax3.set_ylabel('Angle [degrees]')
    ax3.set_xlabel('Time [s]')
    ax4.set_title('VLR Angle from Vertical Axis')


    plt.show()


    ax5 = plt.subplot(projection='3d')
    ax5.plot(z_i[:,0],z_i[:,1],z_i[:,2])
    ax5.plot(z_i[-1,0],z_i[-1,1],z_i[-1,2],'ro')
    ax5.plot([0,z_i[-1,0]],[0,z_i[-1,1]],[0,z_i[-1,2]],'k', label='Final Orientation')
    ax5.plot([0,0],[0,0],[0,1],'g--',label='Vertical Reference')
    ax5.set_xlim(-0.5,0.5)
    ax5.set_ylim(-0.5,0.5)
    ax5.set_zlim(0,1)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')
    ax5.set_title('VLR Unit Vector Orientation')
    ax5.legend()
 

    plt.show()




#-------- Single Simulation --------%
#Simulation setup
t_ignite = 0.4                       # time between rocket release and rocket ignition (sec)
drop_height = 20                    # meters
# drop_height = 30                    # meters
wind = np.array([0.75,0.5,0.])      # (m/s)
# wind = np.array([0.,0.,0.])      # (m/s)
omega_init = np.array([0.2, 0.3, 0])  # initial angular velocity rocket at release (rad/sec)
fin_area = 0.0065366                # area of one fin [m^2]
lever_f = np.array([0,0,0.32])      # vector from cg of fin to cp (meters)
lever_c = np.array([0,0,0.08671])   # vector from cg of body/legs/cap to cp (meters)
Kp = 0.01                          # proportional gain
Kd = 0.01                            # derivative gain


p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim = simulate(t_ignite, Kp, Kd, drop_height, fin_area, lever_c, lever_f,wind,omega_init)

print(v[:,2])

#-------- Visualization --------%
plotOneSim(p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim)
