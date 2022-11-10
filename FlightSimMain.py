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
            # print(f"time: {t_sim[i]})")
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

    fig, ((ax1, ax2)) = plt.subplots(2, 1, sharex=True)    

    line_alt, = ax1.plot(t,p[:,2],label='Altitude [m]')
    line_vel, = ax1.plot(t,v[:,2],label='Vertical Velocity [m/s]')
    line_acc, = ax1.plot(t,acc[:,2],label='Vertical Acceleration [m/s^2]')
    line_gnd, = ax1.plot(t,np.zeros(len(t)),'k--',label='Ground')
    ax1.legend()
    ax1.set_title('Simulated Flight')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylim(-20, 40)
    ax1.set_xlim(0, 5)

    line_angle, = ax2.plot(t,np.rad2deg(z_angle[:]))
    ax2.set_ylabel('Angle [degrees]')
    ax2.set_ylim(0, 20)
    angle_text = ax2.set_title('VLR Angle from Vertical Axis')

    fig.subplots_adjust(bottom=0.25)
    ax_ignition = fig.add_axes([0.25, 0.20, 0.65, 0.03])
    ax_height = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    ax_p = fig.add_axes([0.25, 0.10, 0.65, 0.03])
    ax_d = fig.add_axes([0.25, 0.05, 0.65, 0.03])

    ignition_slider = Slider(
        ax=ax_ignition,
        label='Ignition Delay [s]',
        valmin=0.0,
        valmax=3,
        valinit=1.2,
    )  
    height_slider = Slider(
        ax=ax_height,
        label='Drop Height [m]',
        valmin=0.1,
        valmax=30,
        valinit=drop_height,
    )
    p_slider = Slider(
        ax=ax_p,
        label='P gain',
        valmin=0.0,
        valmax=3,
        valinit=0.01,
    )
    d_slider = Slider(
        ax=ax_d,
        label='D gain',
        valmin=-0.5,
        valmax=3,
        valinit=0,
    )



    # The function to be called anytime a slider's value changes
    def update(val):
        max_wind = 1.7
        wind = np.random.rand(3)*2*max_wind/np.sqrt(3) - (max_wind/np.sqrt(3))*np.ones(3)
        omega_init = np.random.rand(3)*0.2 - 0.1*np.ones(3)
        p,v,acc,z_i,z_angle,q,phi_save,theta_save,_,burn_time,t_sim = simulate(ignition_slider.val, p_slider.val, d_slider.val, height_slider.val, fin_area, lever_c, lever_f, wind, omega_init)
        t = t_sim[:len(p)]
        line_alt.set_data(t, p[:,2])
        line_vel.set_data(t,v[:,2])
        line_acc.set_data(t,acc[:,2])
        line_gnd.set_data(t, np.zeros_like(t))

        line_angle.set_data(t,np.rad2deg(z_angle[:]))
        angle_text.set_text(f"Angle from vertical. Wind: {np.round(np.linalg.norm(wind),2)}, Omegainit: {np.round(np.linalg.norm(omega_init),2)},")
        print("update")

        fig.canvas.draw_idle()
        plt.draw()


    # register the update function with each slider
    ignition_slider.on_changed(update)
    height_slider.on_changed(update)
    p_slider.on_changed(update)
    d_slider.on_changed(update)




    plt.show()




#-------- Single Simulation --------%
#Simulation setup
t_ignite = 1.2                       # time between rocket release and rocket ignition (sec)
drop_height = 20                    # meters
# drop_height = 30                    # meters
wind = np.array([1.4,1,0.])      # (m/s)
# wind = np.array([0.,0.,0.])      # (m/s)
omega_init = np.array([0.2, 0.3, 0])  # initial angular velocity rocket at release (rad/sec)
fin_area = 0.0065366                # area of one fin [m^2]
# lever_f = np.array([0,0,0.32])      # vector from cg of fin to cp (meters)
lever_f = np.array([0,0,0.26])      # vector from cg of fin to cp (meters) 11/10/22
# lever_c = np.array([0,0,0.08671])   # vector from cg of body/legs/cap to cp (meters)
lever_c = np.array([0,0,0.08671])   # vector from cg of body/legs/cap to cp (meters)
Kp = 0.01                          # proportional gain
Kd = 0.01                            # derivative gain


p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim = simulate(t_ignite, Kp, Kd, drop_height, fin_area, lever_c, lever_f,wind,omega_init)

print(v[:,2])

#-------- Visualization --------%
plotOneSim(p,v,acc,z_i,z_angle,q,phi_save,theta_save,t_ignite,burn_time,t_sim)
