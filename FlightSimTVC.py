import numpy as np
from scipy.interpolate import interp1d

class tvc_simulator:
    def __init__(self,dt,Kp,Kd):
        self.lever_gimbal = 0.08           # lever arm from gimbal to cg (meters)
        self.servo_limit = np.deg2rad(3)   # maximum servo angle, rad. +-3 deg from https://www.youtube.com/watch?v=mA7TwcemOh0
        self.angle_initial = 0

        # load thrust curve
        '''
        Note: 
            Thrust curve data should be formated in 2 columns where the first column is time and the second column is thrust.
            The file should start when the signal for ignition was sent. i.e. it contains any delay inherent to the ignitor.
        '''
        file_name = open('Quest_D20W_real.csv')
        motor = np.loadtxt(file_name, delimiter=",")
        
        # store thrust curve
        self.t_thrust = motor[:,0]    #propulsion time vector
        thrust = motor[:,1]           #propulsion force vector
            
        # interpolation
        # increase resolution of thrust curve to match simulation time steps
        self.t = np.arange(self.t_thrust[0], self.t_thrust[-1], dt)     # new interpolation step
        self.thrust_curve = interp1d(self.t_thrust,thrust,'linear');    # derive interpolated thrust curve
        self.thrust_curve = self.thrust_curve(self.t)
        #print(self.thrust_curve)

        # store controller gains locally
        self.Kp = Kp
        self.Kd = Kd

    '''
        apply PD controller, calculate force and torque based on servo.
        @param q: quaternion
        @param omega: angular velocity, body frame, rad/s
        @param j: thrust curve location
        @return thrust_torque: real thrust torque under body frame
        @return thrust_force: real thrust force under body frame
    '''
    def tvc_control( self, q, omega, j,phi_prev,theta_prev ):
        # get current thrust
        f_real = self.thrust_curve[j] 
       
        if f_real == 0:
            thrust_force = np.array([[0],[0],[0]])
            thrust_torque = np.array([[0],[0],[0]])
            phi = 0
            theta = 0   #Define values for when engine is off
        else:
            #%-------- Onboard Controller Start --------%
            # run PD controller to get desired torque
                
            #Simulate gyro readings of passed data: 

            tau_x_d = -self.Kp*q[2] - self.Kd*omega[1]  #ideal torque about x body axis
            tau_y_d = -self.Kp*q[1] - self.Kd*omega[2]    #ideal torque about y body axis

            # get desired forces from torque
            f_x_d = -tau_y_d / self.lever_gimbal
            f_y_d = tau_x_d / self.lever_gimbal

            # get real servo angle phi (around x axis)
            if abs(-f_y_d/f_real) > np.sin(self.servo_limit):
                phi = np.sign(-f_y_d) * self.servo_limit    
            
            else:
                phi = np.arcsin(-f_y_d/f_real)  #asin return -pi/2 ~ pi/2
                  

            # get real servo angle theta (around y axis)
                
            if abs( f_x_d/(f_real*np.cos(phi)) ) > np.sin(self.servo_limit):
                     theta = np.sign(f_x_d) * self.servo_limit  # Multiply by proportion of count to stall the servo commands and mirror real software

            else:
                theta = np.arcsin( f_x_d/(f_real*np.cos(phi)) ) #asin return -pi/2 ~ pi/2

            
                
            #%-------- Onboard Controller End --------%

            # get real force under body frame
            f_x_real = f_real * np.sin(theta) * np.cos(phi)
            f_y_real = -f_real * np.sin(phi)
            f_z_real = f_real * np.cos(theta) * np.cos(phi)
            thrust_force = np.array([[f_x_real],[ f_y_real],[ f_z_real]])
            
                
            # get real torque under body frame
            thrust_torque = np.array([[f_y_real * self.lever_gimbal],[ -f_x_real * self.lever_gimbal], [0]])


        return thrust_torque.T, thrust_force.T, phi, theta

            


