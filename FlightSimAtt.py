import numpy as np
from scipy.spatial.transform import Rotation as R
class attitude_simulator:
    def __init__(self, q, w, dt):
        self.moi = np.array([[0.01966931, 0, 0],[0, 0.01965769, 0],[0, 0, 0.0020481811]]) #MOI-Principal Axis
        self.q = np.array([q[3],q[2],q[1],q[0]])     # quaternion, the first element is scalar
        self.w = w     # angular velocity under the body frame
        self.dt = dt   # integration step size

    '''
         Ref: CubeSat Attitude Determination via Kalman Filtering of Magnetometer and Solar Cell Data
         param q: quaternion
         return A: rotation matrix from body frame to inertial frame.
    '''
    def QtoR(self, q):
        
        q = np.array([q[3],q[2],q[1],q[0]])
        A = np.zeros((3,3))
        A[0,0] = q[0]**2+q[1]**2-q[2]**2-q[3]**2
        A[0,1] = 2*(q[1]*q[2]-q[0]*q[3])
        A[0,2] = 2*(q[1]*q[3]+q[0]*q[2])
        A[1,0] = 2*(q[1]*q[2]+q[0]*q[3])
        A[1,1] = q[0]**2-q[1]**2+q[2]**2-q[3]**2
        A[1,2] = 2*(q[2]*q[3]-q[0]*q[1])
        A[2,0] = 2*(q[1]*q[3]-q[0]*q[2])
        A[2,1] = 2*(q[2]*q[3]+q[0]*q[1])
        A[2,2] = q[0]**2-q[1]**2-q[2]**2+q[3]**2

        return A
    '''
        Euler rotation equation.
        Ref: Markley, Landis & Crassidis, John. (2014). Fundamentals of Spacecraft Attitude Determination and Control
        param omega: angular velocity.
        param T: torque
    '''
    def Dynamics(self, omega, T):
        wdot = np.linalg.inv(self.moi)@( T - np.cross( omega,(self.moi @ omega) ) )
        return wdot

    '''
       quaternion kinematics.
       Ref: CubeSat Attitude Determination via Kalman Filtering of Magnetometer and Solar Cell Data
       param q: quaternion
       param wquat: angular velocity under the body frame
       reutrn qdot: rate of change of quaternion 
    '''
    def quatKin(self, q, wquat):
       #quatermions are represented as q1 being the scalar component
       E = np.zeros((4,3))
       E[0,0] = -q[1]
       E[0,1] = -q[2]
       E[0,2] = -q[3]
       E[1,0] = q[0]
       E[1,1] = -q[3]
       E[1,2] = q[2]
       E[2,0] = q[3]
       E[2,1] = q[0]
       E[2,2] = -q[1]
       E[3,0] = -q[2]
       E[3,1] = q[1]
       E[3,2] = q[0]
       qdot = 0.5 * E @ (wquat.T)
       return qdot

    '''
       4th fixed step runge-kutta attitude simulator
       update obj.q and obj.w
       param torque_input: current torque
       Ref: http://ai.stanford.edu/~varung/rigidbody.pdf
    '''
    def att_RK4(self, torque_input):
       # euler rotation dynamics
       k1 = self.dt * self.Dynamics( self.w, torque_input )
       k2 = self.dt * self.Dynamics( self.w + 0.5*k1, torque_input )
       k3 = self.dt * self.Dynamics( self.w + 0.5*k2, torque_input )
       k4 = self.dt * self.Dynamics( self.w + k3, torque_input )
  
       # quaternion kinematics
       k1q = self.dt * self.quatKin( self.q, self.w )
       k2q = self.dt * self.quatKin( self.q + 0.5*k1q, self.w + 0.5*k1 )
       k3q = self.dt * self.quatKin( self.q + 0.5*k2q, self.w + 0.5*k2 )
       k4q = self.dt * self.quatKin( self.q + k3q, self.w + k3 )

       self.w = self.w + (1/6) * ( k1 + 2*k2 + 2*k3 + k4 )
       self.q = self.q + (1/6) * ( k1q + 2*k2q + 2*k3q + k4q )
            
       #renormalize quaternions
       self.q = self.q / np.linalg.norm(self.q);


