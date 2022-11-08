import numpy as np
from scipy.spatial.transform import Rotation as R

class aero_simulator:
    def __init__(self, lever_f, lever_c, fin_area, side_area):
        self.lever_f = lever_f    #length from cp of fin to cg (meters)
        self.lever_c = lever_c #length from cp of cylinder to cg (meters)
        self.fin_area = fin_area
        self.side_area = side_area
        self.rho = 1.225                       #air density (kg/m^3)

    def empirical_torque_force(self, v, omega, z_i, ref_cd, ref_area, wind_area_mod, wind):
        CG = np.array([-0.004,-0.006,0.014])                # Center of gravity referenced from gimbal, meters 
        CV = np.array([-0.3279, -0.4369, 209.9838])* 1/1000 # Center of Volume, same reference, meters           
       
             
        # Calculate vertical aerodynamic force
        vert_drag = (0.5*ref_cd*self.rho*ref_area*v[2]**2)

        # Calculate torque about cg                                                                                       
        area_f = 2*self.fin_area
        area_c = self.side_area 
        ff_lever_c = np.linalg.norm(self.lever_c)*z_i #translate lever arms to fixed frame
        ff_lever_f = np.linalg.norm(self.lever_f)*z_i
        torque_c = np.cross(ff_lever_c, 0.5*ref_cd*self.rho*area_c*v**2)[0] + np.cross(ff_lever_c, 0.5*ref_cd*self.rho*area_c*wind_area_mod*wind**2)[0] 
        torque_f = np.cross(ff_lever_f, 0.5*ref_cd*self.rho*area_f*v**2)[0] + np.cross(ff_lever_f, 0.5*ref_cd*self.rho*area_f*wind_area_mod*wind**2)[0] 
        
        torque_b = (torque_c + torque_f)

        return torque_b, vert_drag

    '''
        param rot: rotation matrix from body frame to inertial frame.
        param v: rocket velocity vector, m/s
        return torque_3d: aero torque vector under the body frame, N*m
        return force_3d: aero force vector,inertial frame, m/s.
    '''
    def aero_3d(self, rot, v, omega, q, cd_base, cd_side, base_area, side_area, wind ): #added alpha as an output
        # get the rocket symmetry axis / z axis under the inertial frame
        z_i = (rot @ np.array([[0],[0],[1]])).T
        
        # calculate modifier to scale ref area to area acted on by wind
        xy_mag = np.linalg.norm(np.array([z_i[0,0],z_i[0,1]]))
        z_mag = z_i[0,2]
        if (xy_mag > 0.):
            xy_hat  = np.array([z_i[0,0],z_i[0,1],0])/np.linalg.norm(np.array([z_i[0,0],z_i[0,1],0]))
            side_normal = np.array([z_mag*xy_hat[0],z_mag*xy_hat[1],-xy_mag])
            side_normal = side_normal / np.linalg.norm(side_normal)
            wind_area_mod = np.dot(side_normal,xy_hat)
        else: wind_area_mod = 1

        z_angle = np.arccos((np.dot(z_i,np.array([[0],[0],[1]])) / np.linalg.norm(z_i) )+0j)[0,0]
        z_angle = z_angle.real

        # get the reference area for vertical drag calculation
        ref_area = base_area*np.cos(z_angle) + side_area*np.sin(z_angle)
        # approximate the reference drag coeff between base and side cd
        ref_cd = cd_base + (cd_side-cd_base)*(z_angle/(np.pi/2))
        
        # get torque vector and force scalar
        torque_b, vert_drag = self.empirical_torque_force(v, omega, z_i, ref_cd, ref_area, wind_area_mod, wind)
        
        #---- Torque ----%
        torque_3d = rot @ torque_b

        #---- Force ----%
        force_3d = np.array([0,0,vert_drag])
        
        return torque_3d, force_3d, z_i, z_angle
