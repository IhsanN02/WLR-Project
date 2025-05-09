import mujoco as mj
import mujoco.viewer
import numpy as np
from InverseKinematics import ik_2d, compute_leg_angles
from PIDGait import GaitController
from Filters import LowPassFilter
from Filters import ComplementaryFilter



xml_path = "scene.xml"

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)


lowpass_filter1 = LowPassFilter(alpha=0.9999887)
lowpass_filter2 = LowPassFilter(alpha=0.9999875)


def set_initial_pose():
    data.qpos[:] = [
            0, 0, 0.244, 1, 0, 0, 0,  
            -0.9, 1.8, 0, 
            -0.9, 1.8, 0, 
            -0.9, 1.8, 0, 
            -0.9, 1.8, 0
        ]
    data.ctrl[:] = [-0.9, -0.9, -0.9, -0.9, 
                    1.8, 1.8, 1.8, 1.8, 
                    0, 0, 0, 0]  

    mj.mj_forward(model, data)
    

set_initial_pose()

controller = GaitController(dt=0.001)




simulation_duration = 40.0  # seconds
dt = model.opt.timestep
# Run simulation

pitch_roll = ComplementaryFilter()


def pitch_dependent_wheel_control(pitch_estimate, base_torque=2.0, gain=10.0):
   
    return base_torque + gain * pitch_estimate

    


with mj.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < simulation_duration:
        
        
        accel = data.sensordata[0:3]     
        gyro = data.sensordata[3:6]      
        
        pitch_est, roll_est = pitch_roll.update(accel, gyro, dt, controller.in_full_stance)
       

        t = data.time
        
        if data.time > 1.0:
            
         
               
        
            foot_targets = controller.update(lowpass_filter1.update(pitch_est), lowpass_filter2.update(roll_est))
                    
            try:
                    joint_angles = compute_leg_angles(foot_targets)
                    wheel_speed = pitch_dependent_wheel_control(lowpass_filter1.update(pitch_est), base_torque=10.0, gain=20.0)
                    wheel_speeds = 4*[wheel_speed]
                    data.ctrl[:] = joint_angles + wheel_speeds
            except ValueError:
                            print("IK target out of reach")
            
        mj.mj_step(model, data)
        viewer.sync()
        


 

   



