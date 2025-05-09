import numpy as np
import mujoco as mj

def ik_2d(z, y, l1=0.18, l2=0.18):
    L = np.sqrt(z**2 + y**2)
    if L > l1 + l2:
        raise ValueError("Target out of reach")
    knee_angle = np.arccos(np.clip((L**2 - l1**2 - l2**2)/(2*l1*l2), -1.0, 1.0))
    phi = np.arctan2(y, -z)
    psi = np.arccos(np.clip((l1**2 + L**2 - l2**2)/(2*l1*L), -1.0, 1.0))
    hip_angle = phi - psi
    return hip_angle, knee_angle


def compute_leg_angles(foot_targets):
    angles = []
    for leg in ['BR', 'FR', 'FL', 'BL']:  
        y, z = foot_targets[leg]
        hip, knee = ik_2d(z, y)
        angles.append(hip)
    for leg in ['BR', 'FR', 'FL', 'BL']:
        y, z = foot_targets[leg]
        hip, knee = ik_2d(z, y)
        angles.append(knee)
    return angles

def rot_x(theta):
    return np.array([
        [1, 0,           0,          0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0,           0,          1]
    ])

def trans_z(length):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -length], 
        [0, 0, 0, 1]
    ])

def fk_leg_matrix(hip_angle, knee_angle, l1=0.18, l2=0.18):

    T_hip = rot_x(hip_angle)
    T_thigh = trans_z(l1)
    T_knee = rot_x(knee_angle)
    T_shank = trans_z(l2)
    
    T = T_hip @ T_thigh @ T_knee @ T_shank

    foot_pos = T[:3, 3]  
    return foot_pos  



class LegCompressionPitchRollEstimator:
    def __init__(self, model, body_length=0.31, body_width=0.2):
        self.body_length = body_length  
        self.body_width = body_width   
        self.legs = ['FR', 'FL', 'BR', 'BL']
        self.sensor_ids = {
            leg: {
                'hip': mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, f"{leg}_hip_pos"),
                'knee': mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, f"{leg}_knee_pos")
            }
            for leg in self.legs
        }

    def estimate_pitch_roll(self, data):
        z = {}

        for leg in self.legs:
            hip_angle = data.sensordata[self.sensor_ids[leg]['hip']]
            knee_angle = data.sensordata[self.sensor_ids[leg]['knee']]
            foot_pos = fk_leg_matrix(hip_angle, knee_angle)
            z[leg] = foot_pos[2]  

       
        front_avg = 0.5 * (z['FR'] + z['FL'])
        back_avg = 0.5 * (z['BR'] + z['BL'])
        left_avg = 0.5 * (z['FL'] + z['BL'])
        right_avg = 0.5 * (z['FR'] + z['BR'])

        pitch = np.arctan2(back_avg - front_avg, self.body_length)
        roll = np.arctan2(right_avg - left_avg, self.body_width)

        return pitch, roll

def quaternion_to_euler(q):
   
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    pitch = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    roll = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
