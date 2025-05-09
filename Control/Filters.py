import numpy as np

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None  

    def update(self, new_val):
        new_val = np.array(new_val)
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_val
        return self.value

class ComplementaryFilter:
    def __init__(self, alpha_pitch=0.68, alpha_roll=0.90,acc_mag_threshold=10.0, gyro_threshold=0.8,
                 lowpass_alpha=0.95, max_pitch_limit=0.6, max_roll_limit=0.6):
        
        self.alpha_pitch = alpha_pitch
        self.alpha_roll = alpha_roll
        self.acc_mag_threshold = acc_mag_threshold
        self.gyro_threshold = gyro_threshold
        self.lowpass_alpha = lowpass_alpha
        self.max_pitch_limit = max_pitch_limit
        self.max_roll_limit = max_roll_limit

        self.pitch_estimate = 0.0
        self.roll_estimate = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_roll = 0.0
   
    def update(self, accel, gyro, dt, in_stance=True):
        gyro_x, gyro_y = gyro[0], gyro[1]
        acc_mag = np.linalg.norm(accel)
        pitch_valid = False

        if in_stance and abs(gyro_y) < self.gyro_threshold:
            if abs(acc_mag - 9.81) < self.acc_mag_threshold:
                accel_pitch = np.arctan2(-accel[1], accel[2])
                gyro_pitch = self.pitch_estimate + gyro_y * dt
                raw_pitch = self.alpha_pitch * gyro_pitch + (1 - self.alpha_pitch) * accel_pitch
                
                accel_roll = np.arctan2(accel[0], accel[2])
                gyro_roll = self.roll_estimate + gyro_x * dt
                raw_roll = self.alpha_roll * gyro_roll + (1 - self.alpha_roll) * accel_roll

                if abs(raw_pitch) <= self.max_pitch_limit:
                    self.pitch_estimate = raw_pitch
                    pitch_valid = True
                if abs(raw_roll) <= self.max_roll_limit:
                    self.roll_estimate = raw_roll
                    roll_valid = True
       
        return -1*(self.pitch_estimate) + 0.131, (-1.0* self.roll_estimate)+0.0875 # Output flipped for convention
