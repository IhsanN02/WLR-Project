import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def run(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class GaitPhaseScheduler:
    def __init__(self, leg_order, step_time, dt):
        self.leg_order = leg_order
        self.step_time = step_time
        self.dt = dt
        self.active_leg_idx = 0
        self.swing_phase = 0.0

    def update(self):
        self.swing_phase += self.dt / self.step_time
        if self.swing_phase >= 1.0:
            self.swing_phase = 0.0
            self.active_leg_idx = (self.active_leg_idx + 1) % len(self.leg_order)
        return self.leg_order[self.active_leg_idx], self.swing_phase


class SwingControl:
    def __init__(self, step_length, step_height, default_stance, swing_time):
        self.step_length = step_length
        self.step_height = step_height
        self.default_stance = default_stance
        self.swing_time = swing_time

    def touchdown_location(self, leg):
        forward_offset = 0.02
        y_base = self.default_stance[leg][0] + forward_offset
        z_base = self.default_stance[leg][1]
        return [y_base, z_base]


    def next_foot_position(self, leg, swing_phase):
        y_base, z_base = self.touchdown_location(leg)
        y = y_base + self.step_length * (swing_phase - 0.5)
        z = z_base + self.step_height * np.sin(np.pi * swing_phase)
        return [y, z]


class StanceControl:
    def __init__(self, default_stance):
        self.default_stance = default_stance

    def next_foot_position(self, leg):
        return list(self.default_stance[leg])


class GaitController:
    def __init__(self, step_height=0.05, step_length=0.07, step_time=0.5, dt=0.01, pause_duration=0.1):
        self.leg_order = ['FR', 'BL', 'FL', 'BR']
        self.leg_names = ['FR', 'FL', 'BR', 'BL']

        self.default_stance = {
            'FR': [0.0, -0.224],
            'FL': [0.0, -0.224],
            'BR': [0.0, -0.224],
            'BL': [0.0, -0.224]
        }

        self.step_height = step_height
        self.step_length = step_length
        self.step_time = step_time
        self.dt = dt
        self.pause_duration = pause_duration

        self.mode = "pause"
        self.phase_timer = 0.0

        self.foot_positions = {leg: list(self.default_stance[leg]) for leg in self.leg_names}
        self.leg_phases = {leg: "stance" for leg in self.leg_names}

        self.scheduler = GaitPhaseScheduler(self.leg_order, self.step_time, self.dt)
        self.swing_controller = SwingControl(step_length, step_height, self.default_stance, self.step_time)
        self.stance_controller = StanceControl(self.default_stance)

        self.pitch_pid = PIDController(kp=0.5, ki=0.001, kd=0.1)
        self.roll_pid = PIDController(kp=0.5, ki=0.001, kd=0.1)

    def update(self, des = 0.0, estimated_pitch=0.0, estimated_roll=0.0):
        self.phase_timer += self.dt
       
        pitch_correction = self.pitch_pid.run(des - estimated_pitch, self.dt)
        roll_correction = self.roll_pid.run(des - estimated_roll, self.dt)

        if self.mode == "pause":
            for leg in self.leg_names:
                self.foot_positions[leg] = list(self.default_stance[leg])
                self.leg_phases[leg] = "stance"

            if self.phase_timer >= self.pause_duration:
                self.mode = "swing"
                self.phase_timer = 0.0

        elif self.mode == "swing":
            swing_leg, swing_phase = self.scheduler.update()

            for leg in self.leg_names:
                if leg == swing_leg:
                    foot = self.swing_controller.next_foot_position(leg, swing_phase)
                    self.leg_phases[leg] = "swing" if swing_phase < 0.05 else "stance"
                else:
                    foot = self.stance_controller.next_foot_position(leg)
                    self.leg_phases[leg] = "stance"

                self.foot_positions[leg] = foot

            if swing_phase == 0.0:
                self.mode = "pause"
                self.phase_timer = 0.0

        pitch_offset = 0.2 * np.tan(pitch_correction)
        roll_offset  = 0.1 * np.tan(roll_correction)
        for leg in self.leg_names:
            if self.leg_phases[leg] == "stance":
                if leg in ['FL', 'FR']:
                    self.foot_positions[leg][1] += pitch_offset
                if leg in ['BL', 'BR']:
                    self.foot_positions[leg][1] -= pitch_offset
                if leg in ['FL', 'BL']:
                    self.foot_positions[leg][1] += roll_offset
                if leg in ['FR', 'BR']:
                    self.foot_positions[leg][1] -= roll_offset

        return self.foot_positions

    def in_full_stance(self):
        return self.mode == "pause"

    
    