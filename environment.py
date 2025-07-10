import numpy as np
from scipy.constants import g,pi
import task_managment

class User:
    def __init__(self, area_size, user_id, queue, processor, offloader, power_usage, f_user, available_energy):
        """
        User entity that can either process or offload tasks to a UAV.

        Parameters:
        - area_size (float): Size of the square simulation area [m].
        - user_id (int): Unique ID for the user.
        - queue (Queue): Local task queue.
        - processor (Processor): Local task processor.
        - offloader (Offloader): Logic for task offloading to UAV.
        - power_usage (float): Power usage for task processing [W].
        - f_user (float): CPU frequency of the user [Hz].
        - available_energy (float): Total energy available to the user [J].
        """
        self.area_size = area_size
        self.user_id = user_id
        self.queue = queue
        self.processor = processor
        self.offloader = offloader
        self.power_usage = power_usage
        self.f_user = f_user
        self.initial_energy = available_energy
        self.available_energy = available_energy
        self.available = True

        # Random ground position
        self.x_coord = np.random.uniform(0, area_size)
        self.y_coord = np.random.uniform(0, area_size)
        self.altitude = 0
        self.position = [self.x_coord, self.y_coord, self.altitude]

        # Task statistics
        self.generated_tasks = 0
        self.offloaded_tasks = 0
        self.dropped_local = 0
        self.dropped_remote = 0
        self.completed_local = 0
        self.completed_remote = 0
        self.latency_local = []
        self.latency_remote = []

        # Energy tracking
        self.computational_energy = [0]
        self.energy_saved = 0
        self.energy_used_over_time = []

    def decide_on_task(self, task, uav, snr, bandwidth, tx_power, dropped_tasks):
        """
        Decide whether to offload or process a task locally.
        Falls back to the local queue or drops the task if no resources are available.

        Parameters:
        - task: Task object to handle.
        - uav: UAV object to potentially offload to.
        - snr (float): Signal-to-noise ratio.
        - bandwidth (float): Available bandwidth [Hz].
        - tx_power (float): Transmission power [W].
        - dropped_tasks (list): External list where dropped tasks are recorded.
        """
        if not task:
            return

        if not self.offloader.decide_offload(task, self, uav, snr, bandwidth, tx_power):
            can_enqueue = self.queue.enqueue_task(task,False)
            has_storage = self.processor.processing_storage >= task.input_size

            if not can_enqueue and not has_storage:
                self.dropped_local += 1
                dropped_tasks.append(task)


class UAV:
    def __init__(self, center, radius, altitude, velocity, mass, available_energy,
                 power_usage, speed_ratio, queue, processor, f_user, time_step,
                 propeller_radius=0.3, n_propellers=6):
        """
        UAV model that moves on a circular trajectory, processes tasks,
        and tracks energy usage.

        Parameters:
        - center (list): [x, y] coordinates of the circular trajectory center.
        - radius (float): Radius of circular flight path [m].
        - altitude (float): Fixed UAV flight altitude [m].
        - velocity (float): Flight speed [m/s].
        - mass (float): UAV mass [kg].
        - available_energy (float): Onboard energy [J].
        - power_usage (float): Power for task processing [W].
        - speed_ratio (float): UAV CPU speed relative to user speed.
        - queue (Queue): Queue for offloaded tasks.
        - processor (Processor): Task processor.
        - f_user (float): Base CPU frequency [Hz].
        - time_step (float): Simulation time step [s].
        - propeller_radius (float): Radius of a propeller [m].
        - n_propellers (int): Number of propellers.
        """
        self.center = center
        self.radius = radius
        self.altitude = altitude
        self.velocity = velocity
        self.mass = mass
        self.available_energy = available_energy
        self.power_usage = power_usage
        self.speed_ratio = speed_ratio
        self.queue = queue
        self.processor = processor
        self.f_uav = f_user * speed_ratio
        self.time_step = time_step
        self.r = propeller_radius
        self.n_propellers = n_propellers

        self.kinetic_energy = 0.5 * mass * velocity ** 2
        rho = 1.225  # Air density [kg/m³]
        self.hover_energy = np.sqrt(((mass * g) ** 3) / (2 * rho * (self.r ** 2) * self.n_propellers * pi)) * time_step

        self.computational_energy = [0]
        self.available = True

        # Trajectory-related
        self.trajectory = []
        self.n_points = 0
        self.position = []
        self.current_position = 0
        self.turn_angles = []
        self.tasks = []
        self.completed_task = None

    def compute_trajectory(self, time_step):
        """
        Compute circular 3D trajectory at fixed altitude.
        Calculates turn angles between each segment.
        """
        circumference = 2 * np.pi * self.radius
        total_time = circumference / self.velocity
        self.n_points = int(np.ceil(total_time / time_step))

        angles = np.linspace(0, 2 * np.pi, self.n_points)
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        z = np.full(self.n_points, self.altitude)

        self.trajectory = list(zip(x, y, z))
        self.position = self.trajectory[0]

        self.turn_angles = []
        for i in range(self.n_points):
            prev_point = np.array(self.trajectory[i - 1])
            curr_point = np.array(self.trajectory[i])
            next_point = np.array(self.trajectory[(i + 1) % self.n_points])

            v1 = curr_point - prev_point
            v2 = next_point - curr_point
            norm_product = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2))

            if norm_product == 0:
                angle = 0
            else:
                cos_theta = np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)
                angle = np.arccos(cos_theta)

            self.turn_angles.append(angle)

    def move_drone(self):
        """
        Advance UAV one step along the trajectory and reduce energy.
        """
        self.current_position = (self.current_position + 1) % self.n_points
        self.position = self.trajectory[self.current_position]

        # Energy cost from trajectory turns and hovering
        self.available_energy -= self.mass * (self.velocity ** 2) * self.turn_angles[self.current_position]
        self.available_energy -= self.hover_energy

        # Check availability
        if self.available_energy <= 0:
            self.available = False

    def confirm_velocity(self, time_step):
        """
        Confirm UAV velocity using displacement between trajectory points.

        Returns:
            np.ndarray: Velocities at each point [m/s].
        """
        trajectory_a = np.array(self.trajectory)
        trajectory_b = np.roll(trajectory_a, shift=-1, axis=0)
        delta = trajectory_b - trajectory_a
        displacement = np.linalg.norm(delta, axis=1)
        return displacement / time_step

