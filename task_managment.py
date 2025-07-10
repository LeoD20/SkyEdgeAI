import environment
from channel_model import *
from bisect import insort


class Queue:
    def __init__(self, max_queue_storage):
        """
        A priority queue managing tasks based on current size and latency budget.

        Parameters:
        - max_queue_storage (int): Maximum storage in bits
        """
        self.queue = []
        self.max_queue_storage = max_queue_storage
        self.current_queue_size = 0
        self.waiting_time = 0  # total expected latency of all queued tasks

    def enqueue_task(self, task, offloader):
        """
        Adds a task to the queue if there's enough storage space.

        Returns:
            bool: True if enqueued, False otherwise
        """
        if self.current_queue_size + task.input_size < self.max_queue_storage:
            insort(self.queue, task)  # keeps queue ordered by task.current_time
            self.current_queue_size += task.input_size
            if offloader:
                self.waiting_time += task.transmission_latency
            else:
                self.waiting_time += task.processing_time

            return True
        return False

    def dequeue_task(self, offloader):
        """
        Removes and returns the earliest task from the queue.

        Returns:
            Task or None
        """
        if self.queue:
            task = self.queue.pop(0)
            self.current_queue_size -= task.input_size
            if offloader:
                self.waiting_time -= task.transmission_latency
            else:
                self.waiting_time -= task.processing_time
            return task
        return None


class Processor:
    def __init__(self, processing_storage, time_step):
        """
        Simulates a processor for local or UAV-based task execution.

        Parameters:
        - processing_storage (int): Available memory for tasks [bits]
        - time_step (float): Time step of the simulation [s]
        """
        self.processing_storage = processing_storage
        self.time_step = time_step
        self.current_time = 0
        self.current_time_step = 0
        self.busy = False
        self.task = None

    def update_current_time(self):
        """Updates the internal simulation time."""
        self.current_time += self.time_step
        self.current_time_step += 1

    def process_task(self, device, completed_tasks, dropped_tasks):
        """
        Simulates task processing and updates energy and status.

        Parameters:
        - device: The user or UAV owning this processor
        - completed_tasks (list): Tasks successfully completed
        - dropped_tasks (list): Tasks dropped due to deadline or storage

        Returns:
            bool: True if a task was processed or dropped, False otherwise
        """
        if self.busy:
            self.task.completed_now = False
            self.task.processing_time -= self.time_step

            # Compute energy used this step
            actual_step = self.time_step + min(0, self.task.processing_time)
            energy = device.power_usage * actual_step
            device.computational_energy.append(energy)
            device.available_energy -= energy

            if device.available_energy <= 0:
                device.available = False

            # Check if the task is finished
            if self.task.processing_time <= 0:
                if self.current_time - self.task.creation_time > self.task.latency_deadline:
                    self._drop_task(device, dropped_tasks)
                    return True

                self.task.processing_time = 0
                self.processing_storage += self.task.input_size
                self.busy = False

                # Mark completion
                if isinstance(device, environment.UAV):
                    self.task.user.completed_remote += 1
                    self.task.completed_now = True
                    device.completed_task = self.task
                    self.task.user.latency_remote.append(self.current_time - self.task.creation_time)
                else:
                    self.task.user.completed_local += 1
                    self.task.user.latency_local.append(self.current_time - self.task.creation_time)

                completed_tasks.append(self.task)
                return True
            else:
                # Still processing, but missed deadline
                if self.current_time - self.task.creation_time > self.task.latency_deadline:
                    self._drop_task(device, dropped_tasks)
                return True

        else:
            # Try to fetch a new task from queue
            while not self.busy:
                if device.queue.queue:
                    self.task = device.queue.dequeue_task(False)
                    self.task.current_time = self.current_time
                    if (self.task.current_time - self.task.creation_time < self.task.latency_deadline and
                        self.processing_storage > self.task.input_size):
                        self.busy = True
                        self.processing_storage -= self.task.input_size
                        return True
                    else:
                        self._drop_task(device, dropped_tasks)
                        self.busy = False
                else:
                    self.task = None
                    return False
        return False

    def _drop_task(self, device, dropped_tasks):
        """Helper to mark the task as dropped and update stats."""
        if isinstance(device, environment.UAV):
            self.task.user.dropped_remote += 1
        else:
            self.task.user.dropped_local += 1
        dropped_tasks.append(self.task)
        self.busy = False


class Task:
    def __init__(self, user, input_size, output_size, processing_time, current_time, latency_deadline,
                 energy_threshold, latency_threshold):
        """
        Task object representing user-offloaded or locally processed work.

        Parameters:
        - user (User): The user generating the task
        - input_size (int): Size of the input image file
        - output_size (int): Size of the output inference file
        - processing_time (float): Time it takes the model to process the file
        - current_time (float): Current simulation time [s]
        - latency_deadline (float): Time limit for completion [s]
        - energy_threshold (float): Threshold to classify compute-heavy tasks
        - latency_threshold (float): Threshold to classify latency-sensitive tasks
        """
        self.user = user
        self.user_id = user.user_id
        self.input_size = input_size
        self.output_size = output_size
        self.processing_time = processing_time
        self.creation_time = current_time
        self.current_time = current_time
        self.latency_deadline = latency_deadline
        self.latency_threshold = latency_threshold
        self.true_tot_latency = None
        self.offloaded = False
        self.completed_now = False

        # Energy estimate (used for classification and scheduling decisions)
        self.estimated_energy = user.power_usage * self.processing_time

        # Task classification based on latency & energy thresholds
        if latency_deadline < latency_threshold and self.estimated_energy > energy_threshold:
            self.task_type = "hybrid"
        elif latency_deadline < latency_threshold:
            self.task_type = "latency"
        elif self.estimated_energy > energy_threshold:
            self.task_type = "compute"
        else:
            self.task_type = "normal"

        # Network latency placeholders
        self.transmission_latency = 0
        self.downlink_latency = 0

    def __lt__(self, other):
        """
        Comparator for sorting tasks by current_time.
        Enables priority queuing.
        """
        return self.current_time < other.current_time


class Offloader:
    def __init__(self, queue, time_step):
        """
        Manages the task offloading decisions and packet transfer to UAV.

        Parameters:
        - queue (Queue): Queue used to stage tasks waiting for offload
        - time_step (float): Simulation time step [s]
        """
        self.queue = queue
        self.task = None
        self.busy = False
        self.time_step = time_step
        self.current_time = 0
        self.current_time_step = 0

    def update_current_time(self):
        """Advance internal simulation clock."""
        self.current_time += self.time_step
        self.current_time_step += 1

    def decide_offload(self, task, user, uav, snr, bandwidth, tx_power):
        """
        Decides whether to offload a task based on latency and energy criteria.

        Parameters:
        - task (Task): The task being considered for offloading
        - user (User): The task originator
        - uav (UAV): UAV to potentially receive the task
        - snr (float): Current Signal-to-Noise Ratio
        - bandwidth (float): Channel bandwidth [Hz]
        - tx_power (float): User transmission power [W]

        Returns:
            bool: True if the task is offloaded, False otherwise
        """
        offload = False
        threshold_snr = -9.478
        T_s = 1/bandwidth # considering the ideal square spectrum case
        if snr <= threshold_snr:
            return offload

        # Communication rates
        spectral_efficiency = get_spectral_efficiency(snr)
        uplink_rate = bandwidth * spectral_efficiency
        downlink_rate = uplink_rate  # symmetric assumption

        # Latency components
        transmission_latency = task.input_size / uplink_rate
        downlink_latency = task.output_size / downlink_rate
        uav_processing_time = task.processing_time / uav.speed_ratio
        total_offloading_latency = (self.queue.waiting_time + transmission_latency +
                                    uav.queue.waiting_time + uav_processing_time + downlink_latency)
        local_latency = task.processing_time + user.queue.waiting_time

        # Energy costs
        E_local = task.estimated_energy
        E_tx = tx_power * transmission_latency
        E_downlink = tx_power * downlink_latency

        if uav.processor.processing_storage <= task.input_size:
            return offload

        # Decision logic
        task.processing_time = uav_processing_time

        if task.task_type in ["latency", "normal"]:
            if total_offloading_latency < local_latency and self.queue.enqueue_task(task,True):
                offload = True

        elif task.task_type == "compute":
            if (E_tx) <= 0.9 * E_local and self.queue.enqueue_task(task,True):
                offload = True

        elif task.task_type == "hybrid":
            if total_offloading_latency < local_latency and (E_tx) < 0.9 * E_local:
                if self.queue.enqueue_task(task,True):
                    offload = True

        if offload:
            task.offloaded = True
            task.transmission_latency = transmission_latency
            task.downlink_latency = downlink_latency
            energy_saved = E_local - (E_tx)
            user.energy_saved += max(0, energy_saved)
            user.offloaded_tasks += 1
        else:
            # Restore original values
            task.processing_time *= uav.speed_ratio
            task.offloaded = False

        return offload

    def offload_task(self, uav):
        """
        Simulates the transmission of an offloaded task to the UAV.

        Parameters:
        - uav (UAV): Destination UAV

        Returns:
            bool: True if the task was successfully delivered, False if rejected or not ready
        """
        if not self.busy:
            if self.queue.queue:
                self.task = self.queue.dequeue_task(True)
                self.task.current_time = self.current_time - self.time_step
                self.busy = True
            else:
                self.task = None
                return False

        self.task.transmission_latency -= self.time_step
        self.task.current_time += self.time_step

        if self.task.transmission_latency <= 0:
            self.task.current_time += self.task.transmission_latency
            self.task.transmission_latency = 0
            self.busy = False

            accepted = uav.queue.enqueue_task(self.task,False)
            if not accepted:
                self.task.user.dropped_remote += 1
            return accepted

        return False
