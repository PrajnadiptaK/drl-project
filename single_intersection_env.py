import numpy as np


class Lane:
    """
    Simple lane holding number of waiting vehicles.
    """
    def __init__(self, lane_id, max_capacity=100):
        self.id = lane_id
        self.max_capacity = max_capacity
        self.queue = 0

    def add_vehicles(self, n):
        self.queue = min(self.queue + n, self.max_capacity)

    def remove_vehicles(self, n):
        self.queue = max(self.queue - n, 0)


class SingleIntersectionEnv:
    """
    Traffic intersection with 2 approaches:
      - North-South (NS)
      - East-West (EW)

    Signal phases:
      Phase 0: NS green, EW red
      Phase 1: EW green, NS red

    Actions:
      0 = EXTEND current phase
      1 = SWITCH phase
    """

    def __init__(
        self,
        dt=5.0,
        max_time=3600.0,
        lam_ns=0.3,
        lam_ew=0.3,
        sat_rate=0.8,
        max_green_time=60.0
    ):
        self.dt = dt
        self.max_time = max_time
        self.lam_ns = lam_ns
        self.lam_ew = lam_ew
        self.sat_rate = sat_rate
        self.max_green_time = max_green_time

        # Episode time
        self.time = 0.0

        # Two lanes
        self.lane_ns = Lane("NS")
        self.lane_ew = Lane("EW")

        # Signal settings
        self.current_phase = 0  # start with NS green
        self.phase_elapsed = 0.0

    def _sample_arrivals(self):
        arrivals_ns = np.random.poisson(self.lam_ns)
        arrivals_ew = np.random.poisson(self.lam_ew)
        return arrivals_ns, arrivals_ew

    def _apply_departures(self):
        capacity = self.sat_rate  # per timestep

        if self.current_phase == 0:
            depart_ns = min(self.lane_ns.queue, capacity)
            self.lane_ns.remove_vehicles(depart_ns)
            depart_ew = 0
        else:
            depart_ew = min(self.lane_ew.queue, capacity)
            self.lane_ew.remove_vehicles(depart_ew)
            depart_ns = 0

        return depart_ns, depart_ew

    def _get_state(self):
        q_ns = self.lane_ns.queue
        q_ew = self.lane_ew.queue

        phase_0 = 1.0 if self.current_phase == 0 else 0.0
        phase_1 = 1.0 if self.current_phase == 1 else 0.0

        phase_norm = min(self.phase_elapsed / self.max_green_time, 1.0)

        return np.array([q_ns, q_ew, phase_0, phase_1, phase_norm], dtype=np.float32)

    def _compute_reward(self):
        total_queue = self.lane_ns.queue + self.lane_ew.queue
        return -float(total_queue)

    def reset(self):
        self.time = 0.0
        self.lane_ns.queue = 0
        self.lane_ew.queue = 0
        self.current_phase = 0
        self.phase_elapsed = 0.0
        return self._get_state()

    def step(self, action):
        if action == 1:
            self.current_phase = 1 - self.current_phase
            self.phase_elapsed = 0.0
        else:
            self.phase_elapsed += self.dt

        # force phase change if green too long
        if self.phase_elapsed > self.max_green_time:
            self.current_phase = 1 - self.current_phase
            self.phase_elapsed = 0.0

        # arrivals
        arrivals_ns, arrivals_ew = self._sample_arrivals()
        self.lane_ns.add_vehicles(arrivals_ns)
        self.lane_ew.add_vehicles(arrivals_ew)

        # departures
        self._apply_departures()

        # time update
        self.time += self.dt
        done = self.time >= self.max_time

        reward = self._compute_reward()
        next_state = self._get_state()
        info = {
            "time": self.time,
            "queue_ns": self.lane_ns.queue,
            "queue_ew": self.lane_ew.queue,
        }

        return next_state, reward, done, info
