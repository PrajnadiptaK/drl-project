import numpy as np
from single_intersection_env import SingleIntersectionEnv

class TrafficNetworkEnv:
    """
    Stylized 4-intersection network:

        (0) ---- (1)
         |        |
        (2) ---- (3)

    """

    def __init__(self):
        self.num_nodes = 4

        self.node_names = {
            0: "Main arterial × Main arterial",
            1: "Collector × Main arterial",
            2: "Main arterial × Collector",
            3: "Collector × Collector"
        }

        # Network topology (same adjacency for CoLight)
        self.adj_matrix = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1]
        ], dtype=np.float32)

        # Traffic intensities — realistic-ish
        lam = {
            0: (0.55, 0.50),  # high NS, high EW
            1: (0.30, 0.45),  # moderate NS, high EW
            2: (0.45, 0.25),  # high NS, lower EW
            3: (0.30, 0.30)   # moderate NS & EW
        }

        # Single-lane equivalent saturation rates
        cap = {
            0: 2.5,
            1: 2.0,
            2: 2.2,
            3: 1.8
        }

        # Build intersections consistent with your SingleIntersectionEnv signature
        self.intersections = {}

        for i in range(self.num_nodes):
            lam_ns, lam_ew = lam[i]
            self.intersections[i] = SingleIntersectionEnv(
                lam_ns=lam_ns,
                lam_ew=lam_ew,
                sat_rate=cap[i],
                max_green_time=60
            )

    def reset(self):
        states = {}
        for i in range(self.num_nodes):
            states[i] = self.intersections[i].reset()
        return states

    def step(self, actions):
        next_states = {}
        rewards = {}
        done_flags = []

        for i in range(self.num_nodes):
            ns, r, d, _ = self.intersections[i].step(actions[i])
            next_states[i] = ns
            rewards[i] = r
            done_flags.append(d)

        done = any(done_flags)
        return next_states, rewards, done
