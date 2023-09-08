import unittest

import numpy as np

from stochastic.processes.continuous import WienerProcess

from TrajectoryClassifier import trajtest

class TestETC(unittest.TestCase):
    def test_maximal_excursion_test_stats(self):
        std_bm = WienerProcess(t = 10)
        std_bm_traj = std_bm.sample(10)
        x = trajtest.maximal_excursion_test_stats(std_bm_traj, d = 1)
        self.assertTrue(x > 0)

        

