import os

import numpy as np
import pytest

from src.ising import Ising
from src.observable import Observable

UPDATE_NUM_STEPS = 5_000
LOW_NUM_STEPS = 1_000_000
MID_NUM_STEPS = 10 * LOW_NUM_STEPS
HIGH_NUM_STEPS = 10 * MID_NUM_STEPS
LEVELS = 10


def checkboard_ising(temperature, size):
    spins = np.ones((size, size), dtype=np.int64)
    spins[::2, ::2] = 1
    spins[1::2, 1::2] = -1
    return Ising(temperature, spins)


def align_ising(temperature, size, sign=1):
    spins = sign * np.ones((size, size), dtype=np.int64)
    return Ising(temperature, spins)


@pytest.mark.parametrize(
    "grid_size, steps",
    [(n, LOW_NUM_STEPS) for n in [4, 12, 32]],
)
def test_ising_at_zero_temperature(grid_size, steps):
    np.random.seed(0)
    expected_magnetization = grid_size**2
    obs = Observable(LEVELS)
    ising = checkboard_ising(temperature=1e-6, size=grid_size)
    ising.simulation(steps)
    while not obs.is_filled:
        ising.simulation(UPDATE_NUM_STEPS)
        obs.add_measurement(ising.magnetization)
    np.testing.assert_allclose(obs.mean(), expected_magnetization, rtol=0.02)


@pytest.mark.parametrize(
    "grid_size, steps, sign",
    [(n, LOW_NUM_STEPS, sign) for n in [4, 12, 32] for sign in [-1, 1]],
)
def test_ising_at_infinite_temperature(grid_size, steps, sign):
    np.random.seed(0)
    expected_magnetization = 0.0
    ising = align_ising(temperature=np.inf, size=grid_size, sign=sign)
    ising.simulation(steps)
    obs = Observable(LEVELS)
    while not obs.is_filled:
        ising.simulation(UPDATE_NUM_STEPS)
        obs.add_measurement(ising.magnetization)
    np.testing.assert_allclose(obs.mean(), expected_magnetization, atol=grid_size)


class TestIsing:
    energy_array = np.load(
        os.path.join(os.path.dirname(__file__), "data", "energy.npy"), allow_pickle=True
    )
    magn_array = np.load(
        os.path.join(os.path.dirname(__file__), "data", "magnetization.npy"),
        allow_pickle=True,
    )
    tau_array = np.load(
        os.path.join(os.path.dirname(__file__), "data", "temps_correlation.npy"),
        allow_pickle=True,
    )
    T = energy_array[:, 0]
    E, dE = energy_array[:, 1], energy_array[:, 2]
    M, dM = magn_array[:, 1], magn_array[:, 2]
    tau_E, tau_M = tau_array[:, 1], tau_array[:, 2]

    RTOL = 0.1
    ATOL = 0.1
    N_TESTS = 1

    @pytest.fixture
    def ising(self):
        np.random.seed(0)
        ising_obj = checkboard_ising(temperature=np.inf, size=32)
        return ising_obj

    def simulate_t(self, idx, ising):
        t = self.T[idx]
        ising.temperature = t
        ising.simulation(LOW_NUM_STEPS)

        energy_obs = Observable(LEVELS)
        magn_obs = Observable(LEVELS)
        while not (energy_obs.is_filled and magn_obs.is_filled):
            ising.simulation(UPDATE_NUM_STEPS)
            energy_obs.add_measurement(ising.calculate_energy())
            magn_obs.add_measurement(ising.magnetization)
        return energy_obs, magn_obs

    @pytest.mark.parametrize("idx", list(range(len(T))) * N_TESTS)
    def test_ising_energy_at_specific_temperature(self, idx, ising):
        np.random.seed(0)
        norm = ising.size**2
        energy_obs, magn_obs = self.simulate_t(idx, ising)
        np.testing.assert_allclose(
            energy_obs.mean() / norm, self.E[idx], atol=self.ATOL, rtol=self.RTOL
        )
        np.testing.assert_allclose(
            magn_obs.mean() / norm, self.M[idx], atol=self.ATOL, rtol=self.RTOL
        )

    @pytest.mark.parametrize("_", list(range(N_TESTS)))
    def test_ising_correlation_derivative_time(self, _, ising):
        np.random.seed(0)
        energy_corr_times = np.zeros(len(self.T))
        magn_corr_times = np.zeros(len(self.T))
        for idx in range(len(self.T)):
            energy_obs, magn_obs = self.simulate_t(idx, ising)
            energy_corr_times[idx] = energy_obs.correlation_time()
            magn_corr_times[idx] = magn_obs.correlation_time()
        atol = np.ceil(0.2 * len(self.T))
        np.testing.assert_allclose(
            np.sign(np.diff(energy_corr_times)), np.sign(np.diff(self.tau_E)), atol=atol
        )
        np.testing.assert_allclose(
            np.sign(np.diff(magn_corr_times)), np.sign(np.diff(self.tau_M)), atol=atol
        )
