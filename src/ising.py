import numba as nb
import numpy as np


def random_ising(temperature, size):
    """Generates a random spin grid.

    :param temperature: System temperature.
    :type temperature: float
    :param size: The grid has a size of size x size.
    :type size: int
    :return: Initialized Ising object.
    :rtype: Ising
    """
    # Randomly initialize spins with values -1 or +1
    # on a grid of shape (size x size).
    spins = np.ones((size, size), dtype=np.int64)
    rn_signs_mask = np.random.randint(0, 2, (size, size), dtype=np.int64).astype(
        np.bool_
    )
    spins[rn_signs_mask] = -1
    return Ising(temperature, spins)


# Numba allows compiling the class for faster execution.
# Some operations are not permitted, so attention is needed.
@nb.experimental.jitclass(
    [
        ("temperature", nb.float64),
        ("spins", nb.int64[:, :]),
        ("size", nb.uint64),
        ("energy", nb.int64),
        ("itr", nb.uint64),
    ]
)  # type: ignore
class Ising:
    """2-dimensional paramagnetic Ising model.

    Represents a grid of classical spins with coupling J = +1 between
    nearest neighbors.

    :param temperature: System temperature.
    :type temperature: float
    :param spins: Square array of spin values.
    :type spins: numpy.ndarray
    """

    def __init__(self, temperature: float, spins: np.ndarray):
        self.temperature = temperature
        self.spins = spins
        self.size = np.shape(spins)[0]
        self.energy = self.calculate_energy()

    @property
    def magnetization(self):
        """Returns the current magnetization (in absolute value) of the spin grid."""
        raise NotImplementedError("This function is not implemented yet.")

    def energy_difference(self, x, y):
        """Returns the energy difference if the spin at position (x, y)
        were reversed.

        :param x: x-coordinate of the spin.
        :type x: int
        :param y: y-coordinate of the spin.
        :type y: int
        :return: Energy difference.
        :rtype: int
        """
        raise NotImplementedError("This function is not implemented yet.")

    def random_iteration(self):
        """Reverses a random spin with probability ~ e^(-ΔE / T).

        This function updates the grid with the new spin value.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def simulation(self, num_iterations):
        """Simulates the system by performing random iterations.

        :param num_iterations: Number of iterations to perform.
        :type num_iterations: int
        """
        for _ in range(num_iterations):
            self.random_iteration()

    def calculate_energy(self):
        """Returns the current energy of the spin grid.

        :return: Current energy.
        :rtype: int
        """
        # TODO: Do not simply compute the energy with nested loops on all elements.
        # This function should be optimized.
        raise NotImplementedError("This function is not implemented yet.")
