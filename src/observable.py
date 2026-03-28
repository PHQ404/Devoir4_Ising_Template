import numpy as np


class Observable:
    """Uses binning method to compute statistics for an observable.

    :param num_levels: The number of levels for the algorithm.
                       The number of measurements grows exponentially with
                       the number of levels.
    :type num_levels: int
    """

    def __init__(self, num_levels):
        self.num_levels = num_levels

        # Statistics for each level
        self.num_values = np.zeros(num_levels + 1, int)
        self.sums = np.zeros(num_levels + 1)
        self.sum_squares = np.zeros(num_levels + 1)

        # The last added value for each level.
        self.previous_values = np.zeros(num_levels + 1)

        # The level we are going to use.
        # A difference of 6 gives good results.
        # See lecture notes for more details.
        self.error_level = self.num_levels - 6
        self._is_filled = False

    def add_measurement(self, value, level=0):
        """Adds a measurement.

        :param value: Measurement value.
        :type value: float
        :param level: Level at which to add the measurement.
                      By default, the level should always be 0.
                      Other levels are only used for recursion.
        :type level: int
        """
        # TODO: Is the observable filled?

        self.num_values[level] += 1
        self.sums[level] += value
        self.sum_squares[level] += value**2
        # If an even number of values has been added,
        # a simplification can be made.
        if self.num_values[level] % 2 == 0:
            mean = (value + self.previous_values[level]) / 2
            self.add_measurement(mean, level + 1)
        else:
            self.previous_values[level] = value

    @property
    def is_filled(self):
        """Returns True if binning is completed."""
        raise NotImplementedError("This function is not implemented yet.")

    def error(self):
        """Returns the error on the average measurement of the observable.

        The last level must be filled before using this function.

        :return: Error on the measurement.
        :rtype: float

        :raises ValueError: If the observable is not filled yet.
        """
        if not self.is_filled():
            raise ValueError("The observable is not filled yet.")
        return np.sqrt(self.variance(self.error_level))

    def variance(self, level: int) -> float:
        """Computes the variance of the observable at a given level.

        :param level: The level at which to compute the variance.
        :type level: int

        :return: The variance of the observable.
        :rtype: float
        """
        N = self.num_values[level]
        var = 1 / (N - 1) * (self.sum_squares[level] - self.sums[level] ** 2 / N)
        return var / N

    def correlation_time(self):
        """
        Returns the correlation time.

        :return: The correlation time.
        :rtype: float

        :raises ValueError: If the observable is not filled yet.
        """
        # Hint: Similar to the error function
        raise NotImplementedError("This function is not implemented yet.")

    def mean(self):
        """Returns the mean of the measurements.

        :return: The mean of the measurements.
        :rtype: float

        :raises ValueError: If the observable is not filled yet.
        """
        raise NotImplementedError("This function is not implemented yet.")
