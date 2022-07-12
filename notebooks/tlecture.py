import numpy as np
rng = np.random.default_rng()


def sinusoid(times, period, shift=0, amplitude=1.0):
    r"""Create a sinuoidal trend for given input parameters.

    .. math::

        d(t)= A \sin\left( \frac{2 \pi}{T} [t - s]\right)\ .


    Parameters
    ----------
    times : ndarray
      Times (days).

    period : float
      Period of the signal (days).

    shift : float
      Shift of the signal (days).

    amplitude : float
      Amplitude of the signal (days).


    Returns
    -------
    data : ndarray
      Synthetic data

    """

    # Angular frequency from period: w = 2\pi f = 2 \pi / T .
    omega = 2 * np.pi / period

    # Return sinusoid for given period, shift, and amplitude.
    return amplitude * np.sin(omega * (times - shift))


def synthetic_data(times, periods, shifts, amplitudes, random=1.0):
    """Synthetic data of linearly combined sinusoids with random data.

    Returns the sum of sinusoids defined in ``periods``, ``shifts``, and
    ``amplitudes`` plus random noise of amplitude ``random``.


    Parameters
    ----------
    times : ndarray
      Times (days).

    periods : list of floats
      Periods of the signal (days).

    shifts : list of floats
      Shifts of the signal (days); must be of same length as ``periods``.

    amplitudes : list of floats
      Amplitudes of the signal (days); must be of same length as ``periods``.


    Returns
    -------
    data : ndarray
      Synthetic data

    """

    # We could do some input checks, e.g., that `periods`, `shifts`,
    # and `amplitudes` are lists of the same length.

    # Pre-allocated output array.
    data = np.zeros(times.size)

    # Loop over provided periods, shifts, and amplitudes.
    for p, s, a in zip(periods, shifts, amplitudes):
        data += sinusoid(times, p, s, a)

    # Add random data.
    data += random*rng.random(times.size)

    # Return synthetic data.
    return data
