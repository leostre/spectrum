from enumerations import Scale
from numpy import square, sqrt, log, exp, zeros, std, mean
import pickle

def scale_change(scale_type):
    """
    Define the function of wavenumbers recalculation.
    params
    scale_type: Scale - determines the scale units

    rtype: (str|float,) -> float
    """
    if scale_type == Scale.WAVELENGTH_nm:
         return lambda x: 10_000_000. / float(x)
    elif scale_type == Scale.WAVELENGTH_um:
        return lambda x: 10_000. / float(x)
    else:
        return lambda x: float(x) / 1.


width_sigma = 2 * sqrt(log(2))  # * np.sqrt(2)
width_lambda = 2.


def gauss(x, amp, mu, w):
    """
    :param x: iterable of numeric - wavelengths axis
    :param amp: float - Gaussian bell amplitude
    :param mu: float - peak position
    :param w: float - band width
    :return: numpy.array of floats

    Gauss curve
    """
    sigma = w / width_sigma
    return amp * exp(-square((x - mu) / sigma))


def lorentz(x, amp, x0, w):
    """
    :param x: iterable of numeric - wavelengths axis
    :param amp: float - pseudo-Voigt bell amplitude
    :param x0: float - peak position
    :param w: float - band width
    :param gauss_prop:
    :return: numpy.array of floats

    Lorentz curve
    """
    return amp / (square(2 * (x - x0) / w) + 1.)


def voigt(x, amp, x0, w, gauss_prop):
    """
    :param x: iterable of numeric - wavelengths axis
    :param amp: float - pseudo-Voigt bell amplitude
    :param x0: float - peak position
    :param w: float - band width
    :param gauss_prop:
    :return: numpy.array of floats

    Pseudo-Voigt curve
    """
    return gauss_prop * gauss(x, amp, x0, w) + (1 - gauss_prop) * lorentz(x, amp, x0, w)

def summ_voigts(x, params):
    """
    :param x: iterable of numeric - wavelengths axis
    :param params: iterable of (amplitudes, positions, widths, gauss proportions)
    :return: numpy.array of floats - intensities
    """
    data = zeros(len(x))
    for amp, mu, w, g in params:
        data += voigt(x, amp, mu, w, g)
    return data

def n_sigma_filter(sequence, n=1):
    """
    :param sequence: iterable of numeric
    :param n: float - number of sigmas
    :return: filtered sequence
    """
    sigma = std(sequence)
    mu = mean(sequence)
    lo = mu - sigma * n
    hi = mu + sigma * n
    return [lo <= sequencei <= hi for sequencei in sequence]

def filter_opus(path):
    """
    Check whether the file is a valid opus one.
    params:
    path: str - the path to the destination file
    rtype: bool
    """
    ext = path[path.rfind('.') + 1:]
    if not ext.isdigit():
        return False
    with open(path, 'r') as f:
        try:
            f.read()
            return False
        except:
            return True

def save_model(model, path):
    """
    Pickles the object to the path.
    """
    with open(path, 'wb') as out:
      pickle.dump(model, out)

def load_model(path):
    """
    Inpickles the object aquired from the path
    """
    with open(path, 'rb') as inp:
        tmp = pickle.load(inp)
    return tmp


