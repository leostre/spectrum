import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from spectrum import Spectrum
from output import show_spectra
from enumerations import LossFunc
from miscellaneous import gauss, lorentz, voigt, n_sigma_filter
import os
from output import show_curve_approx
from tqdm import tqdm


class Deconvolutor:
    vseq = ['amps',  'mus', 'widths', 'voi']
    v_pattern = {v: [1] for v in vseq}
    pipeline_fixed = [
        ('voi', 'mus'),
        ('voi', 'mus', 'widths'),
        ('voi', 'mus', 'amps'),
        # ('voi', 'widths', 'amps'),
        # ('voi', 'mus', 'amps'),
        # ('voi', 'mus', 'widths'),
        # ('voi', 'mus', 'amps'),
        # ('voi', 'widths', 'amps'),
    ]
    n_sigmas = 1

    def __init__(self, spectrum, loss_func=LossFunc.RMSE, threshold=0.01):
        self.spectrum = spectrum * 1
        self.threshold = threshold
        self.minima = None
        self.loss_func = loss_func

    def peaks_by_2der(self):
        """
        :return: (wavenumbers, intensities) of peaks defined with th 2nd derivative
        """
        orig = self.spectrum * 1
        der = orig * 1
        der.get_derivative(2)
        maxdata = orig[der.get_extrema(locals=False, minima=False)[0]][1]
        if not isinstance(maxdata, (float, int)):
            maxdata = maxdata[0]
        indices = list(filter(lambda x: orig[x][1] > self.threshold * maxdata,
                              der.get_extrema(minima=True, locals=True)[0]))
        wavenumbers = list(map(lambda x: orig[x][0], indices))
        data = list(map(lambda x: orig[x][1], indices))
        return wavenumbers, data

    @staticmethod
    def __split_in_equal(v, to_app):
        """
        Support
        :param v: iterable - vector of parameters
        :param to_app: list of str - paameters names for approximation step; determines the order
        :return: dict (str: iterable)
        """
        res = {}
        division = len(v) // len(to_app)
        def chunks(v, division):
            for i in range(0, len(v), division):
                yield v[i: i + division]
        ch = chunks(v, division)
        for param in Deconvolutor.vseq:
            if param in to_app:
                res[param] = next(ch)
        return res

    @staticmethod
    def __melt(d, to_app):
        """
        :param d: dict (str: iterable) - parameters to melt
        :param to_app: list of str - paameters names for approximation step; determines the order
        :return: list
        """
        res = []
        for param in Deconvolutor.vseq:
            if param in to_app:
                res.extend(d[param])
        return res


    def __approximate(self, params: dict):
        """
        :param params: iterable of (amplitudes, positions, widths, gauss proportions)
        :return: numpy.array of floats

        Summarize the individual bands
        """
        assert set(params) == set(Deconvolutor.vseq), 'Some of parameters are not specified!'
        x = self.spectrum.wavenums
        data = np.zeros(len(x))
        amps = params['amps']
        widths = params['widths']
        mus = params['mus']
        voi = params['voi']
        for i in range(len(amps)):
            data += voigt(x, amps[i], mus[i], widths[i], voi[i])
        return data

    def _deconvolute(self, guess: dict, fixed: dict, penalty=None):
        """
        :param guess: iterable of numeric - initial guess on the bands characteristics
        :param fixed: dict (str - name of band characteristic: iterable of float - fixed values) -
         values not involved into approximation
        :param penalty: (dict (str - name of band characteristic: iterable of float - values)) -> float
        - penalty on band ensemble characteristics
        :return: (list of (amplitudes, positions, widths, gauss proportions),
        dict (str - name of band characteristic: iterable of float - values))
        """
        to_app = [x for x in Deconvolutor.vseq if x not in fixed]
        guess_v: list = self.__melt(guess, to_app)
        deconvolutor = self
        def __residues(v, *args, **kwargs):
            to_app = [x for x in Deconvolutor.vseq if x not in fixed]
            d = deconvolutor.__split_in_equal(v, to_app)
            d = dict(**fixed, **d)
            data = deconvolutor.__approximate(d)
            return np.sum(np.square(deconvolutor.spectrum.data - data)) + (penalty(d)
                if penalty and 'widths' not in fixed else 0)
        res = least_squares(
            __residues, guess_v,
            xtol=10e-9,
            ftol=10e-9,
            bounds=list(zip(*[(0, np.inf)] * len(guess_v)))
        )
        res = res.x
        params = dict(**self.__split_in_equal(res, to_app), **fixed)
        peaks = list(zip(*[params[param] for param in Deconvolutor.vseq]))
        return peaks, params

    def deconvolute(self, pipeline_fixed=None, penalty=None, n_sigmas=None, verbose=False, save_directory=None):
        """
        :param pipeline_fixed: list of tuples of characteristics names - the instruction
         whicn characteristics to keep untouched on every single iteration of the algorithm
        :param penalty: (dict (str - name of band characteristic: iterable of float - values)) -> float
        - penalty on band ensemble characteristics
        :param n_sigmas: float - number of sigmas
        :param verbose: bool
        :param save_directory: str - path - if verbose is set True, saves the images of curve approximation by this path
        :return: (list of (amplitudes, positions, widths, gauss proportions),
        dict (str - name of band characteristic: iterable of float - values))
        """
        if not pipeline_fixed:
            pipeline_fixed = Deconvolutor.pipeline_fixed
        w, d = self.peaks_by_2der()
        n = len(w)
        params = {
            'mus': w,
            'voi': np.ones(n),
            'amps': d,
            'widths': np.ones(n)
        }

        for i, fixed in tqdm(enumerate(pipeline_fixed), total=len(pipeline_fixed)):
            if fixed == 'split':
                peaks, params = self.__split_broad_bands(peaks, params, n_sigmas)
                if verbose:
                    show_curve_approx(self.spectrum, peaks)
                continue
            fixed_params = {f: params[f] for f in fixed}
            guess_params = {f: params[f] for f in Deconvolutor.vseq if f not in fixed}
            peaks, inner_params = self._deconvolute(
                guess_params,
                fixed=fixed_params,
                penalty=penalty
            )
            params = inner_params
            if verbose:
                show_curve_approx(self.spectrum, peaks, path=os.path.join(save_directory, str(fixed) + str(i) + '.jpg')
                if save_directory else None)
                show_spectra([self.spectrum, Spectrum(self.spectrum.wavenums, peaks=peaks)])
                plt.show()
                plt.hist(params['widths'])
                plt.show()
        return peaks, params

    @staticmethod
    def width_penalty(d):
        """
        :param d: dict (str - name of band characteristic: iterable of float - values)
        :return: float
        """
        tmp = np.array(d['widths'])
        return LossFunc.MSE(tmp, np.array([np.median(tmp)] * len(tmp)))

    @staticmethod
    def symmetric_band_split(a, m, w, v):
        """
        :param a: float - amplitude
        :param m: float - position
        :param w: float - width
        :param v: float - Gauss proportion
        :return: two bands parameters sets
        """
        shift = w / 2
        return (a * 0.7, m - shift, w * 0.4, v), (a * 0.7, m - shift, w * 0.4, v)

    @staticmethod
    def __split_broad_bands(peaks, params, n_sigmas=None):
        """
        :param peaks: (list of (amplitudes, positions, widths, gauss proportions)
        :param params: dict (str - name of band characteristic: iterable of float - values))
        :param n_sigmas: float - number of sigmas
        :return: (list of (amplitudes, positions, widths, gauss proportions),
        dict (str - name of band characteristic: iterable of float - values))

        Replace each of the broad bands by two lesser
        """
        if not n_sigmas:
            n_sigmas = Deconvolutor.n_sigmas
        in_intervals = n_sigma_filter(params['widths'], n_sigmas)
        todel_indices = []
        new_peaks = []
        new_params = {param: [] for param in Deconvolutor.vseq}
        for i, in_interval in enumerate(in_intervals):
            if not in_interval:
                todel_indices.append(i)
                peak1, peak2 = Deconvolutor.symmetric_band_split(*peaks[i])
                new_peaks.append(peak1)
                new_peaks.append(peak2)
                for i, param in enumerate(Deconvolutor.vseq):
                    new_params[param].append(peak1[i])
                    new_params[param].append(peak2[i])
            else:
                for param in new_params:
                    new_params[param].append(params[param][i])
                new_peaks.append(peaks[i])

        for param in new_params:
            new_params[param] = np.array(new_params[param])
        new_peaks = np.array(new_peaks)
        return new_peaks, new_params


