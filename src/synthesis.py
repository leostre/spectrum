from sklearn.svm import SVC
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
from matrix import Matrix
from enumerations import Scale
import pandas as pd
from scipy.stats import expon, mode, geom, norm, uniform, rv_continuous, lognorm, skewnorm

from miscellaneous import load_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from deconvolution import Deconvolutor
from spectrum import Spectrum
from smoothing import Smoother
from time import time_ns

np.random.seed(time_ns() % 2 ** 32 - 1)


class Darwin:
    fitness_functions = []
    epsilon = 0.01
    expon_scale = 2
    additional_transform = lambda x: x  # lambda x: np.exp(-x) if x > 0 else -x**2
    _mutation_area_limits = (10, 21)
    _norm_params = {
        'mus': 0.0005,
        'widths': 0.02,
        'amps': 0.03
    }
    _uniform_params = {
        'mus': 0.00002,
        'widths': 0.02,
        'amps': 0.01
    }
    _misclassified_proportion = 0.1

    def __init__(self, estimator_path='../tmp/darwin_estimator.pkl', proba_distr=None):
        self._estimator = load_model(estimator_path)
        self._pca = PCA(30)
        self._separator = SVC()
        self.fitness = None
        self.current_population = None
        self.current_target = None
        self.offspring = None
        self.offspring_target = None
        self.limit = 1000
        self.fitted = False
        self.scale = None
        self.veclen = 0
        self.mutation_proba = 0.02
        self.target_mapping = lambda x: -1 if 'heal' in x else 1
        self.proba_distr = proba_distr if proba_distr else expon(loc=0, scale=3)
        self.peak_lability = norm()
        self.replace_elder_generation = False

    @staticmethod
    def margin_accuracy(margins):
        return np.round(sum(1 for i in margins if i > 0) / len(margins), 4)

    def get_margins(self, X, y, need_fit=False):
        X = self._pca.fit_transform(X) # check whether it is needed to be refitted
        if not self.fitted or need_fit:
            self._separator.fit(X, y)
            self.fitted = True
        return self._separator.decision_function(X) * y

    def download_population(self, path):
        population = pd.read_csv(path)
        y = population.pop(population.columns[0])
        y = y.apply(self.target_mapping)
        self.current_population = population
        self.current_target = y
        self.scale = population.columns.astype(float)
        self.veclen = len(self.scale)

    @classmethod
    def breed(cls, f_body, s_body):
        choke_points = f_body - s_body
        indices = [i for i in range(len(choke_points)) if abs(choke_points[i]) < cls.epsilon]
        times = min(len(indices), int(np.ceil(expon.rvs(loc=1, scale=cls.expon_scale))))
        for _ in range(times):
            ind = np.random.randint(0, len(indices))
            choke = indices[ind]
            f_body = np.concatenate([f_body[:choke], s_body[choke:]])
            s_body = np.concatenate([s_body[:choke], f_body[choke:]])
            del indices[ind]
        return s_body, f_body

    def __select_to_breed(self, distr_generator, max_ind=-1, **kwargs):
        if max_ind == -1:
            max_ind = len(self.current_population) - 1
        while True:
            f, s = np.round(distr_generator.rvs(size=2, **kwargs))
            if f <= max_ind and s <= max_ind:
                break
        return int(f), int(s)

    def __sort_by_fitness(self, X:pd.DataFrame=None, y:pd.Series=None, ascending=False, del_margins=True):
        if X is None or y is None:
            X, y = self.current_population, self.current_target
        X['fitness'] = pd.Series(self.get_margins(X, y))
        X.sort_values(['fitness'], ascending=ascending, inplace=True)
        y = pd.Series([y[i] for i in X.index])
        if del_margins:
            X.drop(['fitness'], axis=1, inplace=True)

    def check(self):
        plt.plot(self.current_population.index)
        self.__sort_by_fitness()
        plt.plot(self.current_population.index)
        plt.show()

    def form_generation(self, gen_size):
        assert gen_size % 2 == 0, 'Please, let the gen_size be even!'
        ng = []
        new_labels = []
        self.__sort_by_fitness()
        for _ in tqdm(range(gen_size // 2)):
            f, s = self.__select_to_breed(self.proba_distr)
            newf, news = self.breed(self.current_population.iloc[f, :], self.current_population.iloc[s, :])
            if np.random.rand() <= self.mutation_proba:
                news = self.mutate(news)
            if np.random.rand() <= self.mutation_proba:
                newf = self.mutate(newf)
            ng.append(newf)
            ng.append(news)
            new_labels.extend(self._estimator.predict(np.vstack([newf, news])))
        self.offspring = pd.DataFrame(ng, columns=self.current_population.columns)
        self.offspring_target = pd.Series(new_labels)
        print(pd.Series(new_labels).value_counts())


    def __generation_transition(self, conversion=0.5, replace=False):
        self.__sort_by_fitness(self.offspring,
                               self.offspring_target,
                               ascending=False,
                               del_margins=False)
        overwallers = self.offspring[self.offspring.fitness <= 0]
        insiders = self.offspring[self.offspring.fitness > 0]
        size = int(round(conversion * len(self.offspring_target)))
        selected_overwallers = self.__choice_skewnorm(overwallers, 0,
                                                      min(overwallers.shape[0],
                                                          int(size * self._misclassified_proportion))
                                                      )
        selected_insiders = self.__choice_skewnorm(insiders, 5,
                                                   min(insiders.shape[0],
                                                       size - selected_overwallers.shape[0])
                                                   )
        print(f'Selected {selected_insiders.shape[0] + selected_overwallers.shape[0]} of {size}')
        selected = pd.concat([selected_insiders, selected_overwallers],
                             ignore_index=True,
                             axis=0,
                             join='inner').drop(['fitness'], axis=1)
        selected_labels = pd.Series([self.offspring_target[label] for label in selected_insiders.index] +
                                    [self.offspring_target[label] for label in selected_overwallers.index])
        if replace:
            self.offspring_target = selected_labels
            self.offspring = selected
        else:
            self.offspring_target = pd.concat([self.current_target, selected_labels],
                             ignore_index=True, axis=0, join='inner')
            self.offspring = pd.concat([self.current_population, selected],
                             ignore_index=True, axis=0, join='inner')


    @staticmethod
    def __choice_skewnorm(population: pd.DataFrame, skewness: int, n_select: int) -> pd.DataFrame:
        maxval = len(population) - 1
        random = skewnorm.rvs(a=skewness, loc=maxval, size=len(population))
        random -= min(random)
        random /= max(random) * maxval
        random /= sum(random)
        indices = np.arange(population.shape[0])
        indices = np.random.choice(indices, n_select, p=random)
        return population.iloc[indices, :]

    def save_population(self, path):
        pd.concat([self.current_target, self.current_population], join='outer', axis=1).to_csv(path, index=False)

    def smooth_mutate(self):
        pass

    def __peak_mutate(self, spc):
        d = Deconvolutor(spc)
        orig = spc * 1

        # define the mutation region
        wavenums = d.peaks_by_2der()[0]
        i = np.random.randint(0, len(wavenums))
        while True:
            j = np.random.randint(i - self._mutation_area_limits[1], i + 1 + self._mutation_area_limits[1])
            if abs(j - i) > self._mutation_area_limits[0]:
                break
        tmp_spc = spc.range(wavenums[i], wavenums[j])

        # deconvolute the region
        d.spectrum = tmp_spc
        peaks, params = d.deconvolute([
            ('voi', 'amps', 'mus'),
            ('voi', 'mus'),
            ('amps', 'voi',),
            ('mus', 'voi',)
        ])
        peaks.sort(key=lambda x: x[1])
        peaks = peaks[1:-1]
        deconvoluted_band = Spectrum(orig.wavenums, peaks=peaks)

        # from output import show_curve_approx
        # show_curve_approx(Spectrum(tmp_spc.wavenums, peaks=peaks), peaks)
        _, wavenums = tmp_spc.get_extrema(locals=True, minima=True, include_edges=True)
        # form new region
        cnt = 0
        for i in range(int(geom.rvs(0.5))):
            cnt += 1
            ind = np.random.randint(0, len(peaks))
            peaks[ind] = self.__change_band(peaks[ind])
        print('Mutatuions: ', cnt)
        return deconvoluted_band, Spectrum(wavenums=orig.wavenums, peaks=peaks)

    def mutate(self, data):
        # from output import show_spectra, show_curve_approx
        spc = Spectrum(self.scale, data)
        orig = spc * 1
        deconvoluted_band, reconstructed = self.__peak_mutate(spc)
        spc -= deconvoluted_band
        spc += reconstructed
        # show_spectra([spc, orig])
        return spc.data

    @staticmethod
    def __change_band(band):
        band = list(band)
        ind = np.random.randint(1, 3)
        rng = (band[ind] * (1 - Darwin._uniform_params[Deconvolutor.vseq[ind]]),
               band[ind] * (1 + Darwin._uniform_params[Deconvolutor.vseq[ind]]))
        print('RANGE: ', rng)
        noise = uniform.rvs(rng[0], rng[1] - rng[0])
        print('NOISE: ', noise)
        band[ind] = noise
        print(Deconvolutor.vseq[ind])
        return tuple(band)

    def run(self, epochs, generation_size, conversion, verbose=False):
        for i in range(epochs):
            self.form_generation(generation_size)
            if verbose:
                counts = self.offspring_target.value_counts()
                print(f'EPOCH: {i} generated 1: {counts[1]}, -1:{counts[-1]}')
            self.__generation_transition(conversion, replace=self.replace_elder_generation)



if __name__ == '__main__':
    print('SYNTHESIS')
    d = Darwin('../tmp/darwin_estimator.pkl')
    d.mutation_proba = 0.0
    d.download_population('../tmp/EH_preproc.csv')
    d.run(3, 100, 0.8, verbose=True)
    d.save_population('../tmp/population.csv')
    # d.mutate(d.current_population.iloc[0, :].to_numpy())
    # d.check()
    # d.form_generation(1000)
    # plt.hist(Darwin.chokes, bins=100)
    # plt.show()










