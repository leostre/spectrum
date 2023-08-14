# spectrumgen

This library is design to:
* simplify the spectral data processing
* perform the deconvolution algorithm
* generate artificial spectra instances for further machine learning purposes

To achieve these goals, the following packages are implemented (You  may get more info about each of them in corresponging docstrings)
## Preprocessing
### Baseline
Scalable package baseline.py contains the basic functions to delete the baseline from a spectrum.
### Deconvolution
Here you may find one of possible realizations for the bandwise decomposition. It is founded on the least squares approximation of bands vector. The present peaks are revealed by 2nd derivative analysis.

The useful fields are:
* vseq = ('amps',  'mus', 'widths', 'voi') - this defines the common parameter order
* pipeline_fixed
  Since the number of values being estimated might be enormous the convergence point might not be reached. The partial gradual approximation is essential. The specific combination must be figured out experimantally pursuant to the required accuracy.
  Contains the list of tuples describing the fixed parameters on the each iteration. Another option: to place the 'split' string instead of a tuple. This leads to the broad bend splitting.
  By default: 
        [('voi', 'mus'),
        ('voi', 'mus', 'widths'),
        ('voi', 'mus', 'amps'),]
    
### Enumerations
Plethora of methods are higly flexible and adaptive. To avoid an inextricable doc reading, the modes are implemented as enumerations.
### Exceptions
An extensible package which doesn't cover all possible errors. Yet to be completed
### Matrix 
This package performs the way of the uniform spectra processing and collection of statistical data.
### Miscellaneous 
According to the name, this package contains various functions such as:
* Scale switcher
* Different curves equations
* Filters
* Serialization tools
### Output
Most of the functions aimed to plot the results of spectra processing are placed here.
### Scan
On the contrary, this package united various imput methods to obtain spectral data from files.
### Smoothing
Since the noise pollution approximately always does not play into our hands, the delicate elimination of it is an essential issue. Here you can find some methods of data smoothing and parameters search located in ParamGrid class.
### Spectrum
That is the basic embodiment of an individual spectrum in spectrumgen. This realization provides the foundamental capabilities, such as spectra:
* scaling
* sum and subtraction
* normalizing and standartization
* differentiation and integration
* similarity estimation
* smoothing and baseline processing
* cropping and auc calculation

		Resulting spectrum inherits all the attributes of the first argument

## Synthesis
The problem of small sample size is acute. The constructing of a fairly complex model in the medicine, for instance, tends to reach under- or overfitting, therefore, the accuracy of classification is not adequate. The collection of samples is accompanied by a huge amount of the side-work and may be too costly. 
This section focuses on the artificial sinthesis implemented by the crossingover-like spectra mixing.

### Darwin
This is a baseclass for all synthesis strategies which also performs classic method by itself.
#### Class fields:
  * epsilon = 0.01 
    the maximum difference between the same points at two spectra to be considered as the crossing point
  * expon_scale = 2
  * additional_transform 
    function (margin) -> float applied to the margin to alter the probability of the spectrum selection
  * _mutation_area_limits = (7, 13)
    The number range of peaks chosen for partial deconvolution.
  * _norm_params = {
        'mus': 0.0005,
        'widths': 0.02,
        'amps': 0.03
    }
    Dispersions for parameters to alter according to the non-biased normal distribution *(The 'voi' parameter may be added)*
  * _uniform_params = {
        'mus': 0.00002,
        'widths': 0.02,
        'amps': 0.01
    }
    The percent divations for the uniform distribution *(The 'voi' parameter may be added)*
  * _misclassified_proportion = 0.1
    In order to prevent the overfitting some portion of misclassified spectra should be kept. The small non-negative values      less than 0.1 are recommended.
  * _inbreeding_threshold = 0.99
    sets the upper limit of spectra similarity during the breeding. 
    The crossing of two nearly identical spectra may lead to the gradual population deterioration. Whereas the too low level     dramatically slows the process down.

#### Instance fields
  * _estimator
    we have no idea about the class of mutant spectra, that's why the generated spectrum has to undergo the alternative estimation.
  * _separator
    works in assumption of linear separability and provides the basic [PCA(20), SVC()] pipeline
  * fitted
    reflects the current state of the estimator
  * scale = None
    the common scale for all synthesized spectra
  * veclen = 0
    the length of a spectrum
  * mutation_proba = 0.02
    probability of mutation occurance
  * target_mapping
    Synthesis is defined only for a binary separation. The margins calculation requires the {-1, 1} labels. So the additional transform may be necessary.
  * proba_distr = scipy.stats.expon(loc=0, scale=3)
    The probability distribution for selection based on the fitness
  * replace_elder_generation = False
    Flag demonstrating if the start population is carried through the generations.
  * important_score = f1_score
    The metric to track the population development
  * current_quality = None
  * random_state = 2104
  * interclass_breeding = True
    Flag allowing the breed the objects of the same class
### BatchDarwin
Whereas the Darwin selection uses the random choice, the BatchDarwin goes further and finds the offspring subsamples improving the estimator quality. The metrics are calculated on a hold out test sample.

### SexlessDarwin
Express method of data generation. Each deconvolution stage is followed by *multiplication_coef* mutation stages leading to *multiplication_coef* new spectra. 
