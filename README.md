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
  Contains the list of tuples describing the fixe
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

## Synthesis
The problem of small sample size is acute. The constructing of a fairly complex model in the medicine, for instance, tends to reach under- or overfitting, therefore, the accuracy of classification is not adequate. The collection of samples is accompanied by a huge amount of the side-work and may be too costly. 
This section focuses on the artificial sinthesis implemented by the crossingover-like spectra mixing. The
