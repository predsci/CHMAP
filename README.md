# CHMAP: Coronal Hole Mapping and Analysis Pipeline

CHMAP is a python package focused on the detection, mapping, tracking, and
general analysis of coronal holes in the solar corona.

CHMAP includes methods for:

- Data Acquisition
- Data Reduction
- Limb Brightening Correction
- Inter-Instrument Scaling
- Flexible Map Creation
- Coronal Hole Detection
- Coronal Hole Tracking

Although CHMAP was designed for analyzing coronal hole evolution from seconds to
years, many of the database, image processing, and mapping procedures are
relevant a broad range of solar features and scientific analysis.

## Requirements

This package requires a custom python environment:

```
conda env create --file conda_recipe_chmap.yml
```

Raw data reduction steps may also
require [SSW/IDL](https://www.lmsal.com/solarsoft/) (for STEREO/EUVI) and
a [GPU deconvolution algorithm](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory).

## Database

CHMAP uses database methods (SQLite, MySQL) to facilitate flexible interaction
with solar imaging data and their byproducts (maps, time-series, etc).

Create your own database using CHMAP or use
our [example database](http://www.predsci.com/chmap/example_db/CHMAP_DB_example.zip),
which includes sample data and 14+ years of data-derived image correction
coefficients (5.7GB).

## Documentation and Examples

Pipeline documentation and examples can be found
here: [predsci.github.io/CHMAP](https://predsci.github.io/CHMAP/).

## Acknowledgements

Copyright 2021 Predictive Science Incorporated (Apache 2.0 License).

A research paper describing the initial release of CHMAP and some applications
is currently in preparation. In the meantime, if you use CHMAP code or concepts
in your own work, please refer to our prior work on this topic 
([Caplan et al., 2016](http://adsabs.harvard.edu/abs/2016ApJ...823...53C)) and cite
the corresponding Zenodo release for CHMAP: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5039439.svg)](https://doi.org/10.5281/zenodo.5039439).

The development of CHMAP was primarily supported by the NASA Heliophysics Guest
Investigators program (grant NNX17AB78G). Additional support is from the
Heliophysics Supporting Research program (grants 80NSSC18K0101 & 80NSSC18K1129).

## Contact

Cooper Downs ([cdowns@predsci.com](mailto:cdowns@predsci.com))

## Contributors

- Cooper Downs
- James Turtle
- Tamar Ervin
- Opal Issan
- Ronald M. Caplan
- Jon A. Linker