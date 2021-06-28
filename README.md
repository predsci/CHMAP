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