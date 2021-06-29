# PSF Deconvolution

Raw data is fully reduced using python/Sunpy methods whenever possible. Part of the STEREO/EUVI reduction steps require
require [SSW/IDL](https://www.lmsal.com/solarsoft/) for `secchi_prep`, which are wrapped in python command line calls.

Then the reduced data is processed using a Richardson Lucy [GPU deconvolution code](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory).
Currently this is done using a remote script to send the data to a GPU enabled machine where this code is installed, deconvolve it, and send it back. More details on this process will be provided soon.
