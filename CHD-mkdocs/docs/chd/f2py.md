## Creating Wrapper Function for Ezseg Algorithm

How to create a wrapper function for Fortran code:  
> more information can be found [here](https://numpy.org/devdocs/f2py/f2py.getting-started.html#the-smart-way)  


* 1.) <code>python -m numpy.f2py ezseg.f -m ezsegwrapper -h ezseg.pyf</code>   
    * creates <code>ezseg.pyf</code> file with ezsegwrapper function 
* 2.) <code>cp ezseg.pyf ezsegwrapper.pyf</code>  
    * copy file and update to be python compatible  
* 3.) <code>python -m numpy.f2py -c ezsegwrapper.pyf ezseg.f</code>  
    * creates shared module <code>ezsegwrapper.so</code>
   