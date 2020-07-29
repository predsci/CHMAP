#Script to compile EZSEG module into linkable object file.
#
# Example use of object file:
#
#   gfortran my_code_that_calls_ezseg.f ezseg.o
#
# Copyright (c) 2015 Predictive Science Inc.
#
#Distributed as part of the ezseg software package at:
#http://www.predsci.com/chd
#

echo 'gfortran -c -mtune=native -fPIC -O3 -DNDEBUG -Wall -Wextra -fopenmp ezseg.f'
gfortran -c -mtune=native -fPIC -O3 -DNDEBUG -Wall -Wextra -fopenmp ezseg.f
