#Script to compile EZSEG_CUDA module into linkable object file.
#
# Example use of object file:
#
#   gfortran my_code_that_calls_ezseg_cuda.f ezseg_cuda.o -L${CUDA_HOME}/lib64 -lcudart
#
# Copyright (c) 2015 Predictive Science Inc.
#
#Distributed as part of the ezseg software package at:
#http://www.predsci.com/chd
#
# Note:  You may need to add/replace gencode options to reflect your GPU's compute capability.
#
echo 'nvcc -m64 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 -O3 -Xcompiler -fPIC -c ezseg_cuda.cu'
nvcc -m64 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 -O3 -Xcompiler -fPIC -c ezseg_cuda.cu

