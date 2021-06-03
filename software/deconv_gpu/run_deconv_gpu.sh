#!/bin/bash
#
# Script to run ALGO SGP deconvolution algorithm on a remote host
# in order to make use of GPU accelerated code.

# Usage:
# run_deconv_gpu.sh instrument_choice
# where instrument_choice is one of the options below:
verb=0

valid_instrument_options="STA_195_2048
STB_195_2048
STA_195_2048_SHEARER
STB_195_2048_SHEARER
AIA_193_2048
AIA_94_4096
AIA_131_4096
AIA_171_4096
AIA_193_4096
AIA_211_4096
AIA_304_4096
AIA_335_4096
AIA_94_4096_PSF2
AIA_131_4096_PSF2
AIA_171_4096_PSF2
AIA_193_4096_PSF2
AIA_211_4096_PSF2
AIA_304_4096_PSF2
AIA_335_4096_PSF2"

instrument_choice=$1

instok=0
for inst_test in ${valid_instrument_options}; do
  if [[ ${inst_test} == ${instrument_choice} ]]; then
    instok=1
    break
  fi
done

if [ ${instok} -eq 0 ]
then
  echo "ERROR! Invalid instrument choice!  You tried to use: ${instrument_choice}"
  exit 1
fi

# pick one of the working remote hosts.
#remote_host="Q.predsci.com"
remote_host="varda.predsci.com"

if [ "$remote_host" ==  "Q.predsci.com" ]; then
  psf_location="/usr/local/psf_compute"
  conf_string="./conf/${instrument_choice}.conf"
  lib_cmd='LD_LIBRARY_PATH=/usr/local/pgi/linux86-64/2018/cuda/9.1/lib64:${LD_LIBRARY_PATH} ; '

elif [ "$remote_host" ==  "varda.predsci.com" ]; then
  psf_location="/usr/local/psf_compute"
  conf_string="./conf/${instrument_choice}.conf"
  lib_cmd='export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64 ; '
fi

# note system libraries are usually too new for algo_image_gpu, lib_cmd is used to
# link to older versions by setting the library paths manually

echo "Selected PSF is: ${instrument_choice}"

# Transfer original image to remote compute directory:
if [ $verb -eq 1 ] 
then
  echo "Transfering Image_orig.dat to ${remote_host}..."
fi
if ! scp ./Image_orig.dat ${remote_host}:${psf_location}; then
  echo "ERROR: scp of Image_orig.dat did not run correctly!"
  exit 1
fi

# Run algo_image_gpu ./conf/INS.conf
if [ $verb -eq 1 ] 
then
  echo "SSHing into ${remote_host} and running algo..."
fi
ssh_exec='cd '${psf_location}" ; ${lib_cmd} ./algo_image_gpu "${conf_string}
echo $ssh_exec
if ! ssh ${remote_host} ${ssh_exec}; then
  echo "ERROR: ssh did not run correctly!"
  exit 1
fi

# Transfer resulting image back to local directory:
if [ $verb -eq 1 ] 
then
  echo "Transfering Image_new.dat from ${remote_host} to here..."
fi
if ! scp ${remote_host}:${psf_location}/Image_new.dat .; then
  echo "ERROR: scp of Image_new.dat did not run correctly!"
  exit 1
fi




