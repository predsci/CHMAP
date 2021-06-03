#!/bin/bash
#
# Script to run DECURLOG deconvolution algorithm on a remote GPU-enabled host.
#
# This is Mark Cheung's GPU code, modified in 2020 by Ron to fix some calling/IO stuff

# Usage:
# run_remote_deconv_gpu.sh <instrument_choice> <resolution> <input_fits_file>
#    where <instrument_choice> is one of the options listed below,
#    <resolution> is either 2048 or 4096 and matches the <instrument_choice>,
#    and <input_file_fits> is a fits file of the input image.

remote_host="q.predsci.com"
remote_psf_compute_location="/work/psi/psf_compute/tmp"

verb=0

if [ $verb -eq 1 ] 
then
  echo "Remote host:  $remote_host"
  echo "Remote PSF computation directory:  $remote_psf_compute_location"
  echo ""
fi

valid_instrument_options="
STA_171_2048
STA_171_2048_SHEARER
STA_195_2048
STA_195_2048_SHEARER
STA_284_2048
STA_284_2048_SHEARER
STA_304_2048
STA_304_2048_SHEARER
STB_171_2048
STB_171_2048_SHEARER
STB_195_2048
STB_195_2048_SHEARER
STB_284_2048
STB_284_2048_SHEARER
STB_304_2048
STB_304_2048_SHEARER
AIA_94_4096
AIA_94_4096_PSF2
AIA_131_4096
AIA_131_4096_PSF2
AIA_171_4096
AIA_171_4096_PSF2
AIA_193_4096
AIA_193_4096_PSF2
AIA_211_4096
AIA_211_4096_PSF2
AIA_304_4096
AIA_304_4096_PSF2
AIA_335_4096
AIA_335_4096_PSF2"

instrument_choice=$1
res=$2
inputfile=$3

outfile="$(basename $inputfile)"
outfile=${outfile%.fits}_deconvolved.fits

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

#Make sure resoltion is ok:
if [[ "$instrument_choice" != *"$res"* ]]; then
  echo "ERROR! Resolution does not match instrument choice!  You tried to use: ${instrument_choice} with resolution: ${res}"
  exit 1
fi

echo "Selected PSF is: ${instrument_choice}"

# Transfer original image to remote compute directory:
if [ $verb -eq 1 ] 
then
  echo "Transfering ${inputfile} to ${remote_host}..."
fi
if ! scp ./${inputfile} ${remote_host}:${remote_psf_compute_location}; then
  echo "ERROR: scp of ${inputfile} did not run correctly!"
  exit 1
fi

ifile_local="$(basename ${inputfile})"


# Run decurlog
if [ $verb -eq 1 ] 
then
  echo "SSHing into ${remote_host} and running algo..."
fi
ssh_exec='cd '${remote_psf_compute_location}" ; ../src/decurlog_${res} --psf=../psf/PSF_${instrument_choice}_SHIFTED.fits --i=${ifile_local} --o=${outfile}"
echo $ssh_exec
if ! ssh ${remote_host} ${ssh_exec}; then
  echo "ERROR: ssh did not run correctly!"
  exit 1
fi

# Transfer resulting image back to local directory:
if [ $verb -eq 1 ] 
then
  echo "Transfering $outfile from ${remote_host} to here..."
fi
if ! scp ${remote_host}:${remote_psf_compute_location}/${outfile} .; then
  echo "ERROR: scp of $outfile did not run correctly!"
  exit 1
fi




