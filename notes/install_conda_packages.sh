#!/bin/bash

PYTHON=python
CONDA=conda


PYTHON_PKG_LIST="sunpy
drms
sqlalchemy
setuptools
scipy
sys
signal
subprocess
argparse
kiwisolver
numpy
jdcal
cycler
pyhdf
netCDF4
bottleneck
matplotlib
python-dateutil
pyparsing
pytz
six
h5py"

PYTHON_PKG_TEST_LIST=(sunpy
drms
sqlalchemy
sqlalchemy-utils
setuptools
scipy
sys
signal
subprocess
argparse
kiwisolver
numpy
jdcal
cycler
pyhdf.SD
netCDF4
bottleneck
matplotlib
dateutil
pyparsing
pytz
six
h5py)

TSLOG=setup_python_CONDA.log

trap ctrl_c INT

function ctrl_c() {
  ${echo} "${cR}==> Caught CTRL-C, shutting down!${cX}"
  exit 1
}

cX="\033[0m"
cR="\033[1;31m"
cB="\033[1;34m"
cG="\033[32m"
cC="\033[1;96m"
cM="\033[35m"
cY="\033[1;93m"
Bl="\033[1;5;96m"
echo="echo -e"

${echo} "Checking software requirements..."
#Check that python is installed:
PTEST=$(which python)
if [ -z "${PTEST}" ]
then
  ${echo} "${cR}==> ERROR! ${PYTHON} does not seem to be installed!${cX}"
  ${echo} "${cR}    Please have your system administrator install it, or install it from http://www.python.org/${cX}"
  exit 1
fi
${echo} "${cG}==> ${PYTHON} is installed!${cX}"

PTEST=$(which ${CONDA})
if [ -z "${PTEST}" ]
then
  ${echo} "${cY}==> WARNING! ${CONDA} does not seem to be installed!${cX}"
  ${echo} "${cY}    Missing packages (if any) will not be auto-installed!${cX}"
  CONDAavail=0
else
  CONDAavail=1
  ${echo} "${cG}==> ${PYTHON} ${CONDA} is installed!${cX}"
fi

i=0
for pypkg in $PYTHON_PKG_LIST
do
  ${PYTHON} -c "import ${PYTHON_PKG_TEST_LIST[$i]}" 2>/dev/null
  pychk=$?
  if [ $pychk -eq 1 ]; then
    if [ $CONDAavail -eq 1 ]; then
      ${echo} "${cY}==>        package ${pypkg} not found, installing it locally...${cX}"
      ${CONDA} install -y ${pypkg} 1>>${TSLOG} 2>>${TSLOG}
      ${PYTHON} -c "import ${PYTHON_PKG_TEST_LIST[$i]}" 2>/dev/null
      pychk=$?
      if [ $pychk -eq 1 ]; then
        ${echo} "${cR}==> ERROR! Could not install package ${pypkg}.${cX}"
      else
        ${echo} "==>        package ${cG}${pypkg}${cX} found!"
      fi
    else
      ${echo} "${cR}==> ERROR! Missing required package ${pypkg}.  Please install it manually and try again.${cX}"
    fi
  else
    ${echo} "==>        package ${cG}${pypkg}${cX} found!"
  fi
  i=$(($i+1))
done

${echo} "${cG}==> Done!${cX}"









