#!/bin/bash
#------------------------------------------------------------------------------
# Script     : setup.sh.
# Description: Sets up the Coronal Hole Detection Python Package.
#
# *** This was copied from the setup script for the fr_designer (RBSL) webapp.
#     - Here the functionality is very simple, but we can add things back into
#       it later, including python venv or conda environment installation/setup.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

declare -ir STATUS_OK=0
declare -ir STATUS_ERROR=1

declare -ir NUM_REQUIRED_ARGS=1

function print_usage_msg {
  printf "
 Script     : setup.sh.
 Description: Sets up the Coronal Hole Detection Python Package.

 Usage: setup.sh <conf_file>

 conf_file: Configuration file (e.g. setup.conf).
 \n"
}

# Reads and parses the configuration file.
function process_configuration_file {
  declare conf_file="$1"

  while read field_name field_value; do
    [[ -z "${field_name}" || "${field_name:0:1}" = "#" ]] && continue

    field_name="${field_name:0:${#field_name}-1}"
    if [[ "${field_name}" = "IDL_DIR" ]]; then
      idl_dir="${field_value}"
    elif [[ "${field_name}" = "SSWIDL_DIR" ]]; then
      sswidl_dir="${field_value}"
    elif [[ "${field_name}" = "PS_EXT_DEPS_HOME" ]]; then
      ps_ext_deps_home="${field_value}"
    elif [[ "${field_name}" = "PS_TOOLS_HOME" ]]; then
      ps_tools_home="${field_value}"
    elif [[ "${field_name}" = "RAW_DATA_HOME" ]]; then
      raw_data_home="${field_value}"
    elif [[ "${field_name}" = "PROCESSED_DATA_HOME" ]]; then
      processed_data_home="${field_value}"
    elif [[ "${field_name}" = "MAP_FILE_HOME" ]]; then
      map_file_home="${field_value}"
    elif [[ "${field_name}" = "DATABASE_HOME" ]]; then
      database_home="${field_value}"
    elif [[ "${field_name}" = "DATABASE_FILENAME" ]]; then
      database_filename="${field_value}"
    elif [[ "${field_name}" = "TMP_HOME" ]]; then
      tmp_home="${field_value}"
    fi
  done < "${conf_file}"

  if [[ -z "${idl_dir}" ||\
        -z "${sswidl_dir}" ||\
        -z "${ps_ext_deps_home}" ||\
        -z "${ps_tools_home}" ||\
        -z "${raw_data_home}" ||\
        -z "${processed_data_home}" ||\
        -z "${map_file_home}" ||\
        -z "${database_home}" ||\
        -z "${database_filename}" ||\
        -z "${tmp_home}" ]]; then
    return ${STATUS_ERROR}
  fi

  return ${STATUS_OK}
}

# Sets up the CH Detection application settings.
function setup_application_directory {

  app_home=`pwd`

  sed -e "s|__IDL_DIR__|'${idl_dir}'|"\
      -e "s|__SSWIDL_DIR__|'${sswidl_dir}'|"\
      -e "s|__PS_EXT_DEPS_HOME__|'${ps_ext_deps_home}'|"\
      -e "s|__PS_TOOLS_HOME__|'${ps_tools_home}'|"\
      -e "s|__APP_HOME__|'${app_home}'|"\
      -e "s|__RAW_DATA_HOME__|'${raw_data_home}'|"\
      -e "s|__PROCESSED_DATA_HOME__|'${processed_data_home}'|"\
      -e "s|__MAP_FILE_HOME__|'${map_file_home}'|"\
      -e "s|__DATABASE_HOME__|'${database_home}'|"\
      -e "s|__DATABASE_FILENAME__|'${database_filename}'|"\
      -e "s|__TMP_HOME__|'${tmp_home}'|"\
      settings/app.py.template > settings/app.py
  [[ $? -ne ${STATUS_OK} ]] && return ${STATUS_ERROR}

  return ${STATUS_OK}
}

# Sets up the database/data directories.
function setup_database_directories {
  folders=(${raw_data_home} ${processed_data_home} ${database_home} ${tmp_home})
  for dir in ${folders[@]}; do
    if [ ! -d $dir ]; then
      printf "### Creating Folder: ${dir}\n"
      mkdir $dir
      [[ $? -ne ${STATUS_OK} ]] && return ${STATUS_ERROR}
    fi
  done

  return ${STATUS_OK}
}

#------------------------------------------------------------------------------

if [[ "$1" = "-h" || "$1" = "-help" || "$1" = "--help" ]]; then
  print_usage_msg
  exit ${STATUS_OK}
fi

if [[ $# -ne ${NUM_REQUIRED_ARGS} ]]; then
  print_usage_msg
  exit ${STATUS_ERROR}
fi

declare configuration_file="$1"

if [[ ! -f "${configuration_file}" || ! -r "${configuration_file}" ]]; then
  printf "ERROR: Configuration file not found: ${configuration_file}.\n" >&2
  exit ${STATUS_ERROR}
fi

process_configuration_file "${configuration_file}"
if [[ $? -ne ${STATUS_OK} ]]; then
  printf "ERROR: Failed processing configuration file.\n" >&2
  exit ${STATUS_ERROR}
fi

setup_application_directory
if [[ $? -ne ${STATUS_OK} ]]; then
  printf "ERROR: Failed setting up Python application directory.\n" >&2
  exit ${STATUS_ERROR}
fi

setup_database_directories
if [[ $? -ne ${STATUS_OK} ]]; then
  printf "ERROR: Failed setting up the database directories.\n" >&2
  exit ${STATUS_ERROR}
fi

printf "\n"
printf "Coronal Hole Detection project setup was successfully completed!\n"
printf "\n"

exit ${STATUS_OK}
