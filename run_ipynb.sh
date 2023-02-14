#/bin/bash

: ${root_path:=`pwd`}
: ${local_port:=10001}

export PYTHONPATH="${root_path}"
jupyter notebook --no-browser --port=${local_port} --ip=0.0.0.0 # --NotebookApp.token=''
# jupyter lab --no-browser --port=${local_port} --ip=0.0.0.0 --NotebookApp.token=''
