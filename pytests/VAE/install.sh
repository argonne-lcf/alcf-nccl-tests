#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
module load conda
conda activate
pip3 install rdkit --prefix=$SCRIPT_DIR/local/
