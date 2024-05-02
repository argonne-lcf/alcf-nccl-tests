#!/bin/sh
export SIZE=$PMI_SIZE
export LOCAL_RANK=$PMI_LOCAL_RANK
export RANK=$PMI_RANK
export WORLD_SIZE=$SIZE
if [ -z "${WORLD_SIZE}" ]; then
    export WORLD_SIZE=1
fi
if [ -z "${RANK}" ]; then
    export RANK=0
    export LOCAL_RANK=0
fi
echo "I am $RANK of $SIZE: $LOCAL_RANK on `hostname`"
$@
