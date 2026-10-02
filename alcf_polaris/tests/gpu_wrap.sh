#!/bin/bash
export CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID
exec "$@"
