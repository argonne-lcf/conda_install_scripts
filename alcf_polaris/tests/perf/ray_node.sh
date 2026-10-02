#!/bin/bash
# Started on each non-head node via mpiexec; blocks until ray stop / job end.
export VLLM_HOST_IP=$(hostname -i | awk '{print $1}')
exec ray start --address=$1 --num-gpus=4 --num-cpus=32 --temp-dir=/tmp/ray_$USER --block
