# Module smoke test (2 nodes, 8 ranks)

Checks torch CUDA, NCCL all-reduce across nodes, CUDA-aware mpi4py (cupy), parallel h5py
MPI-IO (h5py imported before mpi4py on purpose: GTL regression check), TensorFlow, JAX,
flash-attn, TransformerEngine, DeepSpeed, vLLM, torch.compile, mpi4jax.

```
cd alcf_polaris/tests          # must be on a shared filesystem
sed -i 's|module load conda/.*;|module load conda/<date>;|' job.pbs
qsub job.pbs && tail -f job.out
```

`gpu_wrap.sh` sets `CUDA_VISIBLE_DEVICES` from `PALS_LOCAL_RANKID` before Python starts;
this must not be done inside Python (the GTL initializes CUDA during MPI_Init).
The torch.distributed MPI-backend check is expected to FAIL (no CUDA-aware support with Cray MPICH).
