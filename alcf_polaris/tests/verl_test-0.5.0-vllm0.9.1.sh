#######!/bin/bash -l

# based on: https://verl.readthedocs.io/en/latest/start/quickstart.html

# Before running this script, execute this on a compute node 1x:
# > cd $HOME
# > git clone https://huggingface.co/datasets/openai/gsm8k
# > module use /soft/modulefiles; module load conda/2025-09-25; conda activate
# > python3 /soft/applications/conda/2025-09-25/verl/examples/data_preprocess/gsm8k.py --local_dir $HOME/huggingface/openai/gsm8k
# > hf auth login
#
# and input a token for reading Qwen/Qwen2.5-0.5B-Instruct model, to avoid:
#
# We had to rate limit your IP (140.221.69.69). To continue using our service, create a HF
#  account or login to your existing account, and make sure you pass a HF_TOKEN if you're using the API., retrying 1 of 2

# Run outside this script (whenever running the test in a new job):
# > module use /soft/modulefiles/ && module load conda/2025-09-26 && conda activate
# > ./verl_test-0.5.0-vllm0.9.1.sh

export RAY_TMPDIR="/tmp/raytmp"
mkdir -p $RAY_TMPDIR

rm -rfd $HOME/checkpoints/verl_examples/

# if not set in modulefile:
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/huggingface/openai/gsm8k/train.parquet \
 data.val_files=$HOME/huggingface/openai/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.ray_wait_register_center_timeout=300 \
 +ray_kwargs.ray_init.num_cpus=2 \
 +ray_kwargs.ray_init.num_gpus=1 \
 trainer.total_epochs=1 2>&1 | tee verl_test.log

ray stop
