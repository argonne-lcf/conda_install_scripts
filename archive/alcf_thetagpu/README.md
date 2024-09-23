# How to deploy new a module on ThetaGPU
<!-- https://github.com/deephyper/deephyper/wiki/Todo-list:-deploy-new-module-with-latest-DeepHyper-version-on-ThetaGPU -->

Members of the `software` Unix group with the ALCF are able to build and deploy shared environments within the Lmod environment module system on ThetaGPU. 

The scripts used to build the Anaconda-based environments are version controlled here: https://github.com/felker/conda_install_scripts/tree/master/alcf_thetagpu

At the time of writing (July 2021), the two Bash scripts that are currently maintained are `install_dh_hvd_tf_torch.sh` and `install_dh_hvd_tf_torch_latest.sh`, and they build TensorFlow, PyTorch, Horovod, and DeepHyper for the target environment. The former script 

1. Login to ThetaGPU
2. `git clone git@github.com:felker/conda_install_scripts.git $HOME/conda_install_scripts`
3. Grab an interactive compute node allocation, preferably from the `full-node` queue. Installation takes about 1 hr, so to be safe: `qsub -t 360 -n 1 -q full-node -A datascience -I`
4. `cd /lus/theta-fs0/software/thetagpu/conda/`
5. `cp ~/conda_install_scripts/alcf_thetagpu/*.sh ./`
6. Assuming you intend to run `install_dh_hvd_tf_torch.sh`, edit this file. 
  - Update the first 25 or so lines of the file to have the desired Git tags in the variables `DH_REPO_TAG`, `TF_REPO_TAG`, etc. 
  - Update the value of `DH_INSTALL_SUBDIR='2021-06-26/'` to the desired name of the target folder, e.g. today's date. (*TODO: modify the scripts to automatically use the date, if that is the naming scheme we are keeping*). 
  - If the versions of CUDA, cuDNN, NCCL, TensorRT changed on the system, you should update those variables in the script (up to line 60 or so)
7. `./install_dh_hvd_tf_torch.sh > module_build_log.txt 2>&1 &` and then `watch tail module_build_log.txt`
8. When the installation completes successfully, move the build log out of the directory, preferably saving it for future reference.
9. `cd /lus/theta-fs0/software/environment/thetagpu/lmod/modulefiles/conda/`
10. `cp 2021-06-26.lua  YYYY-MM-DD.lua` or whatever name you gave the target directory in the earlier step
11. Edit `YYYY-MM-DD.lua`:
  - Change values of DeepHyper, PyTorch, TensorFlow, Horovod tags in the help message to match the variable values from the install script.
  - Edit `local conda_dir = "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3"` line to point to the target environment directory
  - Edit `prepend_path("LD_LIBRARY_PATH", ...` and similar `PATH` lines to reflect any changes to CUDA, cuDNN, NCCL, TensorRT dependency locations and versions. 
12. Test the environment. (*TODO*: make this automatic; integrate with CI). 
  - PyTorch single GPU
  - TensorFlow single GPU
  - PyTorch DDP
  - PyTorch Horovod
  - TensorFlow Horovod
  - DeepHyper with Ray
13. Upstream new `.lua` file and changes to `install_dh_hvd_tf_torch.sh` to repository via pull request (and consider adding log file) 
