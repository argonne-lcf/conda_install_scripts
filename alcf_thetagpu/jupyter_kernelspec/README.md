Contact Tommie Jackson to get any global kernels installed. E.g.

conda-2021-09-22
/soft/systems/anaconda202105/share/jupyter/kernels/conda-2021-09-22/kernelspec.json

The /soft/systems/anaconda202105/share/jupyter/kernels/ (deprecated kernels) ath is only available on ThetaGPU
compute nodes--- not on service nodes. (/lus/theta-fs0/software/systems/, cant ls, system
denied on thetagpusn1)

/lus/theta-fs0/software/thetagpu/systems/miniconda3/share/jupyter/kernels is available on
service nodes


- Find new conda-2021-11-30 kernelspec.json. FOUND
- "/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/bin/jupyter kernelspec list" 
shadowing the "/soft/systems/anaconda202105/bin/jupyter kernelspec list". Only the latter
knows about conda-2021-09-22, which even isnt the correct environment! (TJ: it may have
been left during the initial set up before moving the miniconda3). 

The actual jupyter binary is
/lus/theta-fs0/software/thetagpu/systems/miniconda3/bin/jupyter
and the kernels are located:
/lus/theta-fs0/software/thetagpu/systems/miniconda3/share/jupyter/kernels

Change kernelspec PATH modficiation to append to $PATH
instead of prepending to avoid this? (TJ:  I will correct the path during the next PM)


/lus/theta-fs0/software/thetagpu/systems/miniconda3/bin/jupyterhub-singleuser
(KGF: what is this used for?)
https://jupyterhub.readthedocs.io/en/stable/reference/config-user-env.html
> Since the jupyterhub-singleuser server extends the standard Jupyter notebook server, most configuration and documentation that applies to Jupyter Notebook applies to the single-user environments. Configuration of user environments typically does not occur through JupyterHub itself, but rather through system- wide configuration of Jupyter, which is inherited by jupyterhub-singleuser.



this is the script that kicks off a jupyter instances on the compute node:
/lus/theta-fs0/software/thetagpu/systems/jupyterhub/spawn_thetagpu_jupyter.sh

TJ: Please do not spread this to everyone but . . . if you want to import user specific
settings you can create the following file ${HOME}/.alcfjupyter.thetagpu to add variables
and overwrite how jupyter start.  I would advice not to at this time but I can see it
being needed in the future . . . I also use this for testing so I do not break jupyter for
other users.




- Watch that the OpenMPI 4.0.5 dependency of the base JupyterHub installation doesnt
  conflict wih the OpenMPI 4.1.1 and newer UCX that conda/2021-11-30 depends on.
     - JupyterHub might not actually depend on OpenMPI 4.0.5 but just inherits 
	 the default loaded module on the compute nodes. 
- Cannot currently launch multiple node jobs through JupyterHub. Users are requesting this
  all the time, but resisting adding support. Can still use mpi4py in sub-node
  parallelism, should check that this + other DL parallelism works in new 2021-11-30 kernel

!/lus/theta-fs0/software/thetagpu/systems/miniconda3/bin/jupyter kernelspec list

is the actual 


E.g. PATH for conda-2021-11-30:

/usr/local/cuda/bin:/opt/bin/:/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/bin:/lus/theta-fs0/software/thetagpu/cuda/nccl_2.11.4-1+cuda11.4_x86_64/include:/lus/theta-fs0/software/thetagpu/ucx/ucx-1.11.2_cuda-11.4_gcc-9.3.0/bin:/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0/bin/usr/local/cuda/bin:/soft/systems/anaconda202105/bin:/usr/local/cuda/bin:/opt/bin/:/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/ucx-1.9.0rc7/bin:/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/bin:/usr/local/cuda/bin:/opt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

- /usr/local/cuda/bin:/opt/bin/: prepended by Jupyter host launch (redundant with
  JupyterHub base PATH)
- NCCL/include: to OpenMPI 4.1.1/bin/usr/local/cuda/bin:, added by us in kernelspec.json
- /soft/systems/anaconda202105/bin to end, the JupyterHub PATH independent of the kernel is loading the old OpenMPI module/libraries 
