#!/bin/bash -l

# Builds the ALCF Polaris conda module (TensorFlow, PyTorch, JAX, vLLM, DeepSpeed, ...),
# mostly from source. Run it as a PBS job on a Sirius compute node (the result is synced
# verbatim to Polaris):
#   bash -l ./build_monolithic_conda_module.sh /soft/applications/conda/<date>
# Re-running on an existing path resumes (see helpers below).
#
# RESUMABLE VARIANT of build_monolithic_conda_module.sh (kept in sync by hand; diff the two).
# Same packages and versions; two differences, both aimed at re-running this script on a
# half-finished BASE_PATH instead of deleting it and starting over:
#  1. No environment leaks out of skippable blocks. The llvm module (TF), the
#     cray-hdf5-parallel module (h5py) and DeepSpeed's CFLAGS/LDFLAGS are scoped to the
#     step that needs them, so later builds see the same environment whether or not an
#     earlier block ran in this invocation.
#  2. Resume guards are version-aware: a wheel/package is only reused if it matches the
#     pinned tag, so bumping a tag and re-running rebuilds that component.

# KGF: check HARDCODE points for lines that potentially require manual edits to pinned package versions
BASE_PATH=$1
DATE_PATH="$(basename $BASE_PATH)"

export PYTHONNOUSERSITE=1
umask 0022
# Fail fast from the start (was only enabled after the conda setup, so a failed module
# load / Miniforge install / Python pin before that point was not fatal).
set -e

# ---------------------------------------------------------------------------
# Resume + network-resilience helpers (added 2026-09-17 after two attempts died on
# transient proxy failures: bazel "Premature EOF" on llvm-raw, then git clone h5py
# "SSL_read: unexpected eof"). Re-running this script on an existing BASE_PATH now
# skips the TF and PyTorch builds when their wheels are already in $WHEELS_PATH,
# skips the Miniforge/Bazel installers when present, and every git clone is
# retried (after removing any half-cloned dir). Everything else is idempotent
# enough to just re-run.
# ---------------------------------------------------------------------------
retry() {   # retry <n> <cmd...>
    local n=$1; shift
    local i
    for ((i=1; i<=n; i++)); do
        "$@" && return 0
        echo "retry: attempt $i/$n failed: $*"
        [ $i -lt $n ] && sleep 30
    done
    echo "ERROR: giving up after $n attempts: $*"
    return 1
}
gclone() {  # gclone <url> <dir> [git clone flags...]; always starts from a clean dir
    local url=$1 dir=$2; shift 2
    rm -rf "$dir"
    retry 5 git clone "$@" "$url" "$dir"
}
# have_pkg <dist-name>: true if pip metadata for it exists in the env (resume skips for
# the slow from-source builds below; each redo cost ~1.5 h per attempt on 2026-09-17).
have_pkg() { python -c "import importlib.metadata as m; m.version('$1')" >/dev/null 2>&1; }
# have_pkg_ver <dist-name> <version>: exact-version variant, for pinned from-source builds.
have_pkg_ver() { python -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('$1') == '$2' else 1)" 2>/dev/null; }
# have_pkg_prefix <dist-name> <version-prefix>: for builds that append local tags or pad the
# version (DeepSpeed "0.19.7+<sha>", TransformerEngine "2.19.0+<sha>" from tag v2.19).
have_pkg_prefix() { python -c "import importlib.metadata as m, sys; v=m.version('$1'); p='$2'; sys.exit(0 if v == p or v.startswith(p + '+') or v.startswith(p + '.') else 1)" 2>/dev/null; }

# pip_sdist_cxx20 <name> <version> [extras]: torch 2.14's cpp_extension compiles every
# extension as C++20 and its headers require it (c10/util/intrusive_ptr.h uses
# std::strong_ordering), but some sdists still hard-code "-std=c++17" in their nvcc
# flags, which wins and fails with "namespace std has no member strong_ordering"
# (attempt 8, 2026-09-17, flash-attn). Download the sdist, rewrite the flag, install
# from the patched tree with the caller's CC/CXX/MAX_JOBS environment. --no-deps (shared-env
# rule): callers install the package's runtime deps explicitly.
pip_sdist_cxx20() {
    local name=$1 ver=$2 extras=${3:-}
    local mod=${name//-/_}
    if python -c "import importlib.metadata as m, sys; sys.exit(0 if m.version('$name') == '$ver' else 1)" 2>/dev/null; then
        echo "RESUME: $name $ver already installed; skipping sdist rebuild"
        return 0
    fi
    local work=$DOWNLOAD_PATH/sdist-$name-$ver
    rm -rf "$work" && mkdir -p "$work" && pushd "$work" >/dev/null
    retry 3 pip download --no-deps --no-binary=:all: --no-build-isolation -d . "${name}==${ver}"
    tar -xzf ./*.tar.gz
    local src; src=$(find . -mindepth 1 -maxdepth 1 -type d | head -1)
    grep -rl -- '-std=c++17' "$src"/setup.py "$src"/*.py 2>/dev/null | xargs -r sed -i 's/-std=c++17/-std=c++20/g'
    grep -c -- '-std=c++20' "$src"/setup.py || true
    pip install --no-build-isolation --no-deps "${src}${extras}"
    popd >/dev/null
}

# move primary conda packages directory/cache away from ~/.conda/pkgs (4.2 GB currently)
# hardlinks should be preserved even if these files are moved (not across filesystem boundaries)
export CONDA_PKGS_DIRS=/soft/applications/conda/pkgs

#########################################################
# Check for outside communication
# (be sure not to inherit these vars from dotfiles)
#########################################################
unset https_proxy
unset http_proxy

if wget -q --spider -T 10 http://google.com; then
    echo "Network Online"
else
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128
fi

module list
# PE 26.03 (since the 2026-08-19 HPCM upgrade): PrgEnv-gnu + gcc-native/14 (= 14.3.0,
# same as /usr/bin/gcc-14). gcc-native/14.2 no longer exists.
module load PrgEnv-gnu
module load gcc-native/14
module load craype-x86-milan
# darshan: `cc` would otherwise link libdarshan into every extension and users get
# darshan_library_warning on every Python exit. xalt: wraps the linker.
module unload darshan
module unload xalt
# craype-accel-nvidia80 refuses to load under PrgEnv-gnu (HPE Case 5367752190).
# Export its effect by hand so every `cc`-linked extension (mpi4py, h5py, torch,
# mpi4jax) links libmpi_gtl_cuda. A GTL-less lib imported first makes MPI_Init abort
# with "GTL library is not linked" because the modulefile sets MPICH_GPU_SUPPORT_ENABLED=1.
export CRAY_ACCEL_TARGET="nvidia80"
export CRAY_ACCEL_VENDOR="nvidia"
export CRAY_CPU_TARGET="x86-64"
export CRAYPE_LINK_TYPE="dynamic"
export CRAY_TCMALLOC_MEMFS_FORCE="1"
export MPICH_GPU_SUPPORT_ENABLED=1
module list
echo $MPICH_DIR

# -------------------- begin HARDCODE of major built-from-source frameworks etc.
# unset *_TAG variables to build latest master/main branch (or "develop" in the case of DeepHyper)
# KGF (2026-09-17): bumped for the Sirius/Polaris CUDA 13 rebuild. Verify each tag
# against upstream "Latest" before building.
#DH_REPO_TAG="0.13.2"
DH_REPO_URL=https://github.com/deephyper/deephyper.git

# Versions verified against upstream "Latest" release pages on 2026-09-16.
TF_REPO_TAG="v2.21.0"   # 2026-03-06 (still latest)
PT_REPO_TAG="v2.14.0"   # 2026-09-02; pairs with torchvision 0.29.0, triton 3.8.0
# Horovod dropped: 0.28.1 incompatible with PyTorch >=2.1 (C++17), upstream dormant.
TF_REPO_URL=https://github.com/tensorflow/tensorflow.git
PT_REPO_URL=https://github.com/pytorch/pytorch.git

############################
# Manual version checks/changes below that must be made compatible with TF/Torch/CUDA versions above:
# - pytorch vision
# - magma-cuda
# - torch-geometric, pyg-lib
# - cupy
# - jax
###########################

#################################################
# CUDA path and version information
#################################################

# KGF (2026-09-16): CUDA 13.0.x target. Sirius/Polaris driver is 580.65.06 (CUDA 13.0
# driver API). Newer 13.x toolkits are installed under /soft/compilers/cudatoolkit/
# (13.1.2, 13.2.2, 13.3.1, 13.4.2) and would run under minor-version compatibility, but PTX
# emitted by a newer NVRTC/ptxas (cupy, JAX/XLA autotuning, triton fallbacks) cannot be
# JIT-loaded by a 13.0 driver, so stay on 13.0.x until the driver is bumped.
# The PE 26.03 GTL (/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0) links libcudart.so.13, so
# CUDA 13 also removes the need for the pe-26.03-shim dirs the 2025-09 modules carry.
# All CUDA-major references below (cuda-bindings, nvshmem4py-cu13, PyG cu130 wheel URL,
# cupy-cuda13x, magma-cuda130) are coupled to CUDA_VERSION_MAJOR.

CUDA_VERSION_MAJOR=13
CUDA_VERSION_MINOR=0
CUDA_VERSION_MINI=3

CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI

CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda-${CUDA_VERSION_FULL}
CUDA_HOME=${CUDA_TOOLKIT_BASE}

CUDA_DEPS_BASE=/soft/libraries/

# cuDNN 9.26.0.51 (2026-09-09). Download:
#   https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.26.0.51_cuda13-archive.tar.xz
# and rename the extracted dir to the schema below.
CUDNN_VERSION_MAJOR=9
CUDNN_VERSION_MINOR=26.0
CUDNN_VERSION_EXTRA=51
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA

# HARDCODE: manually renaming default cuDNN tarball name to fit this schema:
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-cuda$CUDA_VERSION_MAJOR-linux-x64-v$CUDNN_VERSION

# NCCL 2.30.7-1 is what the PyTorch 2.14 cu13 wheels bundle. NVIDIA only publishes the
# CUDA 13 tarball built against 13.3 (there is no +cuda13.0 flavor), so the tag below is
# decoupled from CUDA_VERSION. It runs on the 580 driver via minor-version compatibility
# (same binary the torch cu130 wheels ship). Download:
#   https://developer.download.nvidia.com/compute/redist/nccl/v2.30.7/nccl_2.30.7-1+cuda13.3_x86_64.txz
# (2.31.2 exists too, as nccl-nccl-stable-cuda-13-linux-x86_64-2.31.2-cuda13.3.tar.gz.)
NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=30.7-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_CUDA_TAG=13.3
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${NCCL_CUDA_TAG}_x86_64

# TensorRT 10.16.1.11. The ONLY consumer in this env is onnxruntime-gpu's bundled
# TensorRT execution provider (libonnxruntime_providers_tensorrt.so dlopens
# libnvinfer.so.10 from LD_LIBRARY_PATH; the modulefile adds $TENSORRT_BASE/lib).
# TF dropped TF-TRT in 2.18 and PyTorch never had a TensorRT build option, so the
# TF_TENSORRT_* / USE_TENSORRT exports that used to live here were removed (2026-09-17).
# Only cuda-12.9 and cuda-13.2 tarballs exist for 10.16.1 (13.0/13.1/13.3 are 404).
# Download from the developer portal (login):
#   TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-13.2.tar.gz
# and keep the tarball's suffix on the extracted dir name, as for the other /soft/libraries/trt entries.
TENSORRT_VERSION_MAJOR=10
TENSORRT_VERSION_MINOR=16.1.11
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
# HARDCODE
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-13.2
echo "TENSORRT_BASE=${TENSORRT_BASE}"

for d in "$CUDA_TOOLKIT_BASE" "$CUDNN_BASE" "$NCCL_BASE"; do
    [ -d "$d" ] || { echo "ERROR: missing $d (download it to /soft/libraries first)"; exit 1; }
done
# TensorRT is runtime-only (onnxruntime's TRT provider dlopens it); nothing below
# compiles against it, so a missing tree just costs that provider until it lands.
[ -d "$TENSORRT_BASE" ] || echo "WARNING: $TENSORRT_BASE missing; onnxruntime TensorrtExecutionProvider will be unavailable until it is installed"

#################################################
# TensorFlow Config flags (for ./configure run)
#################################################
export TF_CUDA_COMPUTE_CAPABILITIES=8.0
# Note that TF_CUDA_VERSION and TF_CUDNN_VERSION should consist of major and minor versions only (e.g. 12.3 for CUDA and 9.1 for CUDNN).
# https://openxla.org/xla/hermetic_cuda
export TF_CUDA_VERSION=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}
export TF_CUDNN_VERSION=${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}
export TF_NCCL_VERSION=${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}
# KGF: double check above changes to syntax
export CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_BASE
export CUDNN_INSTALL_PATH=$CUDNN_BASE
export NCCL_INSTALL_PATH=$NCCL_BASE
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_CUDA_CLANG=0   # KGF?
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
# TF-TRT was removed upstream in TF 2.18 ("TensorRT support is disabled in CUDA builds");
# the old TF_NEED_TENSORRT / TF_TENSORRT_VERSION / TENSORRT_INSTALL_PATH exports are gone.
# TENSORRT_BASE deliberately omitted from TF_CUDA_PATHS: including it let 12.9.0 leak in
# when 12.9.1 was specified in the pre-hermetic Polaris builds.
export TF_CUDA_PATHS=$CUDA_TOOLKIT_BASE,$CUDNN_BASE,$NCCL_BASE
#export GCC_HOST_COMPILER_PATH=$(which gcc)

# TF_PYTHON_VERSION is derived from the Miniforge base python below (after install).
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc-14   # 14.3.0, same as gcc-native/14
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

#################################################
## Installing Miniforge
#################################################

# set Conda installation folder and where downloaded content will stay
CONDA_PREFIX_PATH=$BASE_PATH/mconda3
DOWNLOAD_PATH=$BASE_PATH/DOWNLOADS
WHEELS_PATH=$BASE_PATH/wheels

# Resume on a finished build: the script ends with `chmod -R a-w $BASE_PATH`, so make
# it writable again before anything tries to pip install into it.
if [ -d "$BASE_PATH/mconda3" ] && [ ! -w "$BASE_PATH/mconda3" ]; then
    echo "RESUME: $BASE_PATH is read-only from a previous completed run; chmod -R u+w"
    chmod -R u+w "$BASE_PATH"
fi
mkdir -p $CONDA_PREFIX_PATH
mkdir -p $DOWNLOAD_PATH
mkdir -p $WHEELS_PATH
cd $BASE_PATH
echo "Downloading miniforge installer"
CONDA_DOWNLOAD_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
CONDA_INSTALL_SH="Miniforge3-$(uname)-$(uname -m).sh"
if [ -x "$CONDA_PREFIX_PATH/bin/conda" ]; then
    echo "RESUME: Miniforge already installed at $CONDA_PREFIX_PATH; skipping installer"
else
    retry 3 wget -c $CONDA_DOWNLOAD_URL/$CONDA_INSTALL_SH -P $DOWNLOAD_PATH
    chmod +x $DOWNLOAD_PATH/$CONDA_INSTALL_SH

    echo "Installing Miniforge"
    echo "bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u"
    bash $DOWNLOAD_PATH/$CONDA_INSTALL_SH -b -p $CONDA_PREFIX_PATH -u
fi

cd $CONDA_PREFIX_PATH

#########
# create a setup file
cat > setup.sh << EOF
preferred_shell=\$(basename \$SHELL)

module load PrgEnv-gnu

if [ -n "\$ZSH_EVAL_CONTEXT" ]; then
    DIR=\$( cd "\$( dirname "\$0" )" && pwd )
else  # bash, sh, etc.
    DIR=\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )
fi

eval "\$(\$DIR/bin/conda shell.\${preferred_shell} hook)"

# test network
unset https_proxy
unset http_proxy
if wget -q --spider -T 10 http://google.com; then
    echo "Network Online"
else
   echo "Network Offline, setting proxy envs"
   export https_proxy=http://proxy.alcf.anl.gov:3128
   export http_proxy=http://proxy.alcf.anl.gov:3128
fi

export CUDA_TOOLKIT_BASE=$CUDA_TOOLKIT_BASE
export CUDNN_BASE=$CUDNN_BASE
export NCCL_BASE=$NCCL_BASE
export TENSORRT_BASE=$TENSORRT_BASE
export LD_LIBRARY_PATH=\$CUDA_TOOLKIT_BASE/lib64:\$CUDNN_BASE/lib:\$NCCL_BASE/lib:\$TENSORRT_BASE/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export PATH=\$CUDA_TOOLKIT_BASE/bin:\$PATH
EOF

# HARDCODE: TF 2.21.0's hermetic build only ships requirements_lock files for Python
# 3.10-3.13 (bazel fails at python_version_repo otherwise). Miniforge 26.7.2-0 ships
# Python 3.14.7, which is what killed the first 2026-09-17 attempt. Pin base to 3.13 before
# anything else is installed; conda 26.x itself runs fine on 3.13.
TARGET_PYTHON_VER=3.13
./bin/conda install -y -n base --override-channels -c conda-forge "python=${TARGET_PYTHON_VER}"
PYTHON_VER=$(./bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo PYTHON_VER=$PYTHON_VER
if [[ "$PYTHON_VER" != "$TARGET_PYTHON_VER" ]]; then
    echo "ERROR: base python is $PYTHON_VER after pinning to $TARGET_PYTHON_VER"; exit 1
fi
export TF_PYTHON_VERSION=$PYTHON_VER

cat > .condarc << EOF
channels:
   - conda-forge
env_prompt: "(${DATE_PATH}/{default_env}) "
pkgs_dirs:
   - ${CONDA_PKGS_DIRS}
   - \$HOME/.conda/pkgs
EOF

# move to base install directory
cd $BASE_PATH
echo "cd $BASE_PATH"

# setup conda environment
source $CONDA_PREFIX_PATH/setup.sh
# Miniforge's setup.sh only defines the `conda` shell function; it does *not*
# `conda activate base`. Without that, $CONDA_PREFIX stays empty, and later
# steps that expand ${CONDA_PREFIX}/... (magma extract, CMAKE_PREFIX_PATH,
# etc.) silently become /... and fail.
conda activate base
echo "after sourcing conda"
module unload xalt

echo "CONDA BINARY: $(which conda)"
echo "CONDA VERSION: $(conda --version)"
echo "PYTHON VERSION: $(python --version)"

################################################
### Install TensorFlow
################################################

echo "Conda install some dependencies"

# --override-channels: ignore any `defaults` channel leaking in from ~/.condarc.
# That channel still serves an ancient graphviz=2.38.0 that blocks the solve on
# modern Python.
# pymongo dropped from conda spec: conda-forge has not yet published a py313
# build, which fails the solve under the python=3.13 pin. It is pip-installed
# later in the script.
# NOTE: do NOT add `rust` (conda-forge) to this install. It pulls in
# rust_linux-64 -> gcc_linux-64 -> gcc_impl_linux-64 / binutils_impl_linux-64
# / sysroot_linux-64 / kernel-headers_linux-64, which makes the conda
# _compiler_compat/ld a live symlink to x86_64-conda-linux-gnu-ld. That linker
# is sysroot-locked to conda's CDT and won't search /usr/lib64 or the Cray PE
# search paths, so subsequent mpi4py / h5py / any-MPI-linking build fails to
# resolve transitive NEEDED libs from libmpi.so (link warns "not found",
# undefined refs follow). libprotobuf is innocent (only pulls libstdcxx-ng
# runtime) and isn't needed once we drop SGLang. If a future package needs
# rustc, install it via rustup outside the conda env, not via conda-forge `rust`.
conda install -y --override-channels -c conda-forge zip unzip astunparse setuptools six requests graphviz numba numpy pip libaio
conda install -y --override-channels -c conda-forge mkl mkl-include git-lfs  # onednn mkl-dnn  ### on ThetaGPU

# cmake comes from pip only (not conda): with both installed, a pip `cmake` wheel pulled in
# as a dependency (deepspeed-kernels in attempt 11, vLLM's build requirements in attempt 21)
# replaced conda's bin/cmake with a Python launcher that broke once conda touched the env
# again. With pip as the sole owner, later `cmake>=...` requirements are already satisfied.
# <4: PyTorch 2.14 vendors subprojects (NNPACK, FP16, psimd, protobuf's googletest, ...)
# with cmake_minimum_required < 3.5, which CMake 4 rejects. 2.14 wraps some of them in
# CMAKE_POLICY_VERSION_MINIMUM=3.5 but its own CI still builds with cmake 3.31.6
# (.ci/docker/common/install_conda.sh); lift this when torch CI moves to CMake 4.
pip install "cmake<4"
cmake --version | head -1

# MAGMA (CUDA LAPACK): the magma-cuda{NN} conda package is no longer published
# on any channel as of late 2024 (anaconda.org returns empty for conda-forge /
# pytorch / pytorch-nightly). PyTorch's own CI script switched to extracting the
# prebuilt static tarball from S3 directly.
# https://github.com/pytorch/pytorch/issues/138506
MAGMA_CUDA_TAG="cuda${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}"   # e.g. cuda130 (verified on S3 2026-09-16)
MAGMA_TARBALL="magma-${MAGMA_CUDA_TAG}-2.6.1-1.tar.bz2"
(
    set -x
    cd $DOWNLOAD_PATH
    curl -fOLs "https://ossci-linux.s3.us-east-1.amazonaws.com/${MAGMA_TARBALL}"
    mkdir -p magma_extract && cd magma_extract
    tar -xf ../${MAGMA_TARBALL}
    cp -r include/* ${CONDA_PREFIX}/include/
    cp -r lib/*     ${CONDA_PREFIX}/lib/
)

conda install -y --override-channels -c conda-forge mamba ccache

# Only reuse a wheel built from the pinned tag (any wheel when building master).
TF_WHEEL_GLOB="tensorflow-${TF_REPO_TAG#v}-*.whl"; [ -n "$TF_REPO_TAG" ] || TF_WHEEL_GLOB="tensorflow-*.whl"
TF_WHEEL_EXISTING=$(find $WHEELS_PATH/ -name "$TF_WHEEL_GLOB" -type f | head -1)
if [ -n "$TF_WHEEL_EXISTING" ]; then
    echo "RESUME: found $TF_WHEEL_EXISTING; skipping TensorFlow clone + bazel build"
else
echo "Clone TensorFlow"
cd $BASE_PATH
gclone $TF_REPO_URL tensorflow
cd tensorflow

if [[ -z "$TF_REPO_TAG" ]]; then
    echo "Checkout TensorFlow master"
else
    echo "Checkout TensorFlow tag $TF_REPO_TAG"
    git checkout --recurse-submodules $TF_REPO_TAG
fi
BAZEL_VERSION=$(head -n1 .bazelversion)
echo "Found TensorFlow depends on Bazel version $BAZEL_VERSION"

cd $BASE_PATH
echo "Download Bazel binaries"
BAZEL_DOWNLOAD_URL=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION
BAZEL_INSTALL_SH=bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
BAZEL_INSTALL_PATH=$BASE_PATH/bazel-$BAZEL_VERSION
if [ -x "$BAZEL_INSTALL_PATH/bin/bazel" ]; then
    echo "RESUME: bazel already installed at $BAZEL_INSTALL_PATH"
else
    echo "wget $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH"
    retry 3 wget -c $BAZEL_DOWNLOAD_URL/$BAZEL_INSTALL_SH -P $DOWNLOAD_PATH
    chmod +x $DOWNLOAD_PATH/$BAZEL_INSTALL_SH
    echo "Install Bazel in $BAZEL_INSTALL_PATH"
    bash $DOWNLOAD_PATH/$BAZEL_INSTALL_SH --prefix=$BAZEL_INSTALL_PATH
fi
export PATH=$PATH:/$BAZEL_INSTALL_PATH/bin

cd $BASE_PATH

echo "Install TensorFlow Dependencies"
# numpy/numba come from conda above; a pip -U here installed PyPI copies over them whenever
# PyPI was ahead, leaving duplicate conda+pip records.
pip install -U ninja
pip install -U pip wheel gast portpicker pydot packaging pyyaml

echo "Configure TensorFlow"
cd tensorflow
export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
# Build scratch: /tmp on Sirius login nodes is a 1.5 GB tmpfs (full as of 2026-09-17) and
# compute-node /tmp is RAM-backed; pip's per-package build dirs for vLLM / flash-attn / TE
# run to tens of GB. Keep it under DOWNLOAD_PATH (removed at the end). Override with
# TMPDIR=/path/to/nvme if the node has local scratch.
export TMPDIR="${TMPDIR:-$DOWNLOAD_PATH/tmp}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

# HARDCODE: TF 2.21.0 hermetic build wants Clang as host compiler. The Sophia
# 2026-06 build used clang 22.1.0 successfully; Sirius stages 22.1.4 via
# /soft/modulefiles/llvm/release-22.1.4 (18.1.6 is also still there as a fallback).
module use /soft/modulefiles
module load llvm/release-22.1.4
export GCC_HOST_COMPILER_PATH=$(which gcc-14)
export CC=/soft/compilers/llvm/release-22.1.4/bin/clang
export BAZEL_COMPILER=$CC

# Hermetic CUDA build (XLA): no longer uses TF_CUDA_PATHS at compile time; bazel
# downloads its own CUDA/cuDNN/NCCL/NVSHMEM redistribs selected by the
# HERMETIC_*_VERSION env vars. https://openxla.org/xla/hermetic_cuda
#
# These must be exported BEFORE ./configure so it doesn't prompt for them interactively.
# They must be in TF 2.21.0's internal allowlist, which lags NVIDIA: the combo below is
# the one the Sophia 2026-06 build compiled with (13.0.2 / 9.16.0 / 2.29.2 / 3.4.5).
# It is decoupled from the local /soft versions on purpose; at runtime the wheel
# (override_include_cuda_libs=false, no bundled libs) dlopens our newer local
# cudnn 9.26 / nccl 2.30.7 from LD_LIBRARY_PATH, which is forward compatible.
# If bazel rejects a version it prints "supported CUDA versions are [...]".
export HERMETIC_CUDA_VERSION=13.0.2
export HERMETIC_CUDNN_VERSION=9.16.0
export HERMETIC_NCCL_VERSION=2.29.2
export HERMETIC_NVSHMEM_VERSION=3.4.5
export HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_80"

yes "" | ./configure

echo "Bazel Build TensorFlow"

# Retry loop: the hermetic build fetches ~GBs of external repos (llvm-project alone is
# ~200 MB) through the ALCF proxy, and the 2026-09-17 attempt 2 died on a "Premature EOF"
# on llvm-raw. Bazel keeps already-fetched repos and build outputs between invocations,
# so a re-run only redoes the failed fetch. --experimental_repository_downloader_retries
# adds in-process retries per download on top.
# NOTE: --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 is a holdover from when PyTorch wheels used the
# pre-C++11 std::string ABI. Upstream TF (tf.sysconfig docstring) and our torch 2.14 build
# (196 __cxx11 symbols in libc10.so; libtensorflow_framework.so.2 has 0) both use ABI=1.
# TF and torch never exchange C++ objects, so the mismatch is harmless in-process; it only
# matters to users compiling TF custom ops, who must use tf.sysconfig.get_compile_flags()
# (reports ABI=0 here). Consider dropping it on the next TF rebuild to match upstream.
for attempt in 1 2 3; do
    echo "bazel build attempt $attempt"
    HOME=$DOWNLOAD_PATH bazel build --announce_rc --jobs=128 --loading_phase_threads=6 \
        --verbose_failures --config=cuda --config=cuda_wheel \
        --experimental_repository_downloader_retries=5 \
        --@local_config_cuda//cuda:override_include_cuda_libs=false \
        --repo_env=HERMETIC_CUDA_VERSION=${HERMETIC_CUDA_VERSION} \
        --repo_env=HERMETIC_CUDNN_VERSION=${HERMETIC_CUDNN_VERSION} \
        --repo_env=HERMETIC_NCCL_VERSION=${HERMETIC_NCCL_VERSION} \
        --repo_env=HERMETIC_NVSHMEM_VERSION=${HERMETIC_NVSHMEM_VERSION} \
        --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=${HERMETIC_CUDA_COMPUTE_CAPABILITIES} \
        --copt="-Wno-error=unused-command-line-argument" --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        //tensorflow/tools/pip_package:wheel && break
    [ $attempt -eq 3 ] && { echo "ERROR: bazel build failed 3 times"; exit 1; }
    sleep 30
done

echo "Run wheel building"
cp ./bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl $WHEELS_PATH
unset CC
# Don't leak clang / LLVM 22 runtime libs (LD_LIBRARY_PATH) into the rest of the build.
module unload llvm/release-22.1.4
fi   # end TF build (skipped on resume)
# Leave the TF source tree: on a fresh build we are still in $BASE_PATH/tensorflow, and the
# `import tensorflow.python` below would resolve to the source checkout (2026-10-01 attempt 1:
# "Do not import tensorflow from its source directory").
cd $BASE_PATH
echo "Install TensorFlow"
TF_WHEEL=$(find $WHEELS_PATH/ -name "$TF_WHEEL_GLOB" -type f | head -1)
pip install "$TF_WHEEL"
# TF's pywrap_tensorflow.py imports pywrap_dlopen_global_flags and, if present, loads
# _pywrap_tensorflow_internal with RTLD_GLOBAL, exposing the ~12k LLVM symbols in
# libtensorflow_framework.so.2 to the whole process. triton 3.8's libtriton.so no longer
# hides its own LLVM (3.4 did), so `import tensorflow; import triton` (or flash-attn /
# torch.compile after TF) binds triton's static initializers to TF's LLVM and segfaults
# in llvm::DebugCounter (harness 2026-09-17, core.3460843). TF's own docstring says the
# hook is only meant for static builds; ours is dynamic, so drop it -> RTLD_LOCAL.
# Verified on Sirius: tf->triton, tf->jax->flash-attn OK after this; JAX never clashed.
TF_PY_DIR=$(python -c 'import tensorflow.python as p, os; print(os.path.dirname(p.__file__))')
rm -f "$TF_PY_DIR/pywrap_dlopen_global_flags.py" "$TF_PY_DIR"/__pycache__/pywrap_dlopen_global_flags*.pyc

#################################################
### Install PyTorch
#################################################

cd $BASE_PATH
PT_WHEEL_GLOB="torch-${PT_REPO_TAG#v}-*.whl"; [ -n "$PT_REPO_TAG" ] || PT_WHEEL_GLOB="torch-*.whl"
PT_WHEEL_EXISTING=$(find $WHEELS_PATH/ -name "$PT_WHEEL_GLOB" -type f | head -1)
if [ -n "$PT_WHEEL_EXISTING" ]; then
    echo "RESUME: found $PT_WHEEL_EXISTING; skipping PyTorch clone + build"
    PT_WHEEL=$PT_WHEEL_EXISTING
else
echo "Clone PyTorch"

gclone $PT_REPO_URL pytorch
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo "Checkout PyTorch master"
else
    echo "Checkout PyTorch tag $PT_REPO_TAG"
    git checkout $PT_REPO_TAG
    echo "git submodule sync"
    git submodule sync
    echo "git submodule update"
    retry 5 git submodule update --init --recursive
fi
fi   # end PyTorch clone (skipped on resume); the exports below still run, later stages use them
# HARDCODE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include
export CPATH="${CPATH:+$CPATH:}$CUDNN_INCLUDE_DIR"

echo "Install PyTorch"
# PrgEnv-gnu + gcc-native/14 + the CRAY_ACCEL_* exports were set at the top of the script.
module unload darshan xalt
module list
echo "CRAY_ACCEL_TARGET= $CRAY_ACCEL_TARGET"
echo "CRAYPE_LINK_TYPE = $CRAYPE_LINK_TYPE"

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.0"
echo "CUDNN_ROOT=$CUDNN_BASE"
export CUDNN_ROOT_DIR=$CUDNN_BASE
export CUDNN_INCLUDE_DIR=$CUDNN_BASE/include

export USE_SYSTEM_NCCL=1
export NCCL_ROOT=$NCCL_BASE
export NCCL_INCLUDE_DIR=$NCCL_BASE/include
export NCCL_LIB_DIR=$NCCL_BASE/lib

# (USE_TENSORRT / TENSORRT_ROOT / TENSORRT_INCLUDE_DIR / TENSORRT_LIBRARY were Caffe2-era
# knobs; PyTorch 2.14's CMakeLists has no TENSORRT option at all, so they were dropped.)
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export PYTORCH_BUILD_VERSION="${PT_REPO_TAG:1}"
export PYTORCH_BUILD_NUMBER=1

export USE_CUSPARSELT=1
# HARDCODE: cuSPARSELt 0.9.1.1 (2026-04-29), cuda13 archive. Download:
#   https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.9.1.1_cuda13-archive.tar.xz
export CUSPARSELT_ROOT="/soft/libraries/cusparselt/libcusparse_lt-linux-x86_64-0.9.1.1_cuda${CUDA_VERSION_MAJOR}-archive/"
export CUSPARSELT_INCLUDE_PATH="${CUSPARSELT_ROOT}include"
[ -d "$CUSPARSELT_ROOT" ] || { echo "ERROR: missing $CUSPARSELT_ROOT"; exit 1; }
# -------------

echo "CUSPARSELT_ROOT=${CUSPARSELT_ROOT}"
echo "CUSPARSELT_INCLUDE_PATH=${CUSPARSELT_INCLUDE_PATH}"

echo "PYTORCH_BUILD_VERSION=$PYTORCH_BUILD_VERSION and PYTORCH_BUILD_NUMBER=$PYTORCH_BUILD_NUMBER"

# Build with MPI support via the Cray PE wrappers. -lmpi_gtl_cuda is needed so the
# GPU-Transport-Layer libmpi is linked rather than the non-GTL libmpi_gnu.so.12
# that PrgEnv-gnu would otherwise hand us (would silently break CUDA-aware MPI).
# Check afterwards: ldd $(python -c 'import torch;print(torch.__file__)')/../lib/libtorch_cuda.so | grep gtl
#
# PyTorch 2.14 moved to scikit-build-core; `python setup.py bdist_wheel` is now a
# deprecation shim that prints instructions and exits. Use PEP 517 `build` with
# --no-isolation so it sees our conda cmake/ninja and env-var configuration
# (USE_CUDA, USE_MPI, TORCH_CUDA_ARCH_LIST, ... are still honored).
export USE_MPI=1
if [ -z "$PT_WHEEL_EXISTING" ]; then
pip install build scikit-build-core
BUILD_TEST=0 CUDAHOSTCXX=g++-14 CC=cc CXX=CC \
    LDFLAGS="-L/opt/cray/pe/lib64 -Wl,-rpath,/opt/cray/pe/lib64 -lmpi_gtl_cuda ${LDFLAGS}" \
    python -m build --wheel --no-isolation
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo "copying pytorch wheel file $PT_WHEEL"
cp $PT_WHEEL $WHEELS_PATH/
fi   # end PyTorch build (skipped on resume)
cd $WHEELS_PATH
echo "pip installing $(basename $PT_WHEEL)"
pip install $(basename $PT_WHEEL)

# HARDCODE: triton must match what this torch release pairs with (torch 2.14.0 ->
# triton 3.8.0). The 2025-09-28 build shipped an unpinned triton 3.5.0 against torch
# 2.8 and torch.compile broke (`triton_key` removed). Install it right after torch so
# nothing downstream (vLLM, mamba-ssm) silently picks a different one; re-pinned at the end.
TRITON_VERSION="3.8.0"
pip install "triton==${TRITON_VERSION}"
# https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

cd $BASE_PATH

echo "Pip install TensorBoard profiler plugin"
pip install tensorboard_plugin_profile tensorflow-datasets
# tensorflow-datasets' resolve (attempt 3 log) downgraded setuptools to 69.5.1 (torch 2.14
# needs >=77.0.3) and pulled ruamel-yaml 0.19.1 (conda 26.7.2 needs <0.19, and conda is
# still used below for `conda list`/`conda info`). Put both back.
# Resume: the conda install above re-links conda's setuptools (egg-info) next to pip's, the
# xprof downgrade then swaps pip's files for 69.5.1, and -U sees conda's 84.0.0 metadata as
# satisfied: 69.5.1 code under 84.0.0 metadata, which rejects vLLM's PEP 639 license string
# (attempt 5). Remove every copy before reinstalling.
for _ in 1 2 3; do have_pkg setuptools || break; pip uninstall -y setuptools; done
pip install -U "setuptools>=77.0.3" "ruamel.yaml<0.19"

cd $BASE_PATH
# Tripwire: if anything in the conda install above resurrected the conda
# compiler toolchain (gcc_impl_linux-64 / binutils_impl_linux-64 /
# sysroot_linux-64), _compiler_compat/ld becomes a live symlink to
# x86_64-conda-linux-gnu-ld and the mpi4py link below will fail (sysroot-locked
# linker won't honor the Cray PE / system search paths). Bail loudly here
# instead of churning on the link error. Confirmed culprit historically:
# conda-forge `rust`.
if conda list 2>/dev/null | grep -qE '^(gcc_impl_linux-64|binutils_impl_linux-64|sysroot_linux-64) '; then
    echo "ERROR: conda compiler toolchain detected in env; mpi4py link will fail."
    echo "Check what pulled in gcc_impl/binutils_impl/sysroot_linux-64 (conda-forge"
    echo "\`rust\` is the usual suspect). Do not install rust via conda-forge."
    conda list | grep -E '^(sysroot|binutils|gcc|gxx|kernel-headers|libgcc|libstdcxx|libcxx|compiler_)' || true
    exit 1
fi

# KGF (2022-09-09):
MPICC="cc -shared -target-accel=nvidia80" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
ldd $(python -c 'import mpi4py.MPI as m; print(m.__file__)') | grep -q gtl || { echo "ERROR: mpi4py not linked against libmpi_gtl_cuda"; exit 1; }

echo "Pip install parallel h5py"
cd $BASE_PATH
# HARDCODE: pinned to a release (was git master = 3.17.0.dev0, which also violated
# tensorflow's h5py<3.15 metadata pin). 3.14.0 is the newest release under 3.15.
H5PY_TAG="3.14.0"
# Guard must check more than the version: the tensorflow wheel install above pulls PyPI's
# serial h5py wheel (TF pins h5py<3.15), which is the same version string but has no MPI
# driver and no GTL (resume 2026-09-22 skipped the build on that and failed the ldd check).
h5py_is_ours() {
    have_pkg_ver h5py "$1" && python -c "import h5py, sys; sys.exit(0 if h5py.get_config().mpi else 1)" 2>/dev/null \
      && ldd "$(python -c 'import h5py; print(h5py.h5.__file__)')" | grep -q gtl
}
if h5py_is_ours "$H5PY_TAG"; then echo "RESUME: parallel h5py $H5PY_TAG already installed; skipping build"; else
gclone https://github.com/h5py/h5py.git h5py
cd h5py
git checkout "$H5PY_TAG"
# PE 26.03: cray-hdf5-parallel/1.14.3.9 (libhdf5_parallel_gnu.so.310). The GTL check
# below is the regression test for the Aug-2026 breakage: h5py imported before mpi4py
# with a GTL-less libmpi aborted MPI_Init.
module load cray-hdf5-parallel
export CC=cc
export HDF5_MPI="ON"
pip install .
unset CC
# Don't leak PE_PKGCONFIG_LIBS=hdf5_parallel into later `cc` links (mpi4jax).
module unload cray-hdf5-parallel
fi   # end h5py build (skipped on resume)
# Check from outside the source checkout: `import h5py` inside $BASE_PATH/h5py resolves
# to the source tree and raises "cannot import h5py from inside the install directory".
cd $BASE_PATH
ldd $(python -c 'import h5py; print(h5py.h5.__file__)') | grep -q gtl || { echo "ERROR: h5py not linked against libmpi_gtl_cuda"; exit 1; }

echo "Pip install other packages"
pip install pandas matplotlib scikit-learn scipy pytest
pip install "dask[complete]"
pip install sacred wandb

echo "Adding module snooper so we can tell what modules people are using"
# KGF: TODO, modify this path at the top of the script somehow; pick correct sitecustomize_polaris.py, etc.
# wont error out if first path does not exist; will just make a broken symbolic link
# https://github.com/argonne-lcf/PyModuleSnooper.git (fork of msalim2)
ln -sf /soft/applications/PyModuleSnooper/sitecustomize.py $(python -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py

# DeepHyper stuff
# tensorflow_probability dropped (2026-09-23): last release 0.25.0 (2024-11) predates
# TF 2.21 / Keras 3, nothing in the env depends on it, and no test imported it.

# Likely no further development: last develop commit 2026-01-12, last release 0.13.2
# (2026-01-05), so develop == 0.13.2 in practice.
# Extras as of 0.13.x: core, dev, jax-cpu, jax-cuda, mpi, ray, redis, redis-hiredis, torch.
# Not jax-cuda/torch/core: they would pull PyPI jax/torch over ours. mpi = mpi4py>=3.1.3,
# already satisfied by the Cray build above.
DH_EXTRAS="mpi,ray,redis-hiredis"
if [[ -z "$DH_REPO_TAG" ]]; then
    echo "Clone and checkout DeepHyper develop branch from git"
    cd $BASE_PATH
    gclone $DH_REPO_URL deephyper
    cd deephyper
    git checkout develop
    pip install ".[${DH_EXTRAS}]"
    cd $BASE_PATH
else
    echo "Install DeepHyper tag $DH_REPO_TAG from PyPI"
    pip install "deephyper[${DH_EXTRAS}]==${DH_REPO_TAG}"
fi

pip install 'libensemble'

# HARDCODE: Globus Compute endpoint + Parsl (workflow users). globus-compute-endpoint
# 4.9.0 hard-pins parsl==2026.2.23 (PyPI normalizes 2026.02.23 to that), psutil<6 and
# pyzmq<=26.1.0; installing both in one resolve keeps pip from bouncing between them.
# Later vLLM/verl installs only require unversioned psutil/pyzmq, so these caps should
# survive; `pip check` at the end will say if they didn't. Latest endpoint is 4.17.1
# (2026-09-18), but these must match what Ops runs on the Polaris/Crux endpoints (Globus
# dev, 2026-09-24). parsl matters most: the parsl code run by the user endpoint (Ops' env)
# and by the batch job (this env) checks for version consistency and fails on a mismatch.
# The end of this script asserts both versions. No 4.x release is fully compatible with
# this stack anyway: every one pins dill==0.3.9 (fixed below via multiprocess) and
# caps click (4.9.0: click<8.2; vLLM -> huggingface_hub>=1.28 needs click>=8.4.2, so that stays violated).
pip install "globus-compute-endpoint==4.9.0" "parsl==2026.02.23"

# PyG wheels for current torch+CUDA. https://data.pyg.org/whl/torch-2.14.0+cu130.html
# (checked 2026-09-17) only ships pyg_lib (0.9.0+pt214cu130, cp310-abi3); pyg-team no
# longer builds torch_scatter / torch_sparse / torch_cluster / torch_spline_conv for new
# torch releases (their ops live in pyg_lib + pure-torch fallbacks in torch-geometric).
# Requesting them made pip fall back to sdists that fail in an isolated build env
# ("No module named 'torch'"). --no-deps so the wheel cannot drag in a binary torch;
# --only-binary so a missing wheel fails loudly instead of trying a source build.
# If a future torch bump lags pyg, comment this out and keep torch-geometric (pure Python).
# HARDCODE
pip install --no-deps --only-binary=:all: pyg_lib \
    -f https://data.pyg.org/whl/torch-${PT_REPO_TAG:1}+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.html
pip install torch-geometric
pip install pillow

cd $BASE_PATH
echo "Install PyTorch Vision from source"
# HARDCODE: torchvision version pairs with PyTorch; v0.29.0 matches torch 2.14.0
VISION_TAG="v0.29.0"
if ls $WHEELS_PATH/torchvision-${VISION_TAG#v}-*.whl >/dev/null 2>&1; then
    echo "RESUME: torchvision ${VISION_TAG} wheel already in $WHEELS_PATH; skipping build"
    VISION_WHEEL=$(ls $WHEELS_PATH/torchvision-${VISION_TAG#v}-*.whl | head -1)
else
gclone https://github.com/pytorch/vision.git vision
cd vision
# (torchvision is ABI-stable against later torch 2.x per the release notes.)
git checkout "$VISION_TAG"

# HARDCODE: upstream now documents `pip install . --no-build-isolation`; use pip wheel
# so we keep a copy in $WHEELS_PATH. --no-deps: torchvision's metadata pins torch and
# would otherwise pull a binary wheel over our from-source build.
# BUILD_VERSION: without it vision derives "0.29.0a0+<sha>" from git (attempt 7 log).
BUILD_VERSION=${VISION_TAG#v} CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
    pip wheel . -v --no-build-isolation --no-deps -w dist/
VISION_WHEEL=$(find dist/ -name "torchvision*.whl" -type f)
cp $VISION_WHEEL $WHEELS_PATH/
fi   # end torchvision build (skipped on resume)
cd $WHEELS_PATH
echo "pip installing $(basename $VISION_WHEEL)"
pip install --force-reinstall --no-deps $(basename $VISION_WHEEL)

cd $BASE_PATH

pip install --no-deps timm
pip install opencv-python-headless

# HARDCODE: onnxruntime-gpu 1.30.0's PyPI wheel targets CUDA 13 (its `cuda` extra pins
# nvidia-cuda-runtime~=13.0, cudnn-cu13~=9). Install without the extras so it uses our
# /soft CUDA/cuDNN instead of PyPI runtime wheels.
pip install 'onnx==1.22.0' 'onnxruntime-gpu==1.30.0'
# tf2onnx removed: pulls protobuf~=3.20 which downgrades onnx/protobuf and breaks the env.
#pip install tf2onnx
# onnx-tf dropped (2026-09-23): last release 1.10.0 (2022-03); pip resolved 1.6.0 here.
pip install huggingface-hub
# HARDCODE: transformers window = intersection of verl v0.9.0 (>=5.5.3,!=5.6.0,<5.11)
# and vLLM v0.29.0 (>=5.10.4). Re-pinned with --no-deps after verl at the end.
TRANSFORMERS_VERSION="5.10.4"
pip install "transformers==${TRANSFORMERS_VERSION}" evaluate datasets accelerate
# datasets pulls multiprocess 0.70.19 -> dill>=0.4.1, but globus-compute-sdk 4.9.0 pins
# dill==0.3.9, and functions are dill-serialized between the user's client, the Ops endpoint
# env and the workers here. datasets 5.x also accepts multiprocess 0.70.17 (dill>=0.3.9).
pip install "dill==0.3.9" "multiprocess==0.70.17"
# gcsfs (pulled in by xprof's fsspec[gcs]) is released in lockstep with fsspec; datasets caps
# fsspec, so match gcsfs to whatever fsspec version it settled on.
pip install --no-deps "gcsfs==$(python -c 'import fsspec; print(fsspec.__version__)')"
# xformers dropped (2026-09-23): PyPI 0.0.35 _C.so is built for torch 2.10/cu128/py3.10 and
# won't load here; source build needs a c++20 patch. Use torch SDPA or flash_attn instead.
# Flash-attention: pin to last stable 2.x (2.8.3.post1). fa4-v4.0.0.beta* is the new
# architecture (different API) and still in beta as of Sept 2026 (beta31). 2.8.3 is
# known to compile against torch 2.14 / CUDA 13.x (third-party prebuilt wheel matrices).
#
# Subshell to scope the env tweaks (no pollution into later pip installs):
#   * MAX_JOBS / NVCC_THREADS: flash-attn .cu files are CUTLASS-heavy and each
#     nvcc invocation can use 4-8 GB. Without throttling, torch.cpp_extension
#     spawns nproc parallel nvcc jobs (~64 on a Polaris login node) -> OOM-kill
#     and a stream of bare "FAILED: [code=255]" with no nvcc stderr. flash-attn's
#     README documents MAX_JOBS as the official knob.
#   * TORCH_CUDA_ARCH_LIST=8.0: Polaris is A100 (sm_80) only. Without this pin,
#     flash-attn 2.8.3 also builds sm_90 (Hopper), sm_100 / sm_120 (Blackwell)
#     by default, ~4xing build time and per-file memory for no reachable hardware.
#   * CC/CXX/CUDAHOSTCXX=gcc-14: torch.cpp_extension defaults to `c++`, and
#     gcc-native/14 only provides gcc/g++ (no c++ symlink), so `c++` resolves to
#     SUSE's /usr/bin/c++ 7.5, which lacks <compare> and fails on torch 2.14's C++20
#     headers (attempt 7, 2026-09-17). The Sophia script unset CC/CXX to dodge the
#     openmpi wrappers; on Sirius pin gcc-14 exactly like libtorch was built.
pip install einops   # flash-attn's only runtime dep besides torch
(
    export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
    export MAX_JOBS=16   # node-local TMPDIR + 512 GB RAM; was 4 (Sophia /soft-disk limit)
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the +cuXX wheel-URL guess
    pip_sdist_cxx20 flash-attn 2.8.3.post1
)
# flash-attn 2.8.3's sdist also ships flash_attn/cute, an early FA4 (CuTe DSL, Hopper/Blackwell)
# prototype written against an older nvidia-cutlass-dsl. With cutlass-dsl 4.8 (quack-kernels needs
# >=4.7) importing it raises AttributeError (`cute.core.ThrMma`), and megatron-core 0.19.2's
# attention.py imports it unconditionally but only catches ImportError, so
# `import megatron.core.models.gpt` (and verl's Megatron engine) failed in conda/2026-10-01.
# Nothing here can use it (megatron/TE/transformers only enable FA4 with the separate
# flash-attn-4 dist installed), so remove it; the import then fails cleanly.
rm -rf "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/flash_attn/cute"
pip install scikit-image
pip install ipython
pip install line_profiler
pip install torch-tb-profiler
pip install torchinfo
# HARDCODE
pip install cupy-cuda${CUDA_VERSION_MAJOR}x
pip install lightning # pytorch-lightning
#pip install "git+https://github.com/saforem2/ezpz"
retry 3 pip install "git+https://github.com/saforem2/ezpz.git@v0.27.5"   # 2026-09-23 (latest release)
#pip install "git+https://github.com/saforem2/ezpz.git@saforem2/tests"
# make sure TERM is set, run "wandb login", add API key, then run ezpz-test
pip install ml-collections
pip install gpytorch
#pip install xgboost  # KGF: TODO, this installs "nvidia-nccl-cu12" https://github.com/dmlc/xgboost/blob/master/python-package/pyproject.toml
# https://xgboost.readthedocs.io/en/stable/changes/v2.1.0.html#nccl-is-now-fetched-from-pypi

# xgboost
# HARDCODE: pinned to a release (was git master = 3.5.0-dev). v3.4.2 is 2026-09-15.
XGBOOST_TAG="v3.4.2"
if have_pkg_ver xgboost "${XGBOOST_TAG#v}"; then echo "RESUME: xgboost ${XGBOOST_TAG} already installed; skipping build"; else
gclone https://github.com/dmlc/xgboost xgboost --recursive
cd xgboost
git checkout "$XGBOOST_TAG" && retry 5 git submodule update --init --recursive
cd python-package
# xgboost 3.5 moved python-package to scikit-build-core: the old packager's
# `use_cuda/use_nccl/use_dlopen_nccl` config-settings are gone ("Unrecognized options
# in config-settings", attempt 9 log) and the pip build now runs CMake itself, so the
# separate `cmake -B build && ninja` step is redundant. Pass the same defines through
# scikit-build-core's cmake.define.* keys instead.
# (USE_DLOPEN_NCCL, BUILD_WITH_SHARED_NCCL, PLUGIN_FEDERATED are the other knobs.)
# --no-build-isolation: in pip's isolated build env our pip cmake's Python launcher cannot
# import its module, so scikit-build-core fetched cmake 4.4.3 there instead (2026-10-01
# attempt 2). Build against the env's cmake<4 like every other source build.
pip install "scikit-build-core>=0.11.0"   # xgboost's only build requirement
pip install -v . --no-build-isolation \
    --config-settings cmake.define.USE_CUDA=ON \
    --config-settings cmake.define.USE_NCCL=ON \
    --config-settings cmake.define.USE_DLOPEN_NCCL=ON \
    --config-settings cmake.define.NCCL_ROOT=$NCCL_BASE \
    --config-settings cmake.define.CMAKE_CUDA_ARCHITECTURES=80 \
    --config-settings cmake.define.CMAKE_CXX_COMPILER=/usr/bin/g++-14 \
    --config-settings cmake.define.CMAKE_C_COMPILER=/usr/bin/gcc-14 \
    --config-settings cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
    --config-settings build.verbose=true
# TODO: enable RAPIDS integration https://xgboost.readthedocs.io/en/stable/python/rmm-examples/index.html

# pip still drops nvidia-nccl-cu{12,13} into the env even with the cmake flags above; remove it.
# `|| true` since only one of these will actually be installed for any given CUDA target.
# TODO: test if distributed GPU xgboost with Dask works after removing this PyPI package
# https://xgboost.readthedocs.io/en/stable/gpu/index.html#multi-node-multi-gpu-training
# https://xgboost.readthedocs.io/en/stable/tutorials/dask.html
pip uninstall -y nvidia-nccl-cu12 || true
pip uninstall -y nvidia-nccl-cu13 || true
fi   # end xgboost (skipped on resume)
cd $BASE_PATH

pip install multiprocess py4j
# falkon dropped (2026-09-23): no release newer than master, one commit since 2025-07, untested.
pip install pykeops   # optional backend for gpytorch.kernels.keops; JIT-compiles with nvcc on first use, needs CUDA_PATH (set in modulefile)
# pyright[nodejs]: Node from the nodejs-wheel-binaries wheel instead of a nodeenv download
# on first use (compute nodes have no direct internet). The pyright npm package itself is
# still fetched into ~/.cache on first run.
pip install hydra-core hydra_colorlog arviz "pyright[nodejs]" celerite seaborn xarray bokeh matplotx torchviz rich parse
# pip install aim # no aimrocks wheel 0.5.x for python 3.13.x. Latest is 0.5.2 for PyTorch 3.12
pip install jupyter
pip install tensorboardX

# HARDCODE: re-assert the torch-paired triton in case anything above moved it.
pip install "triton==${TRITON_VERSION}"
cd $BASE_PATH
echo "Install CUTLASS from source"
export CUTLASS_PATH="${BASE_PATH}/cutlass"
# These are consumed by CUTLASS *and* by later CMake builds (TransformerEngine's
# cmake/cuDNN.cmake needs CUDNN_PATH; attempt 17 failed with "Could not find
# cudnn_LIBRARY" once the CUTLASS block was skipped on resume), so keep them outside
# the resume guard.
export CUDNN_PATH=${CUDNN_BASE}
export CUDNN_HOME=${CUDNN_BASE}
export CUDA_INSTALL_PATH=${CUDA_HOME}
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
# HARDCODE: pinned to a release (was git master). v4.8.0 is 2026-09-22. vLLM builds its
# CUTLASS kernels from this tree (VLLM_CUTLASS_SRC_DIR), so bump the two together.
CUTLASS_TAG="v4.8.0"
if [ -x "$CUTLASS_PATH/build/tools/profiler/cutlass_profiler" ] && \
   [ "$(git -C "$CUTLASS_PATH" describe --tags --exact-match 2>/dev/null)" = "$CUTLASS_TAG" ]; then
    echo "RESUME: cutlass_profiler already built at $CUTLASS_TAG; skipping CUTLASS"
else
gclone https://github.com/NVIDIA/cutlass cutlass
cd cutlass
git checkout "$CUTLASS_TAG"
mkdir build && cd build
echo "About to run CMake for CUTLASS python = $(which python)"
conda info
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON
make cutlass_profiler -j32
fi   # end CUTLASS (skipped on resume)

cd $BASE_PATH
echo "Install DeepSpeed from source"
# HARDCODE
DEEPSPEED_TAG="v0.19.7"   # 2026-09-16
if have_pkg_prefix deepspeed "${DEEPSPEED_TAG#v}"; then
    echo "RESUME: deepspeed ${DEEPSPEED_TAG} already installed; skipping build"; else
gclone https://github.com/deepspeedai/DeepSpeed.git DeepSpeed
cd DeepSpeed
git checkout "$DEEPSPEED_TAG"
# conda include/lib (libaio for the async_io/gds ops) are passed on the DeepSpeed pip
# command only; exporting them leaked into every later build (apex, mpi4jax, TE, vLLM, ...).
DS_CFLAGS="-I${CONDA_PREFIX}/include/"
DS_LDFLAGS="-L${CONDA_PREFIX}/lib/ -Wl,--enable-new-dtags,-rpath,${CONDA_PREFIX}/lib"
# --no-deps: deepspeed-kernels declares an unversioned `cmake` dependency (see the pip
# cmake note near the top).
pip install --no-deps deepspeed-kernels

# v0.19.7 (2026-09-16) selects -std=c++20 for CUDA >= 13 and torch >= 2.12 upstream, so the
# builder.py sed needed on v0.19.6 is gone. The DeepCompile op (csrc/compile/*.cpp) includes
# nccl.h but DeepCompileBuilder only adds $CUDA_HOME/include, and our NCCL is under /soft:
# pass it through CPATH (gcc honors it; dc is C++ only) instead of patching op_builder/dc.py.
# The modulefile prepends the same dir to CPATH, so runtime JIT rebuilds find it too.

# pip >= 25.3 deprecated --global-option / --build-option (PEP517 always-on) and dropped
# setup.py bdist_wheel. DeepSpeed is not PEP517-compliant, so we need --no-build-isolation
# plus the new -C config-settings syntax.
# https://pip.pypa.io/en/latest/news/#v25-3
# https://github.com/deepspeedai/DeepSpeed/issues/7031
#
# DS_BUILD_CCL_COMM=0: skip deepspeed.ops.comm.deepspeed_ccl_comm_op (Intel oneCCL
# CPU collective backend). Polaris is NVIDIA-only; we don't ship oneapi/ccl.hpp. The
# CCLCommBuilder.is_compatible() gate evidently regressed in v0.19.0 and the op
# now compiles unconditionally under DS_BUILD_OPS=1, failing with
# "fatal error: oneapi/ccl.hpp: No such file or directory". NCCL is what DeepSpeed
# actually uses for GPU collectives here (via PyTorch's torch.distributed), so
# dropping the CCL backend is harmless.
# DS_BUILD_FP_QUANTIZER=0: skip the FP8/FP6/FP12 quantizer op (same op set as 2026-09-17).
# v0.19.6 only built it with triton 2.3.x/3.0 installed, so it was always skipped here;
# v0.19.7 dropped that gate, and its csrc/fp_quantizer/fp_quantize_impl.cu does not compile:
# upstream PR #7976 (2026-04-15) renamed a template parameter to q_exponent_bits, which
# collides with a local of the same name in apply_dequantization and
# apply_selective_dequantization. nvcc 13.0 rejects that ("template parameter
# q_exponent_bits may not be redeclared", 2026-10-01 attempt 2); a compiler that accepted it
# would let the local shadow the parameter and produce wrong dequantized values. Fixing it
# needs a per-use choice between the quantized and fp16/bf16 exponent widths, and A100 has no
# FP8 hardware, so it is not worth patching. Runtime JIT of this op fails the same way.
# Revisit when upstream fixes it (still on master 2026-09-24; no upstream issue filed).
TORCH_CUDA_ARCH_LIST="8.0" CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 \
    NVCC_PREPEND_FLAGS="--forward-unknown-opts" CPATH="$NCCL_BASE/include${CPATH:+:$CPATH}" \
    CFLAGS="$DS_CFLAGS" LDFLAGS="$DS_LDFLAGS" \
    DS_BUILD_OPS=1 DS_BUILD_CCL_COMM=0 DS_BUILD_FP_QUANTIZER=0 \
    pip install -v . -C="--global-option=build_ext" -C="--build-option=-j8" --no-build-isolation
fi   # end DeepSpeed (skipped on resume)

# > ds_report  -- run this after build to confirm op compilation; expect [YES] for fused_adam,
#   cpu_adam, gds, transformer*, etc. fp_quantizer (disabled above) and sparse_attn will be [NO].
cd $BASE_PATH

# HARDCODE: Apex (fused optimizers/norms; optional fast paths in Megatron-LM, NeMo, etc.)
# Maintained but slowly: a few commits a month, mostly pruning modules superseded by TE /
# PyTorch. Releases are bare git tags with no GitHub release notes, and infrequent (25.08,
# 25.09, then 26.09 on 2026-09-23). The package always reports version 0.1, so the resume
# guard checks the checkout's tag instead.
APEX_TAG="26.09"
if have_pkg apex && \
   [ "$(git -C "$BASE_PATH/apex" describe --tags --exact-match 2>/dev/null)" = "$APEX_TAG" ]; then
    echo "RESUME: apex $APEX_TAG already installed; skipping build"; else
gclone https://github.com/NVIDIA/apex apex
cd apex
git checkout "$APEX_TAG"
#  with CUDA and C++ extensions using environment variables:
CUDAHOSTCXX=g++-14 CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
fi   # end apex (skipped on resume)

cd $BASE_PATH

# Megatron-DeepSpeed dropped (2026-09-23): upstream deepspeedai repo idle since 2025-08, and
# its pip install registered as dist "megatron_core 0.2.0", shadowing real Megatron Core below.
# It is a training-scripts repo; users clone it (ALCF fork: argonne-lcf/Megatron-DeepSpeed,
# mostly AuroraGPT/Aurora launch scripts) and run from the checkout.

# HARDCODE: jax must be a version mpi4jax supports. The 2025-09-28 build shipped an
# unpinned jax 0.8.0 (removed `mlir.custom_call`) and mpi4jax stopped importing.
# mpi4jax v0.9.1.post1 (2026-08-24) tracks jax 0.11.1, the current release.
# cuda13_local = jax-cuda13-plugin + jax-cuda13-pjrt using our /soft CUDA/cuDNN/NCCL
# (no nvidia-*-cu13 runtime wheels).
JAX_VERSION="0.11.1"
MPI4JAX_TAG="v0.9.1.post1"
pip install "jax[cuda${CUDA_VERSION_MAJOR}_local]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pymongo optax flax

# HARDCODE: pinned to a release (was git master). 0.22.0 is 2026-09-18, needs jax>=0.7.
NUMPYRO_TAG="0.22.0"
# numpyro is installed --no-deps; multipledispatch is its one runtime dep nothing else pulls in,
# and numpyro.distributions imports it (2026-10-01 build: `import numpyro` failed).
pip install multipledispatch
if have_pkg_ver numpyro "$NUMPYRO_TAG"; then echo "RESUME: numpyro $NUMPYRO_TAG already installed"; else
gclone https://github.com/pyro-ppl/numpyro.git numpyro
cd numpyro
git checkout "$NUMPYRO_TAG"
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --no-deps .
fi
cd $BASE_PATH

# --- MPI4JAX
# nanobind is now required for mpi4jax's from-source build. Linked through `cc` so it
# picks up the GTL like mpi4py/h5py above.
pip install cython nanobind
if have_pkg_ver mpi4jax "${MPI4JAX_TAG#v}"; then echo "RESUME: mpi4jax ${MPI4JAX_TAG} already installed; skipping build"; else
gclone https://github.com/mpi4jax/mpi4jax.git mpi4jax
cd mpi4jax
git checkout "$MPI4JAX_TAG"
CC=cc CXX=CC CUDA_ROOT=$CUDA_TOOLKIT_BASE pip install --no-build-isolation --no-cache-dir --no-binary=mpi4jax -v .
fi
cd $BASE_PATH
python -c "import jax; assert jax.__version__ == '${JAX_VERSION}', jax.__version__; import mpi4jax; print('mpi4jax OK, jax', jax.__version__)"

###############################################################################
# Inference stack (megatron-core, TransformerEngine, vLLM, FlashInfer, mamba-ssm,
# verl). Ported from Sophia 2026-06 build; tags bumped 2026-09-16.
#
# verl + vLLM + TransformerEngine have tight inter-version coupling.
# 2026-09 combo: vLLM v0.29.0 (pins torch 2.13.0 upstream; we rebuild it against our
# torch 2.14.0 via use_existing_torch.py -- first thing to suspect if vLLM misbehaves),
# FlashInfer v0.6.18.post1 (what vLLM 0.29 pins), TransformerEngine v2.19, verl v0.9.0
# (needs vLLM >= 0.18, transformers < 5.11).
# Previous validated combos: Sophia 2026-06: vLLM v0.22.1 / FlashInfer v0.6.12 /
# TE v2.15 / verl v0.8.0. Polaris Fall 2025: vLLM v0.9.1 / TE v2.7 / verl v0.5.0 /
# FlashInfer v0.3.1 / transformers <4.54.0.
###############################################################################

# HARDCODE: Megatron Core (verl's Megatron backend, NeMo). cp313 wheel; only compiled part is
# a pybind11 dataset helper (no torch ABI). Deps (torch>=2.6, numpy, packaging) already met.
pip install --no-deps "megatron-core==0.19.2"   # 2026-09-18

# TransformerEngine (PyTorch + JAX bindings).
# Note: pip 25.x rejects the old `#egg=name[extras]` fragment; use PEP 508
# direct-URL syntax (`name[extras] @ url`) instead.
#
# TE's common/util/logging.h does `#include "nccl.h"` unconditionally; on Polaris
# NCCL lives under /soft (no system install), so the compiler needs to be told
# where to look. NVTE_NCCL_HOME is TE's documented var; CPATH/LIBRARY_PATH are
# belt-and-suspenders for any subproject that doesn't honor it.
# Same OOM/arch/MPI-leak knobs as flash-attn (see above) - TE has even more .cu
# TUs and will fan out to nproc by default.
# HARDCODE
TE_TAG="v2.19"
if have_pkg_prefix transformer_engine "${TE_TAG#v}"; then echo "RESUME: transformer_engine ${TE_TAG} already installed; skipping build"; else
(
    export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
    export MAX_JOBS=16   # node-local TMPDIR + 512 GB RAM; was 4 (Sophia /soft-disk limit)
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    # TE >= 2.19 refuses to build without the nvidia-cudnn-frontend wheel installed
    # ("nvidia-cudnn-frontend is required to build Transformer Engine", attempt 13).
    # --no-deps: it otherwise pulls nvidia-cutlass-dsl[cu13] and friends.
    pip install --no-deps nvidia-cudnn-frontend
    # TE 2.19's _discover_nccl_home() reads NCCL_HOME ("Could not locate NCCL core
    # (nccl.h + libnccl.so). Set NCCL_HOME", attempt 15); NVTE_NCCL_HOME is the older name.
    export NCCL_HOME="$NCCL_BASE"
    export NVTE_NCCL_HOME="$NCCL_BASE"
    # TE 2.19 builds the NCCL EP (expert-parallel) extension by default; it is Hopper+
    # only (arch >= 90) and its bare `make` cannot find cuda.h (attempt 16). A100 is sm_80,
    # so turn it off and tell TE the arch explicitly.
    export NVTE_WITH_NCCL_EP=0
    export NVTE_CUDA_ARCHS=80
    # common/CMakeLists.txt does find_path(NCCL_INCLUDE_DIR ... REQUIRED) with hints
    # limited to site-packages/nvidia/nccl, /opt/nvidia/nccl and /usr/local/nccl, and
    # setup.py only passes -DNCCL_INCLUDE_DIR when NCCL EP is on (attempt 18:
    # "Could not find NCCL_INCLUDE_DIR"). Hand it over through NVTE_CMAKE_EXTRA_ARGS.
    export NVTE_CMAKE_EXTRA_ARGS="-DNCCL_INCLUDE_DIR=$NCCL_BASE/include -DNCCL_LIBRARY=$NCCL_BASE/lib/libnccl.so ${NVTE_CMAKE_EXTRA_ARGS:-}"
    export CUDNN_PATH=${CUDNN_BASE}
    export CUDNN_HOME=${CUDNN_BASE}
    export CPATH="$CUDA_HOME/include:$NCCL_BASE/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$NCCL_BASE/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    retry 3 pip install --no-build-isolation \
        "transformer_engine[pytorch,jax] @ git+https://github.com/NVIDIA/TransformerEngine.git@${TE_TAG}"
)
fi   # end TE (skipped on resume)
# TE <= 2.7 needed NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include at runtime or `import
# transformer_engine` crashed on the `nvidia` namespace package; the modulefile still
# sets it (harmless on 2.19).

pip install pylatexenc qwen-vl-utils

# cuda-bindings tracks the toolkit (13.0.x for cuda-13.0.3); NVSHMEM4Py is tied to the
# CUDA major. Bump both pins if CUDA_VERSION_FULL changes. cuda-python (the metapackage) is
# pinned with them: nvshmem4py-cu13 and flashinfer-python only ask for cuda-python>=12/13, so
# pip took the newest (13.4.1), which requires cuda-bindings~=13.4.1, and conda/2026-10-01
# ended at cuda-bindings 13.4.3. Installed before the constraints file below so both are frozen
# for vLLM/FlashInfer/TRL too.
pip install "cuda-bindings==${CUDA_VERSION_FULL}" "cuda-python==${CUDA_VERSION_FULL}" nvshmem4py-cu${CUDA_VERSION_MAJOR}

# vLLM from source (use_existing_torch.py reuses our PyTorch build so it doesn't
# pull a binary torch wheel that overrides ours).
# (Generated outside the resume guard: FlashInfer below uses it too.)
# Constraints: everything we built from source or pinned above, frozen at its installed
# version. In attempt 21 vLLM's `uv pip install` re-resolved and reinstalled anything of
# conda/local origin (numpy 2.5.3 -> 2.4.6, conda numba/llvmlite -> pip ones, tilelang,
# apache-tvm-ffi, setuptools) because uv does not treat non-index installs as satisfied.
# pip honors installed dists, and -c makes a conflict an error instead of a silent
# downgrade/replace.
VLLM_CONSTRAINTS=$BASE_PATH/vllm-constraints.txt
python - > "$VLLM_CONSTRAINTS" <<'EOF'
import importlib.metadata as m
for n in ["torch","torchvision","triton","transformers","tokenizers","numpy","numba","llvmlite","scipy",
          "jax","jaxlib","jax-cuda13-plugin","jax-cuda13-pjrt","jax-cuda12-plugin","jax-cuda12-pjrt",
          "tensorflow","flash-attn","transformer-engine","transformer-engine-torch","transformer-engine-jax",
          "deepspeed","apex","mpi4py","mpi4jax","h5py","cupy-cuda13x","cupy-cuda12x","xgboost","pyg-lib",
          "onnx","onnxruntime-gpu","cuda-bindings","cuda-python","cmake","ninja","setuptools",
          "parsl","globus-compute-endpoint","globus-compute-sdk","dill","multiprocess","fsspec","gcsfs"]:
    try: print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError: pass
EOF
echo "vLLM constraints:"; cat "$VLLM_CONSTRAINTS"
# HARDCODE
VLLM_TAG="v0.29.0"   # 2026-09-09
if have_pkg_ver vllm "${VLLM_TAG#v}"; then echo "RESUME: vllm ${VLLM_TAG} already installed; skipping build"; else
gclone https://github.com/vllm-project/vllm.git vllm
cd vllm
git checkout "$VLLM_TAG"
python use_existing_torch.py   # strips torch/torchvision/torchaudio pins from requirements + pyproject
# Trim vLLM's CUDA requirement list to what applies here: torch* handled above; numba pin
# (n-gram speculative decoding only) would drag numpy <2.5; flashinfer is installed
# explicitly below (pinned wheel + AOT jit-cache); cutlass-dsl/quack are Blackwell-only
# kernels; torchcodec/PyNvVideoCodec pin torch versions we do not have.
sed -i -E '/^(torch|torchaudio|torchvision|torchcodec|PyNvVideoCodec|numba|flashinfer|nvidia-cutlass-dsl|quack-kernels|--extra-index-url)/d' requirements/cuda.txt
# common.txt and build/cuda.txt cap setuptools<81; the env already has 84 (torch 2.14 needs >=77.0.3, and it
# is in the constraints file), and conda/2026-09-17 ran vLLM 0.29 on 84 without issue.
# Drop the cap rather than downgrade under the rest of the stack.
sed -i -E '/^setuptools[<>=]/d' requirements/common.txt requirements/build/cuda.txt
echo "trimmed requirements/cuda.txt:"; grep -vE '^\s*#|^\s*$' requirements/cuda.txt
pip install -c "$VLLM_CONSTRAINTS" -r requirements/build/cuda.txt
# Cap build parallelism: vLLM pulls in vllm-flash-attn (CUTLASS-heavy, ~340 TUs).
# Default ninja -j$(nproc) generates transient .o piles that have OOM'd /soft.
# Same sm_80 / MPI-leak hygiene as flash-attn / TE above.
(
    export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
    # vLLM runs ninja with MAX_JOBS/NVCC_THREADS workers; 4/2 gave `ninja -j 2` and a
    # 2.5 h+ silent crawl on a 64-core node (attempt 19). The old MAX_JOBS=4 was a
    # Sophia /soft-disk constraint; here TMPDIR is node-local (PBS sets it to
    # /var/tmp/pbs.<jobid>, ~250 GB free) and the node has 512 GB RAM, so 32 nvcc
    # jobs x 2 threads (~8 GB each worst case) is comfortable.
    export MAX_JOBS=32
    export NVCC_THREADS=2
    export CMAKE_BUILD_PARALLEL_LEVEL=32
    export TORCH_CUDA_ARCH_LIST="8.0"
    # use_existing_torch.py dirties the tree, so setuptools-scm reported 0.29.1.dev0+g...;
    # pin the version string to the tag we checked out.
    export SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_TAG#v}
    VLLM_CUTLASS_SRC_DIR=$CUTLASS_PATH pip install -v . --no-build-isolation --no-deps
)
# Runtime deps, resolved by pip under the constraints (common.txt is pulled in by cuda.txt).
pip install -c "$VLLM_CONSTRAINTS" -r requirements/cuda.txt
cd $BASE_PATH   # from inside the source tree, `import vllm` finds its unbuilt vllm/ package
python -c "import vllm; print('vllm', vllm.__version__)"
fi   # end vLLM (skipped on resume)
cd $BASE_PATH

# FlashInfer: pinned wheels (2026-09-22). flashinfer-python is pure Python + JIT; the
# AOT kernels ship separately as flashinfer-jit-cache on flashinfer.ai (no flashinfer-cubin
# for >= 0.6.10). Version must match vLLM v0.29.0's pin (0.6.18). The from-source build
# of earlier scripts is gone: it produced the same JIT-only package plus a jit-cache
# wheel that took longer to build than the whole of vLLM.
FLASHINFER_VERSION="0.6.18"
export FLASHINFER_CUDA_ARCH_LIST="8.0"
# (cuda-bindings / cuda-python / nvshmem4py are pinned before the vLLM constraints file.)
pip install -c "$VLLM_CONSTRAINTS" "flashinfer-python==${FLASHINFER_VERSION}"
pip install --no-deps "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}" \
    --find-links https://flashinfer.ai/whl/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/flashinfer-jit-cache/
python -c "import flashinfer, flashinfer_jit_cache; print('flashinfer', flashinfer.__version__, 'jit-cache OK')"
cd $BASE_PATH

# mamba-ssm + causal-conv1d: fast paths for Mamba/Mamba2/Mamba3 and the hybrid models
# transformers runs through them (Jamba, Falcon-Mamba/H1, Bamba, Granite 4, Nemotron-H).
# torch CUDA extensions whose setup.py imports torch, hence --no-build-isolation. Installed
# after vLLM/FlashInfer and with --no-deps so their exact pins cannot move the env:
# mamba-ssm 2.3.2.post1 wants tilelang==0.1.8 and apache-tvm-ffi<=0.1.9, vLLM installs
# tilelang 0.1.12 / apache-tvm-ffi 0.1.11. Both are only used by the Mamba-3 kernels, so
# `pip check` flags this and the harness (Mamba v1 selective scan) is unaffected.
# Remaining runtime deps: einops, triton, transformers (present), tilelang (vLLM),
# quack-kernels (Mamba-3 step kernels; vLLM's copy is stripped above, so add it here).
# mamba main made the CUDA selective-scan build opt-in (#977, 2026-07); once that ships,
# the sdist patch below should become unnecessary.
(
    export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
    export MAX_JOBS=16   # node-local TMPDIR + 512 GB RAM; was 4 (Sophia /soft-disk limit)
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST="8.0"
    export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE   # skip prebuilt-wheel URL guess
    # HARDCODE: causal-conv1d 1.7.0 (2026-08-20) builds as C++20 without patching.
    have_pkg_ver causal-conv1d 1.7.0 && echo "RESUME: causal-conv1d 1.7.0 already installed" || \
        retry 3 pip install --no-build-isolation --no-deps "causal-conv1d==1.7.0"
    pip install --no-deps quack-kernels
    # HARDCODE: mamba-ssm 2.3.2.post1 (2026-05-09) hard-codes -std=c++17 for nvcc.
    pip_sdist_cxx20 mamba-ssm 2.3.2.post1
)

# SGLang -- DISABLED (KGF 2026-06-10, still as of 2026-09-16)
# SGLang's transitive dep outlines_core has no py3.13 wheel and falls back to a
# source build that needs rustc. Installing conda-forge `rust` pulls in the
# full conda compiler toolchain (gcc_impl/binutils_impl/sysroot_linux-64),
# which sysroot-locks _compiler_compat/ld and breaks mpi4py / h5py MPI linking
# (see the early conda install comment). Until we move rustc to rustup or
# outlines_core ships a py3.13 wheel, just skip SGLang. SGLang 0.5.12 also
# pins torch==2.11.0 hard, which would clobber our from-source torch
# without --no-deps gymnastics anyway.
# git clone -b v0.5.12 https://github.com/sgl-project/sglang.git
# cd sglang/python
# uv pip install . --system --no-deps
# cd $BASE_PATH

# verl: install with --no-deps to avoid stomping on our vLLM version pins,
# then add back the few deps verl actually needs.
# HARDCODE
VERL_TAG="v0.9.0"   # 2026-08-14
if have_pkg_ver verl "${VERL_TAG#v}"; then echo "RESUME: verl ${VERL_TAG} already installed"; else
gclone https://github.com/volcengine/verl.git verl
cd verl
git checkout "$VERL_TAG"
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 pip install --no-deps .
fi
cd $BASE_PATH
# peft: verl declares it as a dep (LoRA/PEFT workflows) but we installed verl
# with --no-deps, so add it back explicitly. tensordict window is verl's.
# orjson: verl/utils/tracking.py imports it but verl does not declare it, so with --no-deps
# every trainer entry point (main_ppo, ray_trainer, sft_trainer, engine_workers, vLLM rollout)
# failed to import in conda/2026-09-17 and 2026-10-01.
pip install torchdata codetiming "tensordict>=0.8.0,<=0.10.0,!=0.9.0" peft orjson

# RL libraries next to verl (added 2026-09-25). -c keeps them from moving anything we built/pinned.
# TRL: pure Python; its deps (accelerate, datasets, transformers) are already satisfied.
pip install -c "$VLLM_CONSTRAINTS" trl
# RLlib: an extra of the Ray that vLLM already installed; pin to it so pip adds gymnasium and
# friends instead of switching Ray versions.
pip install -c "$VLLM_CONSTRAINTS" "ray[rllib]==$(python -c 'import ray; print(ray.__version__)')"
# HARDCODE: torchtitan 0.3.0 (2026-09-03; pure Python). --no-deps: it caps datasets<4.8.0 and
# the env has datasets 5.x (from transformers/evaluate above); its HF text dataloader (streaming
# load_dataset, split_dataset_by_node, state_dict resume) passes on datasets 5.0.1 (isolation test
# `titan`), so the cap stays violated like click for globus-compute. Its other deps are in the env
# except the three below. This is the standalone trainer, not verl's `model_engine=torchtitan`
# (that wants a torchtitan nightly and a torch nightly with the spmd_types backend).
pip install -c "$VLLM_CONSTRAINTS" torch_checkpointing tyro "spmd_types==0.2.3"
pip install --no-deps "torchtitan==0.3.0"
# TorchRL NOT installed (tested 2026-09-25). 0.14.0 (the torch 2.14 release; cp313 wheel) needs
# tensordict>=0.14.2,<0.15, overriding verl's tensordict<=0.10.0 pin above. In a venv on top of
# conda/2026-10-01, verl v0.9.0's own tests: protocol v1 (DataProto) 37/37 pass on both, but
# protocol v2 (TensorDict-based, used by the new engine workers / v1 trainer) fails 3/38 on
# 0.14.2 (test_chunk_concat, test_concat_tensordict: non-tensor fields concat into LinkedLists;
# test_contiguous no longer raises). TorchRL itself works there (PRB C++ ops load against our
# torch). Revisit when verl lifts its tensordict cap; users can install it in a venv.
#pip install -c "$VLLM_CONSTRAINTS" "torchrl==0.14.0"
# OpenRLHF (0.11.2, 2026-09) deliberately NOT installed: PyPI has cp310-cp312 wheels only (no
# sdist, so nothing for py3.13), and it hard-pins transformers==5.15.0 (verl needs <5.11),
# deepspeed==0.19.6, flash-attn==2.8.3 and ray[default]==2.55.0, all of which clash with this
# env. Users who need it: a venv on top of this module with its own transformers, or a container.
# Re-assert the pins that vLLM's runtime deps / verl deps are most likely to have moved.
pip install --no-deps "transformers==${TRANSFORMERS_VERSION}" "triton==${TRITON_VERSION}" "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"

# CUDA runtime/compiler wheels from vLLM's humming-kernels[cu13] (13.4.x in conda/2026-10-01).
# They are used, not just installed: torch's _load_global_deps() preloads any nvidia component
# wheels it finds and JAX's cuda13 plugin prefers them, so every process mapped a 13.4 libcudart
# and libnvrtc next to /soft's 13.0.3 (tests/libprobe-2026-10-01.out). It worked, but NVRTC 13.4
# can emit PTX the 580 (CUDA 13.0) driver cannot JIT. Without them everything resolves to
# $CUDA_HOME (humming-kernels checks CUDA_HOME before the wheels; cuda-bindings' pathfinder falls
# back to LD_LIBRARY_PATH). Kept: nvidia-cuda-cccl (headers only; nvidia-nvshmem-cu13 requires
# it), nvidia-nvshmem-cu13 (torch's only NVSHMEM), nvidia-cutlass-dsl, nvidia-ml-py. Last pip
# step on purpose: nothing after this can pull them back.
pip uninstall -y nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm
python - <<'EOF'
import torch, triton, jax, transformers, mpi4jax
print("torch", torch.__version__, "triton", triton.__version__, "jax", jax.__version__, "transformers", transformers.__version__)
EOF
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if ls "$SITE_PACKAGES"/nvidia/cu13/lib/lib{cudart,nvrtc,nvvm}.so* 2>/dev/null; then
    echo "ERROR: CUDA runtime/compiler wheel libs still in nvidia/cu13/lib (listed above)"; exit 1
fi

echo "Cleaning up"
chmod -R u+w $DOWNLOAD_PATH/
# KGF: lot's of NFS errors lately with the following command
rm -rf $DOWNLOAD_PATH || true
# rm: cannot remove '/soft/applications/conda/2025-09-24/DOWNLOADS/.cache/bazel/_bazel_felker/8d63422c7e36f924c4f33033ca2fe451/server': Directory not empty
rm -rf $DOWNLOAD_PATH || true

conda list
# Expected (metadata-only) complaints: mamba-ssm's tilelang/apache-tvm-ffi pins (see the
# mamba-ssm block), globus-compute-endpoint/sdk's click<8.2 and psutil<6 (huggingface_hub
# needs click>=8.4.2, ipython needs psutil>=7), vLLM's numba and setuptools<81 pins, xprof's
# setuptools<70, torchtitan's datasets<4.8.0. Anything on
# parsl/dill/pyzmq/fsspec means a later install moved a pin.
pip check || true
# parsl/globus-compute must match the Ops endpoint env exactly (see the install above).
python - <<'EOF'
import importlib.metadata as m
want = {"parsl": "2026.2.23", "globus-compute-endpoint": "4.9.0", "globus-compute-sdk": "4.9.0", "dill": "0.3.9"}
got = {k: m.version(k) for k in want}
print("workflow pins:", got)
assert got == want, f"workflow pins moved: {got} != {want}"
EOF
CUDA_VERSION_FULL=$CUDA_VERSION_FULL python - <<'EOF'
import importlib.metadata as m, os
want = os.environ["CUDA_VERSION_FULL"]
got = {k: m.version(k) for k in ("cuda-bindings", "cuda-python")}
print("CUDA python pins:", got)
assert all(v == want for v in got.values()), f"cuda-bindings/cuda-python moved off {want}: {got}"
EOF

chmod -R a-w $BASE_PATH/

set +e
