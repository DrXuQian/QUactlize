# Environment for the QUactlize decode GEMV task on this ZW610 host.
# SDK 2.2.1-184ee6 (cuda-12.9 variant): hgcc accepts -arch=ppu_10 and matches torch 2.9/cuda12.9.
# The system SDK /usr/local/PPU_SDK (2.0.0-715aa1) rejects -arch=ppu_10.
sdk=${QGEMV_SDK:-/workspace/actlize-la-gdn-fla20-20261106/sdk/2.2.1-184ee6-cuda129/PPU_SDK}
export PPU_SDK="$sdk" PPU_HOME="$sdk" PPU_PATH="$sdk" PPU_VERSION=v2.2.1
export CUDA_SDK="$sdk/CUDA_SDK" CUDA_HOME="$sdk/CUDA_SDK" CUDA_PATH="$sdk/CUDA_SDK"
export LD_LIBRARY_PATH="$sdk/CUDA_SDK/lib64:$sdk/lib"
export PATH="$sdk/CUDA_SDK/bin:$sdk/bin:$sdk/ppu-smi/bin:$sdk/asight/bin:/usr/local/bin:/usr/bin:/bin"
unset sdk
export QGEMV_ROOT=/workspace/quactlize-decode-gemv-20261108
export QGEMV_SRC=$QGEMV_ROOT/source
export QGEMV_SCRATCH=/dev/shm/qgemv
export TMPDIR=$QGEMV_SCRATCH/tmp
mkdir -p "$TMPDIR"
