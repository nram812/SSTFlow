# SRDN environments

The existing `venv_srdn` was created for the original notebook and contains
TensorFlow 2.6.0 built without CUDA plus an incomplete scientific stack.  It is
not the GPU environment for these experiments.  Create separate environments
so a CPU validation cannot silently become a CPU fallback on the H200:

```bash
cd 2026/SSTDownscaling/SRDN
python3.9 -m venv venv_srdn_cpu
venv_srdn_cpu/bin/python -m pip install --upgrade pip
venv_srdn_cpu/bin/python -m pip install -r requirements-cpu.txt

python3.9 -m venv venv_srdn_gpu
venv_srdn_gpu/bin/python -m pip install --upgrade pip
venv_srdn_gpu/bin/python -m pip install -r requirements-gpu.txt
```

The GPU environment must pass this check on the allocated node:

```bash
venv_srdn_gpu/bin/python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(tf.__version__, gpus, tf.sysconfig.get_build_info())
assert gpus, "TensorFlow GPU visibility failed"
PY
```

The pinned GPU install follows TensorFlow's documented `tensorflow[and-cuda]`
packaging path.  The code additionally checks physical and logical GPU
visibility in `jobs/srdn_gpu_smoke.py`; no training job should be interpreted
as a GPU result unless that gate passes.
