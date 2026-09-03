# Sea-Surface Temperature Super-Resolution Downscaling Network (SRDN)

This directory contains deterministic deep learning architectures for $8\times$ super-resolution downscaling of sea-surface temperature (SST) fields over the Australasian domain ($64\times 64 \to 512\times 512$).

---

## 1. Architecture Overview

### Baseline Model: `SRDCNN_SST_v3`
- **Notebook**: [`Jupyter_SRDCNN_stand.20260901.ipynb`](./Jupyter_SRDCNN_stand.20260901.ipynb)
- **Design**: 3 raw `Conv2DTranspose` layers ($7\times 7$ kernels, stride 2, 64 channels) followed by a $1\times 1$ convolution.
- **Parameters**: **404,801** (~0.40M).
- **Limitations**: High risk of checkerboard artifacts from transpose convolutions, local receptive fields only (no global frequency mixing), lacks background conditioning, and does not preserve coarse physical heat content.

### Revised Model: `SRDN_ResAFNO_v4` (Deterministic)
- **Notebook**: [`Jupyter_SRDCNN_ResAFNO.20260903.ipynb`](./Jupyter_SRDCNN_ResAFNO.20260903.ipynb)
- **Python Module**: [`model_srdn_advanced.py`](./model_srdn_advanced.py)
- **Design**:
  1. **AFNO 2D Spectral Mixing**: Real 2D Fourier Neural Operator blocks (`tf.signal.rfft2d` / `tf.signal.irfft2d`) with complex block-diagonal MLP weights and softshrink thresholding. Provides global token mixing with $\mathcal{O}(N \log N)$ complexity to capture ocean spatial teleconnections and mesoscale eddies.
  2. **FiLM Conditioning**: Feature-wise Linear Modulation layers that dynamically scale and shift intermediate feature maps: $(1 + \gamma) \odot h + \beta$, conditioned on large-scale background SST state.
  3. **Deep Residual Trunk**: 6 stacked `AFNOResBlock` units utilizing LayerNorm, GELU, and residual connections.
  4. **Progressive Artifact-Free Upsampling**: 3-stage $2\times$ progressive upsampling ($64\to 128\to 256\to 512$) with refinement convs, avoiding transpose-convolution artifacts.
  5. **Physical Coarse Skip Connection**: Direct bilinear skip from coarse input to native grid, ensuring large-scale conservation while the network learns sub-grid eddy turbulence.
- **Parameters**: **4,754,529** (~4.75M, **~5 million target**).
- **Capacity Increase**: **11.75×** over baseline.

---

## 2. Model Summary Comparison

| Metric | Baseline (`SRDCNN_SST_v3`) | Revised (`SRDN_ResAFNO_v4`) |
| :--- | :--- | :--- |
| **Input Shape** | `(None, 64, 64, 1)` | `(None, 64, 64, 1)` |
| **Output Shape** | `(None, 512, 512, 1)` | `(None, 512, 512, 1)` |
| **Framework** | TensorFlow 2.6.0 | TensorFlow 2.6.0 |
| **Upsampling** | 3× `Conv2DTranspose` | 3-stage Progressive $2\times$ Bilinear + Refine Conv |
| **Spectral Mixing** | None | 2D AFNO (`rfft2d` / `irfft2d`, 8 blocks) |
| **Modulation** | None | FiLM conditioning at trunk & upsample stages |
| **Physical Skip** | None | Coarse bilinear skip |
| **Total Parameters** | **404,801** (~0.40M) | **4,754,529** (~4.75M) |
| **Trainable Params** | 404,801 | 4,754,529 |

---

## 3. Directory Contents

- **`Jupyter_SRDCNN_ResAFNO.20260903.ipynb`**: Primary updated Jupyter Notebook containing architecture code, pre-rendered side-by-side summaries, CPU smoke test with dummy data, and cluster Horovod execution logic.
- **`Jupyter_SRDCNN_stand.20260901.ipynb`**: Original baseline notebook.
- **`model_srdn_advanced.py`**: Standalone TensorFlow 2.6.0 module defining `FiLMLayer`, `AFNO2D`, `AFNOResBlock`, `ProgressiveUpsampleBlock`, `SRDCNN_SST_v3`, and `SRDN_ResAFNO_v4`.
- **`test_cpu_dummy.py`**: Standalone verification test using synthetic SST batches on CPU.
- **`requirements.txt`**: Python dependencies for the SRDN environment.
- **`venv_srdn/`**: Dedicated local virtual environment (excluded from git tracking).

---

## 4. Quick Start & Verification

### Running the CPU Smoke Test
Activate the local environment and run the verification test:
```bash
./venv_srdn/bin/python3 test_cpu_dummy.py
```
Expected output:
```text
=== Testing Baseline Model (SRDCNN_SST_v3) ===
✓ Forward pass successful! Output shape: (4, 512, 512, 1)
✓ Train on batch successful! Loss: [1.00159, 0.79855]

=== Testing Revised Model (SRDN_ResAFNO_v4) ===
✓ Forward pass successful! Output shape: (4, 512, 512, 1)
✓ Train on batch successful! Loss: [1.44871, 0.95956]

=== Summary Comparison ===
Baseline Params: 404,801
Revised  Params: 4,754,529

ALL SMOKE TESTS PASSED ON CPU!
```

### Inspecting Model Summaries
```bash
./venv_srdn/bin/python3 model_srdn_advanced.py
```

### Running on Multi-GPU Clusters (Horovod)
The notebook and model are designed to run seamlessly with Horovod on cluster GPU nodes:
```bash
mpirun -np 4 python -c "
from model_srdn_advanced import SRDN_ResAFNO_v4
import horovod.tensorflow.keras as hvd
hvd.init()
model = SRDN_ResAFNO_v4()
print(f'Rank {hvd.rank()}/{hvd.size()} initialized with {model.count_params():,} params')
"
```
