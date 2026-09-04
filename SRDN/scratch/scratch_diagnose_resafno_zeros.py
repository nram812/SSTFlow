import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import xarray as xr

# Let's inspect the actual notebook SRDN_ResAFNO_v4 model
from execute_resafno_notebook import NOTEBOOK_PATH
import json

nb = json.loads(NOTEBOOK_PATH.read_text())
# Cell 3 contains dummy()
code_str = "".join(nb["cells"][2]["source"])

env = {}
exec(code_str, env)
dummy = env["dummy"]

print("Running dummy() to inspect ResAFNO outputs at epoch 0 and epoch 1...")
# We will create a small inspection script
res = dummy(epochs=2, batch_size=4, sample_limit=16)

model = res[0]
x_test, y_test, mean_val, std_val = res[2]

pred0 = model.predict(x_test[:2])
print("Prediction min:", np.min(pred0), "max:", np.max(pred0), "mean:", np.mean(pred0))
print("Target min:", np.min(y_test[:2]), "max:", np.max(y_test[:2]), "mean:", np.mean(y_test[:2]))

# Check how many zeros
zero_frac = np.mean(np.abs(pred0) < 1e-6)
print(f"Fraction of near-zero values in pred0: {zero_frac:.4%}")
print(f"Fraction of near-zero values in y_test: {np.mean(np.abs(y_test[:2]) < 1e-6):.4%}")
