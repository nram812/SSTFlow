"""CPU Dummy Data Smoke Test for Baseline and Revised SRDN Models."""

import numpy as np
import tensorflow as tf
from model_srdn_advanced import SRDCNN_SST_v3, SRDN_ResAFNO_v4

def run_smoke_test():
    print("=== Testing Baseline Model (SRDCNN_SST_v3) ===")
    m_base = SRDCNN_SST_v3()
    m_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    
    # Dummy data: batch of 4 samples (64x64) -> (512x512)
    x_dummy = np.random.randn(4, 64, 64, 1).astype(np.float32)
    y_dummy = np.random.randn(4, 512, 512, 1).astype(np.float32)

    # Forward pass
    y_pred_base = m_base(x_dummy, training=False)
    assert y_pred_base.shape == (4, 512, 512, 1), f"Unexpected shape {y_pred_base.shape}"
    print(f"✓ Forward pass successful! Output shape: {y_pred_base.shape}")

    # Training step
    loss_base = m_base.train_on_batch(x_dummy, y_dummy)
    print(f"✓ Train on batch successful! Loss: {loss_base}")

    print("\n=== Testing Revised Model (SRDN_ResAFNO_v4) ===")
    m_rev = SRDN_ResAFNO_v4()
    m_rev.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

    # Forward pass
    y_pred_rev = m_rev(x_dummy, training=False)
    assert y_pred_rev.shape == (4, 512, 512, 1), f"Unexpected shape {y_pred_rev.shape}"
    print(f"✓ Forward pass successful! Output shape: {y_pred_rev.shape}")

    # Gradient step
    loss_rev = m_rev.train_on_batch(x_dummy, y_dummy)
    print(f"✓ Train on batch successful! Loss: {loss_rev}")

    print("\n=== Summary Comparison ===")
    print(f"Baseline Params: {m_base.count_params():,}")
    print(f"Revised  Params: {m_rev.count_params():,}")
    print("\nALL SMOKE TESTS PASSED ON CPU!")

if __name__ == "__main__":
    run_smoke_test()
