"""CPU dummy smoke test for the mask-aware SRDN models."""

import numpy as np
import tensorflow as tf
from model_srdn_advanced import SRDCNN_SST_v3, SRDN_ResAFNO_v4

def run_smoke_test():
    print("=== Testing Baseline Model (SRDCNN_SST_v3) ===")
    m_base = SRDCNN_SST_v3()
    m_base.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

    # Dummy data: 32x32 -> 512x512, with an explicit land mask.
    x_dummy = np.random.randn(2, 32, 32, 1).astype(np.float32)
    coarse_mask = np.ones_like(x_dummy)
    fine_mask = np.ones((2, 512, 512, 1), dtype=np.float32)
    fine_mask[:, :64, :64] = 0.0
    inputs = {
        "coarse_sst": x_dummy,
        "coarse_mask": coarse_mask,
        "fine_mask": fine_mask,
    }
    y_dummy = np.random.randn(2, 512, 512, 1).astype(np.float32) * fine_mask

    # Forward pass
    y_pred_base = m_base(inputs, training=False)
    assert y_pred_base.shape == (2, 512, 512, 1), f"Unexpected shape {y_pred_base.shape}"
    assert np.all(y_pred_base.numpy()[fine_mask == 0.0] == 0.0)
    print(f"✓ Forward pass successful! Output shape: {y_pred_base.shape}")

    # Training step
    loss_base = m_base.train_on_batch(inputs, y_dummy)
    print(f"✓ Train on batch successful! Loss: {loss_base}")

    print("\n=== Testing Revised Model (SRDN_ResAFNO_v4) ===")
    m_rev = SRDN_ResAFNO_v4()
    m_rev.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

    # Forward pass
    y_pred_rev = m_rev(inputs, training=False)
    assert y_pred_rev.shape == (2, 512, 512, 1), f"Unexpected shape {y_pred_rev.shape}"
    assert np.all(y_pred_rev.numpy()[fine_mask == 0.0] == 0.0)
    print(f"✓ Forward pass successful! Output shape: {y_pred_rev.shape}")

    # Gradient step
    loss_rev = m_rev.train_on_batch(inputs, y_dummy)
    print(f"✓ Train on batch successful! Loss: {loss_rev}")

    print("\n=== Summary Comparison ===")
    print(f"Baseline Params: {m_base.count_params():,}")
    print(f"Revised  Params: {m_rev.count_params():,}")
    print("\nALL SMOKE TESTS PASSED ON CPU!")

if __name__ == "__main__":
    run_smoke_test()
