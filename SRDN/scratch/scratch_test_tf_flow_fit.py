import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load our FlowUNetTF from scratch_test_tf_flow
from scratch_test_tf_flow import FlowUNetTF, group_count

def masked_mse(pred, target, mask):
    diff_sq = tf.square(pred - target) * mask
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(diff_sq) / denom

class FlowMatchingTrainer(keras.Model):
    def __init__(self, velocity_net, sigma_min=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
        self.loss_tracker = keras.metrics.Mean(name="velocity_loss")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack
        if len(data) == 2:
            x, y = data
            condition, mask = x
            target = y
        else:
            condition, mask = data[0]
            target = data[1]

        B = tf.shape(target)[0]
        H = tf.shape(target)[1]
        W = tf.shape(target)[2]

        # Continuous-Time OT-CFM
        flow_time = tf.random.uniform((B,), minval=0.0, maxval=1.0, dtype=target.dtype)
        # Noise over ocean only
        noise = tf.random.normal((B, H, W, 1), dtype=target.dtype) * mask
        
        # State interpolation: x_t = (1 - (1 - sigma_min)*t)*noise + t*target
        t_w = flow_time[:, None, None, None]
        state = (1.0 - (1.0 - self.sigma_min) * t_w) * noise + t_w * target
        state = state * mask
        
        # Target velocity: u_t = target - (1 - sigma_min)*noise
        target_velocity = (target - (1.0 - self.sigma_min) * noise) * mask

        with tf.GradientTape() as tape:
            pred_velocity = self.velocity_net((state, condition, mask, flow_time), training=True)
            loss = masked_mse(pred_velocity, target_velocity, mask)

        grads = tape.gradient(loss, self.velocity_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.velocity_net.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"velocity_loss": self.loss_tracker.result()}

    def test_step(self, data):
        if len(data) == 2:
            x, y = data
            condition, mask = x
            target = y
        else:
            condition, mask = data[0]
            target = data[1]

        B = tf.shape(target)[0]
        H = tf.shape(target)[1]
        W = tf.shape(target)[2]

        flow_time = tf.random.uniform((B,), minval=0.0, maxval=1.0, dtype=target.dtype)
        noise = tf.random.normal((B, H, W, 1), dtype=target.dtype) * mask
        t_w = flow_time[:, None, None, None]
        state = (1.0 - (1.0 - self.sigma_min) * t_w) * noise + t_w * target
        state = state * mask
        target_velocity = (target - (1.0 - self.sigma_min) * noise) * mask

        pred_velocity = self.velocity_net((state, condition, mask, flow_time), training=False)
        loss = masked_mse(pred_velocity, target_velocity, mask)
        self.loss_tracker.update_state(loss)
        return {"velocity_loss": self.loss_tracker.result()}

    def sample(self, condition, mask, steps=10, sampler="heun"):
        B = tf.shape(condition)[0]
        H = tf.shape(mask)[1]
        W = tf.shape(mask)[2]
        state = tf.random.normal((B, H, W, 1), dtype=condition.dtype) * mask
        dt = 1.0 / float(steps)

        for step in range(steps):
            t_val = float(step) / float(steps)
            t_tensor = tf.fill((B,), t_val)
            v1 = self.velocity_net((state, condition, mask, t_tensor), training=False)
            
            if sampler == "heun" and step < steps - 1:
                next_state = (state + dt * v1) * mask
                t_next = tf.fill((B,), t_val + dt)
                v2 = self.velocity_net((next_state, condition, mask, t_next), training=False)
                state = (state + 0.5 * dt * (v1 + v2)) * mask
            else:
                state = (state + dt * v1) * mask

        return state

# Test Flow fit
print("Testing FlowMatchingTrainer fit...")
unet = FlowUNetTF(base_channels=16, levels=2)
trainer = FlowMatchingTrainer(unet)
trainer.compile(optimizer=keras.optimizers.Adam(1e-4))

dummy_cond = np.random.randn(4, 16, 16, 2).astype(np.float32)
dummy_mask = np.ones((4, 64, 64, 1), dtype=np.float32)
dummy_target = np.random.randn(4, 64, 64, 1).astype(np.float32)

history = trainer.fit(
    x=(dummy_cond, dummy_mask),
    y=dummy_target,
    batch_size=2,
    epochs=2,
    validation_data=((dummy_cond, dummy_mask), dummy_target),
    verbose=1,
)
print("FlowMatchingTrainer fit successful! History:", history.history)

# Test sampling
sampled = trainer.sample(dummy_cond[:2], dummy_mask[:2], steps=5, sampler="heun")
print("Sampled output shape:", sampled.shape)
