import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load our GANGeneratorTF from scratch_test_tf_gan
from scratch_test_tf_gan import GANGeneratorTF

class PatchCriticTF(layers.Layer):
    def __init__(self, base_channels=32, levels=4, **kwargs):
        super().__init__(**kwargs)
        channels = [base_channels * min(2**i, 8) for i in range(levels)]
        self.convs = []
        for i, ch in enumerate(channels):
            stride = 2
            self.convs.append(layers.Conv2D(ch, 4, strides=stride, padding="same"))
        self.out_conv = layers.Conv2D(1, 3, padding="same")
        self.act = layers.LeakyReLU(0.2)

    def call(self, x):
        h = x
        features = []
        for conv in self.convs:
            h = self.act(conv(h))
            features.append(h)
        logits = self.out_conv(h)
        return logits, features

class DiscriminatorTF(keras.Model):
    def __init__(self, base_channels=32, levels=4, scales=2, **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
        self.critics = [PatchCriticTF(base_channels, levels) for _ in range(scales)]

    def call(self, inputs, training=None):
        # inputs: (field, condition, ocean_mask)
        field, condition, ocean_mask = inputs
        H, W = tf.shape(field)[1], tf.shape(field)[2]
        cond_up = tf.image.resize(condition, (H, W), method="bilinear")
        x = tf.concat([field * ocean_mask, cond_up, ocean_mask], axis=-1)

        all_logits = []
        all_features = []
        for i, critic in enumerate(self.critics):
            if i > 0:
                x = tf.nn.avg_pool2d(x, 2, 2, padding="SAME")
            logits, feats = critic(x)
            all_logits.append(logits)
            all_features.extend(feats)
        return all_logits, all_features

def masked_mse(pred, target, mask):
    diff_sq = tf.square(pred - target) * mask
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(diff_sq) / denom

def masked_gradient_loss(pred, target, mask):
    # Sobel / spatial finite differences
    dx_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    
    dy_pred = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dy_target = target[:, 1:, :, :] - target[:, :-1, :, :]
    dy_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]

    loss_x = tf.reduce_sum(tf.abs(dx_pred - dx_target) * dx_mask) / tf.maximum(tf.reduce_sum(dx_mask), 1.0)
    loss_y = tf.reduce_sum(tf.abs(dy_pred - dy_target) * dy_mask) / tf.maximum(tf.reduce_sum(dy_mask), 1.0)
    return loss_x + loss_y

class GANTrainer(keras.Model):
    def __init__(self, generator, discriminator,
                 lambda_content=5.0,
                 lambda_adversarial=0.05,
                 lambda_gradient=1.0,
                 lambda_spectral=0.2,
                 lambda_feature_matching=1.0,
                 critic_steps=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_content = lambda_content
        self.lambda_adversarial = lambda_adversarial
        self.lambda_gradient = lambda_gradient
        self.lambda_spectral = lambda_spectral
        self.lambda_feature_matching = lambda_feature_matching
        self.critic_steps = critic_steps
        
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv")

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker, self.content_loss_tracker, self.adv_loss_tracker]

    def train_step(self, data):
        # Unpack data
        if len(data) == 2:
            x, y = data
            condition, mask = x
            target = y
        else:
            condition, mask = data[0]
            target = data[1]

        # 1. Train Critic
        for _ in range(self.critic_steps):
            with tf.GradientTape() as d_tape:
                fake = self.generator((condition, mask), training=False)
                real_logits_list, _ = self.discriminator((target, condition, mask), training=True)
                fake_logits_list, _ = self.discriminator((fake, condition, mask), training=True)
                
                d_loss = 0.0
                for r_logits, f_logits in zip(real_logits_list, fake_logits_list):
                    # Hinge loss
                    d_loss += tf.reduce_mean(tf.nn.relu(1.0 - r_logits)) + tf.reduce_mean(tf.nn.relu(1.0 + f_logits))
            
            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_grads, _ = tf.clip_by_global_norm(d_grads, 1.0)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # 2. Train Generator
        with tf.GradientTape() as g_tape:
            fake = self.generator((condition, mask), training=True)
            content_loss = masked_mse(fake, target, mask)
            
            fake_logits_list, fake_feats = self.discriminator((fake, condition, mask), training=False)
            _, real_feats = self.discriminator((target, condition, mask), training=False)
            
            adv_loss = 0.0
            for f_logits in fake_logits_list:
                adv_loss += -tf.reduce_mean(f_logits)
                
            grad_loss = masked_gradient_loss(fake, target, mask)
            
            feat_loss = 0.0
            for ff, rf in zip(fake_feats, real_feats):
                feat_loss += tf.reduce_mean(tf.abs(ff - rf))
                
            g_loss = (self.lambda_content * content_loss
                      + self.lambda_adversarial * adv_loss
                      + self.lambda_gradient * grad_loss
                      + self.lambda_feature_matching * feat_loss)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        g_grads, _ = tf.clip_by_global_norm(g_grads, 1.0)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        self.content_loss_tracker.update_state(content_loss)
        self.adv_loss_tracker.update_state(adv_loss)

        return {
            "g_loss": self.g_loss_tracker.result(),
            "d_loss": self.d_loss_tracker.result(),
            "content": self.content_loss_tracker.result(),
            "adv": self.adv_loss_tracker.result(),
        }

    def test_step(self, data):
        if len(data) == 2:
            x, y = data
            condition, mask = x
            target = y
        else:
            condition, mask = data[0]
            target = data[1]

        fake = self.generator((condition, mask), training=False)
        content_loss = masked_mse(fake, target, mask)
        self.content_loss_tracker.update_state(content_loss)
        return {"content": self.content_loss_tracker.result()}

# Test fit
print("Testing GANTrainer fit...")
gen = GANGeneratorTF(base_channels=16, rrdb_blocks=1, growth_channels=8, levels=2)
disc = DiscriminatorTF(base_channels=16, levels=2, scales=2)
trainer = GANTrainer(gen, disc)
trainer.compile(
    g_optimizer=keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99),
    d_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.0, beta_2=0.99),
)

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
print("GANTrainer fit successful! History:", history.history)
