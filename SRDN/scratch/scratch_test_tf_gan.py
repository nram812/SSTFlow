import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("TensorFlow version:", tf.__version__)

# -------------------------------------------------------------
# 1. GAN Architecture (RRDB Generator + Patch Discriminator)
# -------------------------------------------------------------

class ResidualDenseBlockTF(layers.Layer):
    def __init__(self, channels=48, growth_channels=24, **kwargs):
        super().__init__(**kwargs)
        self.convs = [
            layers.Conv2D(growth_channels, 3, padding="same")
            for _ in range(4)
        ]
        self.fuse = layers.Conv2D(channels, 3, padding="same")
        self.act = layers.LeakyReLU(0.2)

    def call(self, x):
        features = [x]
        for conv in self.convs:
            cur = tf.concat(features, axis=-1)
            features.append(self.act(conv(cur)))
        fused = self.fuse(tf.concat(features, axis=-1))
        return x + 0.2 * fused

class RRDBTF(layers.Layer):
    def __init__(self, channels=48, growth_channels=24, **kwargs):
        super().__init__(**kwargs)
        self.rdb1 = ResidualDenseBlockTF(channels, growth_channels)
        self.rdb2 = ResidualDenseBlockTF(channels, growth_channels)
        self.rdb3 = ResidualDenseBlockTF(channels, growth_channels)

    def call(self, x):
        h = self.rdb1(x)
        h = self.rdb2(h)
        h = self.rdb3(h)
        return x + 0.2 * h

class UpsampleBlockTF(layers.Layer):
    def __init__(self, channels=48, condition_channels=2, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(channels, 3, padding="same")
        self.cond_conv = layers.Conv2D(channels, 1, padding="same")
        self.refine = layers.Conv2D(channels, 3, padding="same")
        self.act = layers.LeakyReLU(0.2)

    def call(self, values, condition, mask):
        # values: [B, H, W, C]
        H, W = tf.shape(values)[1], tf.shape(values)[2]
        up_size = (H * 2, W * 2)
        up_values = tf.image.resize(values, up_size, method="nearest")
        up_values = self.act(self.conv(up_values))

        cond_resized = tf.image.resize(condition, up_size, method="bilinear")
        mask_resized = tf.image.resize(mask, up_size, method="nearest")
        ctx = tf.concat([cond_resized, mask_resized], axis=-1)
        ctx_proj = self.cond_conv(ctx)

        ref = self.act(self.refine(up_values + ctx_proj))
        return up_values + ref

class GANGeneratorTF(keras.Model):
    def __init__(self, base_channels=48, levels=4, condition_channels=2,
                 target_channels=1, noise_channels=4, rrdb_blocks=4,
                 growth_channels=24, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        self.residual = residual
        self.noise_channels = noise_channels
        self.stem = layers.Conv2D(base_channels, 3, padding="same")
        self.trunk = [
            RRDBTF(base_channels, growth_channels) for _ in range(rrdb_blocks)
        ]
        self.trunk_fuse = layers.Conv2D(base_channels, 3, padding="same")
        self.upsample = [
            UpsampleBlockTF(base_channels, condition_channels) for _ in range(levels)
        ]
        self.head_conv1 = layers.Conv2D(base_channels, 3, padding="same")
        self.head_act = layers.LeakyReLU(0.2)
        self.head_conv2 = layers.Conv2D(target_channels, 3, padding="same",
                                        kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, inputs, training=None):
        # inputs can be (condition, mask, noise) or (condition, mask)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                condition, mask, noise = inputs
            else:
                condition, mask = inputs
                noise = None
        else:
            condition = inputs
            mask = tf.ones_like(condition[:, :, :, :1])
            noise = None

        B = tf.shape(condition)[0]
        H_c = tf.shape(condition)[1]
        W_c = tf.shape(condition)[2]

        if noise is None:
            noise = tf.random.normal((B, H_c, W_c, self.noise_channels), dtype=condition.dtype)

        coarse_mask = tf.image.resize(mask, (H_c, W_c), method="nearest")
        stem_in = tf.concat([condition, noise, coarse_mask], axis=-1)
        h = self.stem(stem_in)
        h_trunk = h
        for block in self.trunk:
            h_trunk = block(h_trunk)
        h = h + self.trunk_fuse(h_trunk)

        for up_block in self.upsample:
            h = up_block(h, condition, mask)

        out = self.head_act(self.head_conv1(h))
        out = self.head_conv2(out)

        if self.residual:
            H_fine = tf.shape(mask)[1]
            W_fine = tf.shape(mask)[2]
            res = tf.image.resize(condition[:, :, :, :1], (H_fine, W_fine), method="bilinear")
            out = out + res

        return out * mask

# Test GAN Generator
gen = GANGeneratorTF()
cond = tf.zeros((2, 32, 32, 2))
mask = tf.ones((2, 512, 512, 1))
out = gen((cond, mask))
print("GAN Generator output shape:", out.shape)
print("GAN Generator param count:", gen.count_params())
