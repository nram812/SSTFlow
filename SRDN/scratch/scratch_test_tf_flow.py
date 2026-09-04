import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("TensorFlow version:", tf.__version__)

def group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0 and channels // candidate >= 4:
            return candidate
    return 1

class TimeEmbeddingTF(layers.Layer):
    def __init__(self, dimension: int, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.fc1 = layers.Dense(dimension * 2)
        self.fc2 = layers.Dense(dimension)

    def call(self, flow_time):
        half = self.dimension // 2
        freqs = tf.exp(
            -math.log(10000.0)
            * tf.range(half, dtype=tf.float32)
            / max(half - 1, 1)
        )
        flow_time = tf.cast(tf.reshape(flow_time, (-1, 1)), tf.float32)
        angles = flow_time * 1000.0 * freqs[None, :]
        emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        h = tf.nn.silu(self.fc1(emb))
        return self.fc2(h)

class ChannelAttentionTF(layers.Layer):
    def __init__(self, channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        hidden = max(channels // reduction, 8)
        self.fc1 = layers.Conv2D(hidden, 1, use_bias=False)
        self.fc2 = layers.Conv2D(channels, 1, use_bias=False)

    def call(self, values):
        # values: [B, H, W, C]
        avg_pool = tf.reduce_mean(values, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(values, axis=[1, 2], keepdims=True)
        w = self.fc2(tf.nn.relu(self.fc1(avg_pool))) + self.fc2(tf.nn.relu(self.fc1(max_pool)))
        return values * tf.sigmoid(w)

class ResidualBlockTF(layers.Layer):
    def __init__(self, output_channels: int, time_dimension: int, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.groups_out = group_count(output_channels)
        self.conv1 = layers.Conv2D(output_channels, 3, padding="same")
        self.conv2 = layers.Conv2D(output_channels, 3, padding="same",
                                   kernel_initializer="zeros", bias_initializer="zeros")
        self.time_proj = layers.Dense(output_channels * 2)
        self.ca = ChannelAttentionTF(output_channels)
        self.norm2 = layers.GroupNormalization(groups=self.groups_out)
        self.norm1 = None
        self.skip_conv = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        groups_in = group_count(in_channels)
        self.norm1 = layers.GroupNormalization(groups=groups_in)
        if in_channels != self.output_channels:
            self.skip_conv = layers.Conv2D(self.output_channels, 1, padding="same")
        super().build(input_shape)

    def call(self, values, time_emb):
        h = self.conv1(tf.nn.silu(self.norm1(values)))
        proj = self.time_proj(time_emb)
        scale, shift = tf.split(proj, 2, axis=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(tf.nn.silu(h))
        h = self.ca(h)
        skip = self.skip_conv(values) if self.skip_conv is not None else values
        return h + skip

class SelfAttentionTF(layers.Layer):
    def __init__(self, channels: int, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = layers.GroupNormalization(groups=group_count(channels))
        self.qkv = layers.Conv2D(channels * 3, 1, padding="same")
        self.out_conv = layers.Conv2D(channels, 1, padding="same",
                                      kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, values):
        B = tf.shape(values)[0]
        H = tf.shape(values)[1]
        W = tf.shape(values)[2]
        h = self.norm(values)
        qkv = self.qkv(h)
        q, k, v = tf.split(qkv, 3, axis=-1)
        # Reshape to [B, heads, H*W, head_dim]
        q = tf.transpose(tf.reshape(q, (B, H * W, self.heads, self.head_dim)), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, (B, H * W, self.heads, self.head_dim)), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, (B, H * W, self.heads, self.head_dim)), [0, 2, 1, 3])
        
        # Scaled dot-product
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = tf.matmul(q, k, transpose_b=True) * scale
        weights = tf.nn.softmax(scores, axis=-1)
        att = tf.matmul(weights, v) # [B, heads, H*W, head_dim]
        att = tf.reshape(tf.transpose(att, [0, 2, 1, 3]), (B, H, W, self.channels))
        return values + self.out_conv(att)

class FlowUNetTF(keras.Model):
    def __init__(self, base_channels: int = 32, levels: int = 4,
                 condition_channels: int = 2, target_channels: int = 1,
                 attention: bool = True, attention_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels
        time_dim = base_channels * 8
        channels = [base_channels * min(2**lvl, 8) for lvl in range(levels)]
        self.time_emb = TimeEmbeddingTF(time_dim)
        self.input_block = ResidualBlockTF(channels[0], time_dim)
        self.down = [
            ResidualBlockTF(channels[lvl], time_dim)
            for lvl in range(1, levels)
        ]
        self.middle1 = ResidualBlockTF(channels[-1], time_dim)
        self.attention = SelfAttentionTF(channels[-1], attention_heads) if attention else layers.Layer()
        self.middle2 = ResidualBlockTF(channels[-1], time_dim)
        self.up = [
            ResidualBlockTF(channels[lvl - 1], time_dim)
            for lvl in range(levels - 1, 0, -1)
        ]
        self.out_norm = layers.GroupNormalization(groups=group_count(channels[0]))
        self.out_conv = layers.Conv2D(target_channels, 1, padding="same",
                                      kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, inputs, training=None):
        # inputs: (state, condition, mask, flow_time)
        state, condition, mask, flow_time = inputs
        H, W = tf.shape(state)[1], tf.shape(state)[2]
        emb = self.time_emb(flow_time)

        cond_up = tf.image.resize(condition, (H, W), method="bilinear")
        x = tf.concat([state, cond_up, mask], axis=-1)
        h = self.input_block(x, emb)
        skips = [h]

        for lvl, down_block in enumerate(self.down):
            h_pool = tf.nn.avg_pool2d(h, 2, 2, padding="SAME")
            H_p, W_p = tf.shape(h_pool)[1], tf.shape(h_pool)[2]
            cond_p = tf.image.resize(condition, (H_p, W_p), method="bilinear")
            h = down_block(tf.concat([h_pool, cond_p], axis=-1), emb)
            skips.append(h)

        h_mid = tf.nn.avg_pool2d(h, 2, 2, padding="SAME")
        H_m, W_m = tf.shape(h_mid)[1], tf.shape(h_mid)[2]
        cond_m = tf.image.resize(condition, (H_m, W_m), method="bilinear")
        h = self.middle1(tf.concat([h_mid, cond_m], axis=-1), emb)
        h = self.attention(h)
        h = self.middle2(h, emb)

        for idx, up_block in enumerate(self.up):
            skip = skips[self.levels - 2 - idx]
            H_s, W_s = tf.shape(skip)[1], tf.shape(skip)[2]
            h = tf.image.resize(h, (H_s, W_s), method="bilinear")
            h = up_block(tf.concat([h, skip], axis=-1), emb)

        out = self.out_conv(tf.nn.silu(self.out_norm(h)))
        return out * mask

unet = FlowUNetTF()
st = tf.zeros((2, 512, 512, 1))
cond = tf.zeros((2, 32, 32, 2))
m = tf.ones((2, 512, 512, 1))
t = tf.constant([0.2, 0.7])
v = unet((st, cond, m, t))
print("Flow UNet output shape:", v.shape)
print("Flow UNet param count:", unet.count_params())
