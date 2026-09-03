"""Advanced Deterministic Super-Resolution Downscaling Network (SRDN) in TensorFlow 2.6.0.

Architecture advancements ported and integrated from PyTorch GAN and Flow Matching models:
1. AFNO (Adaptive Fourier Neural Operator) 2D spectral mixing with block-diagonal complex weights.
2. FiLM (Feature-wise Linear Modulation) conditioning on large-scale background state.
3. Deep Residual Convolutional trunk with LayerNorm and GELU activations.
4. Artifact-free progressive 2x upsampling stages with residual refinement.
5. Global physical coarse SST skip connection (residual formulation for SST anomalies).
6. Target parameter complexity: ~5.0 million parameters.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


class FiLMLayer(layers.Layer):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    Dynamically scales and shifts intermediate feature maps based on a global conditioning vector:
        y = (1 + gamma) * x + beta
    """
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.dense = layers.Dense(
            channels * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_dense"
        )

    def call(self, x, condition):
        # x: (B, H, W, C)
        # condition: (B, D)
        gamma_beta = self.dense(condition)
        gamma, beta = tf.split(gamma_beta, num_or_size_splits=2, axis=-1)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.channels])
        beta = tf.reshape(beta, [-1, 1, 1, self.channels])
        return x * (1.0 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class AFNO2D(layers.Layer):
    """Adaptive Fourier Neural Operator (AFNO) 2D Layer.
    
    Performs spectral token mixing in the 2D spatial frequency domain with O(N log N) complexity:
      x (B, H, W, C) -> 2D RFFT -> Complex Block-Diagonal MLP -> Softshrink -> 2D IRFFT -> x_out
    """
    def __init__(self, embed_dim: int, num_blocks: int = 8, sparsity_threshold: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_blocks == 0, f"embed_dim ({embed_dim}) must be divisible by num_blocks ({num_blocks})"
        self.embed_dim = int(embed_dim)
        self.num_blocks = int(num_blocks)
        self.block_size = int(embed_dim // num_blocks)
        self.sparsity_threshold = float(sparsity_threshold)

    def build(self, input_shape):
        scale = 0.02
        self.w1_real = self.add_weight(
            name="w1_real",
            shape=(self.num_blocks, self.block_size, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.w1_imag = self.add_weight(
            name="w1_imag",
            shape=(self.num_blocks, self.block_size, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.b1_real = self.add_weight(
            name="b1_real",
            shape=(1, 1, self.num_blocks, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.b1_imag = self.add_weight(
            name="b1_imag",
            shape=(1, 1, self.num_blocks, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )

        self.w2_real = self.add_weight(
            name="w2_real",
            shape=(self.num_blocks, self.block_size, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.w2_imag = self.add_weight(
            name="w2_imag",
            shape=(self.num_blocks, self.block_size, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.b2_real = self.add_weight(
            name="b2_real",
            shape=(1, 1, self.num_blocks, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        self.b2_imag = self.add_weight(
            name="b2_imag",
            shape=(1, 1, self.num_blocks, self.block_size),
            initializer=tf.random_normal_initializer(stddev=scale),
            trainable=True,
        )
        super().build(input_shape)

    def _complex_mul(self, xr, xi, wr, wi):
        or_ = tf.einsum('...bi,bio->...bo', xr, wr) - tf.einsum('...bi,bio->...bo', xi, wi)
        oi_ = tf.einsum('...bi,bio->...bo', xr, wi) + tf.einsum('...bi,bio->...bo', xi, wr)
        return or_, oi_

    def _softshrink(self, val, lambd):
        return tf.sign(val) * tf.maximum(tf.abs(val) - lambd, 0.0)

    def call(self, x):
        shape = tf.shape(x)
        B, H, W = shape[0], shape[1], shape[2]

        # Reshape to (B, H, W, num_blocks, block_size)
        x_r = tf.reshape(x, [B, H, W, self.num_blocks, self.block_size])
        # Transpose to (B, num_blocks, block_size, H, W) for rfft2d over inner spatial dimensions
        x_p = tf.transpose(x_r, [0, 3, 4, 1, 2])
        x_ft = tf.signal.rfft2d(x_p)

        # Transpose to (B, H, Wf, num_blocks, block_size)
        x_ft = tf.transpose(x_ft, [0, 3, 4, 1, 2])
        shape_ft = tf.shape(x_ft)
        Hf, Wf = shape_ft[1], shape_ft[2]

        xr = tf.reshape(tf.math.real(x_ft), [B, Hf * Wf, self.num_blocks, self.block_size])
        xi = tf.reshape(tf.math.imag(x_ft), [B, Hf * Wf, self.num_blocks, self.block_size])

        # Layer 1: Complex linear transform + bias + soft thresholding
        o_r, o_i = self._complex_mul(xr, xi, self.w1_real, self.w1_imag)
        o_r = o_r + self.b1_real
        o_i = o_i + self.b1_imag
        o_r = self._softshrink(o_r, self.sparsity_threshold)
        o_i = self._softshrink(o_i, self.sparsity_threshold)

        # Layer 2: Complex linear transform + bias
        o_r, o_i = self._complex_mul(o_r, o_i, self.w2_real, self.w2_imag)
        o_r = o_r + self.b2_real
        o_i = o_i + self.b2_imag

        o_r = tf.reshape(o_r, [B, Hf, Wf, self.num_blocks, self.block_size])
        o_i = tf.reshape(o_i, [B, Hf, Wf, self.num_blocks, self.block_size])
        out_ft = tf.complex(o_r, o_i)

        # Inverse 2D FFT back to spatial domain
        out_ft_p = tf.transpose(out_ft, [0, 3, 4, 1, 2])
        x_out = tf.signal.irfft2d(out_ft_p, fft_length=[H, W])
        x_out = tf.transpose(x_out, [0, 3, 4, 1, 2])
        return tf.reshape(x_out, [B, H, W, self.embed_dim])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "sparsity_threshold": self.sparsity_threshold,
        })
        return config


class AFNOResBlock(layers.Layer):
    """Hybrid Residual Block combining AFNO spectral mixing, FiLM conditioning, and Conv2D refinement."""
    def __init__(self, channels: int, num_blocks: int = 8, mlp_ratio: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.afno = AFNO2D(channels, num_blocks=num_blocks)
        self.film = FiLMLayer(channels)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        hidden = int(channels * mlp_ratio)
        self.conv1 = layers.Conv2D(hidden, 3, padding="same", activation="gelu")
        self.conv2 = layers.Conv2D(channels, 3, padding="same")

    def call(self, x, condition=None):
        # 1. AFNO Spectral branch + residual
        h = self.afno(self.norm1(x))
        x = x + h
        # 2. FiLM modulation from global condition
        if condition is not None:
            x = self.film(x, condition)
        # 3. Spatial convolutional refinement + residual
        h = self.conv2(self.conv1(self.norm2(x)))
        return x + 0.2 * h

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class ProgressiveUpsampleBlock(layers.Layer):
    """Artifact-free progressive 2x upsampling with convolutional refinement and FiLM."""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv_in = layers.Conv2D(out_channels, 3, padding="same", activation="gelu")
        self.film = FiLMLayer(out_channels)
        self.conv_refine = layers.Conv2D(out_channels, 3, padding="same", activation="gelu")
        self.conv_out = layers.Conv2D(out_channels, 3, padding="same")

    def call(self, x, condition=None):
        h = self.upsample(x)
        h = self.conv_in(h)
        if condition is not None:
            h = self.film(h, condition)
        res = self.conv_out(self.conv_refine(h))
        return h + 0.2 * res

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        })
        return config


def SRDCNN_SST_v3(numHiddenUnits=64, numResponses=1, numFeatures=1, numLats=512, numLongs=512, shrink=8):
    """Baseline SRDN Model from Jupyter_SRDCNN_stand.20260901.ipynb.
    
    Total parameters: ~404,801.
    """
    reg_val = 1e-9
    in_h = int(numLats / shrink)
    in_w = int(numLongs / shrink)
    inputs = layers.Input(shape=(in_h, in_w, numFeatures), name="input_coarse_sst")

    x = layers.Conv2DTranspose(
        numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
        kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val),
        name="conv2d_transpose_1"
    )(inputs)

    x = layers.Conv2DTranspose(
        numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
        kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val),
        name="conv2d_transpose_2"
    )(x)

    x = layers.Conv2DTranspose(
        numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
        kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val),
        name="conv2d_transpose_3"
    )(x)

    outputs = layers.Conv2D(
        numResponses, (1, 1), activation="linear", padding="same",
        kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val),
        name="conv2d_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SRDCNN_SST_v3_Baseline")
    return model


def SRDN_ResAFNO_v4(numHiddenUnits=128, numResponses=1, numFeatures=1, numLats=512, numLongs=512,
                    shrink=8, trunk_blocks=6, num_freq_blocks=8):
    """Advanced Deterministic SRDN Model with ResAFNO trunk, FiLM, and Progressive Upsampling.
    
    Calibrated to approximately 5.0 million parameters.
    Deterministic: Directly maps coarse SST (64x64) -> high-res SST (512x512).
    """
    in_h = int(numLats / shrink)
    in_w = int(numLongs / shrink)
    inputs = layers.Input(shape=(in_h, in_w, numFeatures), name="input_coarse_sst")

    # 1. Global physical coarse SST skip (bilinear interpolation to native grid)
    # The network learns fine sub-grid eddy residuals on top of the physical coarse field
    coarse_skip = layers.UpSampling2D(size=shrink, interpolation="bilinear", name="physical_coarse_skip")(inputs)

    # 2. Large-scale conditioning encoder for FiLM
    cond_pool = layers.GlobalAveragePooling2D(name="cond_gap")(inputs)
    cond_emb = layers.Dense(numHiddenUnits, activation="gelu", name="cond_mlp_1")(cond_pool)
    cond_emb = layers.Dense(numHiddenUnits, activation="gelu", name="cond_mlp_2")(cond_emb)

    # 3. Stem convolution
    x = layers.Conv2D(numHiddenUnits, 3, padding="same", activation="gelu", name="stem_conv")(inputs)

    # 4. Deep Hybrid ResAFNO Trunk (at 64x64 coarse resolution)
    for i in range(trunk_blocks):
        x = AFNOResBlock(
            channels=numHiddenUnits,
            num_blocks=num_freq_blocks,
            mlp_ratio=2.0,
            name=f"res_afno_block_{i+1}"
        )(x, condition=cond_emb)

    # 5. Progressive 3-Stage 2x Upsampling with FiLM & Refinement:
    # 64x64 -> 128x128 -> 256x256 -> 512x512
    # Stage 1: 64x64 -> 128x128 (128 channels)
    x = ProgressiveUpsampleBlock(
        in_channels=numHiddenUnits, out_channels=128, name="upsample_stage1_128"
    )(x, condition=cond_emb)

    # Stage 2: 128x128 -> 256x256 (96 channels)
    x = ProgressiveUpsampleBlock(
        in_channels=128, out_channels=96, name="upsample_stage2_256"
    )(x, condition=cond_emb)

    # Stage 3: 256x256 -> 512x512 (64 channels)
    x = ProgressiveUpsampleBlock(
        in_channels=96, out_channels=64, name="upsample_stage3_512"
    )(x, condition=cond_emb)

    # 6. High-Resolution Reconstruction Head
    x = layers.Conv2D(32, 3, padding="same", activation="gelu", name="head_conv1")(x)
    residual_detail = layers.Conv2D(
        numResponses, 3, padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="head_conv_detail"
    )(x)

    # 7. Final synthesis: detail residual + physical coarse skip
    outputs = layers.Add(name="final_sst_output")([residual_detail, coarse_skip])

    model = models.Model(inputs=inputs, outputs=outputs, name="SRDN_ResAFNO_v4_Deterministic")
    return model


if __name__ == "__main__":
    print("Building Baseline Model...")
    baseline_model = SRDCNN_SST_v3()
    baseline_model.summary()

    print("\nBuilding Revised ResAFNO Model...")
    revised_model = SRDN_ResAFNO_v4()
    revised_model.summary()

    b_params = baseline_model.count_params()
    r_params = revised_model.count_params()
    print(f"\n=======================================================")
    print(f"Baseline Model Parameters: {b_params:,} (~{b_params/1e6:.2f}M)")
    print(f"Revised  Model Parameters: {r_params:,} (~{r_params/1e6:.2f}M)")
    print(f"=======================================================")
