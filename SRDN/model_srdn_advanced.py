"""Mask-aware deterministic SRDN models with a canonical ResAFNO trunk.

The real OFAM experiment is a 16x problem: a 32x32 coarse SST field is mapped
to a 512x512 field. Every model receives explicit coarse and fine ocean masks.
Invalid values are zero after normalisation and the output is hard-masked.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


class FiLMLayer(layers.Layer):
    """Feature-wise affine modulation conditioned on a batch vector."""

    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.dense = layers.Dense(
            self.channels * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="film_dense",
        )

    def call(self, x, condition):
        gamma_beta = self.dense(condition)
        gamma, beta = tf.split(gamma_beta, num_or_size_splits=2, axis=-1)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.channels])
        beta = tf.reshape(beta, [-1, 1, 1, self.channels])
        return x * (1.0 + gamma) + beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class MaskedGlobalAverage(layers.Layer):
    """Global average over valid coarse cells only."""

    def call(self, inputs):
        values, mask = inputs
        values = tf.cast(values, tf.float32)
        mask = tf.cast(mask, values.dtype)
        weighted = tf.reduce_sum(values * mask, axis=[1, 2])
        count = tf.reduce_sum(mask, axis=[1, 2])
        return weighted / tf.maximum(count, tf.cast(1.0e-8, values.dtype))

    def get_config(self):
        return super().get_config()


class MaskedOutput(layers.Layer):
    """Force invalid high-resolution pixels to exact zero."""

    def call(self, inputs):
        values, mask = inputs
        return values * tf.cast(mask, values.dtype)

    def get_config(self):
        return super().get_config()


class CoarseConsistencyProjection(layers.Layer):
    """Match valid fine-grid block means to the coarse SST condition.

    The correction is constant within each valid coarse block, so fine-scale
    anomalies are preserved while the mask-aware 16x16 mean is exact.
    """

    def __init__(self, shrink: int, **kwargs):
        super().__init__(**kwargs)
        self.shrink = int(shrink)

    def call(self, inputs):
        values, coarse_sst, coarse_mask, fine_mask = inputs
        values = tf.cast(values, tf.float32)
        coarse_sst = tf.cast(coarse_sst, values.dtype)
        coarse_mask = tf.cast(coarse_mask, values.dtype)
        fine_mask = tf.cast(fine_mask, values.dtype)

        shape = tf.shape(values)
        batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        if values.shape.rank != 4:
            raise ValueError("fine values must be rank four NHWC tensors")
        coarse_height = height // self.shrink
        coarse_width = width // self.shrink
        reshaped_values = tf.reshape(
            values * fine_mask,
            [
                batch,
                coarse_height,
                self.shrink,
                coarse_width,
                self.shrink,
                channels,
            ],
        )
        reshaped_mask = tf.reshape(
            fine_mask,
            [batch, coarse_height, self.shrink, coarse_width, self.shrink, 1],
        )
        totals = tf.reduce_sum(reshaped_values, axis=[2, 4])
        counts = tf.reduce_sum(reshaped_mask, axis=[2, 4])
        block_mean = totals / tf.maximum(counts, tf.cast(1.0e-8, values.dtype))
        correction = (coarse_sst - block_mean) * coarse_mask
        correction = tf.repeat(
            tf.repeat(correction, self.shrink, axis=1), self.shrink, axis=2
        )
        return (values + correction * fine_mask) * fine_mask

    def get_config(self):
        config = super().get_config()
        config.update({"shrink": self.shrink})
        return config


class AFNO2D(layers.Layer):
    """Canonical channel-block AFNO spectral mixer.

    The complex channel MLP is shared over Fourier modes, as in the original
    low-parameter AFNO design. The standalone layer is translation-equivariant
    under periodic shifts. Its nonlinear softshrink operation can create
    nonlocal influence, but a zero-threshold impulse test must not be expected
    to activate every output pixel.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_blocks != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_blocks ({num_blocks})"
            )
        if sparsity_threshold < 0.0:
            raise ValueError("sparsity_threshold must be non-negative")
        self.embed_dim = int(embed_dim)
        self.num_blocks = int(num_blocks)
        self.block_size = int(embed_dim // num_blocks)
        self.sparsity_threshold = float(sparsity_threshold)

    def build(self, input_shape):
        scale = 0.02
        weight_shape = (self.num_blocks, self.block_size, self.block_size)
        bias_shape = (1, 1, self.num_blocks, self.block_size)
        for name, shape in (
            ("w1_real", weight_shape),
            ("w1_imag", weight_shape),
            ("b1_real", bias_shape),
            ("b1_imag", bias_shape),
            ("w2_real", weight_shape),
            ("w2_imag", weight_shape),
            ("b2_real", bias_shape),
            ("b2_imag", bias_shape),
        ):
            setattr(
                self,
                name,
                self.add_weight(
                    name=name,
                    shape=shape,
                    initializer=tf.random_normal_initializer(stddev=scale),
                    trainable=True,
                ),
            )
        super().build(input_shape)

    def _complex_mul(self, xr, xi, wr, wi):
        out_real = tf.einsum("...bi,bio->...bo", xr, wr) - tf.einsum(
            "...bi,bio->...bo", xi, wi
        )
        out_imag = tf.einsum("...bi,bio->...bo", xr, wi) + tf.einsum(
            "...bi,bio->...bo", xi, wr
        )
        return out_real, out_imag

    def _softshrink(self, value, lambd):
        return tf.sign(value) * tf.maximum(tf.abs(value) - lambd, 0.0)

    def call(self, x):
        shape = tf.shape(x)
        batch, height, width = shape[0], shape[1], shape[2]
        reshaped = tf.reshape(
            x, [batch, height, width, self.num_blocks, self.block_size]
        )
        packed = tf.transpose(reshaped, [0, 3, 4, 1, 2])
        norm_scale = tf.cast(
            tf.sqrt(tf.cast(height * width, tf.float32)), tf.complex64
        )
        spectrum = tf.signal.rfft2d(packed) / norm_scale
        spectrum = tf.transpose(spectrum, [0, 3, 4, 1, 2])
        spectrum_shape = tf.shape(spectrum)
        height_f, width_f = spectrum_shape[1], spectrum_shape[2]
        real = tf.reshape(
            tf.math.real(spectrum),
            [batch, height_f * width_f, self.num_blocks, self.block_size],
        )
        imag = tf.reshape(
            tf.math.imag(spectrum),
            [batch, height_f * width_f, self.num_blocks, self.block_size],
        )

        real, imag = self._complex_mul(real, imag, self.w1_real, self.w1_imag)
        real = self._softshrink(real + self.b1_real, self.sparsity_threshold)
        imag = self._softshrink(imag + self.b1_imag, self.sparsity_threshold)
        real, imag = self._complex_mul(real, imag, self.w2_real, self.w2_imag)
        real = real + self.b2_real
        imag = imag + self.b2_imag

        real = tf.reshape(
            real, [batch, height_f, width_f, self.num_blocks, self.block_size]
        )
        imag = tf.reshape(
            imag, [batch, height_f, width_f, self.num_blocks, self.block_size]
        )
        output_spectrum = tf.complex(real, imag)
        output_spectrum = tf.transpose(
            output_spectrum, [0, 3, 4, 1, 2]
        ) * norm_scale
        output = tf.signal.irfft2d(output_spectrum, fft_length=[height, width])
        output = tf.transpose(output, [0, 3, 4, 1, 2])
        return tf.reshape(output, [batch, height, width, self.embed_dim])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_blocks": self.num_blocks,
                "sparsity_threshold": self.sparsity_threshold,
            }
        )
        return config


class AFNOResBlock(layers.Layer):
    """AFNO spectral branch plus FiLM and local convolutional refinement."""

    def __init__(
        self,
        channels: int,
        num_blocks: int = 8,
        mlp_ratio: float = 2.0,
        sparsity_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.mlp_ratio = float(mlp_ratio)
        self.sparsity_threshold = float(sparsity_threshold)
        self.norm1 = layers.LayerNormalization(epsilon=1.0e-5)
        self.afno = AFNO2D(
            self.channels,
            num_blocks=self.num_blocks,
            sparsity_threshold=self.sparsity_threshold,
        )
        self.film = FiLMLayer(self.channels)
        self.norm2 = layers.LayerNormalization(epsilon=1.0e-5)
        hidden = int(self.channels * self.mlp_ratio)
        self.conv1 = layers.Conv2D(hidden, 3, padding="same", activation="gelu")
        self.conv2 = layers.Conv2D(self.channels, 3, padding="same")

    def call(self, x, condition=None):
        x = x + self.afno(self.norm1(x))
        if condition is not None:
            x = self.film(x, condition)
        refinement = self.conv2(self.conv1(self.norm2(x)))
        return x + 0.2 * refinement

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "num_blocks": self.num_blocks,
                "mlp_ratio": self.mlp_ratio,
                "sparsity_threshold": self.sparsity_threshold,
            }
        )
        return config


class ProgressiveUpsampleBlock(layers.Layer):
    """Bilinear 2x upsampling followed by convolutional refinement and FiLM."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.conv_in = layers.Conv2D(
            self.out_channels, 3, padding="same", activation="gelu"
        )
        self.film = FiLMLayer(self.out_channels)
        self.conv_refine = layers.Conv2D(
            self.out_channels, 3, padding="same", activation="gelu"
        )
        self.conv_out = layers.Conv2D(self.out_channels, 3, padding="same")

    def call(self, x, condition=None):
        h = self.conv_in(self.upsample(x))
        if condition is not None:
            h = self.film(h, condition)
        residual = self.conv_out(self.conv_refine(h))
        return h + 0.2 * residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {"in_channels": self.in_channels, "out_channels": self.out_channels}
        )
        return config


def _mask_aware_inputs(numLats, numLongs, numFeatures, shrink):
    if int(numFeatures) != 1:
        raise ValueError("SRDN expects one coarse SST channel")
    if int(numLats) % int(shrink) or int(numLongs) % int(shrink):
        raise ValueError("native grid dimensions must be divisible by shrink")
    coarse_shape = (int(numLats // shrink), int(numLongs // shrink), 1)
    inputs = [
        layers.Input(shape=coarse_shape, name="coarse_sst"),
        layers.Input(shape=coarse_shape, name="coarse_mask"),
        layers.Input(shape=(int(numLats), int(numLongs), 1), name="fine_mask"),
    ]
    coarse_sst, coarse_mask, fine_mask = inputs
    condition = layers.Concatenate(name="coarse_condition")(
        [coarse_sst, coarse_mask]
    )
    coarse_skip = layers.UpSampling2D(
        size=int(shrink), interpolation="bilinear", name="physical_coarse_skip"
    )(coarse_sst)
    return inputs, condition, coarse_skip, fine_mask


def _finish_output(
    residual,
    coarse_skip,
    coarse_sst,
    coarse_mask,
    fine_mask,
    shrink,
    enforce_coarse_consistency,
):
    values = layers.Add(name="residual_plus_coarse")([residual, coarse_skip])
    if enforce_coarse_consistency:
        return CoarseConsistencyProjection(
            shrink, name="coarse_consistency_projection"
        )([values, coarse_sst, coarse_mask, fine_mask])
    return MaskedOutput(name="fine_ocean_mask")([values, fine_mask])


def SRDCNN_SST_v3(
    numHiddenUnits=64,
    numResponses=1,
    numFeatures=1,
    numLats=512,
    numLongs=512,
    shrink=16,
    enforce_coarse_consistency=True,
):
    """Mask-aware conventional transpose-convolution SRDN baseline."""
    if int(numResponses) != 1:
        raise ValueError("SRDN currently supports one SST response channel")
    inputs, condition, coarse_skip, fine_mask = _mask_aware_inputs(
        numLats, numLongs, numFeatures, shrink
    )
    reg_val = 1.0e-9
    x = layers.Conv2DTranspose(
        numHiddenUnits,
        (7, 7),
        strides=2,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(reg_val),
        activity_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        name="conv2d_transpose_1",
    )(condition)
    x = layers.Conv2DTranspose(
        numHiddenUnits,
        (7, 7),
        strides=2,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(reg_val),
        activity_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        name="conv2d_transpose_2",
    )(x)
    x = layers.Conv2DTranspose(
        numHiddenUnits,
        (7, 7),
        strides=2,
        activation="relu",
        padding="same",
        kernel_regularizer=l2(reg_val),
        activity_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        name="conv2d_transpose_3",
    )(x)
    residual = layers.Conv2D(
        numResponses,
        (1, 1),
        activation="linear",
        padding="same",
        kernel_regularizer=l2(reg_val),
        activity_regularizer=l2(reg_val),
        bias_regularizer=l2(reg_val),
        name="conv2d_output",
    )(x)
    outputs = _finish_output(
        residual,
        coarse_skip,
        inputs[0],
        inputs[1],
        fine_mask,
        shrink,
        enforce_coarse_consistency,
    )
    return models.Model(
        inputs=inputs, outputs=outputs, name="SRDCNN_SST_v3_Baseline"
    )


def SRDN_ResAFNO_v4(
    numHiddenUnits=128,
    numResponses=1,
    numFeatures=1,
    numLats=512,
    numLongs=512,
    shrink=16,
    trunk_blocks=6,
    num_freq_blocks=8,
    afno_sparsity_threshold=0.01,
    enforce_coarse_consistency=True,
):
    """Mask-aware ResAFNO deterministic SRDN model for the OFAM grid."""
    if int(numResponses) != 1:
        raise ValueError("SRDN currently supports one SST response channel")
    inputs, condition, coarse_skip, fine_mask = _mask_aware_inputs(
        numLats, numLongs, numFeatures, shrink
    )
    coarse_sst, coarse_mask = inputs[:2]
    cond_emb = MaskedGlobalAverage(name="cond_gap")([coarse_sst, coarse_mask])
    cond_emb = layers.Dense(
        numHiddenUnits, activation="gelu", name="cond_mlp_1"
    )(cond_emb)
    cond_emb = layers.Dense(
        numHiddenUnits, activation="gelu", name="cond_mlp_2"
    )(cond_emb)

    x = layers.Conv2D(
        numHiddenUnits, 3, padding="same", activation="gelu", name="stem_conv"
    )(condition)
    for index in range(int(trunk_blocks)):
        x = AFNOResBlock(
            channels=numHiddenUnits,
            num_blocks=num_freq_blocks,
            mlp_ratio=2.0,
            sparsity_threshold=afno_sparsity_threshold,
            name=f"res_afno_block_{index + 1}",
        )(x, condition=cond_emb)

    x = ProgressiveUpsampleBlock(
        in_channels=numHiddenUnits,
        out_channels=128,
        name="upsample_stage1_128",
    )(x, condition=cond_emb)
    x = ProgressiveUpsampleBlock(
        in_channels=128, out_channels=96, name="upsample_stage2_256"
    )(x, condition=cond_emb)
    x = ProgressiveUpsampleBlock(
        in_channels=96, out_channels=64, name="upsample_stage3_512"
    )(x, condition=cond_emb)
    x = layers.Conv2D(32, 3, padding="same", activation="gelu", name="head_conv1")(
        x
    )
    residual = layers.Conv2D(
        numResponses,
        3,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="head_conv_detail",
    )(x)
    outputs = _finish_output(
        residual,
        coarse_skip,
        coarse_sst,
        coarse_mask,
        fine_mask,
        shrink,
        enforce_coarse_consistency,
    )
    return models.Model(
        inputs=inputs, outputs=outputs, name="SRDN_ResAFNO_v4_Deterministic"
    )


if __name__ == "__main__":
    baseline_model = SRDCNN_SST_v3()
    revised_model = SRDN_ResAFNO_v4()
    print("Baseline parameters:", baseline_model.count_params())
    print("ResAFNO parameters:", revised_model.count_params())
    baseline_model.summary()
    revised_model.summary()
