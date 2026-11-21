# ============================================================================
# 1. IMPORTS & CONFIGURATION
# ============================================================================

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import gc
import time
import pickle
import warnings
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from keras import Model
from keras.applications.inception_v3 import InceptionV3

from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_kernels
from sklearn.impute import SimpleImputer

from metrics import *


# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
zdim = 512
epochs = 150
batch_size = 32
initial_lr = 0.0001
beta = 1
gamma = 0.1
alpha = 0.2
d_reg = 5
metric_interval = 10
name = '3facies_2d_80k'

# Latent Diffusion specific parameters
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

# FID/FRD sampling: 10 batches x 1000 = 10000 total
FID_SAMPLE_SIZE = 10000
FID_N_BATCHES = 10
FID_BATCH_SIZE = 1000
SAMPLE_CHUNK_SIZE = 64
FID_EPS = 1e-6

# Adaptive KL config
kl_weight = 0.0001
ADAPTIVE_KL_TARGET = 50.0
ADAPTIVE_KL_TOLERANCE = 0.3
ADAPTIVE_KL_MIN = 1e-7
ADAPTIVE_KL_MAX = 0.005
ADAPTIVE_KL_ADJ_FACTOR = 0.10
ADAPTIVE_KL_WARMUP_EPOCHS = 5

HIGH_RES_CONFIG = {
    'dpi': 300,
    'figsize_multiplier': 1.5,
    'font_size': 20,
    'title_font_size': 24,
    'line_width': 3.0,
    'marker_size': 80,
    'hist_bins': 150,
    'save_formats': ['png', 'svg']
}

# ----------------------------------------------------------------------------
# Environment setup
# ----------------------------------------------------------------------------
warnings.filterwarnings('ignore')

tf.keras.backend.clear_session()
gc.collect()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPUs: {len(gpus)}")
    except RuntimeError as e:
        print(e)

# Global matplotlib configuration
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})

os.makedirs('training_outputs', exist_ok=True)


# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================

X_train = np.load('datasets/Three_facies_2d.npy').astype('float32')

if X_train.max() > 1.0:
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) * 2 - 1

train_size = int(0.9 * len(X_train))
x_train = X_train[:train_size]
x_test = X_train[train_size:]

latent_dim = zdim
ni, nj, nz = X_train.shape[1], X_train.shape[2], X_train.shape[3]
num_classes = 3

print(f"\n{'=' * 60}")
print(f"DATASET INFO")
print(f"{'=' * 60}")
print(f"Image shape: {ni} x {nj} x {nz}")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Latent dimension: {latent_dim}")
print(f"Diffusion steps: {num_timesteps}")
print(f"Number of classes: {num_classes}")
print(f"FID/FRD sample size: {FID_SAMPLE_SIZE} ({FID_N_BATCHES} batches x {FID_BATCH_SIZE})")
print(f"Sample chunk size: {SAMPLE_CHUNK_SIZE}")
print(f"FID epsilon: {FID_EPS}")
print(f"{'=' * 60}\n")


# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 VAE Encoder
# ----------------------------------------------------------------------------
def build_vae_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Conv2D(32, 3, strides=2, padding="same", dtype=tf.float32)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean", dtype=tf.float32)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", dtype=tf.float32)(x)
    return keras.Model(inputs, [z_mean, z_log_var], name="vae_encoder")


encoder = build_vae_encoder((ni, nj, nz), latent_dim)


# ----------------------------------------------------------------------------
# 3.2 Softmax Decoder (used in training)
# ----------------------------------------------------------------------------
def build_decoder_softmax(latent_dim, num_classes):
    latent_inputs = keras.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(6 * 6 * 256, dtype=tf.float32)(latent_inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((6, 6, 256))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax",
                            padding="same", name="output_softmax")(x)
    return keras.Model(latent_inputs, outputs, name="decoder_softmax")


decoder_softmax = build_decoder_softmax(latent_dim, num_classes)


# ----------------------------------------------------------------------------
# 3.3 Continuous Decoder (derived from softmax, for assimilation)
# ----------------------------------------------------------------------------
def build_decoder_continuous(latent_dim, activation='linear'):
    """
    Continuous decoder (1 channel) to serve as an observation operator
    in run_assimilation. The output is naturally in [-1, 1] because
    we combine the 3 classes {-1, 0, +1} via expected value
    (tensordot on the last Conv2D).
    """
    latent_inputs = keras.Input(shape=(latent_dim,), dtype=tf.float32)
    x = layers.Dense(6 * 6 * 256, dtype=tf.float32)(latent_inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((6, 6, 256))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", dtype=tf.float32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(dtype=tf.float32)(x)
    outputs = layers.Conv2D(1, 3, activation=activation, padding="same",
                            name="output_continuous")(x)
    return keras.Model(latent_inputs, outputs, name="decoder_continuous")


decoder_continuous = build_decoder_continuous(latent_dim, activation='linear')


# ----------------------------------------------------------------------------
# 3.4 1D UNet Denoiser
# ----------------------------------------------------------------------------
def build_unet_model(latent_dim, time_emb_dim=256, base_filters=64):
    assert latent_dim % 8 == 0, \
        f"latent_dim must be divisible by 8 for 3 pooling levels (got {latent_dim})"

    def sinusoidal_embedding(x):
        half_dim = time_emb_dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        x = tf.cast(x, tf.float32)
        emb = x * emb
        return tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)

    def time_mlp(t_emb):
        x = layers.Dense(time_emb_dim, activation="swish", name="t_mlp_1")(t_emb)
        x = layers.Dense(time_emb_dim, activation="swish", name="t_mlp_2")(x)
        return x

    def res_block(x, t_emb, filters, name):
        h = layers.Conv1D(filters, 3, padding="same", name=f"{name}_conv1")(x)
        h = layers.LayerNormalization(name=f"{name}_ln1")(h)
        h = layers.Activation("swish", name=f"{name}_act1")(h)
        t = layers.Dense(filters, name=f"{name}_tproj")(t_emb)
        t = layers.Reshape((1, filters), name=f"{name}_treshape")(t)
        h = layers.Add(name=f"{name}_tadd")([h, t])
        h = layers.Conv1D(filters, 3, padding="same", name=f"{name}_conv2")(h)
        h = layers.LayerNormalization(name=f"{name}_ln2")(h)
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding="same", name=f"{name}_skip_proj")(x)
        out = layers.Add(name=f"{name}_resadd")([x, h])
        return layers.Activation("swish", name=f"{name}_out")(out)

    def downsample(x, filters, name):
        return layers.Conv1D(filters, 4, strides=2, padding="same", name=f"{name}_down")(x)

    def upsample_with_skip(x, skip, filters, name):
        x = layers.UpSampling1D(size=2, name=f"{name}_up")(x)
        assert x.shape[1] == skip.shape[1]
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
        x = layers.Conv1D(filters, 3, padding="same", name=f"{name}_conv")(x)
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        return layers.Activation("swish", name=f"{name}_act")(x)

    latent_input = keras.Input(shape=(latent_dim,), dtype=tf.float32, name="latent_input")
    time_input = keras.Input(shape=(1,), dtype=tf.float32, name="time_input")

    t_emb = sinusoidal_embedding(time_input)
    t_emb = time_mlp(t_emb)

    x = layers.Reshape((latent_dim, 1), name="reshape_input")(latent_input)
    x = layers.Conv1D(base_filters, 3, padding="same", name="input_conv")(x)

    e0 = res_block(x, t_emb, base_filters, "enc0")
    d1 = downsample(e0, base_filters * 2, "enc1_down")
    e1 = res_block(d1, t_emb, base_filters * 2, "enc1")
    d2 = downsample(e1, base_filters * 4, "enc2_down")
    e2 = res_block(d2, t_emb, base_filters * 4, "enc2")
    d3 = downsample(e2, base_filters * 8, "enc3_down")
    e3 = res_block(d3, t_emb, base_filters * 8, "enc3")

    b = res_block(e3, t_emb, base_filters * 8, "bottleneck_1")
    b = res_block(b, t_emb, base_filters * 8, "bottleneck_2")

    u3 = upsample_with_skip(b, e2, base_filters * 4, "dec3")
    u3 = res_block(u3, t_emb, base_filters * 4, "dec3_res")
    u2 = upsample_with_skip(u3, e1, base_filters * 2, "dec2")
    u2 = res_block(u2, t_emb, base_filters * 2, "dec2_res")
    u1 = upsample_with_skip(u2, e0, base_filters, "dec1")
    u1 = res_block(u1, t_emb, base_filters, "dec1_res")

    out = layers.Conv1D(base_filters, 3, padding="same", name="out_conv1")(u1)
    out = layers.Activation("swish", name="out_act")(out)
    out = layers.Conv1D(1, 3, padding="same", name="out_conv2")(out)
    out = layers.Reshape((latent_dim,), name="output_reshape")(out)

    return keras.Model([latent_input, time_input], out, name="unet_1d")


unet = build_unet_model(latent_dim, time_emb_dim=256, base_filters=64)


# ----------------------------------------------------------------------------
# 3.5 Diffusion Model (composite)
# ----------------------------------------------------------------------------
class DiffusionModel(keras.Model):
    """Latent Diffusion (VAE encoder + 1D UNet denoiser + softmax decoder).
    The continuous decoder is NOT part of training — it is derived afterwards.
    """

    def __init__(self, encoder, decoder_softmax, unet, num_timesteps=1000,
                 beta_start=0.0001, beta_end=0.02, kl_weight=0.0001,
                 num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder_softmax = decoder_softmax
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes

        self.kl_weight = tf.Variable(float(kl_weight), trainable=False,
                                     dtype=tf.float32, name="kl_weight")

        self.betas = tf.cast(tf.linspace(beta_start, beta_end, num_timesteps), tf.float32)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = tf.sqrt(1.0 / self.alphas)
        alphas_cumprod_prev = tf.concat([[1.0], self.alphas_cumprod[:-1]], axis=0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss", dtype=tf.float32)
        self.diffusion_loss_tracker = keras.metrics.Mean(name="diffusion_loss", dtype=tf.float32)
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss", dtype=tf.float32)
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss", dtype=tf.float32)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.diffusion_loss_tracker,
                self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.5, clipnorm=1.0)

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean), dtype=tf.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def compute_kl_loss(self, z_mean, z_log_var):
        kl = -0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return tf.reduce_mean(kl)

    def sample_timesteps(self, batch_size):
        return tf.random.uniform(shape=(batch_size,), minval=0,
                                 maxval=self.num_timesteps, dtype=tf.int32)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start), dtype=tf.float32)
        a = tf.gather(self.sqrt_alphas_cumprod, t)
        b = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        a = tf.reshape(a, [-1] + [1] * (len(x_start.shape) - 1))
        b = tf.reshape(b, [-1] + [1] * (len(x_start.shape) - 1))
        return a * x_start + b * noise

    def p_sample(self, x, t, t_index):
        betas_t = tf.gather(self.betas, t)
        sqrt_one_minus = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip = tf.gather(self.sqrt_recip_alphas, t)
        betas_t = tf.reshape(betas_t, [-1] + [1] * (len(x.shape) - 1))
        sqrt_one_minus = tf.reshape(sqrt_one_minus, [-1] + [1] * (len(x.shape) - 1))
        sqrt_recip = tf.reshape(sqrt_recip, [-1] + [1] * (len(x.shape) - 1))
        t_batch = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        model_output = self.unet([x, t_batch], training=False)
        model_mean = sqrt_recip * (x - betas_t * model_output / sqrt_one_minus)
        if t_index == 0:
            return model_mean
        posterior_variance_t = tf.gather(self.posterior_variance, t)
        posterior_variance_t = tf.reshape(posterior_variance_t, [-1] + [1] * (len(x.shape) - 1))
        noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32)
        return model_mean + tf.sqrt(posterior_variance_t) * noise

    def sample(self, num_samples, training=False, verbose=True, chunk_size=SAMPLE_CHUNK_SIZE):
        if verbose:
            print(f"\nGenerating {num_samples} discrete samples (chunk_size={chunk_size})...")
        all_gen = []
        n_chunks = (num_samples + chunk_size - 1) // chunk_size
        for c in range(n_chunks):
            cur = min(chunk_size, num_samples - c * chunk_size)
            x = tf.random.normal((cur, latent_dim), dtype=tf.float32)
            for i in range(self.num_timesteps - 1, -1, -1):
                t = tf.ones((cur,), dtype=tf.int32) * i
                x = self.p_sample(x, t, i)
                if verbose and (i % 100 == 0 or i < 10):
                    print(f"  [chunk {c+1}/{n_chunks}] Timestep {i + 1}/{self.num_timesteps}")
            softmax_output = self.decoder_softmax(x, training=training)
            if isinstance(softmax_output, tf.Tensor):
                softmax_output = softmax_output.numpy()
            discrete_output = np.argmax(softmax_output, axis=-1)
            generated = discrete_output.astype(np.float32) - 1.0
            generated = np.expand_dims(generated, axis=-1)
            all_gen.append(generated)
            del x, generated, softmax_output
            gc.collect()
        out = np.concatenate(all_gen, axis=0)
        if verbose:
            print(f"Sample generation complete.")
            print(f"Unique values in generated samples: {np.unique(out)}")
        return out

    def _classification_loss(self, images, softmax_output):
        target_classes = tf.cast((images + 1) / 2 * (self.num_classes - 1), tf.int32)
        target_classes = tf.squeeze(target_classes, axis=-1)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target_classes, softmax_output, from_logits=False
            )
        )

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(images, training=True)
            latents = self.reparameterize(z_mean, z_log_var)

            t = self.sample_timesteps(batch_size)
            noise = tf.random.normal(tf.shape(latents), dtype=tf.float32)
            noisy = self.q_sample(latents, t, noise)
            t_r = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
            pred = self.unet([noisy, t_r], training=True)
            diff_loss = tf.reduce_mean(tf.square(pred - noise))

            reconstructed_softmax = self.decoder_softmax(latents, training=True)
            recon_loss = self._classification_loss(images, reconstructed_softmax)

            kl_loss = self.compute_kl_loss(z_mean, z_log_var)
            total = diff_loss + beta * recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads if g is not None]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total)
        self.diffusion_loss_tracker.update_state(diff_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]
        z_mean, z_log_var = self.encoder(images, training=False)
        latents = self.reparameterize(z_mean, z_log_var)

        t = self.sample_timesteps(batch_size)
        noise = tf.random.normal(tf.shape(latents), dtype=tf.float32)
        noisy = self.q_sample(latents, t, noise)
        t_r = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        pred = self.unet([noisy, t_r], training=False)
        diff_loss = tf.reduce_mean(tf.square(pred - noise))

        reconstructed_softmax = self.decoder_softmax(latents, training=False)
        recon_loss = self._classification_loss(images, reconstructed_softmax)

        kl_loss = self.compute_kl_loss(z_mean, z_log_var)
        total = diff_loss + beta * recon_loss + self.kl_weight * kl_loss

        self.total_loss_tracker.update_state(total)
        self.diffusion_loss_tracker.update_state(diff_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def call(self, inputs):
        images = inputs[0] if isinstance(inputs, tuple) else inputs
        images = tf.cast(images, tf.float32)
        z_mean, z_log_var = self.encoder(images, training=False)
        latents = self.reparameterize(z_mean, z_log_var)
        bs = tf.shape(images)[0]
        t = tf.random.uniform((bs,), 0, self.num_timesteps, dtype=tf.int32)
        noisy = self.q_sample(latents, t)
        t_r = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        return self.unet([noisy, t_r])


diffusion_model = DiffusionModel(
    encoder, decoder_softmax, unet,
    num_timesteps=num_timesteps,
    beta_start=beta_start,
    beta_end=beta_end,
    kl_weight=kl_weight,
    num_classes=num_classes,
)

dummy_input = tf.random.normal((1, ni, nj, nz), dtype=tf.float32)
_ = diffusion_model(dummy_input)
diffusion_model.compile()


# ============================================================================
# 4. METRICS & UTILITIES
# ============================================================================

# ----------------------------------------------------------------------------
# 4.1 Memory optimizer
# ----------------------------------------------------------------------------
class MemoryOptimizer:
    @staticmethod
    def memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 ** 2

    @staticmethod
    def clear_memory():
        tf.keras.backend.clear_session()
        gc.collect()

    @staticmethod
    def check_memory(threshold_mb=8000):
        current_memory = MemoryOptimizer.memory_usage()
        if current_memory > threshold_mb:
            MemoryOptimizer.clear_memory()
            print(f"Memory cleared. Current usage: {current_memory:.2f} MB")
            return True
        return False


# ----------------------------------------------------------------------------
# 4.2 Static geostatistical metrics
# ----------------------------------------------------------------------------
class StaticMetrics:
    @staticmethod
    def variogram_2d_optimized(data, lag_distances=None, max_lag=None):
        if len(data.shape) == 4:
            data = data[..., 0]
        n_samples, height, width = data.shape[:3]
        if lag_distances is None:
            if max_lag is None:
                max_lag = min(height, width) // 2
            lag_distances = np.arange(1, max_lag + 1)
        variogram = np.zeros(len(lag_distances))
        counts = np.zeros(len(lag_distances))
        for k, lag in enumerate(lag_distances):
            if lag < width:
                diff_h = data[:, :, :-lag] - data[:, :, lag:]
                variogram[k] += np.sum(diff_h ** 2) * n_samples
                counts[k] += diff_h.size
            if lag < height:
                diff_v = data[:, :-lag, :] - data[:, lag:, :]
                variogram[k] += np.sum(diff_v ** 2) * n_samples
                counts[k] += diff_v.size
        variogram = variogram / (2 * counts + 1e-10)
        return lag_distances, variogram

    @staticmethod
    def connectivity_function(data, threshold, lag_distances=None):
        if len(data.shape) == 4:
            data = data[..., 0]
        n_samples, height, width = data.shape[:3]
        binary_data = (data > threshold).astype(np.float32)
        if lag_distances is None:
            max_lag = min(height, width) // 2
            lag_distances = np.arange(1, max_lag + 1)
        connectivity = np.zeros(len(lag_distances), dtype=np.float64)
        for k, lag in enumerate(lag_distances):
            conn = 0.0
            total = 0.0
            if lag < width:
                a = binary_data[:, :, :-lag]
                b = binary_data[:, :, lag:]
                conn += np.sum(a * b)
                total += a.size
            if lag < height:
                a = binary_data[:, :-lag, :]
                b = binary_data[:, lag:, :]
                conn += np.sum(a * b)
                total += a.size
            if total > 0:
                connectivity[k] = conn / total
        return lag_distances, connectivity

    @staticmethod
    def calculate_pca_features(data, n_components=50):
        n_samples, height, width, channels = data.shape
        data_flat = data.reshape(n_samples, -1)
        pca = PCA(n_components=min(n_components, min(data_flat.shape)))
        pca_features = pca.fit_transform(data_flat)
        return pca_features, pca.explained_variance_ratio_

    @staticmethod
    def calculate_mds_features(data, n_components=2, subsample=1000):
        if len(data) > subsample:
            idx = np.random.choice(len(data), subsample, replace=False)
            data_subsample = data[idx]
        else:
            data_subsample = data
        n_samples = len(data_subsample)
        data_flat = data_subsample.reshape(n_samples, -1)
        pca = PCA(n_components=min(50, data_flat.shape[1]))
        data_pca = pca.fit_transform(data_flat)
        mds = MDS(n_components=min(n_components, data_pca.shape[1]),
                  random_state=42, n_jobs=-1)
        mds_features = mds.fit_transform(data_pca)
        return mds_features


# ----------------------------------------------------------------------------
# 4.3 Feature extractors
# ----------------------------------------------------------------------------
def get_feature_extractors():
    iv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))
    f_extraction_iv3 = Model(inputs=iv3.input, outputs=iv3.layers[-1].output)
    RC_full = load_model('Reservoir_classifier.h5')
    f_extraction_rc = Model(inputs=RC_full.input,
                            outputs=layers.GlobalAveragePooling2D()(RC_full.layers[-5].output))
    return f_extraction_iv3, f_extraction_rc


f_extraction_iv3, f_extraction_rc = get_feature_extractors()

print(f"\n[EXTRACTOR INFO]")
print(f"  InceptionV3 input_shape : {f_extraction_iv3.input_shape}")
print(f"  ReservoirClassifier input_shape: {f_extraction_rc.input_shape}")
print("")


# ----------------------------------------------------------------------------
# 4.4 Frechet Distance utilities
# ----------------------------------------------------------------------------
def frechet_distance_stable(act_real, act_gen, eps=FID_EPS):
    mu_r = np.mean(act_real, axis=0)
    mu_g = np.mean(act_gen, axis=0)
    sigma_r = np.cov(act_real, rowvar=False)
    sigma_g = np.cov(act_gen, rowvar=False)
    sigma_r = np.atleast_2d(sigma_r)
    sigma_g = np.atleast_2d(sigma_g)
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps
    sigma_g = sigma_g + np.eye(sigma_g.shape[0]) * eps
    diff = mu_r - mu_g
    mean_diff = float(np.dot(diff, diff))
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace_term = float(np.trace(sigma_r + sigma_g - 2.0 * covmean))
    fid = mean_diff + trace_term
    if not np.isfinite(fid):
        return 1e12
    return max(fid, 0.0)


def frechet_distance_batched(real_features, fake_features,
                              n_batches=10, batch_size=1000, eps=FID_EPS):
    """
    Computes the Frechet Distance using batched computation over 10 batches of 1000 samples.
    Accumulates the statistics (mean and covariance) incrementally between batches,
    without keeping all features at once to compute the statistics.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    real_features = np.nan_to_num(real_features, nan=0.0, posinf=0.0, neginf=0.0)
    fake_features = np.nan_to_num(fake_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Accumulate statistics for the real features
    real_sum = np.zeros(real_features.shape[1], dtype=np.float64)
    real_sum_sq = np.zeros((real_features.shape[1], real_features.shape[1]), dtype=np.float64)
    real_count = 0

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = real_features[start:end]
        if batch.shape[0] == 0:
            continue
        real_sum += batch.sum(axis=0)
        real_sum_sq += batch.T @ batch
        real_count += batch.shape[0]

    mu_r = real_sum / real_count
    sigma_r = (real_sum_sq / real_count) - np.outer(mu_r, mu_r)

    # Accumulate statistics for the generated features
    fake_sum = np.zeros(fake_features.shape[1], dtype=np.float64)
    fake_sum_sq = np.zeros((fake_features.shape[1], fake_features.shape[1]), dtype=np.float64)
    fake_count = 0

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = fake_features[start:end]
        if batch.shape[0] == 0:
            continue
        fake_sum += batch.sum(axis=0)
        fake_sum_sq += batch.T @ batch
        fake_count += batch.shape[0]

    mu_g = fake_sum / fake_count
    sigma_g = (fake_sum_sq / fake_count) - np.outer(mu_g, mu_g)

    sigma_r = np.atleast_2d(sigma_r)
    sigma_g = np.atleast_2d(sigma_g)
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps
    sigma_g = sigma_g + np.eye(sigma_g.shape[0]) * eps

    diff = mu_r - mu_g
    mean_diff = float(np.dot(diff, diff))
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    trace_term = float(np.trace(sigma_r + sigma_g - 2.0 * covmean))
    fid = mean_diff + trace_term
    if not np.isfinite(fid):
        return 1e12
    return max(fid, 0.0)


def kernel_distance_stable(act_real, act_gen, num_blocks=50, max_block_size=500):
    n = min(len(act_real), len(act_gen), max_block_size)
    x = act_real[:n]
    y = act_gen[:n]
    d = x.shape[1]
    k = 3
    coef = 1.0 / d

    def poly_kernel(a, b):
        return (a @ b.T / d + coef) ** k

    k_xx = poly_kernel(x, x).mean()
    k_yy = poly_kernel(y, y).mean()
    k_xy = poly_kernel(x, y).mean()
    mmd2 = k_xx + k_yy - 2 * k_xy
    return max(float(mmd2), 0.0)


# ----------------------------------------------------------------------------
# 4.5 Plotting utilities
# ----------------------------------------------------------------------------
def save_figure_multiple_formats(fig, base_filename, dpi=300, formats=None):
    if formats is None:
        formats = HIGH_RES_CONFIG['save_formats']
    saved_files = []
    for fmt in formats:
        filename = f"{base_filename}.{fmt}"
        if fmt == 'png':
            fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        elif fmt == 'svg':
            fig.savefig(filename, format='svg', bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        else:
            continue
        saved_files.append(filename)
        print(f"  Saved: {filename}")
    return saved_files


def plot_histograms_and_densities(original, reconstructed, latent, epoch=None, save_interval=10):
    figsize = (24 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig = plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 26
    })
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    latent_flat = latent.flatten()

    plt.subplot(2, 3, 1)
    plt.imshow(original[0].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.title('Original Sample', fontsize=24, fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed[0].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.title('Reconstructed Sample', fontsize=24, fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(2, 3, 3)
    sns.histplot(original_flat, color='blue', label='Original',
                 kde=True, stat='density', alpha=0.5, bins=HIGH_RES_CONFIG['hist_bins'],
                 linewidth=HIGH_RES_CONFIG['line_width'])
    sns.histplot(reconstructed_flat, color='red', label='Reconstructed',
                 kde=True, stat='density', alpha=0.5, bins=HIGH_RES_CONFIG['hist_bins'],
                 linewidth=HIGH_RES_CONFIG['line_width'])
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Pixel Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.title('Pixel Value Distribution', fontweight='bold', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(2, 3, 4)
    sns.kdeplot(original_flat, color='blue', label='Original',
                linewidth=HIGH_RES_CONFIG['line_width'])
    sns.kdeplot(reconstructed_flat, color='red', label='Reconstructed',
                linewidth=HIGH_RES_CONFIG['line_width'])
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Pixel Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.title('Density Comparison', fontweight='bold', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(2, 3, 5)
    sns.histplot(latent_flat, color='green', label='Latent Space',
                 kde=True, stat='density', bins=HIGH_RES_CONFIG['hist_bins'],
                 linewidth=HIGH_RES_CONFIG['line_width'])
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Latent Variable Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.title('Latent Space Distribution', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(2, 3, 6)
    sns.kdeplot(original_flat, color='blue', label='Original',
                linewidth=HIGH_RES_CONFIG['line_width'])
    sns.kdeplot(latent_flat, color='green', label='Latent Space',
                linewidth=HIGH_RES_CONFIG['line_width'])
    plt.title('Comparison of Densities', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.suptitle('Latent Diffusion Analysis: Original vs Reconstructed Distributions',
                 fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    should_save = False
    if epoch is not None:
        if (epoch % save_interval == 0) or (epoch == epochs):
            should_save = True
            base_filename = f"training_outputs/latent_diffusion_distributions_epoch_{epoch:03d}"
    else:
        should_save = True
        base_filename = f"training_outputs/latent_diffusion_final_distributions"

    if should_save:
        saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
        print(f"Saved distribution plots for epoch {epoch}")
    else:
        saved_files = []
        plt.pause(0.01)

    plt.close(fig)
    return saved_files


def analyze_latent_space(encoder, data, num_samples=1000):
    z_mean, _ = encoder.predict(data[:num_samples], verbose=0)
    return z_mean.flatten()


# ----------------------------------------------------------------------------
# 4.6 Decoder transfer utilities
# ----------------------------------------------------------------------------
def transfer_softmax_to_continuous(decoder_softmax, decoder_continuous,
                                     class_values=(-1.0, 0.0, 1.0)):
    """
    Copies the weights from the softmax decoder to the continuous decoder.
    """
    w_sm = decoder_softmax.get_weights()
    w_ct = decoder_continuous.get_weights()

    assert len(w_sm) == len(w_ct), (
        f"Weight count mismatch: softmax has {len(w_sm)}, "
        f"continuous has {len(w_ct)}. Check that the architectures "
        f"match up to the last layer."
    )

    n_shared = len(w_ct) - 2
    for i in range(n_shared):
        w_ct[i] = w_sm[i].copy()

    class_values = np.array(class_values, dtype=np.float32)
    w_ct[-2] = np.tensordot(
        w_sm[-2], class_values, axes=([-1], [0])
    )[..., np.newaxis].astype(np.float32)
    w_ct[-1] = np.array([np.dot(w_sm[-1], class_values)], dtype=np.float32)

    decoder_continuous.set_weights(w_ct)
    return decoder_continuous


def sanity_check_decoders(encoder, decoder_softmax, decoder_continuous,
                          x_test, n=20):
    """
    Confirms that the weight transfer preserved the information.
    """
    idx = np.random.choice(len(x_test), size=min(n, len(x_test)), replace=False)
    x_batch = x_test[idx]

    z_mean, _ = encoder.predict(x_batch, verbose=0)
    y_sm = decoder_softmax.predict(z_mean, verbose=0)
    y_ct = decoder_continuous.predict(z_mean, verbose=0)

    cls_sm = np.argmax(y_sm, axis=-1) - 1
    cls_ct = np.round(y_ct[..., 0]).clip(-1, 1).astype(int)
    agreement = float(np.mean(cls_sm == cls_ct))

    print("\n" + "=" * 60)
    print("SANITY CHECK: SOFTMAX DECODER vs CONTINUOUS DECODER")
    print("=" * 60)
    print(f"  y_softmax.shape      : {y_sm.shape}")
    print(f"  y_continuous.shape   : {y_ct.shape}")
    print(f"  y_softmax range      : [{y_sm.min():.3f}, {y_sm.max():.3f}]")
    print(f"  y_continuous range   : [{y_ct.min():.3f}, {y_ct.max():.3f}]")
    print(f"  Agreement cls_sm == cls_ct: {agreement:.4f}")
    print(f"  (expected > 0.9 if the transfer preserved the information)")
    print("=" * 60 + "\n")

    return agreement


# ----------------------------------------------------------------------------
# 4.7 Final report generator
# ----------------------------------------------------------------------------
def generate_final_report(filepath, params, execution_time, loss_history, metrics_history,
                          static_metrics_results, adaptive_kl_history=None,
                          decoder_transfer_agreement=None):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("               FINAL LATENT DIFFUSION TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("TRAINING PARAMETERS\n")
        f.write("-" * 80 + "\n")
        for key, value in params.items():
            f.write(f"  {key:<35}: {value}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("EXECUTION TIME\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Time (seconds) : {execution_time:.2f}s\n")
        f.write(f"  Total Time (minutes) : {execution_time / 60:.2f}min\n")
        f.write(f"  Total Time (hours)   : {execution_time / 3600:.2f}h\n\n")

        f.write("-" * 80 + "\n")
        f.write("TRAINING METRICS (FINAL VALUES)\n")
        f.write("-" * 80 + "\n")

        if loss_history.total_losses:
            last_epoch = len(loss_history.total_losses)
            f.write(f"  Total epochs trained: {last_epoch}\n\n")
            f.write("  Training Losses (last epoch):\n")
            f.write(f"    Total Loss       : {loss_history.total_losses[-1]:.6f}\n")
            f.write(f"    Diffusion Loss   : {loss_history.diffusion_losses[-1]:.6f}\n")
            f.write(f"    Reconstruction   : {loss_history.recon_losses[-1]:.6f}\n")
            f.write(f"    KL Loss          : {loss_history.kl_losses[-1]:.6f}\n\n")

            f.write("  Validation Losses (last epoch):\n")
            f.write(f"    Total Loss       : {loss_history.val_total_losses[-1]:.6f}\n")
            f.write(f"    Diffusion Loss   : {loss_history.val_diffusion_losses[-1]:.6f}\n")
            f.write(f"    Reconstruction   : {loss_history.val_recon_losses[-1]:.6f}\n")
            f.write(f"    KL Loss          : {loss_history.val_kl_losses[-1]:.6f}\n\n")

            f.write("  Best values during training:\n")
            f.write(
                f"    Lowest Total Loss (train)      : {min(loss_history.total_losses):.6f} (epoch {np.argmin(loss_history.total_losses) + 1})\n")
            f.write(
                f"    Lowest Total Loss (val)        : {min(loss_history.val_total_losses):.6f} (epoch {np.argmin(loss_history.val_total_losses) + 1})\n")
            f.write(
                f"    Lowest Diffusion Loss (train)  : {min(loss_history.diffusion_losses):.6f} (epoch {np.argmin(loss_history.diffusion_losses) + 1})\n")
            f.write(
                f"    Lowest Recon Loss (train)      : {min(loss_history.recon_losses):.6f} (epoch {np.argmin(loss_history.recon_losses) + 1})\n")
            f.write(
                f"    Lowest KL Loss (train)         : {min(loss_history.kl_losses):.6f} (epoch {np.argmin(loss_history.kl_losses) + 1})\n\n")

        f.write("-" * 80 + "\n")
        f.write("ADAPTIVE KL WEIGHT HISTORY (first/last 10 epochs)\n")
        f.write("-" * 80 + "\n")
        if adaptive_kl_history and adaptive_kl_history.get('epoch'):
            n = len(adaptive_kl_history['epoch'])
            idx_show = list(range(min(10, n))) + ([None] if n > 20 else []) + list(range(max(10, n - 10), n))
            for i in idx_show:
                if i is None:
                    f.write("  ...\n")
                    continue
                f.write(f"  Epoch {adaptive_kl_history['epoch'][i]:4d}: "
                        f"kl_loss={adaptive_kl_history['kl_loss'][i]:9.4f}  "
                        f"kl_weight={adaptive_kl_history['kl_weight'][i]:.8f}  "
                        f"{adaptive_kl_history['action'][i]}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("TRADITIONAL METRICS (LAST EVALUATION)\n")
        f.write("-" * 80 + "\n")

        if hasattr(metrics_history, 'metrics'):
            for key, values in metrics_history.metrics.items():
                if values:
                    f.write(f"  {key:<15}: {values[-1]:.4f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("GEOSTATISTICAL METRICS (LAST EVALUATION)\n")
        f.write("-" * 80 + "\n")

        if hasattr(metrics_history, 'static_metrics'):
            for key, values in metrics_history.static_metrics.items():
                if values:
                    f.write(f"  {key:<20}: {values[-1]:.6f}\n")
            f.write("\n")

        if static_metrics_results:
            f.write("-" * 80 + "\n")
            f.write("FINAL GEOSTATISTICAL METRICS (POST-TRAINING)\n")
            f.write("-" * 80 + "\n")
            for key, value in static_metrics_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key:<25}: {value:.6f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 80 + "\n")
        f.write("  Encoder (VAE):\n")
        f.write(f"    Input Shape        : ({ni}, {nj}, {nz})\n")
        f.write(f"    Latent Dim         : {latent_dim}\n")
        f.write("    Conv Layers        : 4 (32, 64, 128, 256 filters)\n")
        f.write("    Dense Layer        : 256\n")
        f.write("    Dropout            : 0.4 / 0.3\n")
        f.write("    BatchNormalization : Yes\n")
        f.write("    Output Heads       : z_mean, z_log_var\n\n")

        f.write("  Decoder (softmax) — used in training:\n")
        f.write(f"    Input Dim          : {latent_dim}\n")
        f.write("    ConvTranspose      : 3 (128, 64, 32 filters)\n")
        f.write(f"    Num Classes        : {num_classes}\n")
        f.write("    Output Act         : softmax\n")
        f.write(f"    Output Shape       : (None, {ni}, {nj}, {num_classes})\n\n")

        f.write("  Decoder (continuous) — derived from softmax (NOT trained):\n")
        f.write(f"    Input Dim          : {latent_dim}\n")
        f.write("    ConvTranspose      : 3 (128, 64, 32 filters)\n")
        f.write("    Output Act         : linear\n")
        f.write(f"    Output Shape       : (None, {ni}, {nj}, 1)\n")
        f.write("    Weights            : transferred from softmax via tensordot\n")
        f.write("    Usage              : observation operator in run_assimilation\n")
        if decoder_transfer_agreement is not None:
            f.write(f"    Sanity check acc   : {decoder_transfer_agreement:.4f}\n")
            f.write(f"    (agreement softmax+argmax vs continuous+round)\n")
        f.write("\n")

        f.write("  UNet (Diffusion, 1D):\n")
        f.write(f"    Input Dim          : {latent_dim}\n")
        f.write(f"    Time Emb Dim       : 256\n")
        f.write("    Encoder Channels   : 64 / 128 / 256 / 512\n")
        f.write("    Bottleneck Ch.     : 512 (x2 residual blocks)\n")
        f.write("    Skip Connections   : Yes\n")
        f.write("    Activation         : swish\n\n")

        f.write("  Diffusion:\n")
        f.write(f"    Timesteps          : {num_timesteps}\n")
        f.write(f"    Beta Start         : {beta_start}\n")
        f.write(f"    Beta End           : {beta_end}\n\n")

        f.write("-" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"  Training status         : {'COMPLETED' if len(loss_history.total_losses) == epochs else 'INTERRUPTED (Early Stopping)'}\n")
        f.write(f"  Epochs executed         : {len(loss_history.total_losses)}\n")
        f.write(f"  Total time              : {execution_time / 3600:.2f} hours\n")
        f.write(
            f"  Average time per epoch  : {(execution_time / len(loss_history.total_losses)) if loss_history.total_losses else 0:.2f} seconds\n")
        f.write(f"  Dataset                 : {name}\n")
        f.write(f"  Training samples        : {len(x_train)}\n")
        f.write(f"  Test samples            : {len(x_test)}\n")
        f.write(f"  FID/FRD sampling        : {FID_N_BATCHES} batches x {FID_BATCH_SIZE} samples ({FID_N_BATCHES * FID_BATCH_SIZE} total)\n")
        f.write(f"  Sample chunk size       : {SAMPLE_CHUNK_SIZE}\n")
        f.write(f"  FID epsilon             : {FID_EPS}\n")
        f.write(f"  Image generation        : Discrete (softmax + argmax)\n\n")

        f.write("-" * 80 + "\n")
        f.write("GENERATED FILES\n")
        f.write("-" * 80 + "\n")
        saved_files_list = [
            "training_outputs/latent_diffusion_encoder_weights.h5",
            "training_outputs/latent_diffusion_decoder_weights.h5",
            "training_outputs/latent_diffusion_decoder_softmax_weights.h5",
            "training_outputs/latent_diffusion_decoder_continuous_weights.h5",
            "training_outputs/latent_diffusion_unet_weights.h5",
            "training_outputs/latent_diffusion_model_weights.h5",
            "training_outputs/latent_diffusion_encoder_model.h5",
            "training_outputs/latent_diffusion_decoder_model.h5",
            "training_outputs/latent_diffusion_decoder_softmax_model.h5",
            "training_outputs/latent_diffusion_decoder_continuous_model.h5",
            "training_outputs/latent_diffusion_unet_model.h5",
            "training_outputs/latent_diffusion_best_weights.h5",
            "training_outputs/latent_diffusion_loss_history.txt",
            "training_outputs/latent_diffusion_training_metrics.txt",
            "training_outputs/latent_diffusion_all_metrics.csv"
        ]
        for file in saved_files_list:
            f.write(f"  - {file}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("                    END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Final report saved to: {filepath}")


def plot_static_metrics_comparison(real_data, diffusion_model, save_path_base=None,
                                   sample_size=FID_SAMPLE_SIZE):
    n_samples = min(sample_size, len(real_data))
    fake_data = diffusion_model.sample(n_samples, training=False, verbose=False)
    if isinstance(fake_data, tf.Tensor):
        fake_data = fake_data.numpy()
    if np.any(np.isnan(fake_data)):
        fake_data = np.nan_to_num(fake_data, nan=0.0)
    from sklearn.metrics.pairwise import rbf_kernel
    metrics = {}
    lags, variogram_real = StaticMetrics.variogram_2d_optimized(real_data[:100], max_lag=25)
    _, variogram_fake = StaticMetrics.variogram_2d_optimized(fake_data[:100], max_lag=25)
    metrics['variogram_mse'] = np.mean((variogram_real - variogram_fake) ** 2)
    _, connectivity_real = StaticMetrics.connectivity_function(real_data[:100], 0.5, lag_distances=lags)
    _, connectivity_fake = StaticMetrics.connectivity_function(fake_data[:100], 0.5, lag_distances=lags)
    metrics['connectivity_mse'] = np.mean((connectivity_real - connectivity_fake) ** 2)
    real_flat = real_data.flatten()
    fake_flat = fake_data.flatten()
    all_data = np.concatenate([real_flat, fake_flat])
    data_min, data_max = np.min(all_data), np.max(all_data)
    hist_real, bin_edges = np.histogram(real_flat, bins=100, range=(data_min, data_max), density=False)
    hist_fake, _ = np.histogram(fake_flat, bins=100, range=(data_min, data_max), density=False)
    hist_real = hist_real.astype(float) / np.sum(hist_real)
    hist_fake = hist_fake.astype(float) / np.sum(hist_fake)
    epsilon = 1e-10
    hist_real = np.clip(hist_real, epsilon, 1)
    hist_fake = np.clip(hist_fake, epsilon, 1)
    metrics['hist_kl'] = np.sum(hist_real * np.log(hist_real / hist_fake))
    pca_real, _ = StaticMetrics.calculate_pca_features(real_data[:500])
    pca_fake, _ = StaticMetrics.calculate_pca_features(fake_data[:500])
    if pca_real.shape[0] == pca_fake.shape[0]:
        n_components = min(2, pca_real.shape[1], pca_fake.shape[1])
        pca_correlations = []
        for i in range(n_components):
            corr = np.corrcoef(pca_real[:, i], pca_fake[:, i])[0, 1]
            pca_correlations.append(corr)
        metrics['pca_correlation'] = np.mean(pca_correlations)
    else:
        metrics['pca_correlation'] = 0.0
    mds_real = StaticMetrics.calculate_mds_features(real_data[:500])
    mds_fake = StaticMetrics.calculate_mds_features(fake_data[:500])
    n_samples_mds = min(len(mds_real), len(mds_fake), 100)
    mds_real_small = mds_real[:n_samples_mds]
    mds_fake_small = mds_fake[:n_samples_mds]
    if len(mds_real_small) > 0 and len(mds_fake_small) > 0:
        X = np.vstack([mds_real_small, mds_fake_small])
        pairwise_dists = pdist(X)
        gamma = 1.0 / (2.0 * np.median(pairwise_dists) ** 2) if len(pairwise_dists) > 0 else 1.0
        K_XX = rbf_kernel(mds_real_small, mds_real_small, gamma=gamma)
        K_YY = rbf_kernel(mds_fake_small, mds_fake_small, gamma=gamma)
        K_XY = rbf_kernel(mds_real_small, mds_fake_small, gamma=gamma)
        mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        metrics['mds_mmd'] = np.sqrt(max(mmd_squared, 0))
    else:
        metrics['mds_mmd'] = 0.0
    figsize = (22 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    fig.suptitle('Latent Diffusion: Static Geostatistical Metrics Comparison',
                 fontsize=26, fontweight='bold', y=1.02)
    axes[0, 0].plot(lags, variogram_real, 'b-', linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    axes[0, 0].plot(lags, variogram_fake, 'r--', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    axes[0, 0].set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    axes[0, 0].set_ylabel('Variogram', fontweight='bold', fontsize=22)
    axes[0, 0].set_title(f'Variogram (MSE: {metrics["variogram_mse"]:.4f})', fontweight='bold', fontsize=24)
    axes[0, 0].legend(fontsize=18)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    axes[0, 1].plot(lags, connectivity_real, 'b-', linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    axes[0, 1].plot(lags, connectivity_fake, 'r--', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    axes[0, 1].set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    axes[0, 1].set_ylabel('Connectivity', fontweight='bold', fontsize=22)
    axes[0, 1].set_title(f'Connectivity (MSE: {metrics["connectivity_mse"]:.4f})', fontweight='bold', fontsize=24)
    axes[0, 1].legend(fontsize=18)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_centers[1] - bin_centers[0]
    axes[0, 2].bar(bin_centers, hist_real, width=bar_width, alpha=0.6, label='Real', color='blue')
    axes[0, 2].plot(bin_centers, hist_fake, 'r-', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    axes[0, 2].set_xlabel('Value', fontweight='bold', fontsize=22)
    axes[0, 2].set_ylabel('Density', fontweight='bold', fontsize=22)
    axes[0, 2].set_title(f'Histogram (KL: {metrics["hist_kl"]:.4f})', fontweight='bold', fontsize=24)
    axes[0, 2].legend(fontsize=18)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    axes[1, 0].scatter(pca_real[:100, 0], pca_real[:100, 1], alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
                       color='blue')
    axes[1, 0].scatter(pca_fake[:100, 0], pca_fake[:100, 1], alpha=0.6, label='Generated',
                       s=HIGH_RES_CONFIG['marker_size'], color='red')
    axes[1, 0].set_xlabel('PC1', fontweight='bold', fontsize=22)
    axes[1, 0].set_ylabel('PC2', fontweight='bold', fontsize=22)
    axes[1, 0].set_title(f'PCA (Corr: {metrics["pca_correlation"]:.4f})', fontweight='bold', fontsize=24)
    axes[1, 0].legend(fontsize=18)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    axes[1, 1].scatter(mds_real[:100, 0], mds_real[:100, 1], alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
                       color='blue')
    axes[1, 1].scatter(mds_fake[:100, 0], mds_fake[:100, 1], alpha=0.6, label='Generated',
                       s=HIGH_RES_CONFIG['marker_size'], color='red')
    axes[1, 1].set_xlabel('MDS 1', fontweight='bold', fontsize=22)
    axes[1, 1].set_ylabel('MDS 2', fontweight='bold', fontsize=22)
    axes[1, 1].set_title(f'MDS (MMD: {metrics["mds_mmd"]:.4f})', fontweight='bold', fontsize=24)
    axes[1, 1].legend(fontsize=18)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    stats_text = (
        f'Variogram MSE: {metrics["variogram_mse"]:.6f}\n'
        f'Connectivity MSE: {metrics["connectivity_mse"]:.6f}\n'
        f'Histogram KL: {metrics["hist_kl"]:.6f}\n'
        f'PCA Correlation: {metrics["pca_correlation"]:.4f}\n'
        f'MDS MMD: {metrics["mds_mmd"]:.4f}'
    )
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=18, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_base:
        saved_files = save_figure_multiple_formats(fig, save_path_base, HIGH_RES_CONFIG['dpi'])
    else:
        saved_files = []
    plt.close(fig)
    return metrics, saved_files


# ============================================================================
# 5. TRAINING
# ============================================================================

# ----------------------------------------------------------------------------
# 5.1 Callbacks
# ----------------------------------------------------------------------------
class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()


class DistributionPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, test_data, plot_interval=10, save_interval=10):
        super().__init__()
        self.model = model
        self.test_data = test_data
        self.plot_interval = plot_interval
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        show_plot = True
        save_plot = ((epoch + 1) % self.save_interval == 0) or ((epoch + 1) == epochs)
        if show_plot:
            sample_idx = np.random.randint(0, len(self.test_data), 1)
            test_samples = self.test_data[sample_idx]
            z_mean, _ = self.model.encoder.predict(test_samples, verbose=0)
            softmax_out = self.model.decoder_softmax.predict(z_mean, verbose=0)
            discrete_out = np.argmax(softmax_out, axis=-1)
            reconstructions = discrete_out.astype(float) - 1.0
            reconstructions = np.expand_dims(reconstructions, axis=-1)
            latent_samples = analyze_latent_space(self.model.encoder, self.test_data)
            saved_files = plot_histograms_and_densities(
                test_samples, reconstructions, latent_samples,
                epoch + 1, save_interval=self.save_interval
            )
            if save_plot:
                print(f"Saved distribution plots at epoch {epoch + 1}: {len(saved_files)} files")
            else:
                print(f"Displayed distribution plots at epoch {epoch + 1} (not saved)")


class LivePlotCallback(keras.callbacks.Callback):
    def __init__(self, model, data, num_images=5, save_interval=10):
        super().__init__()
        self.model = model
        self.data = data
        self.num_images = num_images
        self.save_interval = save_interval
        self.fig_width = num_images * 4 * HIGH_RES_CONFIG['figsize_multiplier']
        self.fig_height = 4 * HIGH_RES_CONFIG['figsize_multiplier']
        self.fig, self.ax = plt.subplots(1, num_images * 2,
                                         figsize=(self.fig_width, self.fig_height),
                                         dpi=HIGH_RES_CONFIG['dpi'])
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 24
        })
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        indices = np.random.randint(0, self.data.shape[0], self.num_images)
        original_imgs = self.data[indices]
        z_mean, _ = self.model.encoder.predict(original_imgs, verbose=0)
        softmax_out = self.model.decoder_softmax.predict(z_mean, verbose=0)
        discrete_out = np.argmax(softmax_out, axis=-1)
        reconstructed_imgs = discrete_out.astype(float) - 1.0
        reconstructed_imgs = np.expand_dims(reconstructed_imgs, axis=-1)
        for i in range(self.num_images):
            self.ax[i * 2].imshow(original_imgs[i].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            self.ax[i * 2].set_title("Original", fontweight='bold', fontsize=8)
            self.ax[i * 2].axis('off')
            self.ax[i * 2 + 1].imshow(reconstructed_imgs[i].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1,
                                      vmax=1)
            self.ax[i * 2 + 1].set_title(f"Reconstructed", fontweight='bold', fontsize=8)
            self.ax[i * 2 + 1].axis('off')
        self.fig.suptitle(f'Latent Diffusion - Epoch {epoch + 1}: Original vs Reconstructed Samples',
                          fontsize=18, fontweight='bold')
        if ((epoch + 1) % self.save_interval == 0) or ((epoch + 1) == epochs):
            base_filename = f"training_outputs/latent_diffusion_epoch_{epoch + 1:03d}"
            saved_files = save_figure_multiple_formats(self.fig, base_filename, HIGH_RES_CONFIG['dpi'])
            print(f"Saved epoch {epoch + 1} plot: {len(saved_files)} files")
        else:
            print(f"Displayed epoch {epoch + 1} plot (not saved)")
        plt.pause(0.01)
        self.fig.canvas.draw()


class EnhancedMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, diffusion_model, encoder, x_test, metric_interval=metric_interval,
                 save_interval=10, sample_size=FID_SAMPLE_SIZE,
                 log_file='latent_diffusion_training_metrics.txt'):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.encoder = encoder
        self.x_test = x_test
        self.metric_interval = metric_interval
        self.save_interval = save_interval
        self.sample_size = sample_size
        self.log_file = log_file
        self._real_features_iv3 = None
        self._real_features_rc = None
        self._cached_size = None
        self.metrics = {
            'FID_10k': [], 'FRD_10k': [], 'KID_10k': [], 'KRD_10k': [],
            'FID_500': [], 'FRD_500': [], 'KID_500': [], 'KRD_500': []
        }
        self.static_metrics = {
            'Variogram_MSE': [], 'Connectivity_MSE': [], 'Histogram_KL': [],
            'PCA_Correlation': [], 'MDS_MMD': []
        }
        with open(self.log_file, 'w') as f:
            f.write("Epoch\tTotal_Loss\tDiffusion_Loss\tRecon_Loss\tKL_Loss\t"
                    "FID_10k\tFRD_10k\tKID_10k\tKRD_10k\t"
                    "FID_500\tFRD_500\tKID_500\tKRD_500\t"
                    "Variogram_MSE\tConnectivity_MSE\tHistogram_KL\tPCA_Correlation\tMDS_MMD\n")

    def _prepare_real_features(self, size):
        """Extracts real features for exactly `size` samples (10k in batches)."""
        print(f"\n[EnhancedMetricCallback] Pre-computing real features (size={size})...")
        t0 = time.time()
        real_resized = tf.image.resize(np.repeat(self.x_test[:size], 3, axis=3), [75, 75])
        self._real_features_iv3 = f_extraction_iv3.predict(real_resized, batch_size=128, verbose=0)
        self._real_features_rc = f_extraction_rc.predict(self.x_test[:size], batch_size=128, verbose=0)
        del real_resized
        gc.collect()
        self._cached_size = size
        print(f"[EnhancedMetricCallback] Real features cached. Time: {time.time() - t0:.1f}s\n")

    def on_train_begin(self, logs=None):
        size = min(self.sample_size, self.x_test.shape[0])
        self._prepare_real_features(size)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.metric_interval == 0:
            print(f'\n{"=" * 60}')
            print(f'Latent Diffusion (3 facies): Computing metrics at epoch {epoch + 1}')
            print(f'{"=" * 60}')
            MemoryOptimizer.check_memory(threshold_mb=8000)

            t0 = time.time()
            size = min(self.sample_size, self.x_test.shape[0])
            fake_data = self.diffusion_model.sample(num_samples=size, training=False, verbose=False)
            if isinstance(fake_data, tf.Tensor):
                fake_data = fake_data.numpy()
            if np.any(np.isnan(fake_data)):
                fake_data = np.nan_to_num(fake_data, nan=0.0)
            t_sample = time.time() - t0
            print(f"  Sampling: {t_sample:.1f}s")

            static_results = self.compute_static_metrics_corrected(self.x_test[:size], fake_data[:size])
            training_metrics = {
                'total_loss': logs.get('loss', 0),
                'diffusion_loss': logs.get('diffusion_loss', 0),
                'recon_loss': logs.get('reconstruction_loss', 0),
                'kl_loss': logs.get('kl_loss', 0)
            }

            t1 = time.time()
            try:
                if self._real_features_iv3 is None or self._cached_size != size:
                    self._prepare_real_features(size)

                # Ensure exactly FID_N_BATCHES * FID_BATCH_SIZE samples for batched FID/FRD
                n_fid = FID_N_BATCHES * FID_BATCH_SIZE
                n_fid = min(n_fid, len(self._real_features_iv3), len(fake_data))

                # Batch adjustment if needed (fewer than 10k available)
                n_b = FID_N_BATCHES
                b_sz = FID_BATCH_SIZE
                if n_fid < FID_N_BATCHES * FID_BATCH_SIZE:
                    n_b = max(1, n_fid // FID_BATCH_SIZE)
                    if n_b == 0:
                        n_b = 1
                        b_sz = n_fid
                    else:
                        b_sz = n_fid // n_b
                n_fid = n_b * b_sz

                # --- InceptionV3 (FID) and RC (FRD) in feature batches ---
                # We need the fake_data features for the first n_fid
                fake_fid_subset = fake_data[:n_fid]
                fake_resized = tf.image.resize(np.repeat(fake_fid_subset, 3, axis=3), [75, 75])
                act_gen_iv3 = f_extraction_iv3.predict(fake_resized, batch_size=128, verbose=0)
                act_gen_rc = f_extraction_rc.predict(fake_fid_subset, batch_size=128, verbose=0)
                del fake_resized
                gc.collect()

                # FID / FRD batched: 10 batches x 1000 (accumulated statistics)
                fid_10k = frechet_distance_batched(
                    self._real_features_iv3[:n_fid], act_gen_iv3[:n_fid],
                    n_batches=n_b, batch_size=b_sz
                )
                frd_10k = frechet_distance_batched(
                    self._real_features_rc[:n_fid], act_gen_rc[:n_fid],
                    n_batches=n_b, batch_size=b_sz
                )

                # KID / KRD (previous behavior, subset of 500)
                n = min(500, len(self._real_features_iv3), len(act_gen_iv3))
                kid_10k = kernel_distance_stable(self._real_features_iv3[:n], act_gen_iv3[:n],
                                                 num_blocks=50, max_block_size=n)
                krd_10k = kernel_distance_stable(self._real_features_rc[:n], act_gen_rc[:n],
                                                 num_blocks=50, max_block_size=n)

                # 500-sample versions (previous behavior)
                n500 = min(500, len(self._real_features_iv3), len(act_gen_iv3))
                fid_500 = frechet_distance_stable(self._real_features_iv3[:n500], act_gen_iv3[:n500])
                frd_500 = frechet_distance_stable(self._real_features_rc[:n500], act_gen_rc[:n500])
                kid_500 = kernel_distance_stable(self._real_features_iv3[:n500], act_gen_iv3[:n500],
                                                 num_blocks=50, max_block_size=n500)
                krd_500 = kernel_distance_stable(self._real_features_rc[:n500], act_gen_rc[:n500],
                                                 num_blocks=50, max_block_size=n500)

                traditional_results = {
                    'FID_10k': fid_10k, 'FRD_10k': frd_10k,
                    'KID_10k': kid_10k, 'KRD_10k': krd_10k,
                    'FID_500': fid_500, 'FRD_500': frd_500,
                    'KID_500': kid_500, 'KRD_500': krd_500
                }
                del act_gen_iv3, act_gen_rc
                gc.collect()
            except Exception as e:
                print(f"Warning: FID/FRD failed: {e}. Using placeholder values.")
                traditional_results = {
                    'FID_10k': 0, 'FRD_10k': 0, 'KID_10k': 0, 'KRD_10k': 0,
                    'FID_500': 0, 'FRD_500': 0, 'KID_500': 0, 'KRD_500': 0
                }

            t_fid = time.time() - t1
            print(f"  FID/FRD/KID/KRD: {t_fid:.1f}s")
            print(f"  Total callback time: {t_sample + t_fid:.1f}s")

            for key in ['FID_10k', 'FRD_10k', 'KID_10k', 'KRD_10k',
                        'FID_500', 'FRD_500', 'KID_500', 'KRD_500']:
                self.metrics[key].append(traditional_results[key])
            for key in self.static_metrics:
                self.static_metrics[key].append(static_results[key])
            self.save_metrics_to_file(epoch + 1, training_metrics, traditional_results, static_results)
            self.display_metrics(epoch + 1, training_metrics, traditional_results, static_results)
            self.save_numpy_files()
            self.plot_static_metrics(epoch + 1, static_results, save_interval=self.save_interval)

            del fake_data
            gc.collect()

    def compute_static_metrics_corrected(self, real_data, fake_data):
        print("Calculating static geostatistical metrics...")
        n_samples = min(500, len(real_data), len(fake_data))
        real_subset = real_data[:n_samples]
        fake_subset = fake_data[:n_samples]
        metrics = self.compute_all_metrics_corrected(
            real_subset, fake_subset, threshold=0.5, max_lag=25)
        return {
            'Variogram_MSE': metrics['variogram_mse'],
            'Connectivity_MSE': metrics['connectivity_mse'],
            'Histogram_KL': metrics['hist_kl'],
            'PCA_Correlation': metrics['pca_correlation'],
            'MDS_MMD': metrics['mds_mmd']
        }

    def compute_all_metrics_corrected(self, real_data, fake_data, threshold=0.5, max_lag=25):
        metrics = {}
        print("Calculating variograms...")
        lags, variogram_real = StaticMetrics.variogram_2d_optimized(real_data[:100], max_lag=max_lag)
        _, variogram_fake = StaticMetrics.variogram_2d_optimized(fake_data[:100], max_lag=max_lag)
        metrics['variogram_lags'] = lags
        metrics['variogram_real'] = variogram_real
        metrics['variogram_fake'] = variogram_fake
        metrics['variogram_mse'] = np.mean((variogram_real - variogram_fake) ** 2)
        print("Calculating connectivity functions...")
        _, connectivity_real = StaticMetrics.connectivity_function(real_data[:100], threshold, lag_distances=lags)
        _, connectivity_fake = StaticMetrics.connectivity_function(fake_data[:100], threshold, lag_distances=lags)
        metrics['connectivity_real'] = connectivity_real
        metrics['connectivity_fake'] = connectivity_fake
        metrics['connectivity_mse'] = np.mean((connectivity_real - connectivity_fake) ** 2)
        print("Calculating PCA...")
        pca_real, var_ratio_real = StaticMetrics.calculate_pca_features(real_data[:500])
        pca_fake, var_ratio_fake = StaticMetrics.calculate_pca_features(fake_data[:500])
        if pca_real.shape[0] == pca_fake.shape[0]:
            n_components = min(2, pca_real.shape[1], pca_fake.shape[1])
            pca_correlations = []
            for i in range(n_components):
                corr = np.corrcoef(pca_real[:, i], pca_fake[:, i])[0, 1]
                pca_correlations.append(corr)
            metrics['pca_correlation'] = np.mean(pca_correlations)
        else:
            metrics['pca_correlation'] = 0.0
        metrics['pca_real'] = pca_real[:, :2]
        metrics['pca_fake'] = pca_fake[:, :2]
        metrics['pca_var_ratio_real'] = var_ratio_real
        metrics['pca_var_ratio_fake'] = var_ratio_fake
        print("Calculating MDS...")
        mds_real = StaticMetrics.calculate_mds_features(real_data[:500])
        mds_fake = StaticMetrics.calculate_mds_features(fake_data[:500])
        from sklearn.metrics.pairwise import rbf_kernel
        n_samples = min(len(mds_real), len(mds_fake), 100)
        mds_real_small = mds_real[:n_samples]
        mds_fake_small = mds_fake[:n_samples]
        if len(mds_real_small) > 0 and len(mds_fake_small) > 0:
            X = np.vstack([mds_real_small, mds_fake_small])
            pairwise_dists = pdist(X)
            gamma = 1.0 / (2.0 * np.median(pairwise_dists) ** 2) if len(pairwise_dists) > 0 else 1.0
            K_XX = rbf_kernel(mds_real_small, mds_real_small, gamma=gamma)
            K_YY = rbf_kernel(mds_fake_small, mds_fake_small, gamma=gamma)
            K_XY = rbf_kernel(mds_real_small, mds_fake_small, gamma=gamma)
            mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
            metrics['mds_mmd'] = np.sqrt(max(mmd_squared, 0))
        else:
            metrics['mds_mmd'] = 0.0
        metrics['mds_real'] = mds_real
        metrics['mds_fake'] = mds_fake
        print("Calculating histogram statistics (CORRECTED)...")
        real_flat = real_data.flatten()
        fake_flat = fake_data.flatten()
        all_data = np.concatenate([real_flat, fake_flat])
        data_min, data_max = np.min(all_data), np.max(all_data)
        hist_real, bin_edges = np.histogram(real_flat, bins=100, range=(data_min, data_max), density=False)
        hist_fake, _ = np.histogram(fake_flat, bins=100, range=(data_min, data_max), density=False)
        hist_real = hist_real.astype(float) / np.sum(hist_real)
        hist_fake = hist_fake.astype(float) / np.sum(hist_fake)
        epsilon = 1e-10
        hist_real = np.clip(hist_real, epsilon, 1)
        hist_fake = np.clip(hist_fake, epsilon, 1)
        kl_divergence = np.sum(hist_real * np.log(hist_real / hist_fake))
        metrics['hist_kl'] = kl_divergence
        return metrics

    def plot_static_metrics(self, epoch, static_metrics, save_interval=10):
        should_save = ((epoch % save_interval == 0) or (epoch == epochs))
        figsize = (16 * HIGH_RES_CONFIG['figsize_multiplier'],
                   10 * HIGH_RES_CONFIG['figsize_multiplier'])
        fig, axes = plt.subplots(3, 2, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
        plt.rcParams.update({'font.size': 20, 'axes.titlesize': 24})
        axes[0, 0].bar(range(len(static_metrics)), list(static_metrics.values()),
                       edgecolor='black', linewidth=1.5)
        axes[0, 0].set_xticks(range(len(static_metrics)))
        axes[0, 0].set_xticklabels(list(static_metrics.keys()), rotation=45, ha='right', fontsize=18)
        axes[0, 0].set_ylabel('Metric Value', fontweight='bold', fontsize=22)
        axes[0, 0].set_title(f'Static Metrics - Epoch {epoch}', fontweight='bold', fontsize=24)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        static_metrics_list = list(self.static_metrics.items())
        for i, (metric_name, values) in enumerate(static_metrics_list):
            if values:
                if i < 2:
                    row, col = 1, i
                elif i < 4:
                    row, col = 2, i - 2
                else:
                    row, col = 2, 1
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].plot(range(len(values)), values, marker='o',
                                        linewidth=HIGH_RES_CONFIG['line_width'])
                    axes[row, col].set_xlabel('Epoch', fontweight='bold', fontsize=22)
                    axes[row, col].set_ylabel(metric_name, fontweight='bold', fontsize=22)
                    axes[row, col].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
                    axes[row, col].set_title(f'{metric_name} Trend', fontweight='bold', fontsize=24)
        plt.tight_layout()
        if should_save:
            base_filename = f"training_outputs/latent_diffusion_static_metrics_epoch_{epoch:03d}"
            saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
            print(f"Saved static metrics plot at epoch {epoch}: {len(saved_files)} files")
        else:
            plt.pause(0.01)
            saved_files = []
        plt.close(fig)
        return saved_files

    def save_metrics_to_file(self, epoch, training_metrics, traditional_metrics, static_metrics):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch}\t"
                    f"{training_metrics['total_loss']:.6f}\t"
                    f"{training_metrics['diffusion_loss']:.6f}\t"
                    f"{training_metrics['recon_loss']:.6f}\t"
                    f"{training_metrics['kl_loss']:.6f}\t"
                    f"{traditional_metrics['FID_10k']:.4f}\t"
                    f"{traditional_metrics['FRD_10k']:.4f}\t"
                    f"{traditional_metrics['KID_10k']:.4f}\t"
                    f"{traditional_metrics['KRD_10k']:.4f}\t"
                    f"{traditional_metrics['FID_500']:.4f}\t"
                    f"{traditional_metrics['FRD_500']:.4f}\t"
                    f"{traditional_metrics['KID_500']:.4f}\t"
                    f"{traditional_metrics['KRD_500']:.4f}\t"
                    f"{static_metrics['Variogram_MSE']:.4f}\t"
                    f"{static_metrics['Connectivity_MSE']:.4f}\t"
                    f"{static_metrics['Histogram_KL']:.4f}\t"
                    f"{static_metrics['PCA_Correlation']:.4f}\t"
                    f"{static_metrics['MDS_MMD']:.4f}\n")

    def display_metrics(self, epoch, training_metrics, traditional_metrics, static_metrics):
        current_kl_weight = float(self.diffusion_model.kl_weight.numpy())
        output = (
            f"\nLatent Diffusion - Epoch {epoch} Metrics:\n"
            f"{'=' * 40}\n"
            f"Training Losses:\n"
            f"  Total Loss: {training_metrics['total_loss']:.4f}\n"
            f"  Diffusion Loss: {training_metrics['diffusion_loss']:.4f}\n"
            f"  Recon (CE) Loss: {training_metrics['recon_loss']:.4f}\n"
            f"  KL Loss: {training_metrics['kl_loss']:.4f}\n"
            f"  KL Weight (current): {current_kl_weight:.8f}\n\n"
            f"Traditional Metrics (10 batches x 1000 for FID/FRD):\n"
            f"  FID(10k): {traditional_metrics['FID_10k']:.2f}\n"
            f"  FRD(10k): {traditional_metrics['FRD_10k']:.2f}\n"
            f"  KID(10k): {traditional_metrics['KID_10k']:.4f}\n"
            f"  KRD(10k): {traditional_metrics['KRD_10k']:.4f}\n"
            f"  FID(500): {traditional_metrics['FID_500']:.2f}\n"
            f"  FRD(500): {traditional_metrics['FRD_500']:.2f}\n"
            f"  KID(500): {traditional_metrics['KID_500']:.4f}\n"
            f"  KRD(500): {traditional_metrics['KRD_500']:.4f}\n\n"
            f"Static Geostatistical Metrics:\n"
            f"  Variogram MSE: {static_metrics['Variogram_MSE']:.4f}\n"
            f"  Connectivity MSE: {static_metrics['Connectivity_MSE']:.4f}\n"
            f"  Histogram JS: {static_metrics['Histogram_KL']:.4f}\n"
            f"  PCA Correlation: {static_metrics['PCA_Correlation']:.4f}\n"
            f"  MDS MMD: {static_metrics['MDS_MMD']:.4f}\n"
            f"{'=' * 40}"
        )
        print(output)

    def save_numpy_files(self):
        os.makedirs('training_outputs', exist_ok=True)
        for key, values in self.metrics.items():
            if values:
                np.save(f'training_outputs/latent_diffusion_{key}.npy', values)
        for key, values in self.static_metrics.items():
            if values:
                np.save(f'training_outputs/latent_diffusion_{key}.npy', values)
        self.save_comprehensive_csv()

    def save_comprehensive_csv(self):
        all_metrics = {}
        for key, values in self.metrics.items():
            if values:
                all_metrics[key] = values
        for key, values in self.static_metrics.items():
            if values:
                all_metrics[key] = values
        df = pd.DataFrame(all_metrics)
        if not df.empty:
            df.index = [(i + 1) * self.metric_interval for i in range(len(df))]
            df.index.name = 'Epoch'
            csv_path = 'training_outputs/latent_diffusion_all_metrics.csv'
            df.to_csv(csv_path)
            print(f"Comprehensive metrics saved to CSV: {csv_path}")


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.total_losses = []
        self.diffusion_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.val_total_losses = []
        self.val_diffusion_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.total_losses.append(logs.get('loss', 0))
        self.diffusion_losses.append(logs.get('diffusion_loss', 0))
        self.recon_losses.append(logs.get('reconstruction_loss', 0))
        self.kl_losses.append(logs.get('kl_loss', 0))
        self.val_total_losses.append(logs.get('val_loss', 0))
        self.val_diffusion_losses.append(logs.get('val_diffusion_loss', 0))
        self.val_recon_losses.append(logs.get('val_reconstruction_loss', 0))
        self.val_kl_losses.append(logs.get('val_kl_loss', 0))


class AdaptiveKLWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, target_kl=50.0, tolerance=0.3,
                 initial_kl_weight=0.0001, min_kl_weight=1e-7, max_kl_weight=0.005,
                 adjustment_factor=0.10, warmup_epochs=5, log_every=1):
        super().__init__()
        self.model = model
        self.target_kl = target_kl
        self.target_kl_min = target_kl * (1.0 - tolerance)
        self.target_kl_max = target_kl * (1.0 + tolerance)
        self.initial_kl_weight = initial_kl_weight
        self.min_kl_weight = min_kl_weight
        self.max_kl_weight = max_kl_weight
        self.adjustment_factor = adjustment_factor
        self.warmup_epochs = warmup_epochs
        self.log_every = log_every
        self.history = {'epoch': [], 'kl_weight': [], 'kl_loss': [], 'action': []}

    def on_train_begin(self, logs=None):
        self.model.kl_weight.assign(float(self.initial_kl_weight))
        print(f"\n[AdaptiveKL] Configured:")
        print(f"  initial kl_weight : {self.initial_kl_weight:.8f}")
        print(f"  target_kl         : {self.target_kl} (band [{self.target_kl_min:.2f}, {self.target_kl_max:.2f}])")
        print(f"  range kl_weight   : [{self.min_kl_weight}, {self.max_kl_weight}]")
        print(f"  adjustment_factor : {self.adjustment_factor} (±{self.adjustment_factor*100:.0f}% per epoch)")
        print(f"  warmup_epochs     : {self.warmup_epochs}\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        kl_loss = logs.get('kl_loss', None)
        if kl_loss is None or not np.isfinite(kl_loss):
            return
        current_weight = float(self.model.kl_weight.numpy())
        if epoch < self.warmup_epochs:
            new_weight = current_weight
            action = f"warmup ({self.warmup_epochs - epoch - 1} remaining)"
        else:
            if kl_loss < self.target_kl_min:
                new_weight = current_weight * (1.0 - self.adjustment_factor)
                action = "↓ (KL below target → reduce penalty)"
            elif kl_loss > self.target_kl_max:
                new_weight = current_weight * (1.0 + self.adjustment_factor)
                action = "↑ (KL above target → increase penalty)"
            else:
                new_weight = current_weight
                action = "= (within target band)"
            new_weight = max(self.min_kl_weight, min(self.max_kl_weight, new_weight))
        self.model.kl_weight.assign(float(new_weight))
        self.history['epoch'].append(epoch + 1)
        self.history['kl_weight'].append(new_weight)
        self.history['kl_loss'].append(float(kl_loss))
        self.history['action'].append(action)
        if (epoch + 1) % self.log_every == 0:
            print(f"[AdaptiveKL] Epoch {epoch+1:03d}: kl_loss={kl_loss:8.3f} "
                  f"kl_weight={current_weight:.8f} -> {new_weight:.8f}  {action}")


# ----------------------------------------------------------------------------
# 5.2 Training setup and execution
# ----------------------------------------------------------------------------
os.makedirs('training_outputs', exist_ok=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype('float32'), x_train.astype('float32'))).shuffle(
    10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test.astype('float32'), x_test.astype('float32'))).batch(batch_size)

SAVE_INTERVAL = 10

loss_history_callback = LossHistoryCallback()
live_plot_callback = LivePlotCallback(diffusion_model, x_test, num_images=5, save_interval=SAVE_INTERVAL)
distribution_callback = DistributionPlotCallback(diffusion_model, x_test, plot_interval=10, save_interval=SAVE_INTERVAL)

enhanced_metric_callback = EnhancedMetricCallback(
    diffusion_model, encoder, x_test,
    metric_interval=SAVE_INTERVAL, save_interval=SAVE_INTERVAL,
    sample_size=FID_SAMPLE_SIZE,
    log_file='latent_diffusion_training_metrics.txt',
)

adaptive_kl_callback = AdaptiveKLWeightCallback(
    diffusion_model,
    target_kl=ADAPTIVE_KL_TARGET,
    tolerance=ADAPTIVE_KL_TOLERANCE,
    initial_kl_weight=kl_weight,
    min_kl_weight=ADAPTIVE_KL_MIN,
    max_kl_weight=ADAPTIVE_KL_MAX,
    adjustment_factor=ADAPTIVE_KL_ADJ_FACTOR,
    warmup_epochs=ADAPTIVE_KL_WARMUP_EPOCHS,
    log_every=1,
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'training_outputs/latent_diffusion_best_weights.h5',
    save_best_only=True, monitor='val_loss',
    save_weights_only=True, mode='min'
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=100, restore_best_weights=True
)

tensorboard = keras.callbacks.TensorBoard(
    log_dir='training_outputs/latent_diffusion_logs', update_freq='epoch'
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

start_time = time.time()

print("\n" + "=" * 60)
print("STARTING LATENT DIFFUSION TRAINING")
print("=" * 60)
print(f"Encoder: VAE (z_mean + z_log_var, reparameterization)")
print(f"Denoiser: REAL 1D UNET")
print(f"Decoder (trained): Softmax (num_classes={num_classes})")
print(f"Decoder (derived): Continuous linear (1 channel, for assimilation)")
print(f"Image generation: discrete (softmax + argmax)")
print(f"KL weight: ADAPTIVE (initial={kl_weight}, target={ADAPTIVE_KL_TARGET})")
print(f"FID/FRD sampling: {FID_N_BATCHES} batches x {FID_BATCH_SIZE} samples ({FID_N_BATCHES * FID_BATCH_SIZE} total)")
print(f"Sample chunk size: {SAMPLE_CHUNK_SIZE}")
print(f"FID epsilon: {FID_EPS}")
print(f"Traditional metrics: FID/FRD batched (10x1000) + KID/KRD (same as before)")
print(f"  - InceptionV3 input_shape         : {f_extraction_iv3.input_shape}")
print(f"  - ReservoirClassifier input_shape : {f_extraction_rc.input_shape}")
print("=" * 60)

history = diffusion_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[
        live_plot_callback,
        enhanced_metric_callback,
        distribution_callback,
        loss_history_callback,
        adaptive_kl_callback,
        model_checkpoint,
        early_stopping,
        tensorboard,
        reduce_lr
    ],
    verbose=1
)

plt.ioff()

end_time = time.time()
execution_time = end_time - start_time

print(f"\n[INFO] Training completed in {execution_time / 3600:.2f} hours")


# ============================================================================
# 6. EVALUATION & PLOTS
# ============================================================================

print("\n" + "=" * 60)
print("TRANSFERRING WEIGHTS: softmax → continuous")
print("=" * 60)

decoder_continuous = transfer_softmax_to_continuous(
    decoder_softmax, decoder_continuous,
    class_values=(-1.0, 0.0, 1.0)
)
print("Transfer completed.")

# Sanity check
transfer_agreement = sanity_check_decoders(
    encoder, decoder_softmax, decoder_continuous, x_test, n=20
)

print("\n" + "=" * 60)
print("FINAL ANALYSIS AND SAVING")
print("=" * 60)

# Generate final samples
test_samples = x_test[:5]
z_mean_test, _ = diffusion_model.encoder.predict(test_samples, verbose=0)
softmax_out_test = diffusion_model.decoder_softmax.predict(z_mean_test, verbose=0)
discrete_out_test = np.argmax(softmax_out_test, axis=-1)
reconstructions = discrete_out_test.astype(float) - 1.0
reconstructions = np.expand_dims(reconstructions, axis=-1)

latent_samples = analyze_latent_space(diffusion_model.encoder, x_test, num_samples=1000)

# Plot final distributions
print("Saving final distribution plots...")
saved_dist_files = plot_histograms_and_densities(test_samples, reconstructions, latent_samples)
print(f"Saved final distribution plots: {len(saved_dist_files)} files")

# Latent space visualization
figsize = (14 * HIGH_RES_CONFIG['figsize_multiplier'], 7 * HIGH_RES_CONFIG['figsize_multiplier'])
fig = plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
z_samples, _ = diffusion_model.encoder.predict(x_test[:1000], verbose=0)
pca = PCA(n_components=2)
z_2d = pca.fit_transform(z_samples)
plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=HIGH_RES_CONFIG['marker_size'])
plt.title('2D Projection of Latent Space (PCA)', fontsize=24, fontweight='bold')
plt.xlabel('PCA Component 1', fontweight='bold', fontsize=22)
plt.ylabel('PCA Component 2', fontweight='bold', fontsize=22)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
save_figure_multiple_formats(fig, "training_outputs/latent_diffusion_latent_space_2d", HIGH_RES_CONFIG['dpi'])
plt.close(fig)

# Static metrics comparison + final traditional metrics
static_metrics_results, saved_static_files = plot_static_metrics_comparison(
    x_test, diffusion_model,
    save_path_base='training_outputs/latent_diffusion_static_metrics_comparison',
    sample_size=FID_SAMPLE_SIZE,
)


# ============================================================================
# 7. SAVING
# ============================================================================

# ----------------------------------------------------------------------------
# 7.1 Loss history files
# ----------------------------------------------------------------------------
with open('training_outputs/latent_diffusion_loss_history.txt', 'w') as f:
    f.write("Epoch\tTotal_Loss\tDiffusion_Loss\tRecon_Loss\tKL_Loss\t"
            "Val_Total_Loss\tVal_Diffusion_Loss\tVal_Recon_Loss\tVal_KL_Loss\n")
    for i in range(len(loss_history_callback.total_losses)):
        f.write(f"{i + 1}\t"
                f"{loss_history_callback.total_losses[i]:.6f}\t"
                f"{loss_history_callback.diffusion_losses[i]:.6f}\t"
                f"{loss_history_callback.recon_losses[i]:.6f}\t"
                f"{loss_history_callback.kl_losses[i]:.6f}\t"
                f"{loss_history_callback.val_total_losses[i]:.6f}\t"
                f"{loss_history_callback.val_diffusion_losses[i]:.6f}\t"
                f"{loss_history_callback.val_recon_losses[i]:.6f}\t"
                f"{loss_history_callback.val_kl_losses[i]:.6f}\n")

if adaptive_kl_callback.history['epoch']:
    adaptive_df = pd.DataFrame({
        'Epoch': adaptive_kl_callback.history['epoch'],
        'KL_Loss': adaptive_kl_callback.history['kl_loss'],
        'KL_Weight': adaptive_kl_callback.history['kl_weight'],
        'Action': adaptive_kl_callback.history['action'],
    })
    adaptive_df.to_csv('training_outputs/latent_diffusion_adaptive_kl_history.csv', index=False)
    print(f"Adaptive KL history saved to training_outputs/latent_diffusion_adaptive_kl_history.csv")

# ----------------------------------------------------------------------------
# 7.2 Model weights
# ----------------------------------------------------------------------------
print("Saving model weights...")
encoder.save_weights('training_outputs/latent_diffusion_encoder_weights.h5')
decoder_softmax.save_weights('training_outputs/latent_diffusion_decoder_softmax_weights.h5')
decoder_continuous.save_weights('training_outputs/latent_diffusion_decoder_continuous_weights.h5')
# Compatibility with the old "Code 1" (points to softmax, avoids breaking old imports)
decoder_softmax.save_weights('training_outputs/latent_diffusion_decoder_weights.h5')
unet.save_weights('training_outputs/latent_diffusion_unet_weights.h5')
diffusion_model.save_weights('training_outputs/latent_diffusion_model_weights.h5')

# ----------------------------------------------------------------------------
# 7.3 Full models
# ----------------------------------------------------------------------------
print("Saving models...")
encoder.save('training_outputs/latent_diffusion_encoder_model.h5')
decoder_softmax.save('training_outputs/latent_diffusion_decoder_softmax_model.h5')
decoder_continuous.save('training_outputs/latent_diffusion_decoder_continuous_model.h5')
# Compatibility alias
decoder_softmax.save('training_outputs/latent_diffusion_decoder_model.h5')
unet.save('training_outputs/latent_diffusion_unet_model.h5')

print("\nDecoders saved:")
print("  - latent_diffusion_decoder_softmax_model.h5       (softmax, (48,48,3)) — visualization")
print("  - latent_diffusion_decoder_continuous_model.h5    (linear, (48,48,1)) — USE IN run_assimilation")
print("  - latent_diffusion_decoder_model.h5               (alias = softmax; legacy)")

# ----------------------------------------------------------------------------
# 7.4 Final report
# ----------------------------------------------------------------------------
training_params = {
    'Dataset': name,
    'Encoder type': 'VAE (z_mean + z_log_var + reparameterization)',
    'Decoder (trained)': f'Softmax (num_classes={num_classes})',
    'Decoder (derived)': 'Continuous (linear, (48,48,1)) — from softmax',
    'Latent dimension (zdim)': zdim,
    'Configured epochs': epochs,
    'Batch size': batch_size,
    'Initial learning rate': initial_lr,
    'Beta (reconstruction weight)': beta,
    'Gamma (perceptual weight)': gamma,
    'Alpha (LeakyReLU)': alpha,
    'KL weight (mode)': 'ADAPTIVE',
    'KL weight (initial)': kl_weight,
    'KL target': ADAPTIVE_KL_TARGET,
    'KL tolerance': f'±{ADAPTIVE_KL_TOLERANCE*100:.0f}%',
    'KL weight (min/max)': f'[{ADAPTIVE_KL_MIN}, {ADAPTIVE_KL_MAX}]',
    'KL adjustment factor': ADAPTIVE_KL_ADJ_FACTOR,
    'KL warmup epochs': ADAPTIVE_KL_WARMUP_EPOCHS,
    'Image dimensions (ni x nj x nz)': f'{ni} x {nj} x {nz}',
    'Number of classes': num_classes,
    'Diffusion timesteps': num_timesteps,
    'Beta start': beta_start,
    'Beta end': beta_end,
    'Training samples': len(x_train),
    'Test samples': len(x_test),
    'Model': 'Latent Diffusion with VAE encoder + 1D UNet denoiser + softmax decoder (derived continuous)',
    'UNet structure': 'Encoder(3) + Bottleneck(2x) + Decoder(3) with skip connections',
    'UNet base filters': 64,
    'UNet time emb dim': 256,
    'Optimizer': 'Adam (beta1=0.5, clipnorm=1.0)',
    'Early Stopping': 'Yes (patience=100, monitor=val_loss)',
    'LR Scheduler': 'ReduceLROnPlateau (factor=0.5, patience=5)',
    'Image generation': 'Discrete (softmax + argmax)',
    'Metric interval': metric_interval,
    'FID/FRD sampling': f'{FID_N_BATCHES} batches x {FID_BATCH_SIZE} samples ({FID_N_BATCHES * FID_BATCH_SIZE} total)',
    'Sample chunk size': SAMPLE_CHUNK_SIZE,
    'FID epsilon': FID_EPS,
    'InceptionV3 input_shape': str(f_extraction_iv3.input_shape),
    'ReservoirClassifier input_shape': str(f_extraction_rc.input_shape),
    'FID/FRD technique': f'{FID_N_BATCHES} batches x {FID_BATCH_SIZE} (accumulated mean/cov) + scipy sqrtm (stable)',
    'KID/KRD technique': 'MMD^2 with cubic polynomial kernel (n<=500)',
    'Decoder softmax -> continuous transfer': f'agreement={transfer_agreement:.4f}',
}

report_path = os.path.join('training_outputs', 'latent_diffusion_final_report.txt')
generate_final_report(
    filepath=report_path,
    params=training_params,
    execution_time=execution_time,
    loss_history=loss_history_callback,
    metrics_history=enhanced_metric_callback,
    static_metrics_results=static_metrics_results,
    adaptive_kl_history=adaptive_kl_callback.history,
    decoder_transfer_agreement=transfer_agreement,
)

# ----------------------------------------------------------------------------
# 7.5 Final summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Execution time: {execution_time / 3600:.2f} hours")
if len(loss_history_callback.total_losses) > 0:
    print(f"Total epochs trained: {len(loss_history_callback.total_losses)}")
    print(f"Final Training Total Loss: {loss_history_callback.total_losses[-1]:.4f}")
    print(f"Final Validation Total Loss: {loss_history_callback.val_total_losses[-1]:.4f}")
print("=" * 60)
print(f"\nFinal complete report saved to: {report_path}")
print("=" * 60)
