import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.models import load_model
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import time
import os
import seaborn as sns
import pandas as pd
import pickle
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from scipy.stats import entropy
from sklearn.metrics import pairwise_kernels
from sklearn.impute import SimpleImputer
import warnings
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

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

# Fully clear TensorFlow session
tf.keras.backend.clear_session()
gc.collect()

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Available GPUs: {len(gpus)}")
    except RuntimeError as e:
        print(e)

# Configuration
zdim = 512
epochs = 150  # 150
batch_size = 32
initial_lr = 0.0001
beta = 0.2  # KL loss weight (used as reconstruction loss weight in diffusion)
gamma = 0.1  # Not used in diffusion but kept for compatibility
alpha = 0.2  # LeakyReLU alpha
d_reg = 5
metric_interval = 10
name = 'unisim_ii_cropped'

# Latent Diffusion specific parameters
num_timesteps = 1000  # Number of diffusion steps
beta_start = 0.0001
beta_end = 0.02

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

# Load dataset
if name == 'channel2d_large':
    X_train = np.load('datasets/Two_facies_channel_2d.npy').astype('float32')

elif name == '3facies_2d_80k':
    X_train = np.load('datasets/Three_facies_2d.npy').astype('float32')

elif name == 'unisim_ii_cropped':
    X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')
    X_train[X_train < 1] = 1
    X_train = np.log(X_train)
    X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
    X_train = X_train[..., np.newaxis]
    X_train = X_train.astype('float32')

noise = np.random.normal(0, 1, (5 * 5, zdim)).astype('float32')
ni = X_train.shape[1]
nj = X_train.shape[2]
nz = X_train.shape[3]

# Split data into train and test
train_size = int(0.8 * len(X_train))
x_train = X_train[:train_size].astype('float32')
x_test = X_train[train_size:].astype('float32')

latent_dim = zdim


class MemoryOptimizer:
    """Class for memory optimization during training."""

    @staticmethod
    def memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 ** 2

    @staticmethod
    def clear_memory():
        tf.keras.backend.clear_session()
        gc.collect()

    @staticmethod
    def batch_generator(data, batch_size):
        n_samples = len(data)
        for i in range(0, n_samples, batch_size):
            yield data[i:i + batch_size]

    @staticmethod
    def check_memory(threshold_mb=8000):
        current_memory = MemoryOptimizer.memory_usage()
        if current_memory > threshold_mb:
            MemoryOptimizer.clear_memory()
            print(f"Memory cleared. Current usage: {current_memory:.2f} MB")
            return True
        return False


class StaticMetrics:
    """Class for calculating static geostatistical metrics."""

    @staticmethod
    def variogram_2d(data, lag_distances=None, max_lag=None):
        if len(data.shape) == 4:
            data = data[..., 0]
        n_samples, height, width = data.shape[:3]
        if lag_distances is None:
            if max_lag is None:
                max_lag = min(height, width) // 2
            lag_distances = np.arange(1, max_lag + 1)
        variogram = np.zeros(len(lag_distances))
        counts = np.zeros(len(lag_distances))
        for i in range(height):
            for j in range(width):
                for k, lag in enumerate(lag_distances):
                    if j + lag < width:
                        for sample in range(n_samples):
                            diff = data[sample, i, j] - data[sample, i, j + lag]
                            variogram[k] += diff ** 2
                            counts[k] += 1
                    if i + lag < height:
                        for sample in range(n_samples):
                            diff = data[sample, i, j] - data[sample, i + lag, j]
                            variogram[k] += diff ** 2
                            counts[k] += 1
        variogram = variogram / (2 * counts)
        variogram[counts == 0] = 0
        return lag_distances, variogram

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
        binary_data = (data > threshold).astype(float)
        if lag_distances is None:
            max_lag = min(height, width) // 2
            lag_distances = np.arange(1, max_lag + 1)
        connectivity = np.zeros(len(lag_distances))
        for k, lag in enumerate(lag_distances):
            connected_pairs = 0
            total_pairs = 0
            for sample in range(n_samples):
                for i in range(height):
                    for j in range(width):
                        if binary_data[sample, i, j] == 1:
                            if j + lag < width and binary_data[sample, i, j + lag] == 1:
                                connected_pairs += 1
                            total_pairs += 1
                            if i + lag < height and binary_data[sample, i + lag, j] == 1:
                                connected_pairs += 1
                            total_pairs += 1
            if total_pairs > 0:
                connectivity[k] = connected_pairs / total_pairs
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

    @staticmethod
    def calculate_histogram_stats(data, bins=100):
        if len(data.shape) == 4:
            data = data[..., 0].flatten()
        else:
            data = data.flatten()
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        mean = np.mean(data)
        std = np.std(data)
        skew = np.mean(((data - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((data - mean) / std) ** 4) if std > 0 else 0
        return {
            'histogram': hist,
            'bin_edges': bin_edges,
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurtosis
        }

    @staticmethod
    def compute_all_metrics(real_data, fake_data, threshold=0.5, max_lag=25):
        metrics = {}
        print("Calculating variograms...")
        lags, variogram_real = StaticMetrics.variogram_2d_optimized(real_data[:100], max_lag=max_lag)
        _, variogram_fake = StaticMetrics.variogram_2d_optimized(fake_data[:100], max_lag=max_lag)
        metrics['variogram_lags'] = lags
        metrics['variogram_real'] = variogram_real
        metrics['variogram_fake'] = variogram_fake
        metrics['variogram_mse'] = np.mean((variogram_real - variogram_fake) ** 2)
        print("Calculating connectivity functions...")
        _, connectivity_real = StaticMetrics.connectivity_function(
            real_data[:100], threshold, lag_distances=lags
        )
        _, connectivity_fake = StaticMetrics.connectivity_function(
            fake_data[:100], threshold, lag_distances=lags
        )
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
        mean = np.mean(real_flat)
        std = np.std(real_flat)
        skew = np.mean(((real_flat - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((real_flat - mean) / std) ** 4) if std > 0 else 0
        metrics['hist_stats_real'] = {
            'histogram': hist_real,
            'bin_edges': bin_edges,
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurtosis
        }
        fake_mean = np.mean(fake_flat)
        fake_std = np.std(fake_flat)
        fake_skew = np.mean(((fake_flat - fake_mean) / fake_std) ** 3) if fake_std > 0 else 0
        fake_kurtosis = np.mean(((fake_flat - fake_mean) / fake_std) ** 4) if fake_std > 0 else 0
        metrics['hist_stats_fake'] = {
            'histogram': hist_fake,
            'bin_edges': bin_edges,
            'mean': fake_mean,
            'std': fake_std,
            'skewness': fake_skew,
            'kurtosis': fake_kurtosis
        }
        metrics['hist_kl'] = kl_divergence
        return metrics


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


def get_feature_extractors():
    iv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))
    f_extraction_iv3 = Model(inputs=iv3.input, outputs=iv3.layers[-1].output)
    RC_full = load_model('Reservoir_classifier.h5')
    f_extraction_rc = Model(inputs=RC_full.input,
                            outputs=layers.GlobalAveragePooling2D()(RC_full.layers[-5].output))
    return f_extraction_iv3, f_extraction_rc


f_extraction_iv3, f_extraction_rc = get_feature_extractors()


# ==============================================
# LATENT DIFFUSION MODEL ARCHITECTURE
# ==============================================

def build_encoder(input_shape, latent_dim):
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
    latent_output = layers.Dense(latent_dim, name="latent_output", dtype=tf.float32)(x)
    return keras.Model(inputs, latent_output, name="encoder")


encoder = build_encoder((ni, nj, nz), latent_dim)


def build_decoder(latent_dim, output_shape):
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
    outputs = layers.Conv2D(output_shape[-1], 3, activation="tanh", padding="same", dtype=tf.float32)(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


decoder = build_decoder(latent_dim, (ni, nj, nz))


def build_unet_model(latent_dim, time_emb_dim=256):
    def sinusoidal_embedding(x):
        half_dim = time_emb_dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        x_expanded = tf.cast(x, tf.float32)
        emb = x_expanded * emb
        sin_emb = tf.sin(emb)
        cos_emb = tf.cos(emb)
        full_emb = tf.concat([sin_emb, cos_emb], axis=1)
        if time_emb_dim % 2 == 1:
            full_emb = tf.pad(full_emb, [[0, 0], [0, 1]])
        return full_emb

    latent_input = keras.Input(shape=(latent_dim,), dtype=tf.float32)
    time_input = keras.Input(shape=(1,), dtype=tf.float32)
    time_emb = sinusoidal_embedding(time_input)
    time_emb = layers.Dense(time_emb_dim, dtype=tf.float32)(time_emb)
    time_emb = layers.Activation("swish")(time_emb)
    time_emb = layers.Dense(time_emb_dim, dtype=tf.float32)(time_emb)
    x = layers.Dense(time_emb_dim, dtype=tf.float32)(latent_input)
    x = layers.Add()([x, time_emb])
    x = layers.Dense(1024, dtype=tf.float32)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, dtype=tf.float32)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(512, dtype=tf.float32)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(latent_dim, dtype=tf.float32)(x)
    return keras.Model([latent_input, time_input], output, name="unet")


unet = build_unet_model(latent_dim)


class DiffusionModel(keras.Model):
    """Latent Diffusion Model for generation."""

    def __init__(self, encoder, decoder, unet, num_timesteps=1000,
                 beta_start=0.0001, beta_end=0.02, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.unet = unet
        self.num_timesteps = num_timesteps
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

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.diffusion_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=initial_lr, beta_1=0.5, clipnorm=1.0)

    def sample_timesteps(self, batch_size):
        return tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.num_timesteps,
            dtype=tf.int32
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start), dtype=tf.float32)
        sqrt_alphas_cumprod_t = tf.gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_alphas_cumprod_t = tf.reshape(sqrt_alphas_cumprod_t, [-1] + [1] * (len(x_start.shape) - 1))
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(sqrt_one_minus_alphas_cumprod_t,
                                                     [-1] + [1] * (len(x_start.shape) - 1))
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x, t, t_index):
        betas_t = tf.gather(self.betas, t)
        sqrt_one_minus_alphas_cumprod_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        sqrt_recip_alphas_t = tf.gather(self.sqrt_recip_alphas, t)
        betas_t = tf.reshape(betas_t, [-1] + [1] * (len(x.shape) - 1))
        sqrt_one_minus_alphas_cumprod_t = tf.reshape(sqrt_one_minus_alphas_cumprod_t, [-1] + [1] * (len(x.shape) - 1))
        sqrt_recip_alphas_t = tf.reshape(sqrt_recip_alphas_t, [-1] + [1] * (len(x.shape) - 1))
        t_batch = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        model_output = self.unet([x, t_batch], training=False)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = tf.gather(self.posterior_variance, t)
            posterior_variance_t = tf.reshape(posterior_variance_t, [-1] + [1] * (len(x.shape) - 1))
            noise = tf.random.normal(shape=tf.shape(x), dtype=tf.float32)
            return model_mean + tf.sqrt(posterior_variance_t) * noise

    def sample(self, num_samples, training=False):
        print(f"\nGenerating {num_samples} samples...")
        x = tf.random.normal((num_samples, latent_dim), dtype=tf.float32)
        for i in range(self.num_timesteps - 1, -1, -1):
            t = tf.ones((num_samples,), dtype=tf.int32) * i
            x = self.p_sample(x, t, i)
            if i % 100 == 0 or i < 10:
                print(f"  Timestep {i + 1}/{self.num_timesteps}")
        print("Decoding latents to images...")
        generated_images = self.decoder(x, training=training)
        if isinstance(generated_images, tf.Tensor):
            generated_images = generated_images.numpy()
        generated_images = np.clip(generated_images, -1.0, 1.0)
        img_mean = np.mean(np.abs(generated_images))
        print(f"Sample generation complete. Mean absolute value: {img_mean:.4f}")
        print(f"Range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
        return generated_images

    def train_step(self, data):
        if isinstance(data, tuple):
            images = tf.cast(data[0], tf.float32)
        else:
            images = tf.cast(data, tf.float32)
        batch_size = tf.shape(images)[0]
        with tf.GradientTape() as tape:
            latents = self.encoder(images, training=True)
            t = self.sample_timesteps(batch_size)
            noise = tf.random.normal(tf.shape(latents), dtype=tf.float32)
            noisy_latents = self.q_sample(latents, t, noise)
            t_reshaped = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
            pred_noise = self.unet([noisy_latents, t_reshaped], training=True)
            if pred_noise.shape != noise.shape:
                if len(pred_noise.shape) == 3 and len(noise.shape) == 2:
                    pred_noise = tf.squeeze(pred_noise, axis=1)
            diffusion_loss = tf.reduce_mean(tf.square(pred_noise - noise))
            reconstructed = self.decoder(latents, training=True)
            reconstructed = tf.cast(reconstructed, tf.float32)
            reconstruction_loss = tf.reduce_mean(tf.square(images - reconstructed))
            total_loss = diffusion_loss + beta * reconstruction_loss
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.total_loss_tracker.update_state(total_loss)
        self.diffusion_loss_tracker.update_state(diffusion_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            images = tf.cast(data[0], tf.float32)
        else:
            images = tf.cast(data, tf.float32)
        batch_size = tf.shape(images)[0]
        latents = self.encoder(images, training=False)
        t = self.sample_timesteps(batch_size)
        noise = tf.random.normal(tf.shape(latents), dtype=tf.float32)
        noisy_latents = self.q_sample(latents, t, noise)
        t_reshaped = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        pred_noise = self.unet([noisy_latents, t_reshaped], training=False)
        if pred_noise.shape != noise.shape:
            if len(pred_noise.shape) == 3 and len(noise.shape) == 2:
                pred_noise = tf.squeeze(pred_noise, axis=1)
        diffusion_loss = tf.reduce_mean(tf.square(pred_noise - noise))
        reconstructed = self.decoder(latents, training=False)
        reconstructed = tf.cast(reconstructed, tf.float32)
        reconstruction_loss = tf.reduce_mean(tf.square(images - reconstructed))
        total_loss = diffusion_loss + beta * reconstruction_loss
        self.total_loss_tracker.update_state(total_loss)
        self.diffusion_loss_tracker.update_state(diffusion_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "diffusion_loss": self.diffusion_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    def call(self, inputs):
        if isinstance(inputs, tuple):
            images = tf.cast(inputs[0], tf.float32)
        else:
            images = tf.cast(inputs, tf.float32)
        latents = self.encoder(images)
        batch_size = tf.shape(images)[0]
        timesteps = tf.random.uniform(
            (batch_size,), minval=0, maxval=self.num_timesteps, dtype=tf.int32)
        noisy_latents = self.q_sample(latents, timesteps)
        t_reshaped = tf.reshape(tf.cast(timesteps, tf.float32), [-1, 1])
        pred_noise = self.unet([noisy_latents, t_reshaped])
        return pred_noise


# Initialize diffusion model
diffusion_model = DiffusionModel(
    encoder, decoder, unet,
    num_timesteps=num_timesteps,
    beta_start=beta_start,
    beta_end=beta_end
)

dummy_input = tf.random.normal((1, ni, nj, nz), dtype=tf.float32)
_ = diffusion_model(dummy_input)
diffusion_model.compile()


def lr_scheduler(epoch):
    decay_rate = 0.95
    decay_step = 50
    if epoch > 0 and epoch % decay_step == 0:
        return initial_lr * (decay_rate ** (epoch // decay_step))
    return initial_lr


def perceptual_loss(real_images, fake_images):
    real_images_resized = tf.image.resize(real_images, [75, 75])
    fake_images_resized = tf.image.resize(fake_images, [75, 75])
    if real_images.shape[-1] == 1:
        real_images_resized = tf.repeat(real_images_resized, 3, axis=-1)
        fake_images_resized = tf.repeat(fake_images_resized, 3, axis=-1)
    real_features_iv3 = f_extraction_iv3(real_images_resized)
    fake_features_iv3 = f_extraction_iv3(fake_images_resized)
    real_features_rc = f_extraction_rc(real_images)
    fake_features_rc = f_extraction_rc(fake_images)
    loss_iv3 = tf.reduce_mean(tf.square(real_features_iv3 - fake_features_iv3))
    loss_rc = tf.reduce_mean(tf.square(real_features_rc - fake_features_rc))
    return loss_iv3 + loss_rc


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
    latents = encoder.predict(data[:num_samples], verbose=0)
    latent_flat = latents.flatten()
    return latent_flat


# ==============================================
# CALLBACKS
# ==============================================

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
            latents = self.model.encoder.predict(test_samples, verbose=0)
            reconstructions = self.model.decoder.predict(latents, verbose=0)
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
        latents = self.model.encoder.predict(original_imgs)
        reconstructed_imgs = self.model.decoder.predict(latents)
        for i in range(self.num_images):
            self.ax[i * 2].imshow(original_imgs[i].squeeze(), cmap='viridis', interpolation='nearest')
            self.ax[i * 2].set_title("Original", fontweight='bold', fontsize=8)
            self.ax[i * 2].axis('off')
            self.ax[i * 2 + 1].imshow(reconstructed_imgs[i].squeeze(), cmap='viridis', interpolation='nearest')
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
    def __init__(self, diffusion_model, encoder, x_test, metric_interval=metric_interval, save_interval=10,
                 log_file='latent_diffusion_training_metrics.txt'):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.encoder = encoder
        self.x_test = x_test
        self.metric_interval = metric_interval
        self.save_interval = save_interval
        self.log_file = log_file
        self.metrics = {
            'FID_10k': [], 'FRD_10k': [], 'KID_10k': [], 'KRD_10k': [],
            'FID_500': [], 'FRD_500': [], 'KID_500': [], 'KRD_500': []
        }
        self.static_metrics = {
            'Variogram_MSE': [], 'Connectivity_MSE': [], 'Histogram_KL': [],
            'PCA_Correlation': [], 'MDS_MMD': []
        }
        with open(self.log_file, 'w') as f:
            f.write("Epoch\tTotal_Loss\tDiffusion_Loss\tRecon_Loss\t"
                    "FID_500\tFRD_500\tKID_500\tKRD_500\t"
                    "Variogram_MSE\tConnectivity_MSE\tHistogram_KL\tPCA_Correlation\tMDS_MMD\n")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.metric_interval == 0:
            print(f'\n{"=" * 60}')
            print(f'Latent Diffusion: Computing metrics at epoch {epoch + 1}')
            print(f'{"=" * 60}')
            MemoryOptimizer.check_memory(threshold_mb=8000)
            size = min(1000, self.x_test.shape[0])
            fake_data = self.diffusion_model.sample(num_samples=size, training=False)
            if isinstance(fake_data, tf.Tensor):
                fake_data = fake_data.numpy()
            if np.any(np.isnan(fake_data)):
                print(f"Warning: Generated data contains NaN values")
                fake_data = np.nan_to_num(fake_data, nan=0.0)
            static_results = self.compute_static_metrics_corrected(self.x_test[:size], fake_data[:size])
            training_metrics = {
                'total_loss': logs.get('loss', 0),
                'diffusion_loss': logs.get('diffusion_loss', 0),
                'recon_loss': logs.get('reconstruction_loss', 0)
            }
            try:
                from metrics import frechet_distance, kernel_distance
                real_resized = tf.image.resize(np.repeat(self.x_test[:size], 3, axis=3), [75, 75])
                fake_resized = tf.image.resize(np.repeat(fake_data, 3, axis=3), [75, 75])
                act_train_iv3 = f_extraction_iv3.predict(real_resized, verbose=0)
                act_gen_iv3 = f_extraction_iv3.predict(fake_resized, verbose=0)
                act_train_rc = f_extraction_rc.predict(self.x_test[:size], verbose=0)
                act_gen_rc = f_extraction_rc.predict(fake_data, verbose=0)
                fid_500 = frechet_distance(act_train_iv3[:500], act_gen_iv3[:500])
                frd_500 = frechet_distance(act_train_rc[:500], act_gen_rc[:500])
                kid_500 = kernel_distance(act_train_iv3[:500], act_gen_iv3[:500], num_blocks=50, max_block_size=500)
                krd_500 = kernel_distance(act_train_rc[:500], act_gen_rc[:500], num_blocks=50, max_block_size=500)
                traditional_results = {
                    'FID_500': fid_500, 'FRD_500': frd_500,
                    'KID_500': kid_500, 'KRD_500': krd_500
                }
            except ImportError:
                print("Warning: metrics.py not found. Using placeholder values.")
                traditional_results = {
                    'FID_500': 0, 'FRD_500': 0,
                    'KID_500': 0, 'KRD_500': 0
                }
            for key in ['FID_500', 'FRD_500', 'KID_500', 'KRD_500']:
                self.metrics[key].append(traditional_results[key])
            for key in self.static_metrics:
                self.static_metrics[key].append(static_results[key])
            self.save_metrics_to_file(epoch + 1, training_metrics, traditional_results, static_results)
            self.display_metrics(epoch + 1, training_metrics, traditional_results, static_results)
            self.save_numpy_files()
            self.plot_static_metrics(epoch + 1, static_results, save_interval=self.save_interval)

    def compute_static_metrics_corrected(self, real_data, fake_data):
        print("Calculating static geostatistical metrics...")
        n_samples = min(500, len(real_data), len(fake_data))
        real_subset = real_data[:n_samples]
        fake_subset = fake_data[:n_samples]
        metrics = self.compute_all_metrics_corrected(
            real_subset, fake_subset, threshold=0.5, max_lag=25
        )
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
        _, connectivity_real = StaticMetrics.connectivity_function(
            real_data[:100], threshold, lag_distances=lags
        )
        _, connectivity_fake = StaticMetrics.connectivity_function(
            fake_data[:100], threshold, lag_distances=lags
        )
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
        axes[0, 0].tick_params(axis='both', which='major', labelsize=18, width=2)
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
                                        linewidth=HIGH_RES_CONFIG['line_width'],
                                        markersize=HIGH_RES_CONFIG['marker_size'] // 10)
                    axes[row, col].set_xlabel('Epoch', fontweight='bold', fontsize=22)
                    axes[row, col].set_ylabel(metric_name, fontweight='bold', fontsize=22)
                    axes[row, col].grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
                    axes[row, col].set_title(f'{metric_name} Trend', fontweight='bold', fontsize=24)
                    axes[row, col].tick_params(axis='both', which='major', labelsize=18, width=2)
        plt.tight_layout()
        if should_save:
            base_filename = f"training_outputs/latent_diffusion_static_metrics_epoch_{epoch:03d}"
            saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
            print(f"Saved static metrics plot at epoch {epoch}: {len(saved_files)} files")
        else:
            plt.pause(0.01)
            saved_files = []
            print(f"Displayed static metrics at epoch {epoch} (not saved)")
        plt.close(fig)
        return saved_files

    def save_metrics_to_file(self, epoch, training_metrics, traditional_metrics, static_metrics):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch}\t"
                    f"{training_metrics['total_loss']:.6f}\t"
                    f"{training_metrics['diffusion_loss']:.6f}\t"
                    f"{training_metrics['recon_loss']:.6f}\t"
                    f"{traditional_metrics['FID_500']:.4f}\t"
                    f"{traditional_metrics['FRD_500']:.4f}\t"
                    f"{traditional_metrics['KID_500']:.4f}\t"
                    f"{traditional_metrics['KRD_500']:.4f}\t"
                    f"{static_metrics['Variogram_MSE']:.4f}\t"
                    f"{static_metrics['Connectivity_MSE']:.4f}\t"
                    f"{static_metrics['Histogram_KL']:.4f}\t"
                    f"{static_metrics['PCA_Correlation']:.4f}\t"
                    f"{static_metrics['MDS_MMD']:.4f}\n")
        csv_file = self.log_file.replace('.txt', '.csv')
        if epoch == self.metric_interval:
            df_columns = [
                'Epoch', 'Total_Loss', 'Diffusion_Loss', 'Recon_Loss',
                'FID_500', 'FRD_500', 'KID_500', 'KRD_500',
                'Variogram_MSE', 'Connectivity_MSE', 'Histogram_KL',
                'PCA_Correlation', 'MDS_MMD'
            ]
            pd.DataFrame(columns=df_columns).to_csv(csv_file, index=False)
        metrics_row = {
            'Epoch': epoch,
            'Total_Loss': training_metrics['total_loss'],
            'Diffusion_Loss': training_metrics['diffusion_loss'],
            'Recon_Loss': training_metrics['recon_loss'],
            'FID_500': traditional_metrics['FID_500'],
            'FRD_500': traditional_metrics['FRD_500'],
            'KID_500': traditional_metrics['KID_500'],
            'KRD_500': traditional_metrics['KRD_500'],
            'Variogram_MSE': static_metrics['Variogram_MSE'],
            'Connectivity_MSE': static_metrics['Connectivity_MSE'],
            'Histogram_KL': static_metrics['Histogram_KL'],
            'PCA_Correlation': static_metrics['PCA_Correlation'],
            'MDS_MMD': static_metrics['MDS_MMD']
        }
        df = pd.DataFrame([metrics_row])
        df.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"Metrics saved to CSV: {csv_file}")

    def display_metrics(self, epoch, training_metrics, traditional_metrics, static_metrics):
        output = (
            f"\nLatent Diffusion - Epoch {epoch} Metrics:\n"
            f"{'=' * 40}\n"
            f"Training Losses:\n"
            f"  Total Loss: {training_metrics['total_loss']:.4f}\n"
            f"  Diffusion Loss: {training_metrics['diffusion_loss']:.4f}\n"
            f"  Recon Loss: {training_metrics['recon_loss']:.4f}\n\n"
            f"Traditional Metrics:\n"
            f"  FID(500): {traditional_metrics['FID_500']:.2f}\n"
            f"  FRD(500): {traditional_metrics['FRD_500']:.2f}\n"
            f"  KID(500): {traditional_metrics['KID_500']:.4f}\n"
            f"  KRD(500): {traditional_metrics['KRD_500']:.4f}\n\n"
            f"Static Geostatistical Metrics (CORRECTED):\n"
            f"  Variogram MSE: {static_metrics['Variogram_MSE']:.4f}\n"
            f"  Connectivity MSE: {static_metrics['Connectivity_MSE']:.4f}\n"
            f"  Histogram KL: {static_metrics['Histogram_KL']:.4f}\n"
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
        self.val_total_losses = []
        self.val_diffusion_losses = []
        self.val_recon_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.total_losses.append(logs.get('loss', 0))
        self.diffusion_losses.append(logs.get('diffusion_loss', 0))
        self.recon_losses.append(logs.get('reconstruction_loss', 0))
        self.val_total_losses.append(logs.get('val_loss', 0))
        self.val_diffusion_losses.append(logs.get('val_diffusion_loss', 0))
        self.val_recon_losses.append(logs.get('val_reconstruction_loss', 0))


def plot_losses(loss_history, save_prefix="latent_diffusion", save_interval=10, total_epochs=epochs):
    figsize = (22 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig = plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    epochs_range = range(1, len(loss_history.total_losses) + 1)
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, loss_history.total_losses, label='Training Total Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
    plt.plot(epochs_range, loss_history.val_total_losses, label='Validation Total Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Total Loss', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss_history.diffusion_losses, label='Training Diffusion Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
    plt.plot(epochs_range, loss_history.val_diffusion_losses, label='Validation Diffusion Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Diffusion Loss', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, loss_history.recon_losses, label='Training Recon Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
    plt.plot(epochs_range, loss_history.val_recon_losses, label='Validation Recon Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Reconstruction Loss', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.suptitle('Latent Diffusion Training Losses', fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base_filename = f"training_outputs/{save_prefix}_all_losses"
    saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
    plt.close(fig)
    plot_individual_losses(loss_history, save_prefix)
    return saved_files


def plot_individual_losses(loss_history, save_prefix="latent_diffusion"):
    epochs_range = range(1, len(loss_history.total_losses) + 1)
    loss_types = [
        ('total_loss', loss_history.total_losses, loss_history.val_total_losses, 'Total Loss'),
        ('diffusion_loss', loss_history.diffusion_losses, loss_history.val_diffusion_losses, 'Diffusion Loss'),
        ('recon_loss', loss_history.recon_losses, loss_history.val_recon_losses, 'Reconstruction Loss')
    ]
    for loss_name, train_loss, val_loss, title in loss_types:
        fig, ax = plt.subplots(figsize=(12 * HIGH_RES_CONFIG['figsize_multiplier'],
                                        8 * HIGH_RES_CONFIG['figsize_multiplier']),
                               dpi=HIGH_RES_CONFIG['dpi'])
        ax.plot(epochs_range, train_loss, label='Training', linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
        ax.plot(epochs_range, val_loss, label='Validation', linewidth=HIGH_RES_CONFIG['line_width'], color='red')
        ax.set_xlabel('Epochs', fontweight='bold', fontsize=22)
        ax.set_ylabel('Loss', fontweight='bold', fontsize=22)
        ax.set_title(f'Latent Diffusion: {title}', fontsize=24, fontweight='bold')
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2)
        base_filename = f"training_outputs/{save_prefix}_{loss_name}"
        saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
        plt.close(fig)


def plot_static_metrics_comparison(real_data, diffusion_model, save_path_base=None):
    n_samples = min(500, len(real_data))
    fake_data = diffusion_model.sample(n_samples, training=False)
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
    figsize = (22 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    fig.suptitle('Latent Diffusion: Static Geostatistical Metrics Comparison (CORRECTED)',
                 fontsize=26, fontweight='bold', y=1.02)
    for ax in axes.flat:
        ax.tick_params(labelsize=18, width=1.5)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.title.set_size(24)
    ax = axes[0, 0]
    ax.plot(lags, variogram_real, 'b-', linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    ax.plot(lags, variogram_fake, 'r--', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    ax.set_ylabel('Variogram', fontweight='bold', fontsize=22)
    ax.set_title(f'Variogram (MSE: {metrics["variogram_mse"]:.4f})', fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax = axes[0, 1]
    ax.plot(lags, connectivity_real, 'b-', linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    ax.plot(lags, connectivity_fake, 'r--', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    ax.set_ylabel('Connectivity', fontweight='bold', fontsize=22)
    ax.set_title(f'Connectivity Function (MSE: {metrics["connectivity_mse"]:.4f})', fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax = axes[0, 2]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_centers[1] - bin_centers[0]
    ax.bar(bin_centers, hist_real, width=bar_width, alpha=0.6, label='Real', color='blue', edgecolor='black',
           linewidth=1.0)
    ax.plot(bin_centers, hist_fake, 'r-', linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Value', fontweight='bold', fontsize=22)
    ax.set_ylabel('Density', fontweight='bold', fontsize=22)
    ax.set_title(f'Histogram (KL: {metrics["hist_kl"]:.4f})', fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax = axes[1, 0]
    ax.scatter(pca_real[:100, 0], pca_real[:100, 1], alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
               color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(pca_fake[:100, 0], pca_fake[:100, 1], alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
               color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('PC1', fontweight='bold', fontsize=22)
    ax.set_ylabel('PC2', fontweight='bold', fontsize=22)
    ax.set_title(f'PCA Projection (Corr: {metrics["pca_correlation"]:.4f})', fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax = axes[1, 1]
    ax.scatter(mds_real[:100, 0], mds_real[:100, 1], alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
               color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(mds_fake[:100, 0], mds_fake[:100, 1], alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
               color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('MDS Dimension 1', fontweight='bold', fontsize=22)
    ax.set_ylabel('MDS Dimension 2', fontweight='bold', fontsize=22)
    ax.set_title(f'MDS Projection (MMD: {metrics["mds_mmd"]:.4f})', fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax = axes[1, 2]
    stats_text = (
        f'Variogram MSE: {metrics["variogram_mse"]:.6f}\n'
        f'Connectivity MSE: {metrics["connectivity_mse"]:.6f}\n'
        f'Histogram KL: {metrics["hist_kl"]:.6f}\n'
        f'PCA Correlation: {metrics["pca_correlation"]:.4f}\n'
        f'MDS MMD: {metrics["mds_mmd"]:.4f}\n\n'
        f'Real mean: {np.mean(real_flat):.4f}\n'
        f'Fake mean: {np.mean(fake_flat):.4f}\n'
        f'Real std: {np.std(real_flat):.4f}\n'
        f'Fake std: {np.std(fake_flat):.4f}'
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=18, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Summary Statistics', fontweight='bold', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_base:
        saved_files = save_figure_multiple_formats(fig, save_path_base, HIGH_RES_CONFIG['dpi'])
        print(f"Static metrics plots saved: {len(saved_files)} files")
    else:
        plt.show()
        saved_files = []
    plt.close(fig)
    return metrics, saved_files


def create_file_summary(prefix="latent_diffusion"):
    summary = {}
    for fmt in HIGH_RES_CONFIG['save_formats']:
        pattern = f"training_outputs/{prefix}*.{fmt}"
        files = glob.glob(pattern)
        summary[fmt] = len(files)
    other_files = []
    for pattern in [f'{prefix}*.npy', f'{prefix}*.txt', f'{prefix}*.h5', f'{prefix}*.pkl']:
        other_files.extend(glob.glob(f"training_outputs/{pattern}"))
    summary['other'] = len(other_files)
    summary['total'] = sum(summary.values())
    return summary


# ==============================================
# FINAL TRAINING REPORT FUNCTION
# ==============================================
def generate_final_report(filepath, params, execution_time, loss_history, metrics_history, static_metrics_results):
    """
    Generates a complete final training report and saves it to a .txt file
    """
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
            f.write(f"    Reconstruction   : {loss_history.recon_losses[-1]:.6f}\n\n")

            f.write("  Validation Losses (last epoch):\n")
            f.write(f"    Total Loss       : {loss_history.val_total_losses[-1]:.6f}\n")
            f.write(f"    Diffusion Loss   : {loss_history.val_diffusion_losses[-1]:.6f}\n")
            f.write(f"    Reconstruction   : {loss_history.val_recon_losses[-1]:.6f}\n\n")

            f.write("  Best values during training:\n")
            f.write(
                f"    Lowest Total Loss (train)      : {min(loss_history.total_losses):.6f} (epoch {np.argmin(loss_history.total_losses) + 1})\n")
            f.write(
                f"    Lowest Total Loss (val)        : {min(loss_history.val_total_losses):.6f} (epoch {np.argmin(loss_history.val_total_losses) + 1})\n")
            f.write(
                f"    Lowest Diffusion Loss (train)  : {min(loss_history.diffusion_losses):.6f} (epoch {np.argmin(loss_history.diffusion_losses) + 1})\n")
            f.write(
                f"    Lowest Recon Loss (train)      : {min(loss_history.recon_losses):.6f} (epoch {np.argmin(loss_history.recon_losses) + 1})\n\n")

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
        f.write("  Encoder:\n")
        f.write(f"    Input Shape        : ({ni}, {nj}, {nz})\n")
        f.write(f"    Latent Dim         : {latent_dim}\n")
        f.write("    Conv Layers        : 4 (32, 64, 128, 256 filters)\n")
        f.write("    Dense Layer        : 256\n")
        f.write("    Dropout            : 0.4 / 0.3\n")
        f.write("    BatchNormalization : Yes\n\n")

        f.write("  Decoder:\n")
        f.write(f"    Input Dim          : {latent_dim}\n")
        f.write("    ConvTranspose      : 3 (128, 64, 32 filters)\n")
        f.write("    Output Act         : tanh\n")
        f.write("    Dropout            : 0.2\n\n")

        f.write("  UNet (Diffusion):\n")
        f.write(f"    Input Dim          : {latent_dim}\n")
        f.write(f"    Time Emb Dim       : 256\n")
        f.write("    Hidden Layers      : 1024, 1024, 512\n")
        f.write("    Activation         : LeakyReLU (0.2), swish\n")
        f.write("    Dropout            : 0.1\n\n")

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
        f.write(f"  Test samples            : {len(x_test)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("GENERATED FILES\n")
        f.write("-" * 80 + "\n")
        saved_files_list = [
            "training_outputs/latent_diffusion_encoder_weights.h5",
            "training_outputs/latent_diffusion_decoder_weights.h5",
            "training_outputs/latent_diffusion_unet_weights.h5",
            "training_outputs/latent_diffusion_model_weights.h5",
            "training_outputs/latent_diffusion_encoder_model.h5",
            "training_outputs/latent_diffusion_decoder_model.h5",
            "training_outputs/latent_diffusion_unet_model.h5",
            "training_outputs/latent_diffusion_best_weights.h5",
            "training_outputs/latent_diffusion_loss_history.txt",
            "training_outputs/latent_diffusion_training_metrics.txt",
            "training_outputs/latent_diffusion_all_metrics.csv",
            "training_outputs/latent_diffusion_distribution_data.pkl",
            "training_outputs/latent_diffusion_execution_time.txt"
        ]
        for file in saved_files_list:
            f.write(f"  - {file}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("                    END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Final report saved to: {filepath}")


# ==============================================
# TRAINING
# ==============================================

os.makedirs('training_outputs', exist_ok=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype('float32'), x_train.astype('float32'))).shuffle(
    10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test.astype('float32'), x_test.astype('float32'))).batch(
    batch_size)

SAVE_INTERVAL = 10

loss_history_callback = LossHistoryCallback()
live_plot_callback = LivePlotCallback(diffusion_model, x_test, num_images=5, save_interval=SAVE_INTERVAL)
distribution_callback = DistributionPlotCallback(diffusion_model, x_test, plot_interval=10, save_interval=SAVE_INTERVAL)

enhanced_metric_callback = EnhancedMetricCallback(
    diffusion_model, encoder, x_test,
    metric_interval=SAVE_INTERVAL, save_interval=SAVE_INTERVAL,
    log_file='latent_diffusion_training_metrics.txt'
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'training_outputs/latent_diffusion_best_weights.h5',
    save_best_only=True, monitor='val_loss',
    save_weights_only=True, mode='min'
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
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
print(f"Latent dimension: {latent_dim}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {initial_lr}")
print(f"Number of diffusion timesteps: {num_timesteps}")
print(f"Beta schedule: {beta_start} to {beta_end}")
print(f"Save interval: {SAVE_INTERVAL} epochs")
print(f"Metrics calculation interval: {SAVE_INTERVAL} epochs")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
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

print(f"\n[INFO] Latent Diffusion training completed in {execution_time / 3600:.2f} hours")


def save_execution_time(filename, execution_time):
    with open(filename, 'w') as f:
        f.write(f"=== Latent Diffusion Execution Time ===\n")
        f.write(f"Total: {execution_time:.2f} seconds\n")
        f.write(f"Total: {execution_time / 60:.2f} minutes\n")
        f.write(f"Total: {execution_time / 3600:.2f} hours\n")
        f.write(f"\nSettings:\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Latent dim: {latent_dim}\n")
        f.write(f"- Learning rate: {initial_lr}\n")
        f.write(f"- Diffusion timesteps: {num_timesteps}\n")
        f.write(f"- Save interval: {SAVE_INTERVAL} epochs\n")
        f.write(f"- Metrics interval: {SAVE_INTERVAL} epochs\n")


save_execution_time("training_outputs/latent_diffusion_execution_time.txt", execution_time)

saved_loss_files = plot_losses(loss_history_callback, save_prefix="latent_diffusion")
print(f"Saved final loss plots: {len(saved_loss_files)} files")

with open('training_outputs/latent_diffusion_loss_history.txt', 'w') as f:
    f.write("Epoch\tTotal_Loss\tDiffusion_Loss\tRecon_Loss\t"
            "Val_Total_Loss\tVal_Diffusion_Loss\tVal_Recon_Loss\n")
    for i in range(len(loss_history_callback.total_losses)):
        f.write(f"{i + 1}\t"
                f"{loss_history_callback.total_losses[i]:.6f}\t"
                f"{loss_history_callback.diffusion_losses[i]:.6f}\t"
                f"{loss_history_callback.recon_losses[i]:.6f}\t"
                f"{loss_history_callback.val_total_losses[i]:.6f}\t"
                f"{loss_history_callback.val_diffusion_losses[i]:.6f}\t"
                f"{loss_history_callback.val_recon_losses[i]:.6f}\n")

encoder.save_weights('training_outputs/latent_diffusion_encoder_weights.h5')
decoder.save_weights('training_outputs/latent_diffusion_decoder_weights.h5')
unet.save_weights('training_outputs/latent_diffusion_unet_weights.h5')
diffusion_model.save_weights('training_outputs/latent_diffusion_model_weights.h5')

encoder.save('training_outputs/latent_diffusion_encoder_model.h5')
decoder.save('training_outputs/latent_diffusion_decoder_model.h5')
unet.save('training_outputs/latent_diffusion_unet_model.h5')
tf.saved_model.save(diffusion_model, 'training_outputs/latent_diffusion_final_model')

print("\n" + "=" * 60)
print("FINAL ANALYSIS AND SAVING")
print("=" * 60)

test_samples = x_test[:5]
latents = diffusion_model.encoder.predict(test_samples, verbose=0)
reconstructions = diffusion_model.decoder.predict(latents, verbose=0)
latent_samples = analyze_latent_space(diffusion_model.encoder, x_test, num_samples=1000)

print("Saving final distribution plots...")
saved_dist_files = plot_histograms_and_densities(test_samples, reconstructions, latent_samples)
print(f"Saved final distribution plots: {len(saved_dist_files)} files")

print("Saving latent space visualization...")
figsize = (14 * HIGH_RES_CONFIG['figsize_multiplier'],
           8 * HIGH_RES_CONFIG['figsize_multiplier'])
fig = plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})
latent_samples_proj = diffusion_model.encoder.predict(x_test[:1000], verbose=0)
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_samples_proj)
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=HIGH_RES_CONFIG['marker_size'])
plt.title('Latent Diffusion: 2D Projection of Latent Space (PCA)',
          fontsize=24, fontweight='bold')
plt.xlabel('PCA Component 1', fontweight='bold', fontsize=22)
plt.ylabel('PCA Component 2', fontweight='bold', fontsize=22)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
plt.tick_params(axis='both', which='major', labelsize=18, width=2)
base_filename = "training_outputs/latent_diffusion_latent_space_2d"
saved_latent_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
print(f"Saved final latent space plots: {len(saved_latent_files)} files")
plt.close(fig)

print("Saving final static metrics comparison...")
static_metrics_results, saved_static_files = plot_static_metrics_comparison(
    x_test, diffusion_model, save_path_base='training_outputs/latent_diffusion_static_metrics_comparison'
)
print(f"Saved final static metrics plots: {len(saved_static_files)} files")

print("Saving distribution data...")
dist_data = {
    'original': test_samples.flatten(),
    'reconstructed': reconstructions.flatten(),
    'latent': latent_samples.flatten() if isinstance(latent_samples, np.ndarray) else latent_samples,
    'latent_2d_projection': latent_2d
}
with open('training_outputs/latent_diffusion_distribution_data.pkl', 'wb') as f:
    pickle.dump(dist_data, f)
print("Saved distribution data to pickle file")

file_summary = create_file_summary("latent_diffusion")
print("\n" + "=" * 60)
print("LATENT DIFFUSION FILE SAVING SUMMARY")
print("=" * 60)
print(f"PNG files saved: {file_summary['png']}")
print(f"SVG files saved: {file_summary['svg']}")
print(f"Other files (npy, txt, h5, pkl): {file_summary['other']}")
print(f"TOTAL files saved: {file_summary['total']}")
print("=" * 60)

print("Saving final reconstructions comparison...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    axes[0, i].imshow(test_samples[i].squeeze(), cmap='viridis')
    axes[0, i].set_title(f'Original {i + 1}', fontsize=20, fontweight='bold')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructions[i].squeeze(), cmap='viridis')
    axes[1, i].set_title(f'Reconstructed {i + 1}', fontsize=20, fontweight='bold')
    axes[1, i].axis('off')
plt.suptitle('Latent Diffusion: Final Reconstruction Comparison',
             fontsize=26, fontweight='bold')
plt.tight_layout()
saved_final_files = save_figure_multiple_formats(fig, "training_outputs/latent_diffusion_final_reconstructions",
                                                 HIGH_RES_CONFIG['dpi'])
print(f"Saved final reconstruction comparison: {len(saved_final_files)} files")
plt.show()

# ==============================================
# GENERATE AND SAVE FINAL REPORT
# ==============================================

training_params = {
    'Dataset': name,
    'Latent dimension (zdim)': zdim,
    'Configured epochs': epochs,
    'Batch size': batch_size,
    'Initial learning rate': initial_lr,
    'Beta (reconstruction weight)': beta,
    'Gamma (perceptual weight)': gamma,
    'Alpha (LeakyReLU)': alpha,
    'Image dimensions (ni x nj x nz)': f'{ni} x {nj} x {nz}',
    'Diffusion timesteps': num_timesteps,
    'Beta start': beta_start,
    'Beta end': beta_end,
    'Training samples': len(x_train),
    'Test samples': len(x_test),
    'Model': 'Latent Diffusion Model',
    'Optimizer': 'Adam (beta1=0.5, clipnorm=1.0)',
    'Early Stopping': 'Yes (patience=20, monitor=val_loss)',
    'LR Scheduler': 'ReduceLROnPlateau (factor=0.5, patience=5)',
    'Metric interval': metric_interval,
    'Save interval': SAVE_INTERVAL
}

report_path = os.path.join('training_outputs', 'latent_diffusion_final_report.txt')
generate_final_report(
    filepath=report_path,
    params=training_params,
    execution_time=execution_time,
    loss_history=loss_history_callback,
    metrics_history=enhanced_metric_callback,
    static_metrics_results=static_metrics_results
)

print("\n" + "=" * 60)
print("LATENT DIFFUSION TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Training Summary:")
print(f"- Total epochs: {epochs}")
print(f"- Save interval: {SAVE_INTERVAL} epochs")
print(f"- Metrics calculation interval: {SAVE_INTERVAL} epochs")
print(f"- Execution time: {execution_time / 3600:.2f} hours")
print(f"- Total files saved: {file_summary['total']}")
print("=" * 60)
print(f"\nFinal complete report saved to: {report_path}")
print("=" * 60)