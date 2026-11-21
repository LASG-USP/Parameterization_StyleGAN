# ============================================================================
# 1. IMPORTS & CONFIGURATION
# ============================================================================

import os
import gc
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from keras.applications.inception_v3 import InceptionV3
from PIL import Image

from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from metrics import *

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
zdim = 512
epochs = 150
metric_interval = 10
batch_size = 256
lr = 0.003
beta = 1

d_reg = 5
name = 'unisim_ii_cropped'

# FID/FRD sampling: 10 batches x 1000 = 10000 total
FID_N_BATCHES = 10
FID_BATCH_SIZE = 1000
FID_SAMPLE_SIZE = FID_N_BATCHES * FID_BATCH_SIZE   # 10000

HIGH_RES_CONFIG = {
    'dpi': 300,
    'figsize_multiplier': 1.5,
    'font_size': 18,
    'title_font_size': 20,
    'line_width': 2.5,
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

os.makedirs('training_outputs', exist_ok=True)
print("Directory 'training_outputs' created/verified.")


# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================

# Load dataset
if name == 'channel2d_large':
    X_train = np.load('datasets/Two_facies_channel_2d.npy').astype('float')
elif name == '3facies_2d_80k':
    X_train = np.load('datasets/Three_facies_2d.npy').astype('float')
elif name == 'unisim_ii_cropped':
    X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')
    X_train[X_train < 1] = 1
    X_train = np.log(X_train)
    X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
    X_train = X_train[..., np.newaxis]

noise = np.random.normal(0, 1, (5 * 5, zdim))
ni = X_train.shape[1]
nj = X_train.shape[2]
nz = X_train.shape[3]

train_size = int(0.8 * len(X_train))
x_train = X_train[:train_size]
x_test = X_train[train_size:]

latent_dim = zdim

print(f"Data shape: {X_train.shape}")
print(f"Train: {x_train.shape}, Test: {x_test.shape}")
print(f"Data type: {X_train.dtype}")


# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 Encoder
# ----------------------------------------------------------------------------
def build_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


# ----------------------------------------------------------------------------
# 3.2 Decoder
# ----------------------------------------------------------------------------
def build_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(6 * 6 * 128)(latent_inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((6, 6, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = layers.Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)
    return keras.Model(latent_inputs, outputs, name="decoder")


# ----------------------------------------------------------------------------
# 3.3 Discriminator
# ----------------------------------------------------------------------------
def build_discriminator(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, x, name="discriminator")


# ----------------------------------------------------------------------------
# 3.4 Build model instances
# ----------------------------------------------------------------------------
encoder = build_encoder((ni, nj, nz), latent_dim)
decoder = build_decoder(latent_dim)
discriminator = build_discriminator((ni, nj, nz))


# ----------------------------------------------------------------------------
# 3.5 VAE-GAN composite model
# ----------------------------------------------------------------------------
class VAEGAN(keras.Model):
    def __init__(self, encoder, decoder, discriminator, **kwargs):
        super(VAEGAN, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.discriminator.trainable = False

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        real_images = data[0]
        batch_size = tf.shape(real_images)[0]

        self.discriminator.trainable = True

        with tf.GradientTape() as disc_tape:
            z_mean, z_log_var, z = self.encoder(real_images, training=True)
            fake_images = self.decoder(z, training=True)
            real_preds = self.discriminator(real_images, training=True)
            fake_preds = self.discriminator(fake_images, training=True)
            real_labels = tf.ones_like(real_preds)
            fake_labels = tf.zeros_like(fake_preds)
            d_loss_real = tf.reduce_mean(keras.losses.mean_squared_error(real_labels, real_preds))
            d_loss_fake = tf.reduce_mean(keras.losses.mean_squared_error(fake_labels, fake_preds))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        self.discriminator.trainable = False

        with tf.GradientTape() as gen_tape:
            z_mean, z_log_var, z = self.encoder(real_images, training=True)
            fake_images = self.decoder(z, training=True)
            recon_loss = tf.reduce_mean(keras.losses.mean_squared_error(real_images, fake_images))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            vae_loss = recon_loss + beta * kl_loss
            fake_preds = self.discriminator(fake_images, training=False)
            target_score = 0.9
            misleading_labels = tf.ones_like(fake_preds) * target_score
            g_loss = tf.reduce_mean(keras.losses.mean_squared_error(misleading_labels, fake_preds))
            g_weight = 0.5
            total_gen_loss = vae_loss + g_weight * g_loss

        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        g_gradients = gen_tape.gradient(total_gen_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(g_gradients, trainable_vars))

        self.compiled_metrics.update_state(real_images, fake_images)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({
            "loss": total_gen_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "d_loss": d_loss,
            "g_loss": g_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake
        })
        return metrics

    def test_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
        else:
            real_images = data
        z_mean, z_log_var, z = self.encoder(real_images, training=False)
        fake_images = self.decoder(z, training=False)
        recon_loss = tf.reduce_mean(keras.losses.mean_squared_error(real_images, fake_images))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        vae_loss = recon_loss + beta * kl_loss
        target_score = 0.9
        fake_preds = self.discriminator(fake_images, training=False)
        misleading_labels = tf.ones_like(fake_preds) * target_score
        g_loss = tf.reduce_mean(keras.losses.mean_squared_error(misleading_labels, fake_preds))
        total_loss = vae_loss + 0.5 * g_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "g_loss": g_loss
        }


vae_gan = VAEGAN(encoder, decoder, discriminator)


# ============================================================================
# 4. METRICS & UTILITIES
# ============================================================================

# ----------------------------------------------------------------------------
# 4.1 Numerically stable Frechet Distance (FID fix)
# ----------------------------------------------------------------------------
def frechet_distance(features1, features2, eps=1e-6):
    """
    Numerically stable version of FID.

    Solves the "FID exploding" problem caused by:
      - rank deficiency (n_samples << n_features)
      - sqrtm returning complex/NaN values
      - features contaminated by NaN/Inf
    """
    features1 = np.asarray(features1, dtype=np.float64)
    features2 = np.asarray(features2, dtype=np.float64)

    # --- Feature sanitization ---
    features1 = np.nan_to_num(features1, nan=0.0, posinf=0.0, neginf=0.0)
    features2 = np.nan_to_num(features2, nan=0.0, posinf=0.0, neginf=0.0)

    n1, d = features1.shape
    n2, _ = features2.shape

    # --- PCA reduction when there is rank deficiency ---
    if n1 < 3 * d or n2 < 3 * d:
        target_dims = max(32, min(min(n1, n2) // 3, d))
        if target_dims < d:
            pca = PCA(n_components=target_dims)
            pca.fit(np.vstack([features1, features2]))
            features1 = pca.transform(features1)
            features2 = pca.transform(features2)
            d = target_dims

    mu1 = features1.mean(axis=0)
    mu2 = features2.mean(axis=0)

    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    if sigma1.ndim == 0: sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0: sigma2 = np.array([[sigma2]])

    # --- Regularization proportional to the variance ---
    reg1 = eps * max(np.trace(sigma1) / max(d, 1), 1.0)
    reg2 = eps * max(np.trace(sigma2) / max(d, 1), 1.0)
    sigma1 = sigma1 + np.eye(d) * reg1
    sigma2 = sigma2 + np.eye(d) * reg2

    diff = mu1 - mu2

    # --- Robust sqrtm with fallbacks ---
    tr_covmean = None
    try:
        result = sqrtm(sigma1 @ sigma2)
        covmean = result[0] if isinstance(result, tuple) else result
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = np.nan_to_num(covmean, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(covmean)):
            raise ValueError("covmean non-finite")
        tr_covmean = np.trace(covmean)
    except Exception:
        diag1 = np.diag(sigma1)
        diag2 = np.diag(sigma2)
        tr_covmean = np.sum(np.sqrt(np.maximum(diag1 * diag2, 0.0)))

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

    if not np.isfinite(fid):
        return 1e6
    return max(fid, 0.0)


# ----------------------------------------------------------------------------
# 4.2 Batched Frechet Distance (10 batches x 1000 samples)
# ----------------------------------------------------------------------------
def frechet_distance_batched(real_features, fake_features,
                             n_batches=FID_N_BATCHES, batch_size=FID_BATCH_SIZE,
                             eps=1e-6):
    """
    Computes the Frechet Distance using batched computation over 10 batches of 1000 samples.
    Accumulates the statistics (mean and covariance) incrementally between batches,
    without keeping all features at once to compute the statistics.
    At the end, applies the same stable computation (PCA reduction when rank-deficient,
    regularization + robust sqrtm).
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    real_features = np.nan_to_num(real_features, nan=0.0, posinf=0.0, neginf=0.0)
    fake_features = np.nan_to_num(fake_features, nan=0.0, posinf=0.0, neginf=0.0)

    d = real_features.shape[1]

    # Accumulate statistics for the real features
    real_sum = np.zeros(d, dtype=np.float64)
    real_sum_sq = np.zeros((d, d), dtype=np.float64)
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
    fake_sum = np.zeros(d, dtype=np.float64)
    fake_sum_sq = np.zeros((d, d), dtype=np.float64)
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

    # --- rank deficiency check (we use the accumulated counts) ---
    n_eff = min(real_count, fake_count)
    if n_eff < 3 * d:
        target_dims = max(32, min(n_eff // 3, d))
        if target_dims < d:
            n_real_use = min(real_features.shape[0], n_batches * batch_size)
            n_fake_use = min(fake_features.shape[0], n_batches * batch_size)
            pca = PCA(n_components=target_dims)
            pca.fit(np.vstack([real_features[:n_real_use], fake_features[:n_fake_use]]))

            rred = pca.transform(real_features[:n_real_use])
            gred = pca.transform(fake_features[:n_fake_use])

            mu_r = rred.mean(axis=0)
            sigma_r = np.cov(rred, rowvar=False)
            mu_g = gred.mean(axis=0)
            sigma_g = np.cov(gred, rowvar=False)
            d = target_dims

    sigma_r = np.atleast_2d(sigma_r)
    sigma_g = np.atleast_2d(sigma_g)

    reg1 = eps * max(np.trace(sigma_r) / max(d, 1), 1.0)
    reg2 = eps * max(np.trace(sigma_g) / max(d, 1), 1.0)
    sigma_r = sigma_r + np.eye(d) * reg1
    sigma_g = sigma_g + np.eye(d) * reg2

    diff = mu_r - mu_g

    tr_covmean = None
    try:
        result = sqrtm(sigma_r @ sigma_g)
        covmean = result[0] if isinstance(result, tuple) else result
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = np.nan_to_num(covmean, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(covmean)):
            raise ValueError("covmean non-finite")
        tr_covmean = np.trace(covmean)
    except Exception:
        diag1 = np.diag(sigma_r)
        diag2 = np.diag(sigma_g)
        tr_covmean = np.sum(np.sqrt(np.maximum(diag1 * diag2, 0.0)))

    fid = float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * tr_covmean)
    if not np.isfinite(fid):
        return 1e6
    return max(fid, 0.0)


# ----------------------------------------------------------------------------
# 4.3 Feature extractors
# ----------------------------------------------------------------------------
def get_classifier(filepath):
    RC_full = tf.keras.models.load_model(filepath)
    f_extraction = keras.Model(inputs=RC_full.input,
                               outputs=layers.GlobalAveragePooling2D()(RC_full.layers[-5].output))
    return f_extraction


def get_inception(shape):
    iv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(shape[0], shape[1], 3))
    return iv3


# ----------------------------------------------------------------------------
# 4.4 Memory optimizer
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


# ----------------------------------------------------------------------------
# 4.5 Static geostatistical metrics
# ----------------------------------------------------------------------------
class StaticMetrics:
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
        return {'histogram': hist, 'bin_edges': bin_edges, 'mean': mean,
                'std': std, 'skewness': skew, 'kurtosis': kurtosis}

    @staticmethod
    def compute_all_metrics(real_data, fake_data, threshold=0.5, max_lag=25):
        metrics = {}
        print("Calculating variograms...")
        lags, variogram_real = StaticMetrics.variogram_2d(real_data[:100], max_lag=max_lag)
        _, variogram_fake = StaticMetrics.variogram_2d(fake_data[:100], max_lag=max_lag)
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

        print("Calculating histogram statistics...")
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
        mean = np.mean(real_flat); std = np.std(real_flat)
        skew = np.mean(((real_flat - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((real_flat - mean) / std) ** 4) if std > 0 else 0
        metrics['hist_stats_real'] = {'histogram': hist_real, 'bin_edges': bin_edges,
                                      'mean': mean, 'std': std, 'skewness': skew, 'kurtosis': kurtosis}
        fake_mean = np.mean(fake_flat); fake_std = np.std(fake_flat)
        fake_skew = np.mean(((fake_flat - fake_mean) / fake_std) ** 3) if fake_std > 0 else 0
        fake_kurtosis = np.mean(((fake_flat - fake_mean) / fake_std) ** 4) if fake_std > 0 else 0
        metrics['hist_stats_fake'] = {'histogram': hist_fake, 'bin_edges': bin_edges,
                                      'mean': fake_mean, 'std': fake_std,
                                      'skewness': fake_skew, 'kurtosis': fake_kurtosis}
        metrics['hist_kl'] = kl_divergence
        return metrics


# ----------------------------------------------------------------------------
# 4.6 Plotting utilities
# ----------------------------------------------------------------------------
def save_figure_multiple_formats(filename_base, dpi=None, bbox_inches='tight', **kwargs):
    if dpi is None:
        dpi = HIGH_RES_CONFIG['dpi']
    for fmt in HIGH_RES_CONFIG['save_formats']:
        filename = f"{filename_base}.{fmt}"
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"  Figure saved as: {filename}")


def plot_histograms_and_densities(original, reconstructed, latent, epoch=None):
    figsize = (20 * HIGH_RES_CONFIG['figsize_multiplier'],
               12 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18,
        'xtick.labelsize': 16, 'ytick.labelsize': 16,
        'legend.fontsize': 16, 'figure.titlesize': 22
    })
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    latent_flat = latent.flatten()

    plt.subplot(2, 3, 1)
    plt.imshow(original[0].squeeze(), cmap='viridis', interpolation='nearest')
    plt.title('Original Sample', fontsize=20, fontweight='bold')
    plt.axis('off'); plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed[0].squeeze(), cmap='viridis', interpolation='nearest')
    plt.title('Reconstructed Sample', fontsize=20, fontweight='bold')
    plt.axis('off'); plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 3, 3)
    sns.histplot(original_flat, color='blue', label='Original', kde=True,
                 stat='density', alpha=0.5, bins=HIGH_RES_CONFIG['hist_bins'])
    sns.histplot(reconstructed_flat, color='red', label='Reconstructed', kde=True,
                 stat='density', alpha=0.5, bins=HIGH_RES_CONFIG['hist_bins'])
    plt.legend(fontsize=16); plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Pixel Value', fontweight='bold', fontsize=18)
    plt.ylabel('Density', fontweight='bold', fontsize=18)
    plt.title('Pixel Value Distribution', fontweight='bold', fontsize=20)

    plt.subplot(2, 3, 4)
    sns.kdeplot(original_flat, color='blue', label='Original', linewidth=HIGH_RES_CONFIG['line_width'])
    sns.kdeplot(reconstructed_flat, color='red', label='Reconstructed', linewidth=HIGH_RES_CONFIG['line_width'])
    plt.legend(fontsize=16); plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Pixel Value', fontweight='bold', fontsize=18)
    plt.ylabel('Density', fontweight='bold', fontsize=18)
    plt.title('Density Comparison', fontweight='bold', fontsize=20)

    plt.subplot(2, 3, 5)
    sns.histplot(latent_flat, color='green', label='Latent Space', kde=True,
                 stat='density', bins=HIGH_RES_CONFIG['hist_bins'])
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Latent Variable Value', fontweight='bold', fontsize=18)
    plt.ylabel('Density', fontweight='bold', fontsize=18)
    plt.title('Latent Space Distribution', fontweight='bold', fontsize=20)
    plt.legend(fontsize=16)

    plt.subplot(2, 3, 6)
    sns.kdeplot(original_flat, color='blue', label='Original', linewidth=HIGH_RES_CONFIG['line_width'])
    sns.kdeplot(latent_flat, color='green', label='Latent Space', linewidth=HIGH_RES_CONFIG['line_width'])
    plt.title('Comparison of Densities', fontweight='bold', fontsize=20)
    plt.legend(fontsize=16); plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Value', fontweight='bold', fontsize=18)
    plt.ylabel('Density', fontweight='bold', fontsize=18)

    plt.suptitle('VAE-GAN Analysis: Original vs Reconstructed Distributions',
                 fontsize=24, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if epoch is not None:
        filename_base = f"training_outputs/distributions_epoch_{epoch:03d}"
    else:
        filename_base = "training_outputs/final_distributions"
    save_figure_multiple_formats(filename_base, dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def analyze_latent_space(encoder, data, num_samples=1000):
    _, _, z = encoder.predict(data[:num_samples], verbose=0)
    return z.flatten()


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


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.d_losses = []
        self.g_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        self.val_g_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.recon_losses.append(logs.get('recon_loss'))
        self.kl_losses.append(logs.get('kl_loss'))
        self.d_losses.append(logs.get('d_loss'))
        self.g_losses.append(logs.get('g_loss'))
        self.val_recon_losses.append(logs.get('val_recon_loss'))
        self.val_kl_losses.append(logs.get('val_kl_loss'))
        self.val_g_losses.append(logs.get('val_g_loss'))


loss_history_callback = LossHistoryCallback()


class DistributionPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, test_data, plot_interval=10):
        super().__init__()
        self.model = model; self.test_data = test_data; self.plot_interval = plot_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_interval == 0 or (epoch + 1) == epochs:
            sample_idx = np.random.randint(0, len(self.test_data), 1)
            test_samples = self.test_data[sample_idx]
            _, _, z = self.model.encoder.predict(test_samples, verbose=0)
            reconstructions = self.model.decoder.predict(z, verbose=0)
            latent_samples = analyze_latent_space(self.model.encoder, self.test_data)
            plot_histograms_and_densities(test_samples, reconstructions, latent_samples, epoch + 1)


class LivePlotCallback(keras.callbacks.Callback):
    def __init__(self, model, data, num_images=10):
        super().__init__()
        self.model = model; self.data = data; self.num_images = num_images
        self.fig_width = num_images * 3.5 * HIGH_RES_CONFIG['figsize_multiplier']
        self.fig_height = 3.5 * HIGH_RES_CONFIG['figsize_multiplier']
        self.fig, self.ax = plt.subplots(1, num_images * 2,
                                         figsize=(self.fig_width, self.fig_height),
                                         dpi=HIGH_RES_CONFIG['dpi'])
        plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20})
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        indices = np.random.randint(0, self.data.shape[0], self.num_images)
        original_imgs = self.data[indices]
        reconstructed_imgs = self.model.predict(original_imgs)
        for i in range(self.num_images):
            self.ax[i * 2].imshow(original_imgs[i].squeeze(), cmap='viridis', interpolation='nearest')
            self.ax[i * 2].set_title("Original", fontweight='bold', fontsize=8)
            self.ax[i * 2].axis('off')
            self.ax[i * 2 + 1].imshow(reconstructed_imgs[i].squeeze(), cmap='viridis', interpolation='nearest')
            self.ax[i * 2 + 1].set_title(f"Reconstructed", fontweight='bold', fontsize=8)
            self.ax[i * 2 + 1].axis('off')
        self.fig.suptitle(f'Epoch {epoch + 1}: Original vs Reconstructed Samples',
                          fontsize=18, fontweight='bold')
        save_figure_multiple_formats("training_outputs/training_live",
                                     dpi=HIGH_RES_CONFIG['dpi'], bbox_inches='tight',
                                     facecolor='white', edgecolor='none')
        plt.pause(0.01)
        self.fig.canvas.draw()


class EnhancedMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, vae_model, x_test, metric_interval=metric_interval, log_file='training_metrics.txt'):
        super().__init__()
        self.vae_model = vae_model
        self.x_test = x_test
        self.metric_interval = metric_interval
        self.log_file = log_file
        self.metrics = {'FID_10k': [], 'FRD_10k': [], 'KID_10k': [], 'KRD_10k': [],
                        'FID_500': [], 'FRD_500': [], 'KID_500': [], 'KRD_500': []}
        self.static_metrics = {'Variogram_MSE': [], 'Connectivity_MSE': [],
                               'Histogram_KL': [], 'PCA_Correlation': [], 'MDS_MMD': []}
        with open(self.log_file, 'w') as f:
            f.write("Epoch\tFID_10k\tFRD_10k\tKID_10k\tKRD_10k\tFID_500\tFRD_500\tKID_500\tKRD_500\t"
                    "Variogram_MSE\tConnectivity_MSE\tHistogram_KL\tPCA_Correlation\tMDS_MMD\n")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.metric_interval == 0:
            print(f'\n{"=" * 60}')
            print(f'Computing metrics at epoch {epoch + 1}')
            print(f'{"=" * 60}')
            MemoryOptimizer.check_memory(threshold_mb=8000)
            size = min(FID_SAMPLE_SIZE, self.x_test.shape[0])
            X_pred = self.vae_model.decoder.predict(
                np.random.normal(0, 1, (size, latent_dim)),
                verbose=1, batch_size=32)
            static_results = self.compute_static_metrics_corrected(self.x_test[:size], X_pred[:size])
            if MemoryOptimizer.memory_usage() < 6000:
                traditional_results = self.compute_traditional_metrics(self.x_test[:size], X_pred[:size])
            else:
                print("Skipping traditional metrics due to memory constraints")
                traditional_results = {k: 0 for k in self.metrics}
            for key in self.metrics:
                self.metrics[key].append(traditional_results[key])
            for key in self.static_metrics:
                self.static_metrics[key].append(static_results[key])
            self.save_metrics_to_file(epoch + 1, traditional_results, static_results)
            self.display_metrics(epoch + 1, traditional_results, static_results)
            self.save_numpy_files()

    def compute_static_metrics_corrected(self, real_data, fake_data):
        print("Calculating static geostatistical metrics...")
        n_samples = min(500, len(real_data), len(fake_data))
        real_subset = real_data[:n_samples]
        fake_subset = fake_data[:n_samples]
        metrics = self.compute_all_metrics_corrected(real_subset, fake_subset, threshold=0.5, max_lag=25)
        return {'Variogram_MSE': metrics['variogram_mse'], 'Connectivity_MSE': metrics['connectivity_mse'],
                'Histogram_KL': metrics['hist_kl'], 'PCA_Correlation': metrics['pca_correlation'],
                'MDS_MMD': metrics['mds_mmd']}

    def compute_all_metrics_corrected(self, real_data, fake_data, threshold=0.5, max_lag=25):
        metrics = {}
        print("Calculating variograms...")
        lags, variogram_real = StaticMetrics.variogram_2d(real_data[:100], max_lag=max_lag)
        _, variogram_fake = StaticMetrics.variogram_2d(fake_data[:100], max_lag=max_lag)
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
                pca_correlations.append(np.corrcoef(pca_real[:, i], pca_fake[:, i])[0, 1])
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
        from scipy.spatial.distance import pdist
        n_samples = min(len(mds_real), len(mds_fake), 100)
        mds_real_small = mds_real[:n_samples]; mds_fake_small = mds_fake[:n_samples]
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

        print("Calculating histogram statistics...")
        real_flat = real_data.flatten(); fake_flat = fake_data.flatten()
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
        return metrics

    # ============================================================
    # FID FIX: data sanitization + use of the stable frechet_distance
    # + FID/FRD batched (10 batches x 1000) for the *_10k versions
    # ============================================================
    def compute_traditional_metrics(self, real_data, fake_data):
        """Compute traditional metrics (FID, KID, etc.) with anti-explosion protection.

        FID_10k and FRD_10k are computed with 10000 samples split into 10 batches
        of 1000, accumulating the statistics (mean and covariance) incrementally.
        FID_500, FRD_500, KID_*, KRD_* remain with the previous behavior.
        """
        print("Calculating traditional metrics...")

        # --- 1. Data sanitization ---
        real_data = np.asarray(real_data, dtype=np.float32)
        fake_data = np.asarray(fake_data, dtype=np.float32)

        if not np.all(np.isfinite(fake_data)):
            n_bad = int(np.sum(~np.isfinite(fake_data)))
            print(f"  [FID-FIX] Fake data has {n_bad} NaN/Inf -> cleaning")
            fake_data = np.nan_to_num(fake_data, nan=0.0, posinf=1.0, neginf=-1.0)
        if not np.all(np.isfinite(real_data)):
            n_bad = int(np.sum(~np.isfinite(real_data)))
            print(f"  [FID-FIX] Real data has {n_bad} NaN/Inf -> cleaning")
            real_data = np.nan_to_num(real_data, nan=0.0, posinf=1.0, neginf=-1.0)

        real_data = np.clip(real_data, -1.0, 1.0)
        fake_data = np.clip(fake_data, -1.0, 1.0)

        print(f"  Real data: shape={real_data.shape}, range=[{real_data.min():.3f}, {real_data.max():.3f}]")
        print(f"  Fake data: shape={fake_data.shape}, range=[{fake_data.min():.3f}, {fake_data.max():.3f}]")

        # --- 2. Define subset for batched FID/FRD: 10 batches x 1000 (10000 total) ---
        n_available = min(len(real_data), len(fake_data))
        n_fid = min(FID_SAMPLE_SIZE, n_available)
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

        real_fid_data = real_data[:n_fid]
        fake_fid_data = fake_data[:n_fid]
        print(f"  [FID-FIX] FID/FRD batched: {n_b} batches x {b_sz} = {n_fid} samples")

        # --- 3. Feature extraction (InceptionV3) ---
        real_resized = tf.image.resize(np.repeat(real_fid_data, 3, axis=3), [75, 75]).numpy()
        fake_resized = tf.image.resize(np.repeat(fake_fid_data, 3, axis=3), [75, 75]).numpy()

        f_extract_iv3 = get_inception(shape=(75, 75))
        act_train_iv3 = f_extract_iv3.predict(real_resized, verbose=0)
        act_gen_iv3 = f_extract_iv3.predict(fake_resized, verbose=0)

        act_train_iv3 = np.nan_to_num(act_train_iv3, nan=0.0, posinf=0.0, neginf=0.0)
        act_gen_iv3 = np.nan_to_num(act_gen_iv3, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  IV3 real: shape={act_train_iv3.shape}, mean={act_train_iv3.mean():.3f}, "
              f"std={act_train_iv3.std():.3f}, range=[{act_train_iv3.min():.2f}, {act_train_iv3.max():.2f}]")
        print(f"  IV3 fake: shape={act_gen_iv3.shape}, mean={act_gen_iv3.mean():.3f}, "
              f"std={act_gen_iv3.std():.3f}, range=[{act_gen_iv3.min():.2f}, {act_gen_iv3.max():.2f}]")

        # --- 4. Feature extraction (Reservoir Classifier) ---
        f_extract_RC = get_classifier('Reservoir_classifier.h5')
        act_train_rc = f_extract_RC.predict(real_fid_data, verbose=0)
        act_gen_rc = f_extract_RC.predict(fake_fid_data, verbose=0)

        act_train_rc = np.nan_to_num(act_train_rc, nan=0.0, posinf=0.0, neginf=0.0)
        act_gen_rc = np.nan_to_num(act_gen_rc, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 5. FID_10k / FRD_10k via batched computation (10 batches x 1000) ---
        fid_10k = frechet_distance_batched(
            act_train_iv3, act_gen_iv3, n_batches=n_b, batch_size=b_sz
        )
        frd_10k = frechet_distance_batched(
            act_train_rc, act_gen_rc, n_batches=n_b, batch_size=b_sz
        )

        # --- 6. FID_500 / FRD_500 / KID / KRD (previous behavior) ---
        fid_500 = frechet_distance(act_train_iv3[:500], act_gen_iv3[:500])
        frd_500 = frechet_distance(act_train_rc[:500], act_gen_rc[:500])

        kid_10k = kernel_distance(act_train_iv3, act_gen_iv3, num_blocks=50, max_block_size=500)
        kid_500 = kernel_distance(act_train_iv3[:500], act_gen_iv3[:500], num_blocks=50, max_block_size=500)
        krd_10k = kernel_distance(act_train_rc, act_gen_rc, num_blocks=50, max_block_size=500)
        krd_500 = kernel_distance(act_train_rc[:500], act_gen_rc[:500], num_blocks=50, max_block_size=500)

        def _safe(v, default=1e6):
            return float(v) if np.isfinite(v) else default

        return {
            'FID_10k': _safe(fid_10k), 'FID_500': _safe(fid_500),
            'FRD_10k': _safe(frd_10k), 'FRD_500': _safe(frd_500),
            'KID_10k': _safe(kid_10k, 1.0), 'KID_500': _safe(kid_500, 1.0),
            'KRD_10k': _safe(krd_10k, 1.0), 'KRD_500': _safe(krd_500, 1.0),
        }

    def save_metrics_to_file(self, epoch, traditional_metrics, static_metrics):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch}\t"
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

    def display_metrics(self, epoch, traditional_metrics, static_metrics):
        output = (
            f"\nEpoch {epoch} Metrics:\n{'=' * 40}\n"
            f"Traditional Metrics (10 batches x 1000 for FID/FRD):\n"
            f"  FID(10K): {traditional_metrics['FID_10k']:.2f}\n"
            f"  FID(500): {traditional_metrics['FID_500']:.2f}\n"
            f"  KID(10K): {traditional_metrics['KID_10k']:.4f}\n"
            f"  KID(500): {traditional_metrics['KID_500']:.4f}\n"
            f"  FRD(10K): {traditional_metrics['FRD_10k']:.2f}\n"
            f"  FRD(500): {traditional_metrics['FRD_500']:.2f}\n"
            f"  KRD(10K): {traditional_metrics['KRD_10k']:.2f}\n"
            f"  KRD(500): {traditional_metrics['KRD_500']:.2f}\n\n"
            f"Static Geostatistical Metrics:\n"
            f"  Variogram MSE: {static_metrics['Variogram_MSE']:.4f}\n"
            f"  Connectivity MSE: {static_metrics['Connectivity_MSE']:.4f}\n"
            f"  Histogram KL: {static_metrics['Histogram_KL']:.4f}\n"
            f"  PCA Correlation: {static_metrics['PCA_Correlation']:.4f}\n"
            f"  MDS MMD: {static_metrics['MDS_MMD']:.4f}\n{'=' * 40}"
        )
        print(output)

    def save_numpy_files(self):
        os.makedirs('training_outputs', exist_ok=True)
        for key, values in self.metrics.items():
            np.save(f'training_outputs/{key}.npy', values)
        for key, values in self.static_metrics.items():
            np.save(f'training_outputs/{key}.npy', values)


# ----------------------------------------------------------------------------
# 5.2 Training setup and execution
# ----------------------------------------------------------------------------
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    return lr * tf.math.exp(-0.01)


callbacklr = tf.keras.callbacks.LearningRateScheduler(scheduler)

discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=keras.losses.MeanSquaredError(), metrics=['mse'])

optimizer = keras.optimizers.Adam(learning_rate=lr)
vae_gan.compile(optimizer=optimizer)

enhanced_metric_callback = EnhancedMetricCallback(vae_gan, x_test, metric_interval=metric_interval)
distribution_callback = DistributionPlotCallback(vae_gan, x_test, plot_interval=50)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=100, restore_best_weights=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size)

for batch in train_dataset.take(1):
    print("Shape of batch:", batch[0].shape)

start_time = time.time()

print("Starting VAE-GAN training...")
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Beta (KL weight): {beta}")
print(f"Latent dimension: {latent_dim}")
print(f"FID/FRD sampling: {FID_N_BATCHES} batches x {FID_BATCH_SIZE} samples ({FID_SAMPLE_SIZE} total)")

history = vae_gan.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[
        LivePlotCallback(vae_gan, x_test, num_images=5),
        enhanced_metric_callback,
        callbacklr,
        loss_history_callback,
        distribution_callback,
        early_stopping
    ]
)

plt.ioff()
plt.show()

end_time = time.time()
execution_time = end_time - start_time


# ============================================================================
# 6. EVALUATION & PLOTS
# ============================================================================

# ----------------------------------------------------------------------------
# 6.1 Loss plots
# ----------------------------------------------------------------------------
def plot_total_loss(losses, val_losses):
    figsize = (12 * HIGH_RES_CONFIG['figsize_multiplier'],
               7 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20
    })
    epochs_range = range(1, len(losses) + 1)
    plt.plot(epochs_range, losses, label='Training Total Loss', linewidth=3.0, color='blue')
    plt.plot(epochs_range, val_losses, label='Validation Total Loss', linewidth=3.0, color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.title('VAE-GAN: Total Loss (Training vs Validation)', fontsize=24, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=14, width=1.5)
    save_figure_multiple_formats("training_outputs/total_loss", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


def plot_losses(recon_losses, kl_losses, d_losses, g_losses, val_recon_losses, val_kl_losses, val_g_losses):
    figsize = (16 * HIGH_RES_CONFIG['figsize_multiplier'],
               7 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18
    })
    epochs_range = range(1, len(recon_losses) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, recon_losses, label='Training Recon Loss', linewidth=3.0, color='blue')
    plt.plot(epochs_range, val_recon_losses, label='Validation Recon Loss', linewidth=3.0, color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Reconstruction Loss', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    save_figure_multiple_formats("training_outputs/reconstruction_loss", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, kl_losses, label='Training KL Loss', linewidth=3.0, color='blue')
    plt.plot(epochs_range, val_kl_losses, label='Validation KL Loss', linewidth=3.0, color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('KL Loss', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    save_figure_multiple_formats("training_outputs/kl_loss", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.tight_layout()
    save_figure_multiple_formats("training_outputs/losses_combined", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    figsize_gan = (16 * HIGH_RES_CONFIG['figsize_multiplier'],
                   7 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize_gan, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18
    })
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, d_losses, label='Discriminator Loss', linewidth=3.0, color='green')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Discriminator Loss', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, g_losses, label='Generator Loss', linewidth=3.0, color='purple')
    plt.plot(epochs_range, val_g_losses, label='Validation Gen Loss', linewidth=3.0, color='pink', linestyle='--')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Generator Loss', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.tight_layout()
    save_figure_multiple_formats("training_outputs/gan_losses", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    figsize_dvg = (12 * HIGH_RES_CONFIG['figsize_multiplier'],
                   7 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize_dvg, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18
    })
    plt.plot(epochs_range, d_losses, label='Discriminator', color='green', linewidth=3.0)
    plt.plot(epochs_range, g_losses, label='Generator', color='purple', linewidth=3.0)
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Discriminator vs Generator Losses', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    save_figure_multiple_formats("training_outputs/discriminator_vs_generator", dpi=HIGH_RES_CONFIG['dpi'],
                                 bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


# ----------------------------------------------------------------------------
# 6.2 Final analysis plots
# ----------------------------------------------------------------------------
test_samples = x_test[:5]
_, _, z = vae_gan.encoder.predict(test_samples, verbose=0)
reconstructions = vae_gan.decoder.predict(z, verbose=0)
latent_samples = analyze_latent_space(vae_gan.encoder, x_test, num_samples=1000)

plot_histograms_and_densities(test_samples, reconstructions, latent_samples)

# Latent space 2D projection
figsize = (14 * HIGH_RES_CONFIG['figsize_multiplier'],
           7 * HIGH_RES_CONFIG['figsize_multiplier'])
plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
    'xtick.labelsize': 18, 'ytick.labelsize': 18
})
z_samples = vae_gan.encoder.predict(x_test[:1000], verbose=0)[2]
pca = PCA(n_components=2)
z_2d = pca.fit_transform(z_samples)
plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=HIGH_RES_CONFIG['marker_size'])
plt.title('2D Projection of Latent Space (PCA)', fontsize=24, fontweight='bold')
plt.xlabel('PCA Component 1', fontweight='bold', fontsize=22)
plt.ylabel('PCA Component 2', fontweight='bold', fontsize=22)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
plt.tick_params(axis='both', which='major', labelsize=18, width=2)
save_figure_multiple_formats("training_outputs/latent_space_2d", dpi=HIGH_RES_CONFIG['dpi'],
                             bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()


def plot_static_metrics_comparison(real_data, vae_model, save_path_base=None):
    n_samples = min(500, len(real_data))
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    fake_data = vae_model.decoder.predict(noise, verbose=0)
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.spatial.distance import pdist
    metrics = {}
    lags, variogram_real = StaticMetrics.variogram_2d(real_data[:100], max_lag=25)
    _, variogram_fake = StaticMetrics.variogram_2d(fake_data[:100], max_lag=25)
    metrics['variogram_mse'] = np.mean((variogram_real - variogram_fake) ** 2)
    _, connectivity_real = StaticMetrics.connectivity_function(real_data[:100], 0.5, lag_distances=lags)
    _, connectivity_fake = StaticMetrics.connectivity_function(fake_data[:100], 0.5, lag_distances=lags)
    metrics['connectivity_mse'] = np.mean((connectivity_real - connectivity_fake) ** 2)
    real_flat = real_data.flatten(); fake_flat = fake_data.flatten()
    all_data = np.concatenate([real_flat, fake_flat])
    data_min, data_max = np.min(all_data), np.max(all_data)
    hist_real, bin_edges = np.histogram(real_flat, bins=100, range=(data_min, data_max), density=False)
    hist_fake, _ = np.histogram(fake_flat, bins=100, range=(data_min, data_max), density=False)
    hist_real = hist_real.astype(float) / np.sum(hist_real)
    hist_fake = hist_fake.astype(float) / np.sum(hist_fake)
    epsilon = 1e-10
    hist_real = np.clip(hist_real, epsilon, 1); hist_fake = np.clip(hist_fake, epsilon, 1)
    metrics['hist_kl'] = np.sum(hist_real * np.log(hist_real / hist_fake))
    pca_real, _ = StaticMetrics.calculate_pca_features(real_data[:500])
    pca_fake, _ = StaticMetrics.calculate_pca_features(fake_data[:500])
    if pca_real.shape[0] == pca_fake.shape[0]:
        n_components = min(2, pca_real.shape[1], pca_fake.shape[1])
        pca_correlations = [np.corrcoef(pca_real[:, i], pca_fake[:, i])[0, 1] for i in range(n_components)]
        metrics['pca_correlation'] = np.mean(pca_correlations)
    else:
        metrics['pca_correlation'] = 0.0
    mds_real = StaticMetrics.calculate_mds_features(real_data[:500])
    mds_fake = StaticMetrics.calculate_mds_features(fake_data[:500])
    n_samples = min(len(mds_real), len(mds_fake), 100)
    mds_real_small = mds_real[:n_samples]; mds_fake_small = mds_fake[:n_samples]
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

    figsize = (20 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    fig.suptitle('Static Geostatistical Metrics Comparison (VAE-GAN)', fontsize=26, fontweight='bold', y=1.02)
    for ax in axes.flat:
        ax.tick_params(labelsize=16, width=1.5)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.title.set_size(22)

    ax = axes[0, 0]
    ax.plot(lags, variogram_real, 'b-', linewidth=3.0, label='Real')
    ax.plot(lags, variogram_fake, 'r--', linewidth=3.0, label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=20)
    ax.set_ylabel('Variogram', fontweight='bold', fontsize=20)
    ax.set_title(f'Variogram (MSE: {metrics["variogram_mse"]:.4f})', fontweight='bold', fontsize=22)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax = axes[0, 1]
    ax.plot(lags, connectivity_real, 'b-', linewidth=3.0, label='Real')
    ax.plot(lags, connectivity_fake, 'r--', linewidth=3.0, label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=20)
    ax.set_ylabel('Connectivity', fontweight='bold', fontsize=20)
    ax.set_title(f'Connectivity Function (MSE: {metrics["connectivity_mse"]:.4f})', fontweight='bold', fontsize=22)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax = axes[0, 2]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_centers[1] - bin_centers[0]
    ax.bar(bin_centers, hist_real, width=bar_width, alpha=0.6, label='Real',
           color='blue', edgecolor='black', linewidth=1.0)
    ax.plot(bin_centers, hist_fake, 'r-', linewidth=3.0, label='Generated')
    ax.set_xlabel('Value', fontweight='bold', fontsize=20)
    ax.set_ylabel('Density', fontweight='bold', fontsize=20)
    ax.set_title(f'Histogram (KL: {metrics["hist_kl"]:.4f})', fontweight='bold', fontsize=22)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax = axes[1, 0]
    ax.scatter(pca_real[:100, 0], pca_real[:100, 1], alpha=0.6, label='Real',
               s=HIGH_RES_CONFIG['marker_size'], color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(pca_fake[:100, 0], pca_fake[:100, 1], alpha=0.6, label='Generated',
               s=HIGH_RES_CONFIG['marker_size'], color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('PC1', fontweight='bold', fontsize=20)
    ax.set_ylabel('PC2', fontweight='bold', fontsize=20)
    ax.set_title(f'PCA Projection (Corr: {metrics["pca_correlation"]:.4f})', fontweight='bold', fontsize=22)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax = axes[1, 1]
    ax.scatter(mds_real[:100, 0], mds_real[:100, 1], alpha=0.6, label='Real',
               s=HIGH_RES_CONFIG['marker_size'], color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(mds_fake[:100, 0], mds_fake[:100, 1], alpha=0.6, label='Generated',
               s=HIGH_RES_CONFIG['marker_size'], color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('MDS Dimension 1', fontweight='bold', fontsize=20)
    ax.set_ylabel('MDS Dimension 2', fontweight='bold', fontsize=20)
    ax.set_title(f'MDS Projection (MMD: {metrics["mds_mmd"]:.4f})', fontweight='bold', fontsize=22)
    ax.legend(fontsize=16); ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

    ax = axes[1, 2]
    stats_text = (f'Variogram MSE: {metrics["variogram_mse"]:.6f}\n'
                  f'Connectivity MSE: {metrics["connectivity_mse"]:.6f}\n'
                  f'Histogram KL: {metrics["hist_kl"]:.6f}\n'
                  f'PCA Correlation: {metrics["pca_correlation"]:.4f}\n'
                  f'MDS MMD: {metrics["mds_mmd"]:.4f}\n\n'
                  f'Real mean: {np.mean(real_flat):.4f}\n'
                  f'Fake mean: {np.mean(fake_flat):.4f}\n'
                  f'Real std: {np.std(real_flat):.4f}\n'
                  f'Fake std: {np.std(fake_flat):.4f}')
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Summary Statistics', fontweight='bold', fontsize=22)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path_base:
        save_figure_multiple_formats(save_path_base, dpi=HIGH_RES_CONFIG['dpi'],
                                     bbox_inches='tight', facecolor='white', edgecolor='none')
    else:
        plt.show()
    return metrics


static_metrics_results = plot_static_metrics_comparison(
    x_test, vae_gan, save_path_base='training_outputs/static_metrics_comparison_corrected')

dist_data = {'original': test_samples.flatten(), 'reconstructed': reconstructions.flatten(),
             'latent': latent_samples}
with open('training_outputs/distribution_data.pkl', 'wb') as f:
    pickle.dump(dist_data, f)


def plot_discriminator_predictions(discriminator, real_data, fake_data, save_path_base=None):
    figsize = (12 * HIGH_RES_CONFIG['figsize_multiplier'],
               7 * HIGH_RES_CONFIG['figsize_multiplier'])
    plt.figure(figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    plt.rcParams.update({
        'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
        'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18
    })
    real_preds = discriminator.predict(real_data[:100], verbose=0).flatten()
    fake_preds = discriminator.predict(fake_data[:100], verbose=0).flatten()
    plt.hist(real_preds, bins=30, alpha=0.7, label='Real Images', color='green', edgecolor='black')
    plt.hist(fake_preds, bins=30, alpha=0.7, label='Generated Images', color='red', edgecolor='black')
    plt.xlabel('Discriminator Output', fontweight='bold', fontsize=22)
    plt.ylabel('Frequency', fontweight='bold', fontsize=22)
    plt.title('Discriminator Predictions Distribution', fontsize=24, fontweight='bold')
    plt.legend(fontsize=18); plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    real_mean = np.mean(real_preds); fake_mean = np.mean(fake_preds)
    plt.axvline(real_mean, color='green', linestyle='--', linewidth=3, label=f'Real mean: {real_mean:.3f}')
    plt.axvline(fake_mean, color='red', linestyle='--', linewidth=3, label=f'Fake mean: {fake_mean:.3f}')
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    if save_path_base:
        save_figure_multiple_formats(save_path_base, dpi=HIGH_RES_CONFIG['dpi'],
                                     bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    else:
        plt.show()


fake_data_for_disc = vae_gan.decoder.predict(np.random.normal(0, 1, (100, latent_dim)), verbose=0)
plot_discriminator_predictions(discriminator, x_test[:100], fake_data_for_disc,
                               save_path_base='training_outputs/discriminator_predictions')


def generate_final_report(filepath, params, execution_time, loss_history, metrics_history, static_metrics_results):
    with open(filepath, 'w', encoding='utf-8') as f:
        from datetime import datetime
        f.write("=" * 80 + "\n")
        f.write("                    FINAL VAE-GAN TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-" * 80 + "\nTRAINING PARAMETERS\n" + "-" * 80 + "\n")
        for key, value in params.items():
            f.write(f"  {key:<30}: {value}\n")
        f.write("\n")
        f.write("-" * 80 + "\nEXECUTION TIME\n" + "-" * 80 + "\n")
        f.write(f"  Total Time (seconds) : {execution_time:.2f}s\n")
        f.write(f"  Total Time (minutes) : {execution_time / 60:.2f}min\n")
        f.write(f"  Total Time (hours)   : {execution_time / 3600:.2f}h\n\n")
        f.write("-" * 80 + "\nTRAINING METRICS (FINAL VALUES)\n" + "-" * 80 + "\n")
        if loss_history.losses:
            f.write(f"  Total epochs trained: {len(loss_history.losses)}\n\n")
            f.write("  Training Losses (last epoch):\n")
            f.write(f"    Total Loss       : {loss_history.losses[-1]:.6f}\n")
            f.write(f"    Reconstruction   : {loss_history.recon_losses[-1]:.6f}\n")
            f.write(f"    KL Loss          : {loss_history.kl_losses[-1]:.6f}\n")
            f.write(f"    Generator Loss   : {loss_history.g_losses[-1]:.6f}\n")
            f.write(f"    Discriminator Loss: {loss_history.d_losses[-1]:.6f}\n\n")
            f.write("  Validation Losses (last epoch):\n")
            f.write(f"    Total Loss       : {loss_history.val_losses[-1]:.6f}\n")
            f.write(f"    Reconstruction   : {loss_history.val_recon_losses[-1]:.6f}\n")
            f.write(f"    KL Loss          : {loss_history.val_kl_losses[-1]:.6f}\n")
            f.write(f"    Generator Loss   : {loss_history.val_g_losses[-1]:.6f}\n\n")
            f.write("  Best values during training:\n")
            f.write(f"    Lowest Total Loss (train)  : {min(loss_history.losses):.6f} "
                    f"(epoch {np.argmin(loss_history.losses) + 1})\n")
            f.write(f"    Lowest Total Loss (val)    : {min(loss_history.val_losses):.6f} "
                    f"(epoch {np.argmin(loss_history.val_losses) + 1})\n")
            f.write(f"    Lowest Recon Loss (train)  : {min(loss_history.recon_losses):.6f} "
                    f"(epoch {np.argmin(loss_history.recon_losses) + 1})\n")
            f.write(f"    Lowest Recon Loss (val)    : {min(loss_history.val_recon_losses):.6f} "
                    f"(epoch {np.argmin(loss_history.val_recon_losses) + 1})\n")
            f.write(f"    Lowest KL Loss (train)     : {min(loss_history.kl_losses):.6f} "
                    f"(epoch {np.argmin(loss_history.kl_losses) + 1})\n")
            f.write(f"    Lowest KL Loss (val)       : {min(loss_history.val_kl_losses):.6f} "
                    f"(epoch {np.argmin(loss_history.val_kl_losses) + 1})\n\n")
        f.write("-" * 80 + "\nTRADITIONAL METRICS (LAST EVALUATION)\n" + "-" * 80 + "\n")
        if hasattr(metrics_history, 'metrics'):
            for key, values in metrics_history.metrics.items():
                if values:
                    f.write(f"  {key:<15}: {values[-1]:.4f}\n")
            f.write("\n")
        f.write("-" * 80 + "\nGEOSTATISTICAL METRICS (LAST EVALUATION)\n" + "-" * 80 + "\n")
        if hasattr(metrics_history, 'static_metrics'):
            for key, values in metrics_history.static_metrics.items():
                if values:
                    f.write(f"  {key:<20}: {values[-1]:.6f}\n")
            f.write("\n")
        if static_metrics_results:
            f.write("-" * 80 + "\nFINAL GEOSTATISTICAL METRICS (POST-TRAINING)\n" + "-" * 80 + "\n")
            f.write(f"  Variogram MSE      : {static_metrics_results.get('variogram_mse', 'N/A'):.6f}\n")
            f.write(f"  Connectivity MSE   : {static_metrics_results.get('connectivity_mse', 'N/A'):.6f}\n")
            f.write(f"  Histogram KL       : {static_metrics_results.get('hist_kl', 'N/A'):.6f}\n")
            f.write(f"  PCA Correlation    : {static_metrics_results.get('pca_correlation', 'N/A'):.4f}\n")
            f.write(f"  MDS MMD            : {static_metrics_results.get('mds_mmd', 'N/A'):.4f}\n\n")
        f.write("-" * 80 + "\nMODEL ARCHITECTURE\n" + "-" * 80 + "\n")
        f.write("  Encoder:\n")
        f.write(f"    Input Shape    : ({ni}, {nj}, {nz})\n")
        f.write(f"    Latent Dim     : {latent_dim}\n")
        f.write("    Conv Layers    : 3 (32, 64, 128 filters)\n")
        f.write("    Activation     : LeakyReLU (alpha=0.2)\n")
        f.write("    Dense Layer    : 32\n")
        f.write("    Dropout        : 0.2\n\n")
        f.write("  Decoder:\n")
        f.write(f"    Input Dim      : {latent_dim}\n")
        f.write("    Output Shape   : (48, 48, 1)\n")
        f.write("    ConvTranspose  : 3 (128, 64, 32 filters)\n")
        f.write("    Activation     : LeakyReLU (alpha=0.2)\n")
        f.write("    Output Act     : tanh\n\n")
        f.write("  Discriminator:\n")
        f.write(f"    Input Shape    : ({ni}, {nj}, {nz})\n")
        f.write("    Conv Layers    : 3 (32, 64, 128 filters)\n")
        f.write("    Activation     : LeakyReLU (alpha=0.2)\n")
        f.write("    Output Act     : sigmoid\n")
        f.write("    Loss           : MSE\n\n")
        f.write("-" * 80 + "\nFINAL SUMMARY\n" + "-" * 80 + "\n")
        f.write(f"  Training status         : "
                f"{'COMPLETED' if len(loss_history.losses) == epochs else 'INTERRUPTED (Early Stopping)'}\n")
        f.write(f"  Epochs executed         : {len(loss_history.losses)}\n")
        f.write(f"  Total time              : {execution_time / 3600:.2f} hours\n")
        f.write(f"  Average time per epoch  : "
                f"{(execution_time / len(loss_history.losses)) if loss_history.losses else 0:.2f} seconds\n")
        f.write(f"  Dataset                 : {name}\n")
        f.write(f"  Training samples        : {len(x_train)}\n")
        f.write(f"  Test samples            : {len(x_test)}\n\n")
        f.write("\n" + "=" * 80 + "\n                    END OF REPORT\n" + "=" * 80 + "\n")
    print(f"\n✓ Final report saved to: {filepath}")


# ----------------------------------------------------------------------------
# 6.3 Execute loss plots + final report
# ----------------------------------------------------------------------------
plot_losses(loss_history_callback.recon_losses, loss_history_callback.kl_losses,
            loss_history_callback.d_losses, loss_history_callback.g_losses,
            loss_history_callback.val_recon_losses, loss_history_callback.val_kl_losses,
            loss_history_callback.val_g_losses)

training_params = {
    'Dataset': name,
    'Latent dimension (zdim)': zdim,
    'Configured epochs': epochs,
    'Batch size': batch_size,
    'Learning rate': lr,
    'Beta (KL weight)': beta,
    'Image dimensions (ni x nj x nz)': f'{ni} x {nj} x {nz}',
    'Training samples': len(x_train),
    'Test samples': len(x_test),
    'Model': 'VAE-GAN with reduced architecture',
    'LR Scheduler': 'Exponential decay after epoch 50',
    'Optimizer': 'Adam',
    'Discriminator Loss': 'MSE',
    'Early Stopping': 'Yes (patience=100, monitor=val_loss)',
    'Target Score (smooth label)': 0.9,
    'G Weight': 0.5,
    'FID': f'Stable + batched ({FID_N_BATCHES} batches x {FID_BATCH_SIZE} = {FID_SAMPLE_SIZE} samples for FID_10k/FRD_10k)',
}

report_path = os.path.join('training_outputs', 'final_training_report.txt')
generate_final_report(filepath=report_path, params=training_params,
                      execution_time=execution_time, loss_history=loss_history_callback,
                      metrics_history=enhanced_metric_callback,
                      static_metrics_results=static_metrics_results)


# ============================================================================
# 7. SAVING
# ============================================================================

# ----------------------------------------------------------------------------
# 7.1 Execution time and loss history files
# ----------------------------------------------------------------------------
def save_execution_time(filename, execution_time):
    with open(filename, 'w') as f:
        f.write(f"=== Execution Time ===\n")
        f.write(f"Total: {execution_time:.2f} seconds\n")
        f.write(f"Total: {execution_time / 60:.2f} minutes\n")
        f.write(f"Total: {execution_time / 3600:.2f} hours\n")
        f.write(f"\nSettings:\n- Batch size: {batch_size}\n- Epochs: {epochs}\n"
                f"- Latent dim: {latent_dim}\n- Learning rate: {lr}\n"
                f"- Beta (KL weight): {beta}\n"
                f"- Model: VAE-GAN with 10% reduced architecture\n"
                f"- LR Scheduler: Exponential decay after epoch 50\n"
                f"- FID: Stable (PCA-reduced when rank-deficient + eps regularization)\n"
                f"- FID/FRD sampling: {FID_N_BATCHES} batches x {FID_BATCH_SIZE} samples ({FID_SAMPLE_SIZE} total)\n")


def save_metrics_to_file(filename, history, loss_history_callback):
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    recon_loss = history.history.get('recon_loss', [0] * len(loss) if loss else [])
    val_recon_loss = history.history.get('val_recon_loss', [0] * len(loss) if loss else [])
    kl_loss = history.history.get('kl_loss', [0] * len(loss) if loss else [])
    val_kl_loss = history.history.get('val_kl_loss', [0] * len(loss) if loss else [])
    d_loss = history.history.get('d_loss', [0] * len(loss) if loss else [])
    g_loss = history.history.get('g_loss', [0] * len(loss) if loss else [])
    if not loss and hasattr(loss_history_callback, 'recon_losses'):
        print("Warning: Using data from loss_history_callback instead of history")
        loss = loss_history_callback.losses
        val_loss = loss_history_callback.val_losses
        recon_loss = loss_history_callback.recon_losses
        kl_loss = loss_history_callback.kl_losses
        d_loss = loss_history_callback.d_losses
        g_loss = loss_history_callback.g_losses
        val_recon_loss = loss_history_callback.val_recon_losses
        val_kl_loss = loss_history_callback.val_kl_losses
        if not loss: loss = [r + k + g for r, k, g in zip(recon_loss, kl_loss, g_loss)]
        if not val_loss: val_loss = [r + k for r, k in zip(val_recon_loss, val_kl_loss)]
    with open(filename, 'w') as f:
        f.write("Epoch\tLoss\tVal_Loss\tRecon_Loss\tVal_Recon_Loss\tKL_Loss\tVal_KL_Loss\tD_Loss\tG_Loss\n")
        for i in range(len(loss)):
            f.write(f"{i + 1}\t{loss[i]:.6f}\t{val_loss[i]:.6f}\t"
                    f"{recon_loss[i]:.6f}\t{val_recon_loss[i]:.6f}\t"
                    f"{kl_loss[i]:.6f}\t{val_kl_loss[i]:.6f}\t"
                    f"{d_loss[i]:.6f}\t{g_loss[i]:.6f}\n")
    plot_total_loss(loss, val_loss)


def save_losses_to_file(filename, recon_losses, kl_losses, d_losses, g_losses,
                        val_recon_losses, val_kl_losses, val_g_losses):
    with open(filename, 'w') as f:
        f.write("Epoch\tTrain_Recon_Loss\tTrain_KL_Loss\tD_Loss\tG_Loss\tVal_Recon_Loss\tVal_KL_Loss\tVal_G_Loss\n")
        for epoch, (recon_loss, kl_loss, d_loss, g_loss, val_recon, val_kl, val_g) in enumerate(
                zip(recon_losses, kl_losses, d_losses, g_losses, val_recon_losses, val_kl_losses, val_g_losses)):
            f.write(f"{epoch + 1}\t{recon_loss:.6f}\t{kl_loss:.6f}\t{d_loss:.6f}\t{g_loss:.6f}\t"
                    f"{val_recon:.6f}\t{val_kl:.6f}\t{val_g:.6f}\n")


# ----------------------------------------------------------------------------
# 7.2 Execute saves
# ----------------------------------------------------------------------------
os.makedirs('training_outputs', exist_ok=True)
save_execution_time("training_outputs/execution_time.txt", execution_time)
print(f"\n[INFO] Total execution time: {execution_time / 3600:.2f} hours")

save_losses_to_file("training_outputs/loss_history.txt",
                    loss_history_callback.recon_losses, loss_history_callback.kl_losses,
                    loss_history_callback.d_losses, loss_history_callback.g_losses,
                    loss_history_callback.val_recon_losses, loss_history_callback.val_kl_losses,
                    loss_history_callback.val_g_losses)

save_metrics_to_file("training_outputs/training_metrics.txt", history, loss_history_callback)

# ----------------------------------------------------------------------------
# 7.3 Save models
# ----------------------------------------------------------------------------
encoder.save_weights('training_outputs/encoder_weights.h5')
decoder.save_weights('training_outputs/decoder_weights.h5')
discriminator.save_weights('training_outputs/discriminator_weights.h5')

encoder.save('training_outputs/encoder_model.h5')
decoder.save('training_outputs/decoder_model.h5')
discriminator.save('training_outputs/discriminator_model.h5')

# ----------------------------------------------------------------------------
# 7.4 Final summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VAE-GAN TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Execution time: {execution_time / 3600:.2f} hours")
if len(loss_history_callback.losses) > 0:
    print(f"Total epochs trained: {len(loss_history_callback.losses)}")
    print(f"Final Training Total Loss: {loss_history_callback.losses[-1]:.4f}")
    print(f"Final Validation Total Loss: {loss_history_callback.val_losses[-1]:.4f}")
print("=" * 60)
print(f"\nFinal complete report saved to: {report_path}")
print("=" * 60)
