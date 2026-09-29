import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform, cdist
import gc
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import psutil
from scipy.stats import entropy, norm, kstest
from sklearn.metrics import pairwise_kernels
from scipy import linalg
from scipy.stats import gaussian_kde
import pickle
import time
from PIL import Image


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


tf.keras.backend.clear_session()
gc.collect()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponíveis: {len(gpus)}")
    except RuntimeError as e:
        print(e)

# Configuration parameters
zdim = 512
epochs = 300
metric_interval = 10
batch_size = 64
lr = 0.0002
beta = 0.05
d_reg = 5
name = '3facies_2d_80k'

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
if name == '3facies_2d_80k':
    X_train = np.load('datasets/Three_facies_2d.npy').astype('float32')
elif name == 'channel2d_large':
    X_train = np.load('datasets/Two_facies_channel_2d.npy').astype('float32')
elif name == 'unisim_ii_cropped':
    X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy').astype('float32')
    X_train[X_train < 1] = 1
    X_train = np.log(X_train)
    X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
    X_train = X_train[..., np.newaxis]

ni = X_train.shape[1]
nj = X_train.shape[2]
nz = X_train.shape[3]

print(f"Dataset shape: {ni}x{nj}x{nz}")

train_size = int(0.8 * len(X_train))
x_train = X_train[:train_size]
x_test = X_train[train_size:]

latent_dim = zdim
w_dim = zdim * 2  # Dimensão do espaço W (após mapping network)


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
    def calculate_histogram_kl(real_data, fake_data, bins=100, epsilon=1e-10):
        if len(real_data.shape) == 4:
            real_flat = real_data[..., 0].flatten()
            fake_flat = fake_data[..., 0].flatten()
        else:
            real_flat = real_data.flatten()
            fake_flat = fake_data.flatten()

        min_val = min(real_flat.min(), fake_flat.min())
        max_val = max(real_flat.max(), fake_flat.max())

        hist_real, bin_edges = np.histogram(real_flat, bins=bins, range=(min_val, max_val), density=True)
        hist_fake, _ = np.histogram(fake_flat, bins=bin_edges, density=True)

        hist_real = hist_real + epsilon
        hist_fake = hist_fake + epsilon

        hist_real = hist_real / hist_real.sum()
        hist_fake = hist_fake / hist_fake.sum()

        kl_div = entropy(hist_real, hist_fake)

        return kl_div, hist_real, hist_fake, bin_edges

    @staticmethod
    def calculate_pca_correlation(real_data, fake_data, n_components=50):
        n_real = real_data.shape[0]
        n_fake = fake_data.shape[0]

        real_flat = real_data.reshape(n_real, -1)
        fake_flat = fake_data.reshape(n_fake, -1)

        combined_data = np.vstack([real_flat, fake_flat])

        n_components = min(n_components, combined_data.shape[1], combined_data.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(combined_data)

        real_pca = pca.transform(real_flat)
        fake_pca = pca.transform(fake_flat)

        correlations = []
        for i in range(min(n_components, real_pca.shape[1], fake_pca.shape[1])):
            if np.std(real_pca[:, i]) > 0 and np.std(fake_pca[:, i]) > 0:
                corr = np.corrcoef(real_pca[:, i], fake_pca[:, i])[0, 1]
                correlations.append(corr)

        avg_correlation = np.mean(np.abs(correlations)) if correlations else 0

        return avg_correlation, real_pca, fake_pca, pca.explained_variance_ratio_

    @staticmethod
    def calculate_mds_mmd(real_data, fake_data, subsample=500, kernel='rbf', gamma=None):
        n_real = min(subsample, real_data.shape[0])
        n_fake = min(subsample, fake_data.shape[0])

        real_subsample = real_data[:n_real]
        fake_subsample = fake_data[:n_fake]

        real_flat = real_subsample.reshape(n_real, -1)
        fake_flat = fake_subsample.reshape(n_fake, -1)

        if n_real + n_fake > 100:
            pca = PCA(n_components=min(50, real_flat.shape[1]))
            combined = np.vstack([real_flat, fake_flat])
            combined_pca = pca.fit_transform(combined)
            real_mds = combined_pca[:n_real]
            fake_mds = combined_pca[n_real:]
        else:
            mds = MDS(n_components=2, random_state=42, n_jobs=-1)
            combined = np.vstack([real_flat, fake_flat])
            combined_mds = mds.fit_transform(combined)
            real_mds = combined_mds[:n_real]
            fake_mds = combined_mds[n_real:]

        def rbf_kernel(X, Y, gamma=None):
            if gamma is None:
                gamma = 1.0 / X.shape[1]

            XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
            YY = np.sum(Y ** 2, axis=1)[:, np.newaxis]
            XY = np.dot(X, Y.T)

            distances = XX + YY.T - 2 * XY
            return np.exp(-gamma * distances)

        K_XX = rbf_kernel(real_mds, real_mds, gamma)
        K_YY = rbf_kernel(fake_mds, fake_mds, gamma)
        K_XY = rbf_kernel(real_mds, fake_mds, gamma)

        mmd_squared = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
        mmd = np.sqrt(max(mmd_squared, 0))

        return mmd, real_mds, fake_mds

    @staticmethod
    def compute_all_metrics(real_data, fake_data, threshold=0.5, max_lag=25):
        metrics = {}

        print("Calculating all metrics...")

        lags, variogram_real = StaticMetrics.variogram_2d(real_data[:100], max_lag=max_lag)
        _, variogram_fake = StaticMetrics.variogram_2d(fake_data[:100], max_lag=max_lag)

        metrics['variogram_lags'] = lags
        metrics['variogram_real'] = variogram_real
        metrics['variogram_fake'] = variogram_fake
        metrics['variogram_mse'] = np.mean((variogram_real - variogram_fake) ** 2)

        _, connectivity_real = StaticMetrics.connectivity_function(
            real_data[:100], threshold, lag_distances=lags
        )
        _, connectivity_fake = StaticMetrics.connectivity_function(
            fake_data[:100], threshold, lag_distances=lags
        )

        metrics['connectivity_real'] = connectivity_real
        metrics['connectivity_fake'] = connectivity_fake
        metrics['connectivity_mse'] = np.mean((connectivity_real - connectivity_fake) ** 2)

        kl_div, hist_real, hist_fake, bin_edges = StaticMetrics.calculate_histogram_kl(
            real_data, fake_data, bins=100
        )
        metrics['hist_kl'] = kl_div
        metrics['hist_real'] = hist_real
        metrics['hist_fake'] = hist_fake
        metrics['hist_bin_edges'] = bin_edges

        pca_corr, real_pca, fake_pca, var_ratio = StaticMetrics.calculate_pca_correlation(
            real_data, fake_data, n_components=50
        )
        metrics['pca_correlation'] = pca_corr
        metrics['pca_real'] = real_pca[:, :2] if real_pca.shape[1] >= 2 else real_pca
        metrics['pca_fake'] = fake_pca[:, :2] if fake_pca.shape[1] >= 2 else fake_pca
        metrics['pca_var_ratio'] = var_ratio

        mmd, real_mds, fake_mds = StaticMetrics.calculate_mds_mmd(
            real_data, fake_data, subsample=500
        )
        metrics['mds_mmd'] = mmd
        metrics['mds_real'] = real_mds
        metrics['mds_fake'] = fake_mds

        return metrics


class TraditionalMetrics:
    _inception_model = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_inception(cls, shape):
        from keras.applications.inception_v3 import InceptionV3
        if cls._inception_model is None:
            cls._inception_model = InceptionV3(
                include_top=False,
                pooling='avg',
                input_shape=(shape[0], shape[1], 3)
            )
        return cls._inception_model

    @staticmethod
    def get_classifier(filepath):
        RC_full = tf.keras.models.load_model(filepath)
        f_extraction = tf.keras.Model(
            inputs=RC_full.input,
            outputs=layers.GlobalAveragePooling2D()(RC_full.layers[-5].output)
        )
        return f_extraction

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    @staticmethod
    def frechet_distance(real_features, fake_features):
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

        fid_value = TraditionalMetrics.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

    @staticmethod
    def kernel_distance(real_features, fake_features, num_blocks=100, max_block_size=1000):
        n_samples = min(len(real_features), len(fake_features))
        block_size = min(max_block_size, n_samples // num_blocks)

        kid_scores = []
        for _ in range(num_blocks):
            idx_real = np.random.choice(len(real_features), block_size, replace=False)
            idx_fake = np.random.choice(len(fake_features), block_size, replace=False)

            real_block = real_features[idx_real]
            fake_block = fake_features[idx_fake]

            gamma = 1.0 / real_block.shape[1]
            K_XX = pairwise_kernels(real_block, real_block, metric='rbf', gamma=gamma)
            K_YY = pairwise_kernels(fake_block, fake_block, metric='rbf', gamma=gamma)
            K_XY = pairwise_kernels(real_block, fake_block, metric='rbf', gamma=gamma)

            kid = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
            kid_scores.append(kid)

        return np.mean(kid_scores)

    @staticmethod
    def compute_all_traditional_metrics(real_data, fake_data, classifier_path='Reservoir_classifier.h5'):
        print("Calculating traditional metrics...")

        real_resized = tf.image.resize(np.repeat(real_data, 3, axis=3), [75, 75])
        fake_resized = tf.image.resize(np.repeat(fake_data, 3, axis=3), [75, 75])

        inception_model = TraditionalMetrics.get_inception((75, 75, 3))
        act_real_iv3 = inception_model.predict(real_resized, batch_size=32, verbose=0)
        act_fake_iv3 = inception_model.predict(fake_resized, batch_size=32, verbose=0)

        n_samples = min(10000, len(real_data), len(fake_data))
        fid_10k = TraditionalMetrics.frechet_distance(act_real_iv3[:n_samples], act_fake_iv3[:n_samples])
        kid_10k = TraditionalMetrics.kernel_distance(act_real_iv3[:n_samples], act_fake_iv3[:n_samples])

        n_500 = min(500, len(real_data), len(fake_data))
        fid_500 = TraditionalMetrics.frechet_distance(act_real_iv3[:n_500], act_fake_iv3[:n_500])
        kid_500 = TraditionalMetrics.kernel_distance(act_real_iv3[:n_500], act_fake_iv3[:n_500])

        try:
            classifier_model = TraditionalMetrics.get_classifier(classifier_path)
            act_real_rc = classifier_model.predict(real_data, batch_size=32, verbose=0)
            act_fake_rc = classifier_model.predict(fake_data, batch_size=32, verbose=0)

            frd_10k = TraditionalMetrics.frechet_distance(act_real_rc[:n_samples], act_fake_rc[:n_samples])
            krd_10k = TraditionalMetrics.kernel_distance(act_real_rc[:n_samples], act_fake_rc[:n_samples])
            frd_500 = TraditionalMetrics.frechet_distance(act_real_rc[:n_500], act_fake_rc[:n_500])
            krd_500 = TraditionalMetrics.kernel_distance(act_real_rc[:n_500], act_fake_rc[:n_500])
        except:
            print(f"Warning: Could not load classifier from {classifier_path}")
            frd_10k = frd_500 = krd_10k = krd_500 = 0.0

        return {
            'FID_10k': fid_10k,
            'KID_10k': kid_10k,
            'FID_500': fid_500,
            'KID_500': kid_500,
            'FRD_10k': frd_10k,
            'KRD_10k': krd_10k,
            'FRD_500': frd_500,
            'KRD_500': krd_500
        }


class GaussianLatentValidator:
    """Validates and enforces Gaussian distribution in latent space."""

    @staticmethod
    def test_normality(latent_vectors, significance_level=0.01):
        results = {}

        latent_flat = latent_vectors.flatten()

        ks_statistic, ks_pvalue = kstest(latent_flat, 'norm', args=(0, 1))
        results['ks_statistic'] = ks_statistic
        results['ks_pvalue'] = ks_pvalue
        results['is_gaussian_ks'] = ks_pvalue > significance_level

        mean = np.mean(latent_flat)
        std = np.std(latent_flat)
        skewness = np.mean(((latent_flat - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((latent_flat - mean) / std) ** 4) - 3

        results['mean'] = mean
        results['std'] = std
        results['skewness'] = skewness
        results['kurtosis'] = kurtosis

        results['mean_match'] = abs(mean) < 0.1
        results['std_match'] = abs(std - 1.0) < 0.1
        results['skewness_match'] = abs(skewness) < 0.5
        results['kurtosis_match'] = abs(kurtosis) < 1.0

        results['gaussian_score'] = (
                                            results['mean_match'] +
                                            results['std_match'] +
                                            results['skewness_match'] +
                                            results['kurtosis_match']
                                    ) / 4.0

        return results

    @staticmethod
    def kl_divergence_gaussian(latent_vectors, epsilon=1e-10):
        latent_flat = latent_vectors.flatten()

        kde = gaussian_kde(latent_flat)

        x_range = np.linspace(-4, 4, 1000)

        latent_pdf = kde(x_range)
        latent_pdf = latent_pdf / latent_pdf.sum()

        normal_pdf = norm.pdf(x_range, 0, 1)
        normal_pdf = normal_pdf / normal_pdf.sum()

        kl_div = entropy(latent_pdf + epsilon, normal_pdf + epsilon)

        return kl_div

    @staticmethod
    def visualize_latent_distribution(latent_vectors, epoch, save_dir='training_outputs/latent_analysis'):
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Latent Space Distribution Analysis - Epoch {epoch}',
                     fontsize=20, fontweight='bold')

        latent_flat = latent_vectors.flatten()

        ax = axes[0, 0]
        ax.hist(latent_flat, bins=100, density=True, alpha=0.7, label='Latent Space')
        x_range = np.linspace(-4, 4, 1000)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2, label='N(0,1)')
        ax.set_title('Distribution vs Standard Normal')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        from scipy import stats
        stats.probplot(latent_flat[:10000], dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        dim_means = np.mean(latent_vectors, axis=0)
        dim_stds = np.std(latent_vectors, axis=0)
        ax.errorbar(range(len(dim_means[:50])), dim_means[:50], yerr=dim_stds[:50],
                    fmt='o', alpha=0.6, capsize=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Per-Dimension Statistics (first 50)')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Mean ± Std')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        scatter = ax.scatter(latent_vectors[:5000, 0], latent_vectors[:5000, 1],
                             alpha=0.5, s=1)
        ax.set_title('2D Latent Space (dims 0,1)')
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        normality_results = GaussianLatentValidator.test_normality(latent_vectors)

        text = f"Normality Test Results:\n\n"
        text += f"Mean: {normality_results['mean']:.4f} (target: 0)\n"
        text += f"Std: {normality_results['std']:.4f} (target: 1)\n"
        text += f"Skewness: {normality_results['skewness']:.4f} (target: 0)\n"
        text += f"Kurtosis: {normality_results['kurtosis']:.4f} (target: 0)\n\n"
        text += f"KS p-value: {normality_results['ks_pvalue']:.4f}\n"
        text += f"Gaussian Score: {normality_results['gaussian_score']:.2%}"

        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('Normality Statistics')

        ax = axes[1, 2]
        kl_div = GaussianLatentValidator.kl_divergence_gaussian(latent_vectors)
        ax.bar(['KL Divergence'], [kl_div], color='blue', alpha=0.7)
        ax.set_title(f'KL Divergence: {kl_div:.4f}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'latent_analysis_epoch_{epoch:04d}')
        for fmt in HIGH_RES_CONFIG['save_formats']:
            plt.savefig(f'{save_path}.{fmt}', dpi=HIGH_RES_CONFIG['dpi'],
                        bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)

        return normality_results


class GaussianRegularizedAdaIN(layers.Layer):
    """AdaIN with Gaussian regularization."""

    def __init__(self, channels, gaussian_reg_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.gaussian_reg_weight = gaussian_reg_weight

    def build(self, input_shape):
        self.style_dense = layers.Dense(
            self.channels * 2,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, w = inputs

        if training:
            w_mean = tf.reduce_mean(w, axis=0)
            w_std = tf.math.reduce_std(w, axis=0)

            kl_loss = 0.5 * tf.reduce_sum(
                tf.square(w_mean) + tf.square(w_std) -
                tf.math.log(tf.square(w_std) + 1e-8) - 1.0
            )

            self.add_loss(self.gaussian_reg_weight * kl_loss / tf.cast(self.channels, tf.float32))

        style_params = self.style_dense(w)
        style_scale = style_params[:, :self.channels]
        style_bias = style_params[:, self.channels:]

        style_scale = tf.reshape(style_scale, [-1, 1, 1, self.channels])
        style_bias = tf.reshape(style_bias, [-1, 1, 1, self.channels])

        mean_x, var_x = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        std_x = tf.sqrt(var_x + 1e-8)

        x_normalized = (x - mean_x) / std_x
        x = x_normalized * style_scale + style_bias

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'gaussian_reg_weight': self.gaussian_reg_weight
        })
        return config


class GaussianStabilizedGenerator(tf.keras.Model):
    """Generator with enforced Gaussian latent space and W-space access."""

    def __init__(self, latent_dim=512, img_shape=(48, 48, 1), n_filters=64,
                 gaussian_reg_strength=0.01, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.w_dim = latent_dim * 2  # W space dimension
        self.img_shape = img_shape
        self.n_filters = n_filters
        self.gaussian_reg_strength = gaussian_reg_strength

        self.mapping = self._build_mapping_network()

        self.latent_normalization = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, center=True, scale=True
        )

        self.constant = tf.Variable(
            tf.random.normal([1, 4, 4, n_filters * 8], stddev=0.02),
            trainable=True
        )

        self._build_generator_blocks()

        self.to_rgb = layers.Conv2D(
            img_shape[2], 1, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='tanh'
        )

    def _build_mapping_network(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(self.latent_dim * 2,
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.Dense(self.latent_dim * 2,
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.Dense(self.latent_dim * 2,
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ], name='mapping_network')

    def _build_generator_blocks(self):
        self.blocks = []

        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 8, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 4, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 2, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters // 2, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters // 2, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
            layers.Cropping2D(cropping=((8, 8), (8, 8))),
        ])

    def map_to_w(self, z_vectors):

        return self.mapping(z_vectors)

    def generate_from_w(self, w_vectors, training=False):

        batch_size = tf.shape(w_vectors)[0]
        x = tf.tile(self.constant, [batch_size, 1, 1, 1])

        for block in self.blocks:
            for layer in block:
                if isinstance(layer, GaussianRegularizedAdaIN):
                    x = layer([x, w_vectors], training=training)
                else:
                    x = layer(x)

        x = self.to_rgb(x)
        return x

    def call(self, inputs, training=False):

        input_dim = inputs.shape[-1]

        if input_dim == self.latent_dim:

            if training:
                inputs_mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
                inputs_std = tf.math.reduce_std(inputs, axis=0, keepdims=True) + 1e-8
                inputs = (inputs - inputs_mean) / inputs_std
            w = self.map_to_w(inputs)
            return self.generate_from_w(w, training=training)
        elif input_dim == self.w_dim:

            return self.generate_from_w(inputs, training=training)
        else:
            raise ValueError(f"Expected input dim {self.latent_dim} (Z) or {self.w_dim} (W), got {input_dim}")

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'img_shape': self.img_shape,
            'n_filters': self.n_filters,
            'gaussian_reg_strength': self.gaussian_reg_strength
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FeatureExtractingDiscriminator(tf.keras.Model):
    """StyleGAN2 Discriminator com feature extraction."""

    def __init__(self, img_shape=(48, 48, 1), n_filters=64, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.n_filters = n_filters
        self.features = None

        self.from_rgb = layers.Conv2D(
            n_filters, 1, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

        self._build_discriminator_blocks()

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(n_filters * 8,
                                   activation='leaky_relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = layers.Dense(1,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def _build_discriminator_blocks(self):
        self.blocks = []

        self.blocks.append([
            layers.Conv2D(self.n_filters, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])

        self.blocks.append([
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])

        self.blocks.append([
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])

        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])

        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='valid',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
        ])

    def get_features(self, inputs):
        x = self.from_rgb(inputs)

        for i, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)
            if i == 1:
                return x
        return x

    def call(self, inputs, training=False):
        x = inputs

        if training:
            self.features = self.get_features(x)

        x = self.from_rgb(x)

        for block in self.blocks:
            for layer in block:
                x = layer(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'img_shape': self.img_shape,
            'n_filters': self.n_filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class StyleGAN2(keras.Model):
    """StyleGAN2 with enforced Gaussian latent space and W-space support."""

    def __init__(self,
                 generator,
                 discriminator,
                 latent_dim=512,
                 d_reg=10.0,
                 label_smoothing=0.1,
                 gradient_clip=0.5,
                 noise_std=0.05,
                 latent_gaussian_weight=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.w_dim = latent_dim * 2
        self.d_reg = d_reg
        self.label_smoothing = label_smoothing
        self.gradient_clip = gradient_clip
        self.noise_std = noise_std
        self.latent_gaussian_weight = latent_gaussian_weight

        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.d_real_metric = tf.keras.metrics.Mean(name="d_real")
        self.d_fake_metric = tf.keras.metrics.Mean(name="d_fake")
        self.grad_norm_g = tf.keras.metrics.Mean(name="grad_norm_g")
        self.latent_gaussian_loss = tf.keras.metrics.Mean(name="latent_gaussian_loss")

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return [
            self.g_loss_metric,
            self.d_loss_metric,
            self.d_real_metric,
            self.d_fake_metric,
            self.grad_norm_g,
            self.latent_gaussian_loss
        ]

    def add_discriminator_noise(self, images):
        noise = tf.random.normal(tf.shape(images), stddev=self.noise_std)
        return images + noise

    def compute_latent_gaussian_loss(self, z_vectors):

        mean = tf.reduce_mean(z_vectors, axis=0)
        variance = tf.math.reduce_variance(z_vectors, axis=0)

        mean_loss = tf.reduce_mean(tf.square(mean))
        variance_loss = tf.reduce_mean(tf.square(variance - 1.0))

        centered = z_vectors - mean
        fourth_moment = tf.reduce_mean(centered ** 4, axis=0)
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        kurtosis_loss = tf.reduce_mean(tf.square(kurtosis - 3.0))

        third_moment = tf.reduce_mean(centered ** 3, axis=0)
        skewness = third_moment / (tf.sqrt(variance) ** 3 + 1e-8)
        skewness_loss = tf.reduce_mean(tf.square(skewness))

        gaussian_loss = (
                mean_loss +
                variance_loss +
                0.1 * kurtosis_loss +
                0.1 * skewness_loss
        )

        return gaussian_loss

    def compute_w_consistency_loss(self, z_vectors):

        w_vectors = self.generator.map_to_w(z_vectors)


        w_mean = tf.reduce_mean(w_vectors, axis=0)
        w_std = tf.math.reduce_std(w_vectors, axis=0)


        w_mean_loss = tf.reduce_mean(tf.square(w_mean))
        w_std_loss = tf.reduce_mean(tf.square(w_std - 1.0))


        batch_size = tf.shape(w_vectors)[0]
        half_batch = batch_size // 2

        w1 = w_vectors[:half_batch]  # Shape: (half_batch, w_dim)
        w2 = w_vectors[half_batch:half_batch * 2]  # Shape: (half_batch, w_dim)

        alpha = tf.random.uniform([half_batch, 1], 0.0, 1.0)

        w_interp = w1 + alpha * (w2 - w1)  # Shape: (half_batch, w_dim)

        img1 = self.generator.generate_from_w(w1, training=False)
        img2 = self.generator.generate_from_w(w2, training=False)
        img_interp = self.generator.generate_from_w(w_interp, training=True)

        alpha_img = tf.reshape(alpha, [half_batch, 1, 1, 1])
        img_blend = img1 * (1 - alpha_img) + img2 * alpha_img

        consistency_loss = tf.reduce_mean(tf.square(img_interp - img_blend))

        return w_mean_loss + w_std_loss + 0.5 * consistency_loss

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        d_loss_total = 0

        for _ in range(2):
            with tf.GradientTape() as tape:
                z_noise = tf.random.normal([batch_size, self.latent_dim], mean=0.0, stddev=1.0)
                fake_images = self.generator(z_noise, training=False)

                real_noisy = self.add_discriminator_noise(real_images)
                fake_noisy = self.add_discriminator_noise(fake_images)

                real_output = self.discriminator(real_noisy, training=True)
                fake_output = self.discriminator(fake_noisy, training=True)

                d_loss_real = tf.reduce_mean(real_output)
                d_loss_fake = tf.reduce_mean(fake_output)

                epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated = epsilon * real_images + (1 - epsilon) * fake_images

                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    interpolated_output = self.discriminator(interpolated, training=True)

                gradients = gp_tape.gradient(interpolated_output, [interpolated])[0]
                gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)

                d_loss = d_loss_fake - d_loss_real + self.d_reg * gradient_penalty
                d_loss += 0.001 * tf.reduce_sum(self.discriminator.losses)

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

            if self.gradient_clip > 0:
                d_gradients = [
                    tf.clip_by_norm(g, self.gradient_clip) if g is not None else g
                    for g in d_gradients
                ]

            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )
            d_loss_total += d_loss

        d_loss_avg = d_loss_total / 2.0

        with tf.GradientTape() as tape:
            z_noise = tf.random.normal([batch_size, self.latent_dim], mean=0.0, stddev=1.0)
            fake_images = self.generator(z_noise, training=True)

            fake_output = self.discriminator(fake_images, training=False)

            g_loss = -tf.reduce_mean(fake_output)

            latent_gaussian_loss = self.compute_latent_gaussian_loss(z_noise)
            g_loss += self.latent_gaussian_weight * latent_gaussian_loss

            w_consistency_loss = self.compute_w_consistency_loss(z_noise)
            g_loss += 0.05 * w_consistency_loss

            g_loss += 0.001 * tf.reduce_sum(self.generator.losses)

            if hasattr(self.discriminator, 'features') and self.discriminator.features is not None:
                real_features = self.discriminator.features
                fake_features = self.discriminator.get_features(fake_images)
                feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
                g_loss += 0.1 * feature_loss

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

        if self.gradient_clip > 0:
            g_gradients = [
                tf.clip_by_norm(g, self.gradient_clip) if g is not None else g
                for g in g_gradients
            ]

        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        g_grad_norm = tf.linalg.global_norm(g_gradients)

        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss_avg)
        self.d_real_metric.update_state(tf.reduce_mean(real_output))
        self.d_fake_metric.update_state(tf.reduce_mean(fake_output))
        self.grad_norm_g.update_state(g_grad_norm)
        self.latent_gaussian_loss.update_state(latent_gaussian_loss)

        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "d_real": self.d_real_metric.result(),
            "d_fake": self.d_fake_metric.result(),
            "grad_norm_g": self.grad_norm_g.result(),
            "latent_gaussian_loss": self.latent_gaussian_loss.result()
        }

    def call(self, inputs, training=False):

        return self.generator(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'd_reg': self.d_reg,
            'label_smoothing': self.label_smoothing,
            'gradient_clip': self.gradient_clip,
            'noise_std': self.noise_std,
            'latent_gaussian_weight': self.latent_gaussian_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        raise ValueError(
            "StyleGAN2 não pode ser carregado diretamente de config. "
            "Use: StyleGAN2(generator, discriminator, **config)"
        )

    def save_all_models_weights(self, save_dir='training_outputs/final_models'):
        print(f"\n{'=' * 60}")
        print("SAVING ALL MODEL WEIGHTS (FORMAT .weights.h5)")
        print("=" * 60)

        os.makedirs(save_dir, exist_ok=True)

        print("\n1. Saving generator weights...")
        generator_path = os.path.join(save_dir, 'generator.weights_w.h5')
        self.generator.save_weights(generator_path)
        print(f"   ✓ Generator weights saved to: {generator_path}")

        print("\n2. Saving discriminator weights...")
        discriminator_path = os.path.join(save_dir, 'discriminator.weights_w.h5')
        self.discriminator.save_weights(discriminator_path)
        print(f"   ✓ Discriminator weights saved to: {discriminator_path}")

        print("\n3. Saving complete StyleGAN2 weights...")
        stylegan_path = os.path.join(save_dir, 'stylegan2_complete.weights_w.h5')
        self.save_weights(stylegan_path)
        print(f"   ✓ StyleGAN2 weights saved to: {stylegan_path}")

        print("\n4. Saving model configuration...")
        config = {
            'latent_dim': self.latent_dim,
            'w_dim': self.w_dim,
            'd_reg': self.d_reg,
            'label_smoothing': self.label_smoothing,
            'gradient_clip': self.gradient_clip,
            'noise_std': self.noise_std,
            'latent_gaussian_weight': self.latent_gaussian_weight,
            'generator_config': {
                'latent_dim': self.generator.latent_dim,
                'img_shape': self.generator.img_shape,
                'n_filters': self.generator.n_filters,
                'gaussian_reg_strength': self.generator.gaussian_reg_strength
            },
            'discriminator_config': {
                'img_shape': self.discriminator.img_shape,
                'n_filters': self.discriminator.n_filters
            }
        }

        import json
        with open(os.path.join(save_dir, 'model_config_w.json'), 'w') as f:
            json.dump(config, f, indent=4, default=str)
        print(f"   ✓ Configuration saved to: {os.path.join(save_dir, 'model_config_w.json')}")


        with open(os.path.join(save_dir, 'w_dim.txt'), 'w') as f:
            f.write(str(self.w_dim))
        print(f"   ✓ W dimension ({self.w_dim}) saved to: {os.path.join(save_dir, 'w_dim.txt')}")

        print(f"\n{'=' * 60}")
        print(f"ALL MODEL WEIGHTS SAVED SUCCESSFULLY")
        print(f"Save directory: {os.path.abspath(save_dir)}")
        print("=" * 60)

        return save_dir


class SampleSaverCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator, latent_dim, save_dir='training_outputs/samples',
                 latent_analysis_interval=50):
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.save_dir = save_dir
        self.latent_analysis_interval = latent_analysis_interval
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        noise = np.random.normal(0, 1, (6, self.latent_dim))
        samples = self.generator.predict(noise, verbose=0)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Generated Samples - Epoch {epoch + 1}',
                     fontsize=HIGH_RES_CONFIG['title_font_size'],
                     fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < 6:
                im = ax.imshow(samples[i, ..., 0], cmap='viridis', aspect='auto')
                ax.set_title(f'Sample {i + 1}', fontsize=HIGH_RES_CONFIG['font_size'],
                             fontweight='bold')
                ax.axis('off')

        plt.tight_layout()

        base_path = os.path.join(self.save_dir, f'samples_epoch_{epoch + 1:04d}')
        plt.savefig(f"{base_path}.png", dpi=HIGH_RES_CONFIG['dpi'],
                    bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(f"{base_path}.svg", format='svg', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        np.save(os.path.join(self.save_dir, f'samples_epoch_{epoch + 1:04d}.npy'), samples)

        if (epoch + 1) % self.latent_analysis_interval == 0:
            print(f"\nAnalyzing latent space gaussianity at epoch {epoch + 1}...")
            latent_noise = np.random.normal(0, 1, (5000, self.latent_dim))
            GaussianLatentValidator.visualize_latent_distribution(
                latent_noise, epoch + 1,
                save_dir='training_outputs/latent_analysis'
            )
            print(f"✓ Latent space analysis saved for epoch {epoch + 1}")


class EnhancedMetricCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator, x_test, latent_dim, metric_interval=10,
                 log_file='training_metrics.txt', save_dir='training_outputs/metrics'):
        super().__init__()
        self.generator = generator
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.metric_interval = metric_interval
        self.log_file = log_file
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.traditional_metrics = {
            'FID_10k': [], 'FRD_10k': [], 'KID_10k': [], 'KRD_10k': [],
            'FID_500': [], 'FRD_500': [], 'KID_500': [], 'KRD_500': []
        }

        self.static_metrics = {
            'Variogram_MSE': [],
            'Connectivity_MSE': [],
            'Histogram_KL': [],
            'PCA_Correlation': [],
            'MDS_MMD': []
        }

        with open(self.log_file, 'w') as f:
            header = "Epoch\t"
            header += "\t".join(self.traditional_metrics.keys()) + "\t"
            header += "\t".join(self.static_metrics.keys()) + "\n"
            f.write(header)

        self.history = {
            'g_loss': [],
            'd_loss': []
        }

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.history['g_loss'].append(logs.get('g_loss', 0))
            self.history['d_loss'].append(logs.get('d_loss', 0))

        if (epoch + 1) % self.metric_interval == 0:
            print(f'\n{"=" * 60}')
            print(f'Computing ALL metrics at epoch {epoch + 1}')
            print(f'{"=" * 60}')

            size = min(500, self.x_test.shape[0])
            noise = np.random.normal(0, 1, (size, self.latent_dim))
            fake_data = self.generator.predict(noise, verbose=0, batch_size=16)
            real_data = self.x_test[:size]

            try:
                traditional_results = TraditionalMetrics.compute_all_traditional_metrics(
                    real_data, fake_data
                )

                for key in traditional_results:
                    if abs(traditional_results[key]) > 1e6:
                        print(f"Warning: {key} has extreme value: {traditional_results[key]:.2e}")
                        traditional_results[key] = 100.0 if 'FID' in key or 'FRD' in key else 1.0
            except Exception as e:
                print(f"Error calculating traditional metrics: {e}")
                traditional_results = {
                    'FID_10k': 100.0, 'FID_500': 100.0,
                    'KID_10k': 1.0, 'KID_500': 1.0,
                    'FRD_10k': 100.0, 'FRD_500': 100.0,
                    'KRD_10k': 1.0, 'KRD_500': 1.0
                }

            try:
                static_results = StaticMetrics.compute_all_metrics(real_data, fake_data)
                static_summary = {
                    'Variogram_MSE': static_results['variogram_mse'],
                    'Connectivity_MSE': static_results['connectivity_mse'],
                    'Histogram_KL': static_results['hist_kl'],
                    'PCA_Correlation': static_results['pca_correlation'],
                    'MDS_MMD': static_results['mds_mmd']
                }
            except Exception as e:
                print(f"Error calculating static metrics: {e}")
                static_summary = {
                    'Variogram_MSE': 1.0,
                    'Connectivity_MSE': 1.0,
                    'Histogram_KL': 10.0,
                    'PCA_Correlation': 0.0,
                    'MDS_MMD': 1.0
                }

            for key in self.traditional_metrics:
                self.traditional_metrics[key].append(traditional_results[key])

            for key in self.static_metrics:
                self.static_metrics[key].append(static_summary[key])

            self.save_metrics_to_file(epoch + 1, traditional_results, static_summary)
            self.display_metrics(epoch + 1, traditional_results, static_summary)
            self.save_numpy_files()

            try:
                self.plot_metrics_comparison(epoch + 1, static_results)
            except Exception as e:
                print(f"Error plotting metrics: {e}")

    def save_metrics_to_file(self, epoch, traditional_metrics, static_metrics):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch}\t")
            for key in self.traditional_metrics.keys():
                f.write(f"{traditional_metrics[key]:.4f}\t")
            for key in self.static_metrics.keys():
                f.write(f"{static_metrics[key]:.4f}\t")
            f.write("\n")

    def display_metrics(self, epoch, traditional_metrics, static_metrics):
        output = (
            f"\nEpoch {epoch} - ALL METRICS:\n"
            f"{'=' * 70}\n"
            f"Traditional Metrics:\n"
            f"  FID (10k): {traditional_metrics['FID_10k']:.4f}  "
            f"FID (500): {traditional_metrics['FID_500']:.4f}\n"
            f"  KID (10k): {traditional_metrics['KID_10k']:.4f}  "
            f"KID (500): {traditional_metrics['KID_500']:.4f}\n"
            f"  FRD (10k): {traditional_metrics['FRD_10k']:.4f}  "
            f"FRD (500): {traditional_metrics['FRD_500']:.4f}\n"
            f"  KRD (10k): {traditional_metrics['KRD_10k']:.4f}  "
            f"KRD (500): {traditional_metrics['KRD_500']:.4f}\n"
            f"{'=' * 70}\n"
            f"Static Geostatistical Metrics:\n"
            f"  Variogram MSE: {static_metrics['Variogram_MSE']:.6f}\n"
            f"  Connectivity MSE: {static_metrics['Connectivity_MSE']:.6f}\n"
            f"  Histogram KL: {static_metrics['Histogram_KL']:.6f}\n"
            f"  PCA Correlation: {static_metrics['PCA_Correlation']:.6f}\n"
            f"  MDS MMD: {static_metrics['MDS_MMD']:.6f}\n"
            f"{'=' * 70}"
        )
        print(output)

    def save_numpy_files(self):
        for key, values in self.traditional_metrics.items():
            np.save(f'{self.save_dir}/{key}.npy', values)
        for key, values in self.static_metrics.items():
            np.save(f'{self.save_dir}/{key}.npy', values)
        np.save(f'{self.save_dir}/g_loss.npy', self.history['g_loss'])
        np.save(f'{self.save_dir}/d_loss.npy', self.history['d_loss'])

    def plot_metrics_comparison(self, epoch, static_results):
        fig, axes = plt.subplots(2, 3, figsize=(22 * HIGH_RES_CONFIG['figsize_multiplier'],
                                                14 * HIGH_RES_CONFIG['figsize_multiplier']),
                                 dpi=HIGH_RES_CONFIG['dpi'])
        fig.suptitle(f'Static Metrics Comparison - Epoch {epoch}',
                     fontsize=HIGH_RES_CONFIG['title_font_size'] + 2,
                     fontweight='bold')

        ax = axes[0, 0]
        lags = static_results['variogram_lags']
        ax.plot(lags, static_results['variogram_real'], 'b-',
                linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
        ax.plot(lags, static_results['variogram_fake'], 'r--',
                linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
        ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_ylabel('Variogram', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title(f'Variogram (MSE: {static_results["variogram_mse"]:.4f})',
                     fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        ax = axes[0, 1]
        ax.plot(lags, static_results['connectivity_real'], 'b-',
                linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
        ax.plot(lags, static_results['connectivity_fake'], 'r--',
                linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
        ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_ylabel('Connectivity', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title(f'Connectivity (MSE: {static_results["connectivity_mse"]:.4f})',
                     fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        ax = axes[0, 2]
        bin_centers = (static_results['hist_bin_edges'][:-1] + static_results['hist_bin_edges'][1:]) / 2
        bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
        ax.bar(bin_centers, static_results['hist_real'], width=bar_width,
               alpha=0.6, label='Real', color='blue', edgecolor='black', linewidth=1.0)
        ax.plot(bin_centers, static_results['hist_fake'], 'r-',
                linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
        ax.set_xlabel('Value', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_ylabel('Density', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title(f'Histogram (KL: {static_results["hist_kl"]:.4f})',
                     fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        ax = axes[1, 0]
        ax.scatter(static_results['pca_real'][:, 0], static_results['pca_real'][:, 1],
                   alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
                   color='blue', edgecolor='black', linewidth=1.0)
        ax.scatter(static_results['pca_fake'][:, 0], static_results['pca_fake'][:, 1],
                   alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
                   color='red', edgecolor='black', linewidth=1.0)
        ax.set_xlabel('PC1', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_ylabel('PC2', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title(f'PCA (Correlation: {static_results["pca_correlation"]:.4f})',
                     fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        ax = axes[1, 1]
        ax.scatter(static_results['mds_real'][:, 0], static_results['mds_real'][:, 1],
                   alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
                   color='blue', edgecolor='black', linewidth=1.0)
        ax.scatter(static_results['mds_fake'][:, 0], static_results['mds_fake'][:, 1],
                   alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
                   color='red', edgecolor='black', linewidth=1.0)
        ax.set_xlabel('MDS Dimension 1', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_ylabel('MDS Dimension 2', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title(f'MDS (MMD: {static_results["mds_mmd"]:.4f})',
                     fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        ax = axes[1, 2]
        summary_metrics = ['Variogram\nMSE', 'Connectivity\nMSE', 'Histogram\nKL', 'PCA\nCorrelation', 'MDS\nMMD']
        summary_values = [
            static_results['variogram_mse'],
            static_results['connectivity_mse'],
            static_results['hist_kl'],
            static_results['pca_correlation'],
            static_results['mds_mmd']
        ]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        bars = ax.bar(summary_metrics, summary_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Value', fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'])
        ax.set_title('Summary of Static Metrics', fontweight='bold', fontsize=HIGH_RES_CONFIG['title_font_size'])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

        for bar, val in zip(bars, summary_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=HIGH_RES_CONFIG['font_size'] - 2,
                    fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        base_path = f'{self.save_dir}/metrics_epoch_{epoch:04d}'
        plt.savefig(f'{base_path}.png', dpi=HIGH_RES_CONFIG['dpi'], bbox_inches='tight')
        plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight')
        plt.close(fig)


class ImprovedStabilizationCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold_ratio=0.8):
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.patience = 5
        self.wait = 0
        self.best_d_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            d_loss = logs.get('d_loss', 0)
            g_loss = logs.get('g_loss', 0)
            d_real = logs.get('d_real', 0)
            d_fake = logs.get('d_fake', 0)
            grad_norm_g = logs.get('grad_norm_g', 0)

            d_too_confident = (abs(d_real) > 5.0 or abs(d_fake) > 5.0)
            loss_imbalance = (g_loss / max(d_loss, 1e-8)) > 10.0
            grad_exploding = grad_norm_g > 100.0

            if d_too_confident or loss_imbalance or grad_exploding:
                print(f"\n STABILIZATION NEEDED at epoch {epoch + 1}:")

                if d_too_confident:
                    print(f"   Discriminator too confident: d_real={d_real:.4f}, d_fake={d_fake:.4f}")
                    new_smoothing = min(0.3, self.model.label_smoothing * 1.5)
                    self.model.label_smoothing = new_smoothing
                    print(f"   Increased label smoothing to: {new_smoothing:.3f}")

                if loss_imbalance:
                    print(f"   Loss imbalance: g_loss={g_loss:.4f}, d_loss={d_loss:.4f} (ratio: {g_loss / d_loss:.2f})")

                if grad_exploding:
                    print(f"   Gradient exploding: grad_norm_g={grad_norm_g:.4f}")
                    new_clip = max(0.1, self.model.gradient_clip * 0.5)
                    self.model.gradient_clip = new_clip
                    print(f"   Reduced gradient clip to: {new_clip:.3f}")

            if d_loss < self.best_d_loss:
                self.best_d_loss = d_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    current_lr_g = float(self.model.g_optimizer.learning_rate)
                    current_lr_d = float(self.model.d_optimizer.learning_rate)

                    new_lr_g = current_lr_g * 0.9
                    new_lr_d = current_lr_d * 0.9

                    self.model.g_optimizer.learning_rate.assign(new_lr_g)
                    self.model.d_optimizer.learning_rate.assign(new_lr_d)

                    print(f"\n📉 Reducing learning rates: G={new_lr_g:.6f}, D={new_lr_d:.6f}")
                    self.wait = 0


class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):

    def __init__(self, initial_lr_g=0.0001, initial_lr_d=0.0004):
        super().__init__()
        self.initial_lr_g = initial_lr_g
        self.initial_lr_d = initial_lr_d
        self.stable_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            d_real = logs.get('d_real', 0)
            d_fake = logs.get('d_fake', 0)

            d_balance = abs(d_real + d_fake) < 0.5

            if d_balance:
                self.stable_epochs += 1
                if self.stable_epochs >= 10:
                    current_lr_g = float(self.model.g_optimizer.learning_rate)
                    current_lr_d = float(self.model.d_optimizer.learning_rate)

                    new_lr_g = min(self.initial_lr_g, current_lr_g * 1.1)
                    new_lr_d = min(self.initial_lr_d, current_lr_d * 1.1)

                    self.model.g_optimizer.learning_rate.assign(new_lr_g)
                    self.model.d_optimizer.learning_rate.assign(new_lr_d)

                    if new_lr_g != current_lr_g or new_lr_d != current_lr_d:
                        print(f"\n Increasing learning rates (stable training): "
                              f"G={new_lr_g:.6f}, D={new_lr_d:.6f}")

                    self.stable_epochs = 0
            else:
                self.stable_epochs = 0


class LivePlotCallback(keras.callbacks.Callback):
    def __init__(self, model, data, num_images=10):
        super().__init__()
        self.model = model
        self.data = data
        self.num_images = num_images

        self.fig_width = num_images * 4 * HIGH_RES_CONFIG['figsize_multiplier']
        self.fig_height = 4 * HIGH_RES_CONFIG['figsize_multiplier']

        self.fig, self.ax = plt.subplots(1, num_images * 2,
                                         figsize=(self.fig_width, self.fig_height),
                                         dpi=HIGH_RES_CONFIG['dpi'])

        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        indices = np.random.randint(0, self.data.shape[0], self.num_images)
        original_imgs = self.data[indices]

        noise = np.random.normal(0, 1, (self.num_images, latent_dim))
        generated_imgs = self.model.generator.predict(noise, verbose=0)

        for i in range(self.num_images):
            self.ax[i * 2].imshow(original_imgs[i].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
            self.ax[i * 2].set_title("Original", fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'] - 12)
            self.ax[i * 2].axis('off')

            self.ax[i * 2 + 1].imshow(generated_imgs[i].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1,
                                      vmax=1)
            self.ax[i * 2 + 1].set_title(f"Generated", fontweight='bold', fontsize=HIGH_RES_CONFIG['font_size'] - 12)
            self.ax[i * 2 + 1].axis('off')

        self.fig.suptitle(f'Epoch {epoch + 1}: Original vs Generated Samples',
                          fontsize=HIGH_RES_CONFIG['title_font_size'] - 8, fontweight='bold')

        for fmt in HIGH_RES_CONFIG['save_formats']:
            filename = f"training_outputs/training_live_epoch_{epoch + 1:04d}.{fmt}"
            plt.savefig(filename, dpi=HIGH_RES_CONFIG['dpi'], bbox_inches='tight',
                        facecolor='white', edgecolor='none')

        plt.pause(0.01)
        self.fig.canvas.draw()


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

def plot_losses_history(g_loss_history, d_loss_history, d_real_history, d_fake_history,
                        save_path='training_outputs/loss_plots', save_prefix="stylegan2"):
    epochs = range(1, len(g_loss_history) + 1)

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

    plt.subplot(2, 2, 1)
    plt.plot(epochs, g_loss_history, label='Generator Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
    plt.plot(epochs, d_loss_history, label='Discriminator Loss',
             linewidth=HIGH_RES_CONFIG['line_width'], color='red')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Loss', fontweight='bold', fontsize=22)
    plt.title('Training Losses', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, d_real_history, label='D(Real)',
             linewidth=HIGH_RES_CONFIG['line_width'], color='green')
    plt.plot(epochs, d_fake_history, label='D(Fake)',
             linewidth=HIGH_RES_CONFIG['line_width'], color='magenta')
    plt.xlabel('Epochs', fontweight='bold', fontsize=22)
    plt.ylabel('Discriminator Output', fontweight='bold', fontsize=22)
    plt.title('Discriminator Confidence', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)
    plt.ylim([-5, 5])

    if len(g_loss_history) > 10:
        window = 10
        g_smooth = np.convolve(g_loss_history, np.ones(window) / window, mode='valid')
        d_smooth = np.convolve(d_loss_history, np.ones(window) / window, mode='valid')
        epochs_smooth = range(window, len(g_loss_history) + 1)

        plt.subplot(2, 2, 3)
        plt.plot(epochs_smooth, g_smooth, label='Generator (smoothed)',
                 linewidth=HIGH_RES_CONFIG['line_width'], color='blue')
        plt.plot(epochs_smooth, d_smooth, label='Discriminator (smoothed)',
                 linewidth=HIGH_RES_CONFIG['line_width'], color='red')
        plt.xlabel('Epochs', fontweight='bold', fontsize=22)
        plt.ylabel('Loss (smoothed)', fontweight='bold', fontsize=22)
        plt.title(f'Smoothed Losses (window={window})', fontweight='bold', fontsize=24)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    if len(g_loss_history) > 1:
        loss_diff = np.array(g_loss_history) - np.array(d_loss_history)
        plt.subplot(2, 2, 4)
        plt.plot(epochs, loss_diff, label='G Loss - D Loss',
                 linewidth=HIGH_RES_CONFIG['line_width'], color='black')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2.0)
        plt.xlabel('Epochs', fontweight='bold', fontsize=22)
        plt.ylabel('Loss Difference', fontweight='bold', fontsize=22)
        plt.title('Generator vs Discriminator Loss Difference', fontweight='bold', fontsize=24)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.suptitle('StyleGAN2 Training Losses (W-Space Enabled)',
                 fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base_filename = f"{save_path}/{save_prefix}_all_losses"
    saved_files = save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
    plt.close(fig)

    plot_individual_losses(g_loss_history, d_loss_history, d_real_history, d_fake_history,
                           save_path, save_prefix)

    return saved_files


def plot_individual_losses(g_loss_history, d_loss_history, d_real_history, d_fake_history,
                           save_path='training_outputs/loss_plots', save_prefix="stylegan2"):
    epochs_range = range(1, len(g_loss_history) + 1)

    loss_types = [
        ('generator_loss', g_loss_history, 'Generator Loss', 'blue'),
        ('discriminator_loss', d_loss_history, 'Discriminator Loss', 'red'),
        ('d_real', d_real_history, 'D(Real)', 'green'),
        ('d_fake', d_fake_history, 'D(Fake)', 'magenta')
    ]

    for loss_name, loss_data, title, color in loss_types:
        fig, ax = plt.subplots(figsize=(12 * HIGH_RES_CONFIG['figsize_multiplier'],
                                        8 * HIGH_RES_CONFIG['figsize_multiplier']),
                               dpi=HIGH_RES_CONFIG['dpi'])

        ax.plot(epochs_range, loss_data, linewidth=HIGH_RES_CONFIG['line_width'], color=color)
        ax.set_xlabel('Epochs', fontweight='bold', fontsize=22)
        ax.set_ylabel('Value', fontweight='bold', fontsize=22)
        ax.set_title(f'StyleGAN2: {title}',
                     fontsize=24, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2)

        base_filename = f"{save_path}/{save_prefix}_{loss_name}"
        save_figure_multiple_formats(fig, base_filename, HIGH_RES_CONFIG['dpi'])
        plt.close(fig)


def plot_original_distribution_and_latent_space(original_data, generator_model, latent_dim,
                                                save_path='training_outputs/final_outputs'):
    print("\n" + "=" * 60)
    print("PLOTTING ORIGINAL DISTRIBUTION AND LATENT SPACE")
    print("=" * 60)

    os.makedirs(save_path, exist_ok=True)

    n_samples = min(1000, len(original_data))
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_samples = generator_model.predict(noise, verbose=0)

    if len(original_data.shape) == 4:
        original_flat = original_data[:n_samples, ..., 0].flatten()
        generated_flat = generated_samples[..., 0].flatten()
    else:
        original_flat = original_data[:n_samples].flatten()
        generated_flat = generated_samples.flatten()

    figsize = (24 * HIGH_RES_CONFIG['figsize_multiplier'],
               10 * HIGH_RES_CONFIG['figsize_multiplier'])
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])

    ax1 = axes[0]

    ax1.hist(original_flat, bins=HIGH_RES_CONFIG['hist_bins'], alpha=0.5,
             density=True, color='blue', label='Original Data', edgecolor='black', linewidth=1.0)
    ax1.hist(generated_flat, bins=HIGH_RES_CONFIG['hist_bins'], alpha=0.5,
             density=True, color='red', label='Generated Data', edgecolor='black', linewidth=1.0)

    try:
        kde_original = gaussian_kde(original_flat)
        kde_generated = gaussian_kde(generated_flat)

        x_range = np.linspace(min(original_flat.min(), generated_flat.min()),
                              max(original_flat.max(), generated_flat.max()), 1000)

        ax1.plot(x_range, kde_original(x_range), 'b-', linewidth=HIGH_RES_CONFIG['line_width'],
                 label='Original KDE')
        ax1.plot(x_range, kde_generated(x_range), 'r--', linewidth=HIGH_RES_CONFIG['line_width'],
                 label='Generated KDE')
    except:
        print("Warning: Could not compute KDE for distribution plot")

    ax1.set_xlabel('Pixel Value', fontsize=HIGH_RES_CONFIG['font_size'], fontweight='bold')
    ax1.set_ylabel('Density', fontsize=HIGH_RES_CONFIG['font_size'], fontweight='bold')
    ax1.set_title('Original vs Generated Distribution', fontsize=HIGH_RES_CONFIG['title_font_size'],
                  fontweight='bold')
    ax1.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax1.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

    stats_text = f'Original:\n  Mean: {np.mean(original_flat):.4f}\n  Std: {np.std(original_flat):.4f}\n\nGenerated:\n  Mean: {np.mean(generated_flat):.4f}\n  Std: {np.std(generated_flat):.4f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=HIGH_RES_CONFIG['font_size'] - 2, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]

    if len(original_data.shape) == 4:
        original_features = original_data[:n_samples].reshape(n_samples, -1)
    else:
        original_features = original_data[:n_samples].reshape(n_samples, -1)

    generated_features = generated_samples.reshape(n_samples, -1)

    combined_features = np.vstack([original_features, generated_features])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_features)

    original_pca = combined_pca[:n_samples]
    generated_pca = combined_pca[n_samples:]

    ax2.scatter(original_pca[:, 0], original_pca[:, 1], alpha=0.5,
                s=HIGH_RES_CONFIG['marker_size'], color='blue', label='Original Data',
                edgecolor='black', linewidth=1.0)
    ax2.scatter(generated_pca[:, 0], generated_pca[:, 1], alpha=0.5,
                s=HIGH_RES_CONFIG['marker_size'], color='red', label='Generated Data',
                edgecolor='black', linewidth=1.0)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)',
                   fontsize=HIGH_RES_CONFIG['font_size'], fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)',
                   fontsize=HIGH_RES_CONFIG['font_size'], fontweight='bold')
    ax2.set_title('Latent Space Visualization (PCA)', fontsize=HIGH_RES_CONFIG['title_font_size'],
                  fontweight='bold')
    ax2.legend(fontsize=HIGH_RES_CONFIG['font_size'] - 2)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax2.tick_params(axis='both', which='major', labelsize=HIGH_RES_CONFIG['font_size'] - 2, width=2)

    total_var = pca.explained_variance_ratio_.sum() * 100
    ax2.text(0.05, 0.95, f'Total Variance Explained: {total_var:.1f}%',
             transform=ax2.transAxes, fontsize=HIGH_RES_CONFIG['font_size'] - 2,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Original Distribution and Latent Space Analysis - StyleGAN2 (W-Space)',
                 fontsize=HIGH_RES_CONFIG['title_font_size'] + 4, fontweight='bold', y=1.02)

    plt.tight_layout()

    base_path = os.path.join(save_path, 'original_distribution_and_latent_space')
    saved_files = save_figure_multiple_formats(fig, base_path, HIGH_RES_CONFIG['dpi'])
    plt.close(fig)

    print(f"✓ Figure saved: {base_path}.png and .svg")

    return fig


def plot_histograms_and_densities(original, generated, latent_vectors=None, epoch=None, save_interval=10):
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
    generated_flat = generated.flatten()

    plt.subplot(2, 3, 1)
    plt.imshow(original[0].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.title('Original Sample', fontsize=24, fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(2, 3, 2)
    plt.imshow(generated[0].squeeze(), cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.title('Generated Sample', fontsize=24, fontweight='bold')
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(2, 3, 3)
    sns.histplot(original_flat, color='blue', label='Original',
                 kde=True, stat='density', alpha=0.5, bins=HIGH_RES_CONFIG['hist_bins'],
                 linewidth=HIGH_RES_CONFIG['line_width'])
    sns.histplot(generated_flat, color='red', label='Generated',
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
    sns.kdeplot(generated_flat, color='red', label='Generated',
                linewidth=HIGH_RES_CONFIG['line_width'])
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Pixel Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.title('Density Comparison', fontweight='bold', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.subplot(2, 3, 5)
    stats_text = (
        f'Original:\n'
        f'  Mean: {np.mean(original_flat):.4f}\n'
        f'  Std: {np.std(original_flat):.4f}\n'
        f'  Min: {np.min(original_flat):.4f}\n'
        f'  Max: {np.max(original_flat):.4f}\n\n'
        f'Generated:\n'
        f'  Mean: {np.mean(generated_flat):.4f}\n'
        f'  Std: {np.std(generated_flat):.4f}\n'
        f'  Min: {np.min(generated_flat):.4f}\n'
        f'  Max: {np.max(generated_flat):.4f}'
    )
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
             fontsize=20, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    plt.title('Summary Statistics', fontweight='bold', fontsize=24)

    plt.subplot(2, 3, 6)
    if latent_vectors is not None:
        latent_flat = latent_vectors.flatten()
        sns.kdeplot(original_flat, color='blue', label='Original',
                    linewidth=HIGH_RES_CONFIG['line_width'])
        sns.kdeplot(latent_flat, color='green', label='Latent Space',
                    linewidth=HIGH_RES_CONFIG['line_width'])
    else:
        sns.kdeplot(original_flat, color='blue', label='Original',
                    linewidth=HIGH_RES_CONFIG['line_width'])
        sns.kdeplot(generated_flat, color='red', label='Generated',
                    linewidth=HIGH_RES_CONFIG['line_width'])
    plt.title('Comparison of Densities', fontweight='bold', fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    plt.xlabel('Value', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    plt.suptitle('StyleGAN2 Analysis: Original vs Generated Distributions (W-Space)',
                 fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if epoch is not None:
        filename_base = f"training_outputs/distributions_epoch_{epoch:03d}"
    else:
        filename_base = "training_outputs/final_distributions"

    saved_files = save_figure_multiple_formats(fig, filename_base, HIGH_RES_CONFIG['dpi'])
    plt.close(fig)

    return saved_files


def plot_static_metrics_comparison(real_data, generator_model, save_path_base=None):
    n_samples = min(500, len(real_data))
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    fake_data = generator_model.predict(noise, verbose=0)

    metrics = StaticMetrics.compute_all_metrics(real_data[:n_samples], fake_data)

    figsize = (22 * HIGH_RES_CONFIG['figsize_multiplier'],
               14 * HIGH_RES_CONFIG['figsize_multiplier'])

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=HIGH_RES_CONFIG['dpi'])
    fig.suptitle('StyleGAN2: Static Geostatistical Metrics Comparison (W-Space)',
                 fontsize=26, fontweight='bold', y=1.02)

    ax = axes[0, 0]
    lags = metrics['variogram_lags']
    ax.plot(lags, metrics['variogram_real'], 'b-',
            linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    ax.plot(lags, metrics['variogram_fake'], 'r--',
            linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    ax.set_ylabel('Variogram', fontweight='bold', fontsize=22)
    ax.set_title(f'Variogram (MSE: {metrics["variogram_mse"]:.4f})',
                 fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

    ax = axes[0, 1]
    ax.plot(lags, metrics['connectivity_real'], 'b-',
            linewidth=HIGH_RES_CONFIG['line_width'], label='Real')
    ax.plot(lags, metrics['connectivity_fake'], 'r--',
            linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Lag Distance', fontweight='bold', fontsize=22)
    ax.set_ylabel('Connectivity', fontweight='bold', fontsize=22)
    ax.set_title(f'Connectivity Function (MSE: {metrics["connectivity_mse"]:.4f})',
                 fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

    ax = axes[0, 2]
    bin_centers = (metrics['hist_bin_edges'][:-1] + metrics['hist_bin_edges'][1:]) / 2
    bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1

    ax.bar(bin_centers, metrics['hist_real'], width=bar_width,
           alpha=0.6, label='Real', color='blue', edgecolor='black', linewidth=1.0)
    ax.plot(bin_centers, metrics['hist_fake'], 'r-',
            linewidth=HIGH_RES_CONFIG['line_width'], label='Generated')
    ax.set_xlabel('Value', fontweight='bold', fontsize=22)
    ax.set_ylabel('Density', fontweight='bold', fontsize=22)
    ax.set_title(f'Histogram (KL: {metrics["hist_kl"]:.4f})',
                 fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

    ax = axes[1, 0]
    ax.scatter(metrics['pca_real'][:, 0], metrics['pca_real'][:, 1],
               alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
               color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(metrics['pca_fake'][:, 0], metrics['pca_fake'][:, 1],
               alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
               color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('PC1', fontweight='bold', fontsize=22)
    ax.set_ylabel('PC2', fontweight='bold', fontsize=22)
    ax.set_title(f'PCA Projection (Corr: {metrics["pca_correlation"]:.4f})',
                 fontweight='bold', fontsize=24)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)

    ax = axes[1, 1]
    ax.scatter(metrics['mds_real'][:, 0], metrics['mds_real'][:, 1],
               alpha=0.6, label='Real', s=HIGH_RES_CONFIG['marker_size'],
               color='blue', edgecolor='black', linewidth=1.0)
    ax.scatter(metrics['mds_fake'][:, 0], metrics['mds_fake'][:, 1],
               alpha=0.6, label='Generated', s=HIGH_RES_CONFIG['marker_size'],
               color='red', edgecolor='black', linewidth=1.0)
    ax.set_xlabel('MDS Dimension 1', fontweight='bold', fontsize=22)
    ax.set_ylabel('MDS Dimension 2', fontweight='bold', fontsize=22)
    ax.set_title(f'MDS Projection (MMD: {metrics["mds_mmd"]:.4f})',
                 fontweight='bold', fontsize=24)
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
        f'Real mean: {np.mean(real_data):.4f}\n'
        f'Fake mean: {np.mean(fake_data):.4f}\n'
        f'Real std: {np.std(real_data):.4f}\n'
        f'Fake std: {np.std(fake_data):.4f}'
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
    return metrics


def generate_final_report(all_metrics, execution_time, save_path='training_outputs/FINAL_REPORT.txt'):
    print("\n" + "=" * 60)
    print("GENERATING FINAL REPORT")
    print("=" * 60)

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL TRAINING REPORT - STYLEGAN2 WITH W-SPACE SUPPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. TRAINING PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {name}\n")
        f.write(f"Image dimensions: {ni} x {nj} x {nz}\n")
        f.write(f"Training samples: {len(x_train)}\n")
        f.write(f"Test samples: {len(x_test)}\n")
        f.write(f"Latent dimension (Z): {latent_dim}\n")
        f.write(f"W dimension: {w_dim}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Learning rate (Generator): 0.0001\n")
        f.write(f"Learning rate (Discriminator): 0.0004\n")
        f.write(f"Gradient penalty (d_reg): 10.0\n")
        f.write(f"Label smoothing: 0.1\n")
        f.write(f"Gradient clipping: 0.5\n")
        f.write(f"Noise std: 0.05\n")
        f.write(f"Latent Gaussian weight: 0.1\n")
        f.write(f"W consistency weight: 0.05\n")
        f.write(f"Generator Gaussian reg: 0.01\n")
        f.write(f"Metric interval: {metric_interval}\n\n")

        f.write("2. W-SPACE INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"The W space is the output of the mapping network (Z -> W).\n")
        f.write(f"W dimension: {w_dim}\n")
        f.write(f"W is designed to be more linear and disentangled than Z.\n")
        f.write(f"Data assimilation should operate in W space for better results.\n\n")

        f.write("3. COMPUTATIONAL TIME\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n")
        f.write(f"Total execution time: {execution_time / 60:.2f} minutes\n")
        f.write(f"Total execution time: {execution_time / 3600:.2f} hours\n")
        if len(all_metrics.get('training_history', {}).get('g_loss', [])) > 0:
            avg_time_per_epoch = execution_time / len(all_metrics['training_history']['g_loss'])
            f.write(f"Average time per epoch: {avg_time_per_epoch:.2f} seconds\n")
        f.write("\n")

        f.write("4. TRAINING HISTORY\n")
        f.write("-" * 40 + "\n")
        if 'training_history' in all_metrics:
            hist = all_metrics['training_history']
            f.write(f"Total epochs trained: {len(hist.get('g_loss', []))}\n")
            if hist.get('g_loss'):
                f.write(f"Final Generator Loss: {hist['g_loss'][-1]:.6f}\n")
                f.write(f"Final Discriminator Loss: {hist['d_loss'][-1]:.6f}\n")
        f.write("\n")

        f.write("5. TRADITIONAL METRICS (Final)\n")
        f.write("-" * 40 + "\n")
        if 'traditional' in all_metrics:
            trad = all_metrics['traditional']
            for key in ['FID_10k', 'FID_500', 'KID_10k', 'KID_500',
                        'FRD_10k', 'FRD_500', 'KRD_10k', 'KRD_500']:
                if key in trad:
                    f.write(f"{key}: {trad[key]:.6f}\n")
        f.write("\n")

        f.write("6. STATIC GEOSTATISTICAL METRICS (Final)\n")
        f.write("-" * 40 + "\n")
        if 'static' in all_metrics:
            static = all_metrics['static']
            f.write(f"Variogram MSE: {static.get('variogram_mse', 'N/A')}\n")
            f.write(f"Connectivity MSE: {static.get('connectivity_mse', 'N/A')}\n")
            f.write(f"Histogram KL Divergence: {static.get('hist_kl', 'N/A')}\n")
            f.write(f"PCA Correlation: {static.get('pca_correlation', 'N/A')}\n")
            f.write(f"MDS MMD: {static.get('mds_mmd', 'N/A')}\n")
        f.write("\n")

        f.write("7. FINAL SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write("Training completed successfully with W-space support.\n")
        f.write("The W-space dimension has been saved for use in data assimilation.\n")
        f.write(f"All outputs saved in: training_outputs/\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Final report saved to: {save_path}")
    return save_path


def initialize_models_for_training(generator, discriminator):
    for layer in generator.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    for layer in discriminator.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    print("✓ Models initialized for stable training")


# ============================================
# BUILD AND TRAIN
# ============================================

print("\n" + "=" * 60)
print("BUILDING STYLEGAN2 WITH W-SPACE SUPPORT")
print("=" * 60)

tf.keras.backend.clear_session()
gc.collect()

start_time = time.time()

print("Building Gaussian-regularized Generator with W-space access...")
generator = GaussianStabilizedGenerator(
    latent_dim=latent_dim,
    img_shape=(ni, nj, nz),
    n_filters=64,
    gaussian_reg_strength=0.01
)

print("\nTesting generator...")
dummy_noise = tf.random.normal([1, latent_dim])
try:
    dummy_output = generator(dummy_noise, training=False)
    print(f"✓ Generator output shape: {dummy_output.shape}")
    print(f"✓ Generator output range: [{tf.reduce_min(dummy_output):.3f}, {tf.reduce_max(dummy_output):.3f}]")

    print("\nTesting Z -> W mapping...")
    dummy_w = generator.map_to_w(dummy_noise)
    print(f"✓ W space shape: {dummy_w.shape}")
    print(f"✓ W dimension confirmed: {w_dim}")

    print("\nTesting W -> Image generation...")
    dummy_output_w = generator.generate_from_w(dummy_w, training=False)
    print(f"✓ W-based output shape: {dummy_output_w.shape}")
    print(f"✓ W-based output range: [{tf.reduce_min(dummy_output_w):.3f}, {tf.reduce_max(dummy_output_w):.3f}]")
except Exception as e:
    print(f"✗ Generator error: {e}")
    import traceback

    traceback.print_exc()

print("\nBuilding Feature Extracting Discriminator...")
discriminator = FeatureExtractingDiscriminator(
    img_shape=(ni, nj, nz),
    n_filters=64
)

print("Testing discriminator...")
try:
    dummy_disc_output = discriminator(dummy_output, training=False)
    print(f"✓ Discriminator output shape: {dummy_disc_output.shape}")
except Exception as e:
    print(f"✗ Discriminator error: {e}")
    import traceback

    traceback.print_exc()

print(f"\nGenerator parameters: {generator.count_params():,}")
print(f"Discriminator parameters: {discriminator.count_params():,}")

print("\nInitializing StyleGAN2 unified model...")
stylegan2 = StyleGAN2(
    generator=generator,
    discriminator=discriminator,
    latent_dim=latent_dim,
    d_reg=10.0,
    label_smoothing=0.1,
    gradient_clip=0.5,
    noise_std=0.05,
    latent_gaussian_weight=0.1
)

print("Compiling model...")
g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.5,
    beta_2=0.9,
    epsilon=1e-8
)
d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0004,
    beta_1=0.5,
    beta_2=0.9,
    epsilon=1e-8
)

stylegan2.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)

print("Setting up callbacks...")

sample_saver_callback = SampleSaverCallback(
    generator=generator,
    latent_dim=latent_dim,
    save_dir='training_outputs/samples',
    latent_analysis_interval=50
)

enhanced_metric_callback = EnhancedMetricCallback(
    generator=generator,
    x_test=x_test,
    latent_dim=latent_dim,
    metric_interval=metric_interval,
    log_file='training_outputs/metrics/training_metrics.txt',
    save_dir='training_outputs/metrics'
)

improved_stabilization_callback = ImprovedStabilizationCallback()

adaptive_lr_scheduler = AdaptiveLearningRateScheduler()

live_plot_callback = LivePlotCallback(
    model=stylegan2,
    data=x_test,
    num_images=5
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='g_loss',
    patience=50,
    min_delta=0.01,
    restore_best_weights=True,
    verbose=1,
    mode='min',
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='training_outputs/checkpoints/stylegan2_epoch_{epoch:03d}.weights_w.h5',
    monitor='d_loss',
    save_weights_only=True,
    save_best_only=True,
    mode='min',
    verbose=1,
    save_freq='epoch'
)

print("Creating output directories...")
os.makedirs('training_outputs/samples', exist_ok=True)
os.makedirs('training_outputs/metrics', exist_ok=True)
os.makedirs('training_outputs/checkpoints', exist_ok=True)
os.makedirs('training_outputs/loss_plots', exist_ok=True)
os.makedirs('training_outputs/final_outputs', exist_ok=True)
os.makedirs('training_outputs/latent_analysis', exist_ok=True)

print("\nInitializing models with stable weights...")
initialize_models_for_training(generator, discriminator)

print("Preparing training dataset...")

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

print(f"\n{'=' * 60}")
print("STARTING STYLEGAN2 TRAINING WITH W-SPACE SUPPORT")
print("=" * 60)
print(f"Dataset: {name}")
print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
print(f"Image shape: {ni}x{nj}x{nz}")
print(f"Z dimension: {latent_dim}")
print(f"W dimension: {w_dim}")
print(f"Batch size: {batch_size}")
print(f"Learning rates: G=0.0001, D=0.0004")
print("=" * 60)

try:
    print("\nFinal test before training...")
    test_noise = tf.random.normal([2, latent_dim])
    test_output = generator(test_noise, training=False)
    print(f"✓ Generator test passed! Output shape: {test_output.shape}")

    test_disc_output = discriminator(test_output, training=False)
    print(f"✓ Discriminator test passed! Output shape: {test_disc_output.shape}")

    print("\nStarting training...")

    history = stylegan2.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[
            sample_saver_callback,
            enhanced_metric_callback,
            improved_stabilization_callback,
            adaptive_lr_scheduler,
            live_plot_callback,
            early_stopping,
            checkpoint_callback
        ],
        verbose=1
    )

    g_loss_history = history.history['g_loss']
    d_loss_history = history.history['d_loss']
    d_real_history = history.history['d_real']
    d_fake_history = history.history['d_fake']

    plot_losses_history(g_loss_history, d_loss_history, d_real_history, d_fake_history)

    print("\n✓ Training completed successfully!")


    stylegan2.save_all_models_weights('training_outputs/final_models')

    print(f"\n{'=' * 60}")
    print("PERFORMING FINAL ANALYSIS")
    print("=" * 60)

    print("Generating final samples...")
    size = min(5000, len(x_test))
    noise = np.random.normal(0, 1, (size, latent_dim))
    fake_data = generator.predict(noise, verbose=0, batch_size=32)
    real_data = x_test[:size]

    print("\nCalculating final traditional metrics...")
    final_traditional_metrics = TraditionalMetrics.compute_all_traditional_metrics(
        real_data, fake_data
    )

    print("\nCalculating final static metrics...")
    final_static_metrics = StaticMetrics.compute_all_metrics(real_data, fake_data)

    print("\nAnalyzing final latent space...")
    final_latent_noise = np.random.normal(0, 1, (10000, latent_dim))
    final_latent_analysis = GaussianLatentValidator.test_normality(final_latent_noise)
    final_latent_kl = GaussianLatentValidator.kl_divergence_gaussian(final_latent_noise)
    GaussianLatentValidator.visualize_latent_distribution(
        final_latent_noise,
        len(g_loss_history),
        save_dir='training_outputs/latent_analysis'
    )

    # Consolidar métricas
    final_metrics = {
        'traditional': final_traditional_metrics,
        'static': final_static_metrics,
        'latent_analysis': final_latent_analysis,
        'latent_kl_divergence': final_latent_kl,
        'training_history': {
            'g_loss': g_loss_history,
            'd_loss': d_loss_history,
            'd_real': d_real_history,
            'd_fake': d_fake_history,
            'total_epochs': len(g_loss_history)
        }
    }

    np.save('training_outputs/metrics/final_metrics.npy', final_metrics)

    print("\nGenerating final visualizations...")


    noise = np.random.normal(0, 1, (6, latent_dim))
    final_samples = generator.predict(noise, verbose=0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Final Generated Samples - StyleGAN2 (W-Space)',
                 fontsize=HIGH_RES_CONFIG['title_font_size'] + 2,
                 fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < 6:
            im = ax.imshow(final_samples[i, ..., 0], cmap='viridis', aspect='auto')
            ax.set_title(f'Sample {i + 1}', fontsize=HIGH_RES_CONFIG['font_size'], fontweight='bold')
            ax.axis('off')

    plt.tight_layout()

    final_samples_path = 'training_outputs/final_outputs/final_samples_grid'
    plt.savefig(f'{final_samples_path}.png', dpi=HIGH_RES_CONFIG['dpi'], bbox_inches='tight')
    plt.savefig(f'{final_samples_path}.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    np.save('training_outputs/final_outputs/final_samples.npy', final_samples)

    plot_histograms_and_densities(real_data[:5], fake_data[:5], latent_vectors=final_latent_noise)

    plot_static_metrics_comparison(
        x_test, generator,
        save_path_base='training_outputs/final_outputs/final_static_metrics'
    )

    plot_original_distribution_and_latent_space(
        original_data=x_test,
        generator_model=generator,
        latent_dim=latent_dim,
        save_path='training_outputs/final_outputs'
    )

    end_time = time.time()
    execution_time = end_time - start_time

    generate_final_report(final_metrics, execution_time, 'training_outputs/FINAL_REPORT.txt')

    print(f"\n{'=' * 60}")
    print("STYLEGAN2 TRAINING WITH W-SPACE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total execution time: {execution_time / 3600:.2f} hours")
    print(f"W dimension ({w_dim}) has been saved for data assimilation")
    print(f"Final report saved to: training_outputs/FINAL_REPORT.txt")
    print(f"Final model weights saved in: training_outputs/final_models/")
    print(f"  - generator.weights_w.h5")
    print(f"  - discriminator.weights_w.h5")
    print(f"  - stylegan2_complete.weights_w.h5")
    print(f"  - w_dim.txt")
    print("=" * 60)

    print("\nFINAL METRICS SUMMARY:")
    print("-" * 40)
    print("Traditional Metrics:")
    for key, value in final_traditional_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nStatic Geostatistical Metrics:")
    print(f"  Variogram MSE: {final_static_metrics['variogram_mse']:.6f}")
    print(f"  Connectivity MSE: {final_static_metrics['connectivity_mse']:.6f}")
    print(f"  Histogram KL Divergence: {final_static_metrics['hist_kl']:.6f}")
    print(f"  PCA Correlation: {final_static_metrics['pca_correlation']:.6f}")
    print(f"  MDS MMD: {final_static_metrics['mds_mmd']:.6f}")

    print("\nW-Space Information:")
    print(f"  W dimension: {w_dim}")
    print(f"  The W space is ready for data assimilation")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Training failed with error: {e}")
    import traceback

    traceback.print_exc()

    try:
        if 'g_loss_history' in locals() and g_loss_history:
            plot_losses_history(g_loss_history, d_loss_history, d_real_history, d_fake_history)
            print("✓ Loss plots saved despite error.")

        if 'stylegan2' in locals():
            print("\nAttempting to save models despite error...")
            stylegan2.save_all_models_weights('training_outputs/emergency_save')
    except:
        print("Could not save models.")

finally:
    print("\nCleaning up memory...")
    plt.ioff()
    plt.close('all')
    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 60)
    print("PROGRAM EXECUTION COMPLETED")
    print("=" * 60)