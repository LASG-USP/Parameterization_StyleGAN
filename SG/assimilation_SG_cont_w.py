import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pickle
import tensorflow as tf
import os
import shutil
import glob
from main import run_assimilation, run_true_case
from processing import parameter_mapping, vectorization
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU
import json


# ============================================
# CUSTOM OBJECTS FOR STYLEGAN2 WITH W-SPACE
# ============================================

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


class ImprovedWSpaceGeneratorForAssimilation(tf.keras.Model):
    """Generator for assimilation in W space."""

    def __init__(self, latent_dim=512, img_shape=(48, 48, 1), n_filters=64,
                 gaussian_reg_strength=0.01, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.w_dim = latent_dim * 2
        self.img_shape = img_shape
        self.n_filters = n_filters
        self.gaussian_reg_strength = gaussian_reg_strength

        # Mapping network: Z -> W
        self.mapping = self._build_mapping_network()

        # BatchNormalization for the latent space
        self.latent_normalization = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, center=True, scale=True
        )

        # Constant input
        self.constant = tf.Variable(
            tf.random.normal([1, 4, 4, n_filters * 8], stddev=0.02),
            trainable=False
        )

        # Building blocks
        self._build_generator_blocks()

        # To RGB
        self.to_rgb = layers.Conv2D(
            img_shape[2], 1, padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            activation='tanh'
        )

    def _build_mapping_network(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(self.latent_dim * 2, kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.Dense(self.latent_dim * 2, kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.Dense(self.latent_dim * 2, kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ], name='mapping_network')

    def _build_generator_blocks(self):
        self.blocks = []

        # Block 1: 4x4
        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 8, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        # Block 2: 8x8
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 4, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        # Block 3: 16x16
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 2, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        # Block 4: 32x32
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])

        # Block 5: 64x64 -> 48x48
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
        """Maps Z vectors to W space"""
        return self.mapping(z_vectors)

    def generate_from_w(self, w_vectors, training=False):
        """Generates images from W space vectors"""
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
        """
        Generates images from Z or W depending on input dimension.
        """
        input_dim = inputs.shape[-1]

        if input_dim == self.latent_dim:
            # Z -> W -> Image
            w = self.map_to_w(inputs)
            return self.generate_from_w(w, training=training)
        elif input_dim == self.w_dim:
            # W -> Image (path for assimilation)
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


class WSpaceAssimilationWrapper:
    """
    Wrapper for assimilation in W space.
    Compatible with processing2.py: additional.predict(M.T, ...)
    """

    def __init__(self, generator, w_dim):
        self.generator = generator
        self.latent_dim = generator.latent_dim
        self.w_dim = w_dim

    def map_to_w(self, z_vectors):
        """Converts Z -> W"""
        if z_vectors.ndim == 1:
            z_t = z_vectors.reshape(1, -1)
            if z_t.shape[1] == self.latent_dim:
                return self.generator.map_to_w(tf.constant(z_t)).numpy().flatten()
            else:
                raise ValueError(f"Expected {self.latent_dim}, got {z_t.shape[1]}")

        # Detect format
        if z_vectors.shape[0] == self.latent_dim:
            # Format (latent_dim, N)
            z_t = z_vectors.T  # (N, latent_dim)
            w_t = self.generator.map_to_w(tf.constant(z_t)).numpy()  # (N, w_dim)
            return w_t.T  # (w_dim, N)
        elif z_vectors.shape[1] == self.latent_dim:
            # Format (N, latent_dim)
            return self.generator.map_to_w(tf.constant(z_vectors)).numpy()
        else:
            raise ValueError(f"Unexpected shape: {z_vectors.shape}")

    def predict(self, inputs, verbose=0, batch_size=32):
        """
        Unified predict for assimilation.
        Automatically detects input format.
        """
        if inputs.ndim == 1:
            inputs_reshaped = inputs.reshape(1, -1)
            tf_inputs = tf.constant(inputs_reshaped)
            if inputs_reshaped.shape[1] == self.latent_dim:
                return self.generator(tf_inputs, training=False).numpy()
            elif inputs_reshaped.shape[1] == self.w_dim:
                return self.generator.generate_from_w(tf_inputs, training=False).numpy()
            else:
                raise ValueError(f"1D input: Expected {self.latent_dim} or {self.w_dim}, got {len(inputs)}")

        if inputs.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {inputs.shape}")

        dim0, dim1 = inputs.shape[0], inputs.shape[1]

        # Format (w_dim, N) or (latent_dim, N)
        if dim0 == self.w_dim:
            inputs_t = inputs.T  # (N, w_dim)
            tf_inputs = tf.constant(inputs_t)
            return self.generator.generate_from_w(tf_inputs, training=False).numpy()
        elif dim0 == self.latent_dim:
            inputs_t = inputs.T  # (N, latent_dim)
            tf_inputs = tf.constant(inputs_t)
            return self.generator(tf_inputs, training=False).numpy()

        # Format (N, w_dim) or (N, latent_dim)
        elif dim1 == self.w_dim:
            tf_inputs = tf.constant(inputs)
            return self.generator.generate_from_w(tf_inputs, training=False).numpy()
        elif dim1 == self.latent_dim:
            tf_inputs = tf.constant(inputs)
            return self.generator(tf_inputs, training=False).numpy()

        raise ValueError(f"Cannot determine format for shape {inputs.shape}")


def create_assimilation_wrapper(gan_wrapper):
    """
    Creates a wrapper compatible with processing2.py.
    In processing2.py: out = additional.predict(M.T, verbose=0)[..., 0]
    """

    class GANAssimilationWrapper:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.latent_dim = wrapper.latent_dim
            self.w_dim = wrapper.w_dim

        def predict(self, inputs, verbose=0, batch_size=None):
            """Predict method compatible with processing2.py"""
            if inputs.ndim == 2:
                n_samples, n_dims = inputs.shape

                if n_dims == self.w_dim:
                    return self.wrapper.predict(inputs.T, verbose=verbose)
                elif n_dims == self.latent_dim:
                    return self.wrapper.predict(inputs.T, verbose=verbose)
                else:
                    raise ValueError(f"Unexpected dim: {n_dims}")
            else:
                return self.wrapper.predict(inputs, verbose=verbose)

        @property
        def n_filters(self):
            return self.wrapper.generator.n_filters

        @property
        def img_shape(self):
            return self.wrapper.generator.img_shape

    return GANAssimilationWrapper(gan_wrapper)


# ============================================
# FUNCTION TO LOAD THE MODEL
# ============================================

def load_improved_stylegan_with_wspace(model_dir='training_outputs/final_models',
                                       latent_dim=512, img_shape=(48, 48, 1), n_filters=64):
    """Loads the improved StyleGAN2 generator with W-space support."""
    print("=" * 60)
    print("LOADING STYLEGAN2 MODEL WITH W-SPACE")
    print("=" * 60)

    generator_path = os.path.join(model_dir, 'generator.weights_w.h5')
    config_path = os.path.join(model_dir, 'model_config.json')
    w_dim_path = os.path.join(model_dir, 'w_dim.txt')

    # Load W dimension
    w_dim = latent_dim * 2
    if os.path.exists(w_dim_path):
        with open(w_dim_path, 'r') as f:
            w_dim = int(f.read().strip())
        print(f"✓ W dimension loaded: {w_dim}")
    else:
        print(f"⚠ w_dim.txt not found, using default: {w_dim}")

    # Load configuration
    gaussian_reg_strength = 0.01
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Model configuration loaded")
        if 'generator_config' in config:
            gen_config = config['generator_config']
            latent_dim = gen_config.get('latent_dim', latent_dim)
            n_filters = gen_config.get('n_filters', n_filters)
            gaussian_reg_strength = gen_config.get('gaussian_reg_strength', 0.01)
            if 'img_shape' in gen_config:
                img_shape = tuple(gen_config['img_shape'])

    # Check weights file
    print(f"\nChecking file: {generator_path}")
    if not os.path.exists(generator_path):
        print(f"✗ File NOT found: {generator_path}")
        alt_paths = [
            'generator.weights_w.h5',
            'final_models/generator.weights_w.h5',
            '../training_outputs/final_models/generator.weights_w.h5',
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"  ✓ Found at: {alt_path}")
                generator_path = alt_path
                break
        else:
            print("  ✗ No file found")
            return None, None

    file_size_mb = os.path.getsize(generator_path) / 1024 / 1024
    print(f"✓ File found! Size: {file_size_mb:.2f} MB")

    # Create model
    print("\nCreating model architecture...")
    generator = ImprovedWSpaceGeneratorForAssimilation(
        latent_dim=latent_dim,
        img_shape=img_shape,
        n_filters=n_filters,
        gaussian_reg_strength=gaussian_reg_strength
    )

    # Build
    print("Building model...")
    dummy_input = tf.random.normal([1, latent_dim])
    _ = generator(dummy_input, training=False)
    print(f"✓ Model built")

    # Load weights
    print(f"\nLoading weights...")
    try:
        generator.load_weights(generator_path)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
        try:
            print("Trying with by_name=True...")
            generator.load_weights(generator_path, by_name=True)
            print("✓ Weights loaded with by_name=True!")
        except Exception as e2:
            print(f"✗ Error: {e2}")
            return None, None

    # Create wrapper
    gan_wrapper = WSpaceAssimilationWrapper(generator, w_dim)

    # Quick tests
    print("\nTesting wrapper...")
    test_z = np.random.normal(0, 1, (latent_dim, 10))
    test_out = gan_wrapper.predict(test_z, verbose=0)
    print(f"✓ Test Z ({test_z.shape}): Output {test_out.shape}")

    test_w = np.random.normal(0, 1, (w_dim, 10))
    test_out_w = gan_wrapper.predict(test_w, verbose=0)
    print(f"✓ Test W ({test_w.shape}): Output {test_out_w.shape}")

    print("\n" + "=" * 60)
    print("MODEL LOADED SUCCESSFULLY!")
    print(f"Z dimension: {latent_dim}")
    print(f"W dimension: {w_dim}")
    print("=" * 60)

    return gan_wrapper, w_dim


# ============================================
# INITIAL CONFIGURATION
# ============================================

def setup_plots():
    """Configure matplotlib for high quality plots"""
    rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (12, 6),
        'figure.autolayout': False,
        'figure.constrained_layout.use': True,
    })


setup_plots()

# ============================================
# INITIAL CLEANUP
# ============================================

print("\n" + "=" * 60)
print("CLEANING TEMPORARY DIRECTORIES")
print("=" * 60)

temp_patterns = ['sim_*', 'temp_*', '.sim_*', '__pycache__']
for pattern in temp_patterns:
    for path in glob.glob(pattern):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"  Cleaned: {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  Removed: {path}")
        except Exception as e:
            print(f"  ⚠ Could not clean {path}: {e}")

# ============================================
# PARAMETERS
# ============================================

N = 500
latent_dim = 512
ptype = 'gan_uii'

# Create directories
print("\n" + "=" * 60)
print("CREATING DIRECTORIES")
print("=" * 60)

directories = ['outputs']
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory: {directory}")

# ============================================
# LOAD THE MODEL
# ============================================

print("\n" + "=" * 60)
print("LOADING STYLEGAN2 MODEL")
print("=" * 60)

model_dir = 'training_outputs/final_models'
print(f"Searching for model in: {os.path.abspath(model_dir)}")

GAN, W_DIM = load_improved_stylegan_with_wspace(
    model_dir=model_dir,
    latent_dim=latent_dim,
    img_shape=(48, 48, 1),
    n_filters=64
)

if GAN is None:
    print("\n⚠ WARNING: Could not load the model.")
    print("Trying alternative paths...")

    # Try to load directly from current directory
    alt_dirs = ['.', '..', 'training_outputs/final_models']
    for alt_dir in alt_dirs:
        GAN, W_DIM = load_improved_stylegan_with_wspace(
            model_dir=alt_dir,
            latent_dim=latent_dim,
            img_shape=(48, 48, 1),
            n_filters=64
        )
        if GAN is not None:
            break

    if GAN is None:
        print("\n✗ CRITICAL ERROR: Model not found.")
        exit(1)

# Create compatible wrapper
GAN_ASSIMILATION = create_assimilation_wrapper(GAN)
print("✓ Assimilation wrapper created")

# ============================================
# LOAD DATA AND PREPARE ENSEMBLE
# ============================================

print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

# Load training data
print("Loading data...")
X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')
X_train[X_train < 1] = 1
X_train = np.log(X_train)
X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
X_train = X_train[..., np.newaxis]
print(f"✓ Data loaded: {X_train.shape}")

# Generate ensemble in Z space and convert to W
print(f"\nGenerating prior ensemble...")
print(f"  N (models): {N}")
print(f"  Z dimension: {latent_dim}")

z_prior = np.random.normal(0, 1, (latent_dim, N))
print(f"✓ Z prior shape: {z_prior.shape}")
print(f"  Z stats: mean={z_prior.mean():.4f}, std={z_prior.std():.4f}")

# Convert to W space
print(f"\nConverting to W space...")
w_prior = GAN.map_to_w(z_prior)
print(f"✓ W prior shape: {w_prior.shape}")
print(f"  Expected: ({W_DIM}, {N})")
print(f"  W stats: mean={w_prior.mean():.4f}, std={w_prior.std():.4f}")

# True model
print(f"\nLoading true model...")
true_model = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')[0:1]
true_model[true_model < 1] = 1
true_model = vectorization(np.log(true_model), 'vec', (48, 48))
print(f"✓ True model shape: {true_model.shape}")

# ============================================
# ASSIMILATION CONFIGURATION
# ============================================

name_case = 'cont_32_nonloc_wspace_improved'
n_iter = 32
tapering = None

print(f"\nConfiguration:")
print(f"  Case: {name_case}")
print(f"  Iterations: {n_iter}")
print(f"  Tapering: {tapering}")
print(f"  Space: W (dimension {W_DIM})")
print(f"  N models: {N}")

# ============================================
# RUN ASSIMILATION
# ============================================

print("\n" + "=" * 60)
print("RUNNING ASSIMILATION IN W SPACE")
print("=" * 60)

# Run true case
print("Running true case...")
try:
    D_true, D_measurements = run_true_case(parameter_mapping(true_model, type='log', additional=None))
    print("✓ run_true_case executed successfully")
except Exception as e:
    print(f"✗ Error in run_true_case: {e}")
    import traceback

    traceback.print_exc()
    raise

# Run assimilation
print(f"\nRunning assimilation...")
print(f"  Ensemble shape: {w_prior.shape}")
print(f"  W dimension: {w_prior.shape[0]}")
print(f"  N models: {w_prior.shape[1]}")

try:
    Result = run_assimilation(
        w_prior,
        true_model,
        D_measurements,
        n_iter,
        12,
        tapering=tapering,
        ptype=ptype,
        additional=GAN_ASSIMILATION,
        metric_space='latent'
    )
    print("✓ Assimilation completed successfully!")
except Exception as e:
    print(f"✗ Error in assimilation: {e}")
    import traceback

    traceback.print_exc()
    raise

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save main results
with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)
print(f"✓ Results_{name_case}.pickle saved")

# Reload to verify
with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)
print("✓ Results verified")

# Save DM
np.savetxt(f'./outputs/dm_{name_case}.txt', np.array(Result[0]).mean(-1))
print(f"✓ dm_{name_case}.txt saved")

# Save metrics to text files
with open(f'./outputs/DM_{name_case}.txt', 'w', encoding='utf-8') as f:
    f.write("# Data Mismatch per iteration\n")
    for i, dm_values in enumerate(Result[0]):
        if isinstance(dm_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, dm_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {dm_values}\n")
print(f"✓ DM_{name_case}.txt saved")

with open(f'./outputs/RMSE_{name_case}.txt', 'w', encoding='utf-8') as f:
    f.write("# RMSE per iteration\n")
    for i, rsme_values in enumerate(Result[1]):
        if isinstance(rsme_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, rsme_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {rsme_values}\n")
print(f"✓ RMSE_{name_case}.txt saved")

with open(f'./outputs/Spread_{name_case}.txt', 'w', encoding='utf-8') as f:
    f.write("# Spread per iteration\n")
    for i, spread_values in enumerate(Result[2]):
        if isinstance(spread_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, spread_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {spread_values}\n")
print(f"✓ Spread_{name_case}.txt saved")

# Save W-space information
with open(f'./outputs/w_space_info_{name_case}.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("W-SPACE DATA ASSIMILATION INFORMATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"W dimension: {W_DIM}\n")
    f.write(f"Z dimension: {GAN.latent_dim}\n")
    f.write(f"Number of models (N): {N}\n")
    f.write(f"Number of iterations: {n_iter}\n")
    f.write(f"Tapering: {tapering}\n\n")
    f.write("W prior statistics:\n")
    f.write(f"  Mean: {w_prior.mean():.6f}\n")
    f.write(f"  Std: {w_prior.std():.6f}\n")
    f.write(f"  Min: {w_prior.min():.6f}\n")
    f.write(f"  Max: {w_prior.max():.6f}\n")
    f.write(f"  Shape: {w_prior.shape}\n")
print(f"✓ w_space_info_{name_case}.txt saved")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Properties (OPR, WPR, BHP)
print("Generating Plot 1: Properties...")
try:
    fig1, axs = plt.subplots(3, 9, figsize=(18, 5))

    for _1 in range(3):
        for _2 in range(9):
            # Initial ensemble (gray)
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1][:, _2, :],
                             'xkcd:gray', alpha=0.35, linewidth=0.5)
            # Final ensemble (blue)
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1][:, _2, :],
                             'b', alpha=0.35, linewidth=0.5)
            # Observed data (red)
            axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1][:, _2],
                             'or', markersize=2)

    for _2 in range(9):
        axs[0, _2].set_ylabel('OPR (m³/day)', fontsize=8)
        axs[1, _2].set_ylabel('WPR (m³/day)', fontsize=8)
        axs[2, _2].set_ylabel('BHP (kgf/cm²)', fontsize=8)
        for _1 in range(3):
            axs[_1, _2].set_xlabel('Time (day)', fontsize=7)

    plt.suptitle(f'W-Space Assimilation - {name_case}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig1.savefig(f'./outputs/properties_plot_{name_case}.png', dpi=300, bbox_inches='tight')
    fig1.savefig(f'./outputs/properties_plot_{name_case}.svg', format='svg', bbox_inches='tight')
    print(f"✓ properties_plot_{name_case} saved")
    plt.close(fig1)
except Exception as e:
    print(f"⚠ Error in Plot 1: {e}")

# Plot 2: Metrics (DM, RMSE, Spread) - WITHOUT BALANCED ACCURACY
print("Generating Plot 2: Metrics...")
try:
    fig2, axs = plt.subplots(1, 3, figsize=(8, 3))

    metrics_data = [Result[0], Result[1], Result[2]]
    metrics_names = ['DM', 'RMSE', 'Spread']

    for i, (data, name) in enumerate(zip(metrics_data, metrics_names)):
        axs[i].boxplot(data, showfliers=False)
        axs[i].set_xlabel('Iteration', fontsize=9)
        axs[i].set_ylabel(name, fontsize=9)
        axs[i].set_title(name, fontsize=10, fontweight='bold')
        [axs[_].set_xticks(np.linspace(0 + 1, n_iter + 1, 5), np.linspace(0, n_iter, 5, dtype='int')) for _ in range(3)]
        plt.tight_layout()



    fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.svg', format='svg', bbox_inches='tight')
    print(f"✓ metrics_boxplot_{name_case} saved")
    plt.close(fig2)
except Exception as e:
    print(f"⚠ Error in Plot 2: {e}")

# Plot 3: Additional well data
print("Generating Plot 3: Additional data...")
try:
    fig3, axs = plt.subplots(2, 4, figsize=(9, 5))

    for _1 in range(2):
        for _2 in range(4):
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1 + 3][:, _2, :],
                             color=[0.2, 0.2, 0.2], alpha=0.35, linewidth=0.5)
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1 + 3][:, _2, :],
                             color='blue', alpha=0.35, linewidth=0.5)
            axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1 + 3][:, _2],
                             'or', markersize=2)

    for _2 in range(4):
        axs[0, _2].set_ylabel('WIR (m³/day)', fontsize=8)
        axs[1, _2].set_ylabel('BHP (kgf/cm²)', fontsize=8)
        for _1 in range(2):
            axs[_1, _2].set_xlabel('Time (day)', fontsize=7)

    plt.suptitle(f'W-Space Additional Well Data - {name_case}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig3.savefig(f'./outputs/additional_wells_{name_case}.png', dpi=300, bbox_inches='tight')
    fig3.savefig(f'./outputs/additional_wells_{name_case}.svg', format='svg', bbox_inches='tight')
    print(f"✓ additional_wells_{name_case} saved")
    plt.close(fig3)
except Exception as e:
    print(f"⚠ Error in Plot 3: {e}")

# Plot 4: Facies maps (generating from W space)
print("Generating Plot 4: Facies maps...")
try:
    fig4, axs = plt.subplots(1, 7, figsize=(16, 3))

    # Revert facies values for visualization
    true_model_plot = true_model.copy()
    true_model_plot[true_model_plot == 100] = -1  # purple
    true_model_plot[true_model_plot == 1000] = 0  # green
    true_model_plot[true_model_plot == 9000] = 1  # yellow

    # True model
    axs[0].imshow(true_model_plot.reshape(48, 48), vmin=-1, vmax=1, cmap='viridis')

    print("  Generating images with GAN (W-space)...")

    # Result[3][0]: initial ensemble in W space, shape (W_DIM, N)
    # Result[3][-1]: final ensemble in W space, shape (W_DIM, N)
    print(f"  Prior ensemble shape in Result: {Result[3][0].shape}")
    print(f"  Posterior ensemble shape in Result: {Result[3][-1].shape}")

    # Generate prior images - using GAN.predict with format (dims, N)
    prior_images = GAN.predict(Result[3][0], verbose=0)  # (N, 48, 48, 1)
    prior_mean = prior_images.mean(0).squeeze()  # (48, 48)
    prior_std = prior_images.std(0).squeeze()  # (48, 48)
    prior_member1 = prior_images[0].squeeze()  # (48, 48)

    # Generate posterior images
    posterior_images = GAN.predict(Result[3][-1], verbose=0)  # (N, 48, 48, 1)
    posterior_mean = posterior_images.mean(0).squeeze()  # (48, 48)
    posterior_std = posterior_images.std(0).squeeze()  # (48, 48)
    posterior_member1 = posterior_images[0].squeeze()  # (48, 48)

    print(f"  Prior images shape: {prior_images.shape}")
    print(f"  Posterior images shape: {posterior_images.shape}")

    # Plot
    axs[1].imshow(prior_mean, vmin=-1, vmax=1, cmap='viridis')
    axs[2].imshow(posterior_mean, vmin=-1, vmax=1, cmap='viridis')
    axs[3].imshow(prior_std, vmin=0, vmax=0.7, cmap='viridis')
    axs[4].imshow(posterior_std, vmin=0, vmax=0.7, cmap='viridis')
    axs[5].imshow(prior_member1, vmin=-1, vmax=1, cmap='viridis')
    axs[6].imshow(posterior_member1, vmin=-1, vmax=1, cmap='viridis')

    # Turn off axes
    for ax in axs:
        ax.axis('off')

    # Titles
    cols = ['True', 'Prior\nmean (W)', 'Posterior\nmean (W)',
            'Prior\nstd', 'Posterior\nstd',
            'Prior\nMember 1', 'Posterior\nMember 1']
    for i, title in enumerate(cols):
        axs[i].set_title(title, fontsize=8, pad=5)

    plt.suptitle(f'W-Space Facies Maps - {name_case}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig4.savefig(f'./outputs/model_visualization_{name_case}.png', dpi=300, bbox_inches='tight')
    fig4.savefig(f'./outputs/model_visualization_{name_case}.svg', format='svg', bbox_inches='tight')
    print(f"✓ model_visualization_{name_case} saved")
    plt.close(fig4)
except Exception as e:
    print(f"⚠ Error in Plot 4: {e}")
    import traceback

    traceback.print_exc()

# Plot 5: Additional posterior realizations
print("Generating Plot 5: Additional realizations...")
try:
    realizations = GAN.predict(Result[3][-1], verbose=0)

    fig5, axs5 = plt.subplots(1, 7, figsize=(14, 2))
    for j in range(7):
        if j < len(realizations):
            axs5[j].imshow(realizations[j].squeeze(), vmin=-1, vmax=1, cmap='viridis')
        axs5[j].axis('off')
        axs5[j].set_title(f'Member {j + 1} (W)', fontsize=8, pad=5)

    plt.suptitle(f'W-Space Posterior Realizations - {name_case}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    fig5.savefig(f'./outputs/additional_realizations_{name_case}.png', dpi=300, bbox_inches='tight')
    fig5.savefig(f'./outputs/additional_realizations_{name_case}.svg', format='svg', bbox_inches='tight')
    print(f"✓ additional_realizations_{name_case} saved")
    plt.close(fig5)
except Exception as e:
    print(f"⚠ Error in Plot 5: {e}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("W-SPACE ASSIMILATION COMPLETED!")
print("=" * 60)
print(f"Case: {name_case}")
print(f"Assimilation space: W (dimension {W_DIM})")
print(f"Number of models: {N}")
print(f"Number of iterations: {n_iter}")
print(f"\nFiles saved in: {os.path.abspath('outputs')}")
print(f"  - Results_{name_case}.pickle")
print(f"  - w_space_info_{name_case}.txt")
print(f"  - DM_{name_case}.txt")
print(f"  - RMSE_{name_case}.txt")
print(f"  - Spread_{name_case}.txt")
print(f"  - dm_{name_case}.txt")
print(f"  - properties_plot_{name_case}.png/svg")
print(f"  - metrics_boxplot_{name_case}.png/svg")
print(f"  - additional_wells_{name_case}.png/svg")
print(f"  - model_visualization_{name_case}.png/svg")
print(f"  - additional_realizations_{name_case}.png/svg")

# Final metrics
print(f"\nFinal metrics (last iteration):")
try:
    final_dm = np.array(Result[0][-1])
    print(f"  Average DM: {final_dm.mean():.4f} (min: {final_dm.min():.4f}, max: {final_dm.max():.4f})")
except Exception as e:
    print(f"  DM: error - {e}")

try:
    final_rmse = np.array(Result[1][-1])
    print(f"  Average RMSE: {final_rmse.mean():.4f} (min: {final_rmse.min():.4f}, max: {final_rmse.max():.4f})")
except Exception as e:
    print(f"  RMSE: error - {e}")

try:
    final_spread = np.array(Result[2][-1])
    print(f"  Average Spread: {final_spread.mean():.4f} (min: {final_spread.min():.4f}, max: {final_spread.max():.4f})")
except Exception as e:
    print(f"  Spread: error - {e}")

print("\n" + "=" * 60)
print("END OF PROCESSING")
print("=" * 60)