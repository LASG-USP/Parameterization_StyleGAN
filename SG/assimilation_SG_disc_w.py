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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU


# ============================================
# CUSTOM OBJECTS FOR STYLEGAN2 WITH W-SPACE
# ============================================

class StableAdaIN(layers.Layer):
    """Adaptive Instance Normalization - Stabilized Version."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.style_dense = layers.Dense(
            self.channels * 2,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )
        super().build(input_shape)

    def call(self, inputs):
        x, w = inputs
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
        config.update({'channels': self.channels})
        return config


class GaussianRegularizedAdaIN(layers.Layer):
    """AdaIN with Gaussian regularization - Same as training."""

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


class GaussianStabilizedGeneratorForAssimilation(tf.keras.Model):
    """Generator for assimilation - Operates in W space."""

    def __init__(self, latent_dim=512, img_shape=(48, 48, 1), n_filters=64,
                 gaussian_reg_strength=0.01, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.w_dim = latent_dim * 2
        self.img_shape = img_shape
        self.n_filters = n_filters
        self.gaussian_reg_strength = gaussian_reg_strength

        self.mapping = self._build_mapping_network()
        self.constant = tf.Variable(
            tf.random.normal([1, 4, 4, n_filters * 8], stddev=0.02),
            trainable=False
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
        # Block 1
        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 8, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])
        # Block 2
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 4, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])
        # Block 3
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters * 2, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])
        # Block 4
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters, 3, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            GaussianRegularizedAdaIN(self.n_filters, self.gaussian_reg_strength),
            layers.LeakyReLU(0.2),
        ])
        # Block 5
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
                if isinstance(layer, (GaussianRegularizedAdaIN, StableAdaIN)):
                    x = layer([x, w_vectors], training=training)
                else:
                    x = layer(x)

        x = self.to_rgb(x)
        return x

    def call(self, inputs, training=False):
        """
        Generates images.
        If input has dim = latent_dim -> Z, maps to W first
        If input has dim = w_dim -> W, generates directly
        """
        input_dim = inputs.shape[-1]

        if input_dim == self.latent_dim:
            # Z -> W -> Image
            w = self.map_to_w(inputs)
            return self.generate_from_w(w, training=training)
        elif input_dim == self.w_dim:
            # W -> Image (PATH FOR ASSIMILATION)
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


class WSpaceGeneratorWrapper:
    """
    Wrapper that exposes W space for assimilation.

    Compatible with two call forms:
    1. additional.predict(M.T, ...) where M.shape = (dims, N)
       M.T will have shape (N, dims)
    2. additional.predict(vectors, ...) where vectors.shape = (dims, N)

    The wrapper automatically detects the format based on dimension.
    """

    def __init__(self, generator, w_dim):
        self.generator = generator
        self.latent_dim = generator.latent_dim
        self.w_dim = w_dim
        self._n_calls = 0  # Counter for debug

    def map_to_w(self, z_vectors):
        """
        Converts Z -> W
        Accepts: (latent_dim, N) or (N, latent_dim)
        Returns: (w_dim, N) or (N, w_dim) - maintains format
        """
        if z_vectors.ndim == 1:
            # Single vector
            z_t = z_vectors.reshape(1, -1)
            if z_t.shape[1] == self.latent_dim:
                return self.generator.map_to_w(tf.constant(z_t)).numpy().flatten()
            else:
                raise ValueError(f"Expected {self.latent_dim}, got {z_t.shape[1]}")

        # Detect format
        if z_vectors.shape[0] == self.latent_dim and z_vectors.shape[1] != self.latent_dim:
            # Format (latent_dim, N) - common in assimilation
            z_t = z_vectors.T  # (N, latent_dim)
            w_t = self.generator.map_to_w(tf.constant(z_t)).numpy()  # (N, w_dim)
            return w_t.T  # (w_dim, N)
        elif z_vectors.shape[1] == self.latent_dim:
            # Format (N, latent_dim) or (batch, latent_dim)
            return self.generator.map_to_w(tf.constant(z_vectors)).numpy()
        else:
            raise ValueError(f"Unexpected shape: {z_vectors.shape}. Expected dim={self.latent_dim} in axis 0 or 1")

    def predict(self, inputs, verbose=0, batch_size=32):
        """
        Unified predict for assimilation.

        Automatically detects input format:
        - If first dim = w_dim (1024): interprets as (w_dim, N) and generates from W
        - If last dim = w_dim (1024): interprets as (N, w_dim) and generates from W
        - If first dim = latent_dim (512): interprets as (latent_dim, N) and generates from Z
        - If last dim = latent_dim (512): interprets as (N, latent_dim) and generates from Z
        - If none of the cases: ERROR

        Returns: array with shape (N, H, W, C) or (batch, H, W, C)
        """
        self._n_calls += 1

        if inputs.ndim == 1:
            # Single vector - assume Z
            inputs_reshaped = inputs.reshape(1, -1)
            tf_inputs = tf.constant(inputs_reshaped)
            if inputs_reshaped.shape[1] == self.latent_dim:
                return self.generator(tf_inputs, training=False).numpy()
            else:
                raise ValueError(f"1D input: Expected {self.latent_dim} elements, got {len(inputs)}")

        if inputs.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {inputs.shape}")

        # Detect format based on dimensions
        dim0, dim1 = inputs.shape[0], inputs.shape[1]

        # Case 1: (w_dim, N) or (latent_dim, N) - assimilation format
        if dim0 == self.w_dim:
            # Format (w_dim, N) -> transpose to (N, w_dim)
            inputs_t = inputs.T
            tf_inputs = tf.constant(inputs_t)
            result = self.generator.generate_from_w(tf_inputs, training=False).numpy()
            return result

        elif dim0 == self.latent_dim:
            # Format (latent_dim, N) -> transpose to (N, latent_dim)
            inputs_t = inputs.T
            tf_inputs = tf.constant(inputs_t)
            result = self.generator(tf_inputs, training=False).numpy()
            return result

        # Case 2: (N, w_dim) or (N, latent_dim) - batch format
        elif dim1 == self.w_dim:
            # Format (N, w_dim) - generate directly
            tf_inputs = tf.constant(inputs)
            result = self.generator.generate_from_w(tf_inputs, training=False).numpy()
            return result

        elif dim1 == self.latent_dim:
            # Format (N, latent_dim) - Z to W internally
            tf_inputs = tf.constant(inputs)
            result = self.generator(tf_inputs, training=False).numpy()
            return result

        # Case 3: None of the dimensions match
        # Try to infer: if dim0 == dim1, probably a square (N, N)
        if dim0 == dim1:
            # May be a square matrix - check which dimension is more likely
            raise ValueError(
                f"Ambiguous square matrix with shape ({dim0}, {dim1}). "
                f"Expected dimensions: Z={self.latent_dim} or W={self.w_dim}. "
                f"Call #{self._n_calls}: inputs.shape={inputs.shape}"
            )

        raise ValueError(
            f"Cannot determine format for shape {inputs.shape}. "
            f"Expected: first or last dimension to be {self.latent_dim} (Z) or {self.w_dim} (W). "
            f"Call #{self._n_calls}"
        )

    def __call__(self, *args, **kwargs):
        """Support for direct call as function"""
        return self.predict(*args, **kwargs)


# ============================================
# WRAPPER COMPATIBLE WITH PROCESSING2.PY
# ============================================

def create_gan_wrapper_compativel(gan_wrapper):
    """
    Creates a wrapper that is 100% compatible with the call in processing2.py:
    out = additional.predict(M.T, verbose=0)[..., 0]

    where M.shape = (dims, N) and M.T.shape = (N, dims)
    """

    class GANWrapperCompativel:
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self.latent_dim = wrapper.latent_dim

        def predict(self, inputs, verbose=0, batch_size=None):
            """
            Predict method compatible with processing2.py

            Expects inputs in format (N, dims) where dims = 512 (Z) or 1024 (W)
            """
            # Check format
            if inputs.ndim == 2:
                n_samples, n_dims = inputs.shape

                # If (N, 1024) -> W space
                if n_dims == self.wrapper.w_dim:
                    return self.wrapper.predict(inputs.T, verbose=verbose)
                # If (N, 512) -> Z space
                elif n_dims == self.latent_dim:
                    return self.wrapper.predict(inputs.T, verbose=verbose)
                else:
                    # Try to infer
                    if n_dims == n_samples and n_dims not in [self.latent_dim, self.wrapper.w_dim]:
                        # Strange square matrix
                        raise ValueError(
                            f"Ambiguous input shape: ({n_samples}, {n_dims}). "
                            f"Expected last dim to be {self.latent_dim} or {self.wrapper.w_dim}"
                        )
                    raise ValueError(f"Unexpected dim: {n_dims}")
            else:
                return self.wrapper.predict(inputs, verbose=verbose)

        @property
        def n_filters(self):
            return self.wrapper.generator.n_filters

        @property
        def img_shape(self):
            return self.wrapper.generator.img_shape

        @property
        def w_dim(self):
            return self.wrapper.w_dim

    return GANWrapperCompativel(gan_wrapper)


# ============================================
# FUNCTION TO LOAD THE MODEL WITH W-SPACE
# ============================================

def load_stylegan_generator_with_w(model_path='training_outputs/final_models/generator.weights_w.h5',
                                   latent_dim=512, img_shape=(48, 48, 1), n_filters=64):
    """
    Loads the StyleGAN2 generator and returns a wrapper with W space access.
    """
    print("=" * 60)
    print("LOADING STYLEGAN2 MODEL WITH W-SPACE SUPPORT")
    print("=" * 60)

    # Try to read W dimension from saved file
    w_dim = latent_dim * 2  # Default: 1024
    w_dim_file = os.path.join(os.path.dirname(model_path), 'w_dim.txt')
    if os.path.exists(w_dim_file):
        with open(w_dim_file, 'r') as f:
            w_dim = int(f.read().strip())
        print(f"✓ W dimension loaded from file: {w_dim}")
    else:
        print(f"⚠ w_dim.txt not found, using default: {w_dim}")

    # Check if file exists
    print(f"\nChecking file: {model_path}")
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"✓ File found! Size: {file_size_mb:.2f} MB")
    else:
        print(f"✗ File NOT found at path: {model_path}")
        print(f"  Absolute path: {os.path.abspath(model_path)}")
        return None, None

    # Create the model
    print("\nCreating model architecture...")
    generator = GaussianStabilizedGeneratorForAssimilation(
        latent_dim=latent_dim,
        img_shape=img_shape,
        n_filters=n_filters,
        gaussian_reg_strength=0.01
    )

    # Build the model with dummy input
    print("Building the model with dummy input...")
    dummy_input = tf.random.normal([1, latent_dim])
    dummy_output = generator(dummy_input, training=False)
    print(f"✓ Model built! Output shape: {dummy_output.shape}")

    # Test Z -> W
    print("Testing Z -> W mapping...")
    dummy_w = generator.map_to_w(dummy_input)
    print(f"✓ W shape: {dummy_w.shape}")

    # Test W -> Image
    print("Testing generation from W...")
    dummy_output_w = generator.generate_from_w(dummy_w, training=False)
    print(f"✓ W-based output shape: {dummy_output_w.shape}")

    # Load the weights
    print(f"\nLoading weights from: {model_path}")
    try:
        generator.load_weights(model_path)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        try:
            print("\nTrying with by_name=True...")
            generator.load_weights(model_path, by_name=True)
            print("✓ Weights loaded with by_name=True!")
        except Exception as e2:
            print(f"✗ Error also with by_name=True: {e2}")
            import traceback
            traceback.print_exc()
            return None, None

    # Create the wrapper
    gan_wrapper = WSpaceGeneratorWrapper(generator, w_dim)

    # Test the wrapper with assimilation format (dims, N)
    print("\nTesting the wrapper with assimilation format...")

    # Test 1: Z format (latent_dim, N)
    test_z_ensemble = np.random.normal(0, 1, (latent_dim, 10))
    test_output_z = gan_wrapper.predict(test_z_ensemble, verbose=0)
    print(f"✓ Wrapper test Z (format: {test_z_ensemble.shape}): Output shape: {test_output_z.shape}")

    # Test 2: W format (w_dim, N)
    test_w_ensemble = np.random.normal(0, 1, (w_dim, 10))
    test_output_w = gan_wrapper.predict(test_w_ensemble, verbose=0)
    print(f"✓ Wrapper test W (format: {test_w_ensemble.shape}): Output shape: {test_output_w.shape}")

    # Test 3: Z -> W mapping
    test_w_mapped = gan_wrapper.map_to_w(test_z_ensemble)
    print(f"✓ Map Z->W test: Input {test_z_ensemble.shape} -> Output {test_w_mapped.shape}")

    # Test 4: Transposed format (N, w_dim) - as processing2.py calls
    test_transposed = np.random.normal(0, 1, (10, w_dim))
    test_output_transposed = gan_wrapper.predict(test_transposed, verbose=0)
    print(f"✓ Wrapper test transposed (format: {test_transposed.shape}): Output shape: {test_output_transposed.shape}")

    print("\n" + "=" * 60)
    print("MODEL LOADED SUCCESSFULLY!")
    print(f"Z dimension: {latent_dim}")
    print(f"W dimension: {w_dim}")
    print("=" * 60)

    return gan_wrapper, w_dim


# ============================================
# INITIAL CONFIGURATION
# ============================================

def setup_high_res_plots():
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


setup_high_res_plots()

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
ptype = 'gan_3f'

# ============================================
# CREATE ALL DIRECTORIES
# ============================================
print("\n" + "=" * 60)
print("CREATING DIRECTORIES")
print("=" * 60)

directories = [
    'outputs',
    'training_outputs',
    'training_outputs/final_models',
    'training_outputs/samples',
    'training_outputs/metrics',
    'training_outputs/checkpoints',
    'training_outputs/loss_plots',
    'training_outputs/final_outputs'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory created/verified: {directory}")

# ============================================
# LOAD THE MODEL WITH W-SPACE SUPPORT
# ============================================

print("\n" + "=" * 60)
print("STARTING LOADING PROCESS (W-SPACE)")
print("=" * 60)

model_path = 'training_outputs/final_models/generator.weights_w.h5'
print(f"\nCurrent directory: {os.getcwd()}")

GAN, W_DIM = load_stylegan_generator_with_w(
    model_path=model_path,
    latent_dim=512,
    img_shape=(48, 48, 1),
    n_filters=64
)

if GAN is None:
    print("\n⚠ WARNING: Could not load the model.")
    print("Please check if:")
    print("1. The file actually exists at the specified path")
    print("2. The file is a valid weights file (.weights_w.h5)")
    print("3. The first code was executed with the W-space version")
    print("4. The path is correct relative to the execution directory")

    # Try alternative paths
    alternative_paths = [
        'generator.weights_w.h5',
        'final_models/generator.weights_w.h5',
        '../training_outputs/final_models/generator.weights_w.h5',
        os.path.expanduser('~/training_outputs/final_models/generator.weights_w.h5')
    ]

    print("\nTrying alternative paths...")
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            print(f"  ✓ Found at: {alt_path}")
            GAN, W_DIM = load_stylegan_generator_with_w(
                model_path=alt_path,
                latent_dim=512,
                img_shape=(48, 48, 1),
                n_filters=64
            )
            if GAN is not None:
                break

    if GAN is None:
        exit(1)

# Create wrapper compatible with processing2.py
GAN_COMPATIVEL = create_gan_wrapper_compativel(GAN)
print("✓ Wrapper compatible with processing2.py created")

# ============================================
# PREPARE DATA - NOW IN W SPACE
# ============================================

print("\n" + "=" * 60)
print("PREPARING DATA IN W SPACE")
print("=" * 60)

# Sample Z from Gaussian distribution in format (latent_dim, N)
print(f"Creating prior ensemble...")
print(f"  N (number of models): {N}")
print(f"  Z dimension: {GAN.latent_dim}")

z_prior = np.random.normal(0, 1, (GAN.latent_dim, N))  # Shape: (512, 500)
print(f"✓ Z prior shape: {z_prior.shape}")

# Check Z statistics
print(f"  Z statistics:")
print(f"    Mean: {z_prior.mean():.4f} (target: 0.0)")
print(f"    Std: {z_prior.std():.4f} (target: 1.0)")

# Convert to W space
print(f"\nConverting prior ensemble from Z to W...")
w_prior = GAN.map_to_w(z_prior)  # Shape: (W_DIM, N) = (1024, 500)
print(f"✓ W prior shape: {w_prior.shape}")
print(f"  Expected: ({W_DIM}, {N})")

# Check format
assert w_prior.shape[0] == W_DIM, f"ERROR: W prior should have {W_DIM} rows, but has {w_prior.shape[0]}"
assert w_prior.shape[1] == N, f"ERROR: W prior should have {N} columns, but has {w_prior.shape[1]}"

# Check W statistics
print(f"\nW space statistics:")
print(f"  Mean: {w_prior.mean():.4f}")
print(f"  Std: {w_prior.std():.4f}")
print(f"  Min: {w_prior.min():.4f}")
print(f"  Max: {w_prior.max():.4f}")
print(f"  Shape for assimilation: {w_prior.shape[0]} parameters x {w_prior.shape[1]} models")

# Test image generation from prior
print(f"\nTesting image generation from prior...")
test_prior_images = GAN.predict(w_prior[:, :5], verbose=0)  # first 5 models
print(f"✓ Test images generated: {test_prior_images.shape}")

# Load true model
print(f"\nLoading true model...")
true_model = np.load('datasets/Three_facies_2d.npy')[0:1][..., 0]
true_model = vectorization(true_model, 'vec', (48, 48))
true_model[true_model == -1] = 100  # purple -> 100
true_model[true_model == 0] = 1000  # green -> 1000
true_model[true_model == 1] = 9000  # yellow -> 9000
print(f"✓ True model shape: {true_model.shape}")

name_case = 'disc_32_nonloc_wspace'  # Name with wspace suffix
n_iter = 32
tapering = None  # None, 'adaptive_furrer' or 'scaling'

print(f"\nAssimilation configuration:")
print(f"  Case: {name_case}")
print(f"  Iterations: {n_iter}")
print(f"  Tapering: {tapering}")
print(f"  Space: W (dimension {W_DIM})")

# ============================================
# RUN SIMULATION - USING W SPACE
# ============================================

print("\n" + "=" * 60)
print("RUNNING SIMULATION IN W SPACE")
print("=" * 60)

if not os.path.exists('outputs'):
    os.makedirs('outputs', exist_ok=True)
    print("✓ Directory 'outputs' created")

try:
    D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=None, additional=None))
    print("✓ run_true_case executed successfully")
    print(f"  D_measurements type: {type(D_measurements)}")
except Exception as e:
    print(f"✗ Error in run_true_case: {e}")
    import traceback

    traceback.print_exc()
    raise

# IMPORTANT: Pass the ensemble in W space and the compatible wrapper
print(f"\nCalling run_assimilation with:")
print(f"  Ensemble shape: {w_prior.shape}")
print(f"  N parameters: {w_prior.shape[0]}")
print(f"  N models: {w_prior.shape[1]}")
print(f"  Additional type: {type(GAN_COMPATIVEL).__name__}")

try:
    Result = run_assimilation(w_prior, true_model, D_measurements, n_iter, 12,
                              tapering=tapering, ptype=ptype,
                              additional=GAN_COMPATIVEL,  # Compatible wrapper
                              metric_space='latent')
    print("✓ run_assimilation executed successfully in W space")
except Exception as e:
    print(f"✗ Error in run_assimilation: {e}")
    import traceback

    traceback.print_exc()

    # Try with the original wrapper as fallback
    print("\nTrying with original wrapper as fallback...")
    try:
        Result = run_assimilation(w_prior, true_model, D_measurements, n_iter, 12,
                                  tapering=tapering, ptype=ptype,
                                  additional=GAN,  # Original wrapper
                                  metric_space='latent')
        print("✓ run_assimilation executed with original wrapper")
    except Exception as e2:
        print(f"✗ Error also with original wrapper: {e2}")
        raise

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SAVING RESULTS (W-SPACE)")
print("=" * 60)

if not os.path.exists('outputs'):
    os.makedirs('outputs', exist_ok=True)
    print("✓ Directory 'outputs' created")

# Save W-space information
with open(f'./outputs/w_space_info_{name_case}.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("W-SPACE DATA ASSIMILATION INFORMATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Data assimilation performed in W-space\n")
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
    f.write(f"  Shape: {w_prior.shape}\n\n")
    f.write("Result structure:\n")
    f.write(f"  Result[0]: Data Mismatch\n")
    f.write(f"  Result[1]: RMSE\n")
    f.write(f"  Result[2]: Spread\n")
    f.write(f"  Result[3]: Latent vectors (W-space)\n")
    f.write(f"  Result[4]: Generated images\n")
    f.write(f"  Result[5]: Forward model results\n")
    f.write(f"  Result[6]: (varies)\n")
    f.write(f"  Result[7]: Balanced Accuracy\n")
print(f"✓ w_space_info_{name_case}.txt saved")

# Save main results
with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)
print(f"✓ Results saved to ./outputs/Results_{name_case}.pickle")

# Reload to verify
with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)
print("✓ Results loaded and verified")

# Save metrics to text files
try:
    np.savetxt(f'./outputs/dm_{name_case}.txt', np.array(Result[0]).mean(-1))
    print(f"✓ dm_{name_case}.txt saved")
except Exception as e:
    print(f"⚠ Error saving dm: {e}")

try:
    with open(f'./outputs/DM_{name_case}.txt', 'w', encoding='utf-8') as f:
        f.write("# Data Mismatch per iteration\n")
        for i, dm_values in enumerate(Result[0]):
            if isinstance(dm_values, (list, np.ndarray)):
                f.write(f"Iteration {i}: " + " ".join(map(str, dm_values)) + "\n")
            else:
                f.write(f"Iteration {i}: {dm_values}\n")
    print(f"✓ DM_{name_case}.txt saved")
except Exception as e:
    print(f"⚠ Error saving DM: {e}")

try:
    with open(f'./outputs/RMSE_{name_case}.txt', 'w', encoding='utf-8') as f:
        f.write("# RMSE per iteration\n")
        for i, rsme_values in enumerate(Result[1]):
            if isinstance(rsme_values, (list, np.ndarray)):
                f.write(f"Iteration {i}: " + " ".join(map(str, rsme_values)) + "\n")
            else:
                f.write(f"Iteration {i}: {rsme_values}\n")
    print(f"✓ RMSE_{name_case}.txt saved")
except Exception as e:
    print(f"⚠ Error saving RMSE: {e}")

try:
    with open(f'./outputs/Spread_{name_case}.txt', 'w', encoding='utf-8') as f:
        f.write("# Spread per iteration\n")
        for i, spread_values in enumerate(Result[2]):
            if isinstance(spread_values, (list, np.ndarray)):
                f.write(f"Iteration {i}: " + " ".join(map(str, spread_values)) + "\n")
            else:
                f.write(f"Iteration {i}: {spread_values}\n")
    print(f"✓ Spread_{name_case}.txt saved")
except Exception as e:
    print(f"⚠ Error saving Spread: {e}")

try:
    with open(f'./outputs/BA_{name_case}.txt', 'w', encoding='utf-8') as f:
        f.write("# Balanced Accuracy per iteration\n")
        for i, ba_values in enumerate(Result[7]):
            if isinstance(ba_values, (list, np.ndarray)):
                f.write(f"Iteration {i}: " + " ".join(map(str, ba_values)) + "\n")
            else:
                f.write(f"Iteration {i}: {ba_values}\n")
    print(f"✓ BA_{name_case}.txt saved")
except Exception as e:
    print(f"⚠ Error saving BA: {e}")

# ============================================
# PLOTS (WITH W-SPACE)
# ============================================

print("\n" + "=" * 60)
print("GENERATING PLOTS (W-SPACE ASSIMILATION)")
print("=" * 60)

# Plot 1: Measurements comparison
print("Generating Plot 1: Measurements comparison...")
try:
    fig, axs = plt.subplots(3, 9, figsize=(18, 5))

    for _1 in range(3):
        for _2 in range(9):
            # Plot initial ensemble (gray)
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1][:, _2, :],
                             'xkcd:gray', alpha=0.35, linewidth=0.5)
            # Plot final ensemble (blue)
            axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1][:, _2, :],
                             'b', alpha=0.35, linewidth=0.5)
            # Plot observed data (red)
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

    plt.savefig(f'./outputs/measurements_comparison_{name_case}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./outputs/measurements_comparison_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'./outputs/measurements_comparison_{name_case}.pdf', format='pdf', bbox_inches='tight')
    print(f"✓ measurements_comparison_{name_case} saved")
    plt.show()
except Exception as e:
    print(f"⚠ Error in Plot 1: {e}")
    import traceback

    traceback.print_exc()

# Plot 2: Metrics boxplots
print("Generating Plot 2: Metrics boxplots...")
try:
    fig, axs = plt.subplots(1, 4, figsize=(8, 3))

    metrics_data = [Result[0], Result[1], Result[2], Result[7]]
    metrics_names = ['DM', 'RMSE', 'Spread', 'Balanced Accuracy']

    for i, (data, name) in enumerate(zip(metrics_data, metrics_names)):
        axs[i].boxplot(data, showfliers=False)
        axs[i].set_xlabel('Iteration', fontsize=10)
        axs[i].set_ylabel(name, fontsize=10)
        axs[i].set_title(name, fontsize=10, fontweight='bold')
        axs[i].set_xticks(np.linspace(1, n_iter + 1, 5))
        axs[i].set_xticklabels(np.linspace(0, n_iter, 5, dtype='int'))

    #plt.suptitle(f'W-Space Assimilation Metrics - {name_case}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f'./outputs/metrics_boxplots_{name_case}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./outputs/metrics_boxplots_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'./outputs/metrics_boxplots_{name_case}.pdf', format='pdf', bbox_inches='tight')
    print(f"✓ metrics_boxplots_{name_case} saved")
    plt.show()
except Exception as e:
    print(f"⚠ Error in Plot 2: {e}")


# Plot 3: Facies maps (generating from W space)
print("Generating Plot 3: Facies maps...")
try:
    fig, axs = plt.subplots(1, 7, figsize=(16, 3))

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

    plt.savefig(f'./outputs/facies_maps_{name_case}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./outputs/facies_maps_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'./outputs/facies_maps_{name_case}.pdf', format='pdf', bbox_inches='tight')
    print(f"✓ facies_maps_{name_case} saved")
    plt.show()
except Exception as e:
    print(f"⚠ Error in Plot 3: {e}")
    import traceback

    traceback.print_exc()

# Plot 4: Additional realizations
print("Generating Plot 4: Additional realizations...")
try:
    realizations = GAN.predict(Result[3][-1], verbose=0)

    fig2, axs2 = plt.subplots(1, 7, figsize=(16, 3))
    for j in range(7):
        if j < len(realizations):
            axs2[j].imshow(realizations[j].squeeze(), vmin=-1, vmax=1, cmap='viridis')
        axs2[j].axis('off')
        axs2[j].set_title(f'Member {j + 1} (W)', fontsize=8, pad=5)

    plt.suptitle(f'W-Space Posterior Realizations - {name_case}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f'./outputs/additional_realizations_{name_case}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./outputs/additional_realizations_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.savefig(f'./outputs/additional_realizations_{name_case}.pdf', format='pdf', bbox_inches='tight')
    print(f"✓ additional_realizations_{name_case} saved")
    plt.show()
except Exception as e:
    print(f"⚠ Error in Plot 4: {e}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("W-SPACE PROCESSING COMPLETED!")
print("=" * 60)
print(f"Case: {name_case}")
print(f"Assimilation space: W (dimension {W_DIM})")
print(f"Number of models: {N}")
print(f"Number of iterations: {n_iter}")
print(f"All files have been saved to: {os.path.abspath('outputs')}")
print(f"\nGenerated files:")
print(f"  - Results_{name_case}.pickle")
print(f"  - w_space_info_{name_case}.txt")
print(f"  - DM_{name_case}.txt")
print(f"  - RMSE_{name_case}.txt")
print(f"  - Spread_{name_case}.txt")
print(f"  - BA_{name_case}.txt")
print(f"  - dm_{name_case}.txt")
print(f"  - measurements_comparison_{name_case}.png/svg/pdf")
print(f"  - metrics_boxplots_{name_case}.png/svg/pdf")
print(f"  - facies_maps_{name_case}.png/svg/pdf")
print(f"  - additional_realizations_{name_case}.png/svg/pdf")

# Display final metrics
print(f"\nFinal metrics (last iteration):")
try:
    final_dm = np.array(Result[0][-1])
    print(f"  Average DM: {final_dm.mean():.4f} (min: {final_dm.min():.4f}, max: {final_dm.max():.4f})")
except Exception as e:
    print(f"  DM: error calculating - {e}")

try:
    final_rmse = np.array(Result[1][-1])
    print(f"  Average RMSE: {final_rmse.mean():.4f} (min: {final_rmse.min():.4f}, max: {final_rmse.max():.4f})")
except Exception as e:
    print(f"  RMSE: error calculating - {e}")

try:
    final_spread = np.array(Result[2][-1])
    print(f"  Average Spread: {final_spread.mean():.4f} (min: {final_spread.min():.4f}, max: {final_spread.max():.4f})")
except Exception as e:
    print(f"  Spread: error calculating - {e}")

try:
    final_ba = np.array(Result[7][-1])
    print(f"  Average Balanced Accuracy: {final_ba.mean():.4f} (min: {final_ba.min():.4f}, max: {final_ba.max():.4f})")
except Exception as e:
    print(f"  BA: error calculating - {e}")

# Comparison with Z space (if available)
z_results_file = './outputs/Results_disc_32_nonloc.pickle'
if os.path.exists(z_results_file):
    try:
        with open(z_results_file, 'rb') as fp:
            Result_Z = pickle.load(fp)

        final_dm_z = np.array(Result_Z[0][-1]).mean()
        final_rmse_z = np.array(Result_Z[1][-1]).mean()
        final_ba_z = np.array(Result_Z[7][-1]).mean()

        print(f"\nComparison with Z space:")
        print(f"  DM (Z): {final_dm_z:.4f} | DM (W): {final_dm.mean():.4f}")
        print(f"  RMSE (Z): {final_rmse_z:.4f} | RMSE (W): {final_rmse.mean():.4f}")
        print(f"  BA (Z): {final_ba_z:.4f} | BA (W): {final_ba.mean():.4f}")

        improvement_dm = (final_dm_z - final_dm.mean()) / final_dm_z * 100
        improvement_rmse = (final_rmse_z - final_rmse.mean()) / final_rmse_z * 100
        improvement_ba = (final_ba.mean() - final_ba_z) / final_ba_z * 100

        print(f"\nRelative improvement:")
        print(f"  DM: {improvement_dm:+.1f}%")
        print(f"  RMSE: {improvement_rmse:+.1f}%")
        print(f"  BA: {improvement_ba:+.1f}%")
    except Exception as e:
        print(f"  Could not compare with Z space: {e}")

print("\n" + "=" * 60)
print("END OF PROCESSING")
print("=" * 60)