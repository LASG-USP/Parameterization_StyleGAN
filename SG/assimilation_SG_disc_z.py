import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pickle
import tensorflow as tf
import os
from main import run_assimilation, run_true_case
from processing import parameter_mapping, vectorization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU

# ============================================
# CUSTOM OBJECTS FOR STYLEGAN2
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


class StabilizedStyleGAN2Generator(tf.keras.Model):
    """StyleGAN2 Generator - Stabilized Version."""
    def __init__(self, latent_dim=512, img_shape=(48, 48, 1), n_filters=64, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_filters = n_filters

        self.mapping = self._build_mapping_network()

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
            StableAdaIN(self.n_filters * 8),
            layers.LeakyReLU(0.2),
        ])
        # Block 2
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            StableAdaIN(self.n_filters * 4),
            layers.LeakyReLU(0.2),
        ])
        # Block 3
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            StableAdaIN(self.n_filters * 2),
            layers.LeakyReLU(0.2),
        ])
        # Block 4
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            StableAdaIN(self.n_filters),
            layers.LeakyReLU(0.2),
        ])
        # Block 5
        self.blocks.append([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
            layers.Conv2D(self.n_filters // 2, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            StableAdaIN(self.n_filters // 2),
            layers.LeakyReLU(0.2),
            layers.Cropping2D(cropping=((8, 8), (8, 8))),
        ])

    def call(self, inputs, training=False):
        w = self.mapping(inputs)
        batch_size = tf.shape(inputs)[0]
        x = tf.tile(self.constant, [batch_size, 1, 1, 1])

        for block in self.blocks:
            for layer in block:
                if isinstance(layer, StableAdaIN):
                    x = layer([x, w])
                else:
                    x = layer(x)

        x = self.to_rgb(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'img_shape': self.img_shape,
            'n_filters': self.n_filters
        })
        return config


class FeatureExtractingDiscriminator(tf.keras.Model):
    """StyleGAN2 Discriminator with feature extraction."""
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
        # Block 1
        self.blocks.append([
            layers.Conv2D(self.n_filters, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])
        # Block 2
        self.blocks.append([
            layers.Conv2D(self.n_filters * 2, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])
        # Block 3
        self.blocks.append([
            layers.Conv2D(self.n_filters * 4, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])
        # Block 4
        self.blocks.append([
            layers.Conv2D(self.n_filters * 8, 3, padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.1),
            layers.AveragePooling2D(pool_size=(2, 2)),
        ])
        # Block 5
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


class StyleGAN2(keras.Model):
    def __init__(self,
                 generator,
                 discriminator,
                 latent_dim=512,
                 d_reg=10.0,
                 label_smoothing=0.1,
                 gradient_clip=0.5,
                 noise_std=0.05,
                 **kwargs):
        super().__init__(**kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.d_reg = d_reg
        self.label_smoothing = label_smoothing
        self.gradient_clip = gradient_clip
        self.noise_std = noise_std

    def call(self, inputs, training=False):
        if isinstance(inputs, tf.Tensor) and inputs.shape[-1] == self.latent_dim:
            return self.generator(inputs, training=training)
        else:
            return self.discriminator(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'd_reg': self.d_reg,
            'label_smoothing': self.label_smoothing,
            'gradient_clip': self.gradient_clip,
            'noise_std': self.noise_std,
        })
        return config


# ============================================
# FUNCTION TO LOAD THE MODEL
# ============================================

def load_stylegan_generator(model_path='training_outputs/final_models/generator.weights_z.h5',
                            latent_dim=512, img_shape=(48, 48, 1), n_filters=64):



    print(f"\nChecking file: {model_path}")
    if os.path.exists(model_path):
        print(f"✓ File found! Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ File NOT found at path: {model_path}")
        return None

    # Create the model with the same architecture
    print("\nCreating model architecture...")
    generator = StabilizedStyleGAN2Generator(
        latent_dim=latent_dim,
        img_shape=img_shape,
        n_filters=n_filters
    )

    # IMPORTANT: Build the model with a dummy input and store the output
    print("Building the model with dummy input...")
    dummy_input = tf.random.normal([1, latent_dim])
    dummy_output = generator(dummy_input, training=False)
    print(f"✓ Model built! Output shape: {dummy_output.shape}")

    # Show model summary
    print("\nModel summary:")
    print(f"  Latent dim: {generator.latent_dim}")
    print(f"  Output shape: {generator.img_shape}")
    print(f"  N filters: {generator.n_filters}")

    # Load the weights
    print(f"\nLoading weights from: {model_path}")
    try:
        # Try to load the weights
        generator.load_weights(model_path)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")

        # Try with by_name=True as fallback
        try:
            print("\nTrying with by_name=True...")
            generator.load_weights(model_path, by_name=True)
            print("✓ Weights loaded with by_name=True!")
        except Exception as e2:
            print(f"✗ Error also with by_name=True: {e2}")
            return None

    # Test the model after loading weights
    print("\nTesting the model after loading weights...")
    test_noise = tf.random.normal([1, latent_dim])
    test_output = generator(test_noise, training=False)
    print(f"✓ Test OK! Output shape: {test_output.shape}")
    print(f"  Output range: [{tf.reduce_min(test_output):.3f}, {tf.reduce_max(test_output):.3f}]")

    print("\n" + "=" * 60)
    print("MODEL LOADED SUCCESSFULLY!")
    print("=" * 60)

    return generator


# ============================================
# INITIAL CONFIGURATION
# ============================================

# High resolution settings for matplotlib
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


# Call configuration before creating plots
setup_high_res_plots()

# ============================================
# PARAMETERS
# ============================================

N = 500
ptype = 'gan_3f'

# ============================================
# CREATE ALL NECESSARY DIRECTORIES (NEW!)
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
# LOAD THE MODEL
# ============================================

print("\n" + "=" * 60)
print("STARTING LOADING PROCESS")
print("=" * 60)

# Specific path for the file
model_path = 'training_outputs/final_models/generator.weights_z.h5'

# Check current directory
print(f"\nCurrent directory: {os.getcwd()}")

# Load model
GAN = load_stylegan_generator(
    model_path=model_path,
    latent_dim=512,
    img_shape=(48, 48, 1),
    n_filters=64
)

if GAN is None:
    print("\n⚠ WARNING: Could not load the model.")
    print("Please check if:")
    print("1. The file actually exists at the specified path")
    print("2. The file is a valid weights file (.weights.h5)")
    print("3. You have read permission for the file")
    exit(1)

# ============================================
# PREPARE DATA
# ============================================

print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

prior_ensemble = np.random.normal(0, 1, (N, GAN.latent_dim)).T

true_model = np.load('datasets/Three_facies_2d.npy')[0:1][..., 0]
true_model = vectorization(true_model, 'vec', (48, 48))
true_model[true_model == -1] = 100  # purple
true_model[true_model == 0] = 1000  # green
true_model[true_model == 1] = 9000  # yellow

name_case = 'disc_32_nonloc'  # case study name
n_iter = 32  # number of iterations
tapering = None  # None, 'adaptive_furrer' or 'scaling' for localization methods

# ============================================
# RUN SIMULATION
# ============================================

print("\n" + "=" * 60)
print("RUNNING SIMULATION")
print("=" * 60)

# Check if outputs directory exists before proceeding
if not os.path.exists('outputs'):
    os.makedirs('outputs', exist_ok=True)
    print("✓ Directory 'outputs' created")

try:
    D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=None, additional=None))
    print("✓ run_true_case executed successfully")
except Exception as e:
    print(f"✗ Error in run_true_case: {e}")
    raise

# D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=ptype, additional=GAN))

try:
    Result = run_assimilation(prior_ensemble, true_model, D_measurements, n_iter, 12,
                              tapering=tapering, ptype=ptype, additional=GAN, metric_space='latent')
    print("✓ run_assimilation executed successfully")
except Exception as e:
    print(f"✗ Error in run_assimilation: {e}")
    raise

# ============================================
# SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Check if outputs directory exists
if not os.path.exists('outputs'):
    os.makedirs('outputs', exist_ok=True)
    print("✓ Directory 'outputs' recreated")

with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)
print(f"✓ Results saved to ./outputs/Results_{name_case}.pickle")

with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)
print("✓ Results loaded from file")

# Save metrics to .txt files
np.savetxt(f'./outputs/dm_{name_case}.txt', np.array(Result[0]).mean(-1))
print(f"✓ dm_{name_case}.txt saved")

# Check and adjust each metric
with open(f'./outputs/DM_{name_case}.txt', 'w') as f:
    for i, dm_values in enumerate(Result[0]):
        if isinstance(dm_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, dm_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {dm_values}\n")
print(f"✓ DM_{name_case}.txt saved")

with open(f'./outputs/RMSE_{name_case}.txt', 'w') as f:
    for i, rsme_values in enumerate(Result[1]):
        if isinstance(rsme_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, rsme_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {rsme_values}\n")
print(f"✓ RMSE_{name_case}.txt saved")

with open(f'./outputs/Spread_{name_case}.txt', 'w') as f:
    for i, spread_values in enumerate(Result[2]):
        if isinstance(spread_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, spread_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {spread_values}\n")
print(f"✓ Spread_{name_case}.txt saved")

with open(f'./outputs/BA_{name_case}.txt', 'w') as f:
    for i, ba_values in enumerate(Result[7]):
        if isinstance(ba_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, ba_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {ba_values}\n")
print(f"✓ BA_{name_case}.txt saved")

# ============================================
# PLOTS
# ============================================

print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

# Plot 1: Measurements comparison
fig, axs = plt.subplots(3, 9, figsize=(18, 5))
prop_names = ['OPR', 'WPR', 'BHP']
for _1 in range(3):
    for _2 in range(9):
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1][:, _2, :], 'xkcd:gray', alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1][:, _2, :], 'b', alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1][:, _2], 'or', markersize=2)
for _2 in range(9):
    axs[0, _2].set_ylabel('OPR (m³/day)')
    axs[1, _2].set_ylabel('WPR (m³/day)')
    axs[2, _2].set_ylabel('BHP (kgf/cm²)')
    [[axs[_1, _2].set_xlabel('Time (day)') for _1 in range(3)] for _2 in range(9)]
plt.tight_layout()

# Save plot 1 in multiple formats
plt.savefig(f'./outputs/measurements_comparison_{name_case}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./outputs/measurements_comparison_{name_case}.svg', format='svg', bbox_inches='tight')
plt.savefig(f'./outputs/measurements_comparison_{name_case}.pdf', format='pdf', bbox_inches='tight')
print(f"✓ measurements_comparison_{name_case} saved")
plt.show()

# Plot 2: Metrics boxplots
fig, axs = plt.subplots(1, 4, figsize=(8, 3))
axs[0].boxplot(Result[0], showfliers=False)
axs[1].boxplot(Result[1], showfliers=False)
axs[2].boxplot(Result[2], showfliers=False)
axs[3].boxplot(Result[7], showfliers=False)
[axs[_].set_xlabel('iteration') for _ in range(4)]
ynames = ['DM', 'RMSE', 'Spread', 'Balanced Accuracy']
[axs[_].set_ylabel(ynames[_]) for _ in range(4)]
[axs[_].set_xticks(np.linspace(0 + 1, n_iter + 1, 5), np.linspace(0, n_iter, 5, dtype='int')) for _ in range(4)]
plt.tight_layout()

# Save plot 2 in multiple formats
plt.savefig(f'./outputs/metrics_boxplots_{name_case}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./outputs/metrics_boxplots_{name_case}.svg', format='svg', bbox_inches='tight')
plt.savefig(f'./outputs/metrics_boxplots_{name_case}.pdf', format='pdf', bbox_inches='tight')
print(f"✓ metrics_boxplots_{name_case} saved")

# Plot 3: Facies maps
fig, axs = plt.subplots(1, 7, figsize=(14, 2))

# Revert facies values for plotting
true_model_plot = true_model.copy()
true_model_plot[true_model_plot == 100] = -1  # purple
true_model_plot[true_model_plot == 1000] = 0  # green
true_model_plot[true_model_plot == 9000] = 1  # yellow

# Plot images
axs[0].imshow(true_model_plot.reshape(48, 48), vmin=-1, vmax=1)

# Generate images with GAN
print("Generating images with GAN...")
prior_mean = GAN.predict(Result[3][0].T, verbose=0).mean(0)
posterior_mean = GAN.predict(Result[3][-1].T, verbose=0).mean(0)
prior_std = GAN.predict(Result[3][0].T, verbose=0).std(0)
posterior_std = GAN.predict(Result[3][-1].T, verbose=0).std(0)
prior_member1 = GAN.predict(Result[3][0].T, verbose=0)[0]
posterior_member1 = GAN.predict(Result[3][-1].T, verbose=0)[0]

axs[1].imshow(prior_mean, vmin=-1, vmax=1)
axs[2].imshow(posterior_mean, vmin=-1, vmax=1)
axs[3].imshow(prior_std, vmin=0, vmax=0.7)
axs[4].imshow(posterior_std, vmin=0, vmax=0.7)
axs[5].imshow(prior_member1, vmin=-1, vmax=1)
axs[6].imshow(posterior_member1, vmin=-1, vmax=1)

# Turn off axes
[axs[_].axis('off') for _ in range(7)]

# Set titles
cols = ['True', 'Prior\n mean', 'Posterior\n mean', 'Prior\n std', 'Posterior\n std',
        'Prior\n Member 1', 'Posterior\n Member 1']
for i in range(len(cols)):
    axs[i].set_title(cols[i], fontsize=8, pad=5)

plt.tight_layout()

# Save plot 3 in multiple formats
plt.savefig(f'./outputs/facies_maps_{name_case}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./outputs/facies_maps_{name_case}.svg', format='svg', bbox_inches='tight')
plt.savefig(f'./outputs/facies_maps_{name_case}.pdf', format='pdf', bbox_inches='tight')
print(f"✓ facies_maps_{name_case} saved")
plt.show()

# Plot 4: Additional realizations
realizations = GAN.predict(Result[3][-1].T, verbose=0)
fig2, axs2 = plt.subplots(1, 7, figsize=(14, 2))
for j in range(7):
    axs2[j].imshow(realizations[j], vmin=-1, vmax=1)
    axs2[j].axis('off')
    axs2[j].set_title(f'Member {j + 1}', fontsize=8, pad=5)

plt.tight_layout()

# Save plot 4 in multiple formats
plt.savefig(f'./outputs/additional_realizations_{name_case}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'./outputs/additional_realizations_{name_case}.svg', format='svg', bbox_inches='tight')
plt.savefig(f'./outputs/additional_realizations_{name_case}.pdf', format='pdf', bbox_inches='tight')
print(f"✓ additional_realizations_{name_case} saved")
plt.show()

print("\n" + "=" * 60)
print("PROCESSING COMPLETED!")
print(f"All files have been saved to: {os.path.abspath('outputs')}")
print("=" * 60)