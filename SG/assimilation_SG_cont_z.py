import matplotlib.pyplot as plt
import numpy as np
import pickle
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
    """Unified StyleGAN2 - Compatible with save/load"""
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
# INITIAL CONFIGURATION
# ============================================

N = 500
latent_dim = 512
ptype = 'gan_uii'

# Custom objects for loading the model
custom_objects = {
    'StableAdaIN': StableAdaIN,
    'StabilizedStyleGAN2Generator': StabilizedStyleGAN2Generator,
    'FeatureExtractingDiscriminator': FeatureExtractingDiscriminator,
    'StyleGAN2': StyleGAN2
}

# Load the StyleGAN2 model
print("Loading StyleGAN2 model...")
try:
    # Try to load the complete model first
    stylegan = tf.keras.models.load_model(
        'training_outputs/final_models/stylegan2_complete.weights_z.h5',
        custom_objects=custom_objects
    )
    generator = stylegan.generator
    print("✓ StyleGAN2 model loaded successfully (complete format)")
except:
    print("Complete file not found. Trying to load generator separately...")
    # Recreate generator
    generator = StabilizedStyleGAN2Generator(
        latent_dim=latent_dim,
        img_shape=(48, 48, 1),
        n_filters=64
    )
    # Build with dummy input
    _ = generator(tf.random.normal([1, latent_dim]))
    # Load weights
    generator.load_weights('training_outputs/final_models/generator.weights_z.h5')
    print("✓ Generator weights loaded successfully")

# Load data
X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')
X_train[X_train < 1] = 1
X_train = np.log(X_train)
X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
X_train = X_train[..., np.newaxis]

# Generate ensemble in latent space using StyleGAN mapping
print("Generating ensemble in latent space...")
latent_samples = np.random.normal(0, 1, (N, latent_dim))
prior_ensemble = latent_samples.T  # Transpose to have dimensions [latent_dim, N]

# True model
true_model = np.load('simfiles/Perm_ensemble_cropped.npy')[0:1]
true_model[true_model < 1] = 1
true_model = vectorization(np.log(true_model), 'vec', (48, 48))

name_case = 'cont_32_nonloc_stylegan'
n_iter = 32
tapering = None

# Run true case
D_true, D_measurements = run_true_case(parameter_mapping(true_model, type='log', additional=None))

# Run assimilation using StyleGAN as decoder
Result = run_assimilation(
    prior_ensemble,
    true_model,
    D_measurements,
    n_iter,
    12,
    tapering=tapering,
    ptype=ptype,
    additional=generator,  # Use StyleGAN generator as decoder
    metric_space='latent'
)

# Save results
with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)

with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)

# Save DM
np.savetxt(f'dm_{name_case}.txt', np.array(Result[0]).mean(-1))

with open(f'./outputs/DM_{name_case}.txt', 'w') as f:
    for i, dm_values in enumerate(Result[0]):
        if isinstance(dm_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, dm_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {dm_values}\n")

with open(f'./outputs/RMSE_{name_case}.txt', 'w') as f:
    for i, rsme_values in enumerate(Result[1]):
        if isinstance(rsme_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, rsme_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {rsme_values}\n")

with open(f'./outputs/Spread_{name_case}.txt', 'w') as f:
    for i, spread_values in enumerate(Result[2]):
        if isinstance(spread_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, spread_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {spread_values}\n")

# ============================================
# VISUALIZATIONS
# ============================================

# Plot 1: Properties (OPR, WPR, BHP)
fig1, axs = plt.subplots(3, 9, figsize=(18, 5))
prop_names = ['OPR (m³/day)', 'WPR (m³/day)', 'BHP (kgf/cm²)']
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

fig1.savefig(f'./outputs/properties_plot_{name_case}.png', dpi=300, bbox_inches='tight')
fig1.savefig(f'./outputs/properties_plot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig1)

# Plot 2: Metrics (DM, RMSE, Spread) - WITH AUTOMATIC X-AXIS
fig2, axs = plt.subplots(1, 3, figsize=(8, 3))
axs[0].boxplot(Result[0], showfliers=False)
axs[1].boxplot(Result[1], showfliers=False)
axs[2].boxplot(Result[2], showfliers=False)

[axs[_].set_xlabel('Iterations') for _ in range(3)]
ynames = ['DM', 'RMSE', 'Spread']
[axs[_].set_ylabel(ynames[_]) for _ in range(3)]
[axs[_].set_xticks(np.linspace(0 + 1, n_iter + 1, 5), np.linspace(0, n_iter, 5, dtype='int')) for _ in range(3)]
plt.tight_layout()



fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.png', dpi=300, bbox_inches='tight')
fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig2)

# Plot 3: Additional well data
fig3, axs = plt.subplots(2, 4, figsize=(9, 5))
for _1 in range(2):
    for _2 in range(4):
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1+3][:, _2, :], color=[0.2, 0.2, 0.2], alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1+3][:, _2, :], color='blue', alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1+3][:, _2], 'or', markersize=2)

for _2 in range(4):
    axs[0, _2].set_ylabel('WIR (m³/day)')
    axs[1, _2].set_ylabel('BHP (kgf/cm²)')
    [[axs[_1, _2].set_xlabel('Time (day)') for _1 in range(2)] for _2 in range(4)]
plt.tight_layout()
plt.show()

# Plot 4: Model visualization
fig4, axs = plt.subplots(1, 7, figsize=(14, 2))

# Generate samples from latent space for visualization
latent_prior = Result[3][0].T  # [latent_dim, N]
latent_posterior = Result[3][-1].T  # [latent_dim, N]

# Decode using StyleGAN
prior_samples = generator.predict(latent_prior, verbose=0)
posterior_samples = generator.predict(latent_posterior, verbose=0)

axs[0].imshow(true_model.reshape(48, 48), vmin=0, vmax=9)
axs[1].imshow(prior_samples.mean(0), vmin=-1, vmax=1)
axs[2].imshow(posterior_samples.mean(0), vmin=-1, vmax=1)
axs[3].imshow(prior_samples.std(0), vmin=0, vmax=0.7)
axs[4].imshow(posterior_samples.std(0), vmin=0, vmax=0.7)
axs[5].imshow(prior_samples[0], vmin=-1, vmax=1)
axs[6].imshow(posterior_samples[0], vmin=-1, vmax=1)

[axs[_].axis('off') for _ in range(7)]

cols = ['True', 'Prior\n mean', 'Posterior\n mean', 'Prior\n std', 'Posterior\n std', 'Prior\n Member 1',
        'Posterior\n Member 1']
for i in range(len(cols)):
    axs[i].set_title(cols[i], fontsize=8, pad=5)

fig4.savefig(f'./outputs/model_visualization_{name_case}.png', dpi=300, bbox_inches='tight')
fig4.savefig(f'./outputs/model_visualization_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig4)

print(f"- properties_plot_{name_case}.png/.svg")
print(f"- metrics_boxplot_{name_case}.png/.svg")
print(f"- model_visualization_{name_case}.png/.svg")
print(f"\nStyleGAN2 assimilation completed successfully!")