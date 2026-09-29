import matplotlib.pyplot as plt
import numpy as np
import pickle
from main import run_assimilation, run_true_case
from processing import parameter_mapping, vectorization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU

N = 500
latent_dim = 512
num_classes = 3
ptype = 'vae_3f'


# Define sampling function for VAE (needed for loading the model)
def sampling(args):
    """Sampling function for VAE latent space."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_shape, latent_dim):
    """Build encoder for the Latent Diffusion Model."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    latent_output = layers.Dense(latent_dim, name="latent_output")(x)

    return keras.Model(inputs, latent_output, name="encoder")


def build_decoder_softmax(latent_dim, num_classes):

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(6 * 6 * 256)(latent_inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Reshape((6, 6, 256))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)


    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same", name="output_softmax")(x)

    return keras.Model(latent_inputs, outputs, name="decoder_softmax")


def decode_to_discrete(decoder_softmax, latents):

    softmax_output = decoder_softmax.predict(latents, verbose=0)

    # Apply argmax to get discrete classes (0, 1, 2)
    discrete_output = np.argmax(softmax_output, axis=-1)

    # Convert back to original scale (-1, 0, 1)
    # Class 0 -> -1, Class 1 -> 0, Class 2 -> 1
    discrete_images = discrete_output.astype(float) - 1.0

    # Add channel dimension back
    discrete_images = np.expand_dims(discrete_images, axis=-1)

    return discrete_images


# Initialize and load encoder
encoder = build_encoder((48, 48, 1), 512)
encoder.load_weights("training_outputs/latent_diffusion_encoder_weights.h5")

# Load original decoder (continuous)
decoder = tf.keras.models.load_model('training_outputs/latent_diffusion_decoder_model.h5')

# Build and load weights for softmax decoder
decoder_softmax = build_decoder_softmax(latent_dim, num_classes)

# Try to load softmax decoder weights if available
try:
    decoder_softmax.load_weights('training_outputs/latent_diffusion_decoder_softmax_weights.h5')
    print("Loaded softmax decoder weights successfully!")
except:
    print("Warning: Could not load softmax decoder weights. Using original decoder weights as base...")
    # If softmax weights not available, we need to train or convert
    # For now, we'll use the original decoder and convert output
    pass

# Load and process prior ensemble
data = np.load('datasets/Three_facies_2d.npy').astype('float32')[:N]

# Add channel dimension if needed
if data.ndim == 3:  # Shape: (N, 48, 48)
    data = data[..., np.newaxis]  # Shape: (N, 48, 48, 1)

prior_ensemble = encoder.predict(data, verbose=1)

# Check shape of prior_ensemble
print(f"Prior ensemble shape: {prior_ensemble.shape}")

# If prior_ensemble is multi-output, take the latent output
if isinstance(prior_ensemble, list):
    prior_ensemble = prior_ensemble[0]  # Assuming latent is first output

# Transpose for assimilation function
prior_ensemble = prior_ensemble.T  # Shape: (latent_dim, N)

# True model processing
true_model = np.load('datasets/Three_facies_2d.npy')[0:1][..., 0]
true_model = vectorization(true_model, 'vec', (48, 48))

# Map facies to numerical values
true_model = true_model.copy()  # Create a copy to avoid modifying original
true_model[true_model == -1] = 100  # purple
true_model[true_model == 0] = 1000  # green
true_model[true_model == 1] = 9000  # yellow

name_case = 'disc_32_nonloc'
n_iter = 32
tapering = None  # None, 'adaptive_furrer' or 'scaling'

# Run true case and assimilation
D_true, D_measurements = run_true_case(
    parameter_mapping(true_model, type=None, additional=None)
)

Result = run_assimilation(
    prior_ensemble,
    true_model,
    D_measurements,
    n_iter,
    12,
    tapering=tapering,
    ptype=ptype,
    additional=decoder,  # Keep original decoder for assimilation
    metric_space='latent'
)

# Save results
with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)

# Load and process metrics
with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)

    # Save DM values
    dm_array = np.array(Result[0])
    if dm_array.ndim > 1:
        dm_mean = dm_array.mean(-1)
    else:
        dm_mean = dm_array
    np.savetxt(f'./outputs/dm_{name_case}.txt', dm_mean)

    # Save detailed metrics to files
    metrics_data = [
        (Result[0], f'./outputs/DM_{name_case}.txt', "Iteration {i}: {value}\n"),
        (Result[1], f'./outputs/RMSE_{name_case}.txt', "Iteration {i}: {value}\n"),
        (Result[2], f'./outputs/Spread_{name_case}.txt', "Iteration {i}: {value}\n"),
        (Result[7], f'./outputs/BA_{name_case}.txt', "Iteration {i}: {value}\n")
    ]

    for metric_data, filename, template in metrics_data:
        with open(filename, 'w') as f:
            for i, values in enumerate(metric_data):
                if isinstance(values, (list, np.ndarray)):
                    # Join multiple values with spaces
                    f.write(f"Iteration {i}: " + " ".join([f"{v:.6f}" for v in values]) + "\n")
                else:
                    f.write(f"Iteration {i}: {values:.6f}\n")

# Create visualization directory if it doesn't exist
import os

os.makedirs('./outputs', exist_ok=True)

###########################################################################################################

# Figure 1: Properties plot
fig1, axs = plt.subplots(3, 9, figsize=(18, 5))
prop_names = ['OPR (m³/day)', 'WPR (m³/day)', 'BHP (kgf/cm²)']


if len(Result) > 5 and Result[5] is not None:
    for _1 in range(3):
        for _2 in range(9):
            # Plot prior (iteration 0) in gray
            if len(Result[5][0][_1]) > _2:
                axs[_1, _2].plot(
                    np.arange(90, 3000, 90),
                    Result[5][0][_1][:, _2, :],
                    'xkcd:gray',
                    alpha=0.35
                )

            if len(Result[5][-1][_1]) > _2:
                axs[_1, _2].plot(
                    np.arange(90, 3000, 90),
                    Result[5][-1][_1][:, _2, :],
                    'b',
                    alpha=0.35
                )

            if len(D_measurements[_1]) > _2:
                axs[_1, _2].plot(
                    np.arange(90, 3000, 90),
                    D_measurements[_1][:, _2],
                    'or',
                    markersize=2
                )

# Set labels
for _2 in range(9):
    axs[0, _2].set_ylabel('OPR (m³/day)')
    axs[1, _2].set_ylabel('WPR (m³/day)')
    axs[2, _2].set_ylabel('BHP (kgf/cm²)')

# Set x-labels for bottom row
for _2 in range(9):
    axs[0, _2].set_xlabel('Time (day)')
    axs[1, _2].set_xlabel('Time (day)')
    axs[2, _2].set_xlabel('Time (day)')

plt.tight_layout()
fig1.savefig(f'./outputs/properties_plot_{name_case}.png', dpi=300, bbox_inches='tight')
fig1.savefig(f'./outputs/properties_plot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig1)

###########################################################################################################

# Figure 2: Metrics boxplot
fig2, axs = plt.subplots(1, 4, figsize=(8, 3))

# Plot boxplots for each metric
metrics_to_plot = [Result[0], Result[1], Result[2], Result[7]]
ynames = ['DM', 'RMSE', 'Spread', 'Balanced Accuracy']

for i, (ax, metric, yname) in enumerate(zip(axs, metrics_to_plot, ynames)):
    # Ensure metric is list-like for boxplot
    if isinstance(metric, (list, np.ndarray)):
        ax.boxplot(metric, showfliers=False)
    else:
        ax.boxplot([metric], showfliers=False)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(yname)

    # Set x-ticks
    if isinstance(metric, (list, np.ndarray)):
        n_points = len(metric)
    else:
        n_points = 1

    if n_points > 1:
        tick_positions = np.linspace(1, n_points, 5, dtype=int)
        tick_labels = np.linspace(0, n_points - 1, 5, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

plt.tight_layout()
fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.png', dpi=300, bbox_inches='tight')
fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig2)

###########################################################################################################

# Figure 3: Model visualization
fig3, axs = plt.subplots(1, 7, figsize=(14, 2))

# Reset true model to original values for plotting
true_model_plot = np.load('datasets/Three_facies_2d.npy')[0:1][..., 0]
true_model_plot = vectorization(true_model_plot, 'vec', (48, 48))

if decoder is not None:
    # Prepare true model for display
    true_model_display = true_model_plot.copy()
    true_model_display[true_model_display == -1] = -1  # purple
    true_model_display[true_model_display == 0] = 0  # green
    true_model_display[true_model_display == 1] = 1  # yellow

    axs[0].imshow(true_model_display.reshape(48, 48), vmin=-1, vmax=1, cmap='viridis')
    axs[0].set_title('True', fontsize=8, pad=5)

    # Check if we have latent representations
    if len(Result) > 3 and Result[3] is not None:
        prior_latent = Result[3][0].T  # Transpose to (N, latent_dim)
        if prior_latent.shape[0] > 0:
            # Generate discrete images using softmax decoder
            prior_discrete = decode_to_discrete(decoder_softmax, prior_latent)
            prior_mean = prior_discrete.mean(0)
            axs[1].imshow(prior_mean.squeeze(), vmin=-1, vmax=1, cmap='viridis')

        posterior_latent = Result[3][-1].T
        if posterior_latent.shape[0] > 0:
            # Generate discrete images using softmax decoder
            posterior_discrete = decode_to_discrete(decoder_softmax, posterior_latent)
            posterior_mean = posterior_discrete.mean(0)
            axs[2].imshow(posterior_mean.squeeze(), vmin=-1, vmax=1, cmap='viridis')

        # Prior std
        prior_recon = decode_to_discrete(decoder_softmax, prior_latent)
        axs[3].imshow(prior_recon.std(0).squeeze(), vmin=0, vmax=0.7, cmap='viridis')

        # Posterior std
        posterior_recon = decode_to_discrete(decoder_softmax, posterior_latent)
        axs[4].imshow(posterior_recon.std(0).squeeze(), vmin=0, vmax=0.7, cmap='viridis')

        # Prior member 1
        if prior_recon.shape[0] > 0:
            axs[5].imshow(prior_recon[0].squeeze(), vmin=-1, vmax=1, cmap='viridis')

        # Posterior member 1
        if posterior_recon.shape[0] > 0:
            axs[6].imshow(posterior_recon[0].squeeze(), vmin=-1, vmax=1, cmap='viridis')

    # Set titles
    cols = ['True', 'Prior\nmean', 'Posterior\nmean', 'Prior\nstd', 'Posterior\nstd',
            'Prior\nMember 1', 'Posterior\nMember 1']
    for i in range(min(len(cols), len(axs))):
        axs[i].set_title(cols[i], fontsize=8, pad=5)

else:
    # Fallback to direct plotting without decoder
    axs[0].imshow(true_model_plot.reshape(48, 48), vmin=0, vmax=10, cmap='viridis')

    if len(Result) > 3 and Result[3] is not None:
        axs[1].imshow(Result[3][0].mean(-1).reshape(48, 48), vmin=0, vmax=10, cmap='viridis')
        axs[2].imshow(Result[3][-1].mean(-1).reshape(48, 48), vmin=0, vmax=10, cmap='viridis')
        axs[3].imshow(Result[3][0].std(-1).reshape(48, 48), vmin=0, vmax=1.5, cmap='viridis')
        axs[4].imshow(Result[3][-1].std(-1).reshape(48, 48), vmin=0, vmax=1.5, cmap='viridis')

    cols = ['True', 'Prior\nmean', 'Posterior\nmean', 'Prior\nstd', 'Posterior\nstd']
    for i in range(min(len(cols), len(axs))):
        axs[i].set_title(cols[i], fontsize=8, pad=5)

# Turn off axes
for ax in axs:
    ax.axis('off')

plt.tight_layout()
fig3.savefig(f'./outputs/model_visualization_{name_case}.png', dpi=300, bbox_inches='tight')
fig3.savefig(f'./outputs/model_visualization_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig3)

# Figure 4 (NEW): Additional visualization showing discrete class distribution
fig4, axs = plt.subplots(1, 3, figsize=(12, 3))

if len(Result) > 3 and Result[3] is not None:
    prior_latent = Result[3][0].T
    posterior_latent = Result[3][-1].T

    if prior_latent.shape[0] > 0 and posterior_latent.shape[0] > 0:
        prior_discrete = decode_to_discrete(decoder_softmax, prior_latent)
        posterior_discrete = decode_to_discrete(decoder_softmax, posterior_latent)

        # Class distribution - Prior
        axs[0].hist(prior_discrete.flatten(), bins=[-1.5, -0.5, 0.5, 1.5],
                    alpha=0.7, color='blue', edgecolor='black')
        axs[0].set_title('Prior Class Distribution')
        axs[0].set_xlabel('Class')
        axs[0].set_ylabel('Count')
        axs[0].set_xticks([-1, 0, 1])

        # Class distribution - Posterior
        axs[1].hist(posterior_discrete.flatten(), bins=[-1.5, -0.5, 0.5, 1.5],
                    alpha=0.7, color='red', edgecolor='black')
        axs[1].set_title('Posterior Class Distribution')
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Count')
        axs[1].set_xticks([-1, 0, 1])

        # Class distribution comparison
        prior_counts = [np.sum(prior_discrete == c) for c in [-1, 0, 1]]
        posterior_counts = [np.sum(posterior_discrete == c) for c in [-1, 0, 1]]

        x = np.arange(3)
        width = 0.35
        axs[2].bar(x - width / 2, prior_counts, width, label='Prior', color='blue', alpha=0.7)
        axs[2].bar(x + width / 2, posterior_counts, width, label='Posterior', color='red', alpha=0.7)
        axs[2].set_title('Class Distribution Comparison')
        axs[2].set_xlabel('Class')
        axs[2].set_ylabel('Count')
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(['-1', '0', '1'])
        axs[2].legend()

plt.tight_layout()
fig4.savefig(f'./outputs/class_distribution_{name_case}.png', dpi=300, bbox_inches='tight')
fig4.savefig(f'./outputs/class_distribution_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig4)

print("Generated files:")
print(f"- properties_plot_{name_case}.png/.svg")
print(f"- metrics_boxplot_{name_case}.png/.svg")
print(f"- model_visualization_{name_case}.png/.svg")
print(f"- class_distribution_{name_case}.png/.svg (NEW - discrete classes)")
print(f"- Various metric text files in ./outputs/")

# Print summary statistics for discrete outputs
if len(Result) > 3 and Result[3] is not None:
    prior_latent = Result[3][0].T
    posterior_latent = Result[3][-1].T

    if prior_latent.shape[0] > 0 and posterior_latent.shape[0] > 0:
        prior_discrete = decode_to_discrete(decoder_softmax, prior_latent)
        posterior_discrete = decode_to_discrete(decoder_softmax, posterior_latent)

        print("\n" + "=" * 50)
        print("DISCRETE OUTPUT STATISTICS")
        print("=" * 50)
        print("\nPrior:")
        for c in [-1, 0, 1]:
            count = np.sum(prior_discrete == c)
            pct = count / prior_discrete.size * 100
            print(f"  Class {c}: {count} pixels ({pct:.1f}%)")

        print("\nPosterior:")
        for c in [-1, 0, 1]:
            count = np.sum(posterior_discrete == c)
            pct = count / posterior_discrete.size * 100
            print(f"  Class {c}: {count} pixels ({pct:.1f}%)")

        # Calculate class accuracy if true model is available
        true_classes = true_model_display.flatten()
        posterior_flat = posterior_discrete.flatten()

        # Only compare where true model has valid classes
        valid_mask = np.isin(true_classes, [-1, 0, 1])
        if np.sum(valid_mask) > 0:
            accuracy = np.mean(true_classes[valid_mask] == posterior_flat[valid_mask])
            print(f"\nClass Accuracy (Posterior vs True): {accuracy:.4f}")
        print("=" * 50)