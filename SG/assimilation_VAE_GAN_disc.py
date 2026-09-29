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
from tensorflow.keras.layers import LeakyReLU, BatchNormalization

N = 500
latent_dim = 512
ptype = 'vae_3f'
alpha = 0.2  # LeakyReLU alpha


DPI = 300


def build_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)

    # Downsampling blocks
    x = layers.Conv2D(58, 5, strides=2, padding="same")(inputs)  # 64
    x = LeakyReLU(alpha=alpha)(x)
    x = BatchNormalization()(x)

    x = layers.Conv2D(116, 5, strides=2, padding="same")(x)  # 128
    x = LeakyReLU(alpha=alpha)(x)
    x = BatchNormalization()(x)

    x = layers.Conv2D(230, 5, strides=2, padding="same")(x)  # 256
    x = LeakyReLU(alpha=alpha)(x)
    x = BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(922)(x)  # 1024
    x = LeakyReLU(alpha=alpha)(x)
    x = layers.Dropout(0.3)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


vae_encoder = build_encoder((48, 48, 1), 512)
vae_encoder.load_weights("training_outputs/encoder_weights.h5")
vae_decoder = tf.keras.models.load_model('training_outputs/decoder_model.h5')
prior_ensemble = vae_encoder.predict(np.load('datasets/Three_facies_2d.npy').astype('float')[:N])[2].T
true_model = np.load('datasets/Three_facies_2d.npy')[0:1][..., 0]
true_model = vectorization(true_model, 'vec', (48, 48))
true_model[true_model == -1] = 100  # purple
true_model[true_model == 0] = 1000  # green
true_model[true_model == 1] = 9000  # yellow

name_case = 'disc_32_nonloc'
n_iter = 32
tapering = None

D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=None, additional=None))
Result = run_assimilation(prior_ensemble, true_model, D_measurements, n_iter, 12,
                          tapering=tapering, ptype=ptype, additional=vae_decoder, metric_space='latent')

with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:
    pickle.dump(Result, fp)

with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:
    Result = pickle.load(fp)


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

with open(f'./outputs/BA_{name_case}.txt', 'w') as f:
    for i, ba_values in enumerate(Result[7]):
        if isinstance(ba_values, (list, np.ndarray)):
            f.write(f"Iteration {i}: " + " ".join(map(str, ba_values)) + "\n")
        else:
            f.write(f"Iteration {i}: {ba_values}\n")

###########################################################################################################

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


fig1.savefig(f'time_series_{name_case}.png', dpi=DPI, bbox_inches='tight')
fig1.savefig(f'time_series_{name_case}.svg', format='svg', bbox_inches='tight')
plt.show()


fig2, axs = plt.subplots(1, 4, figsize=(8, 3))
axs[0].boxplot(Result[0], showfliers=False)
axs[1].boxplot(Result[1], showfliers=False)
axs[2].boxplot(Result[2], showfliers=False)
axs[3].boxplot(Result[7], showfliers=False)
[axs[_].set_xlabel('iteration') for _ in range(4)]
ynames = ['DM', 'RMSE', 'Spread', 'Balanced Accuracy']
[axs[_].set_ylabel(ynames[_]) for _ in range(4)]
[axs[_].set_xticks(np.linspace(0 + 1, n_iter + 1, 5), np.linspace(0, n_iter, 5, dtype='int')) for _ in range(4)]
plt.tight_layout()


fig2.savefig(f'metrics_boxplot_{name_case}.png', dpi=DPI, bbox_inches='tight')
fig2.savefig(f'metrics_boxplot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.show()


if vae_decoder:
    true_model[true_model == 100] = -1  # purple
    true_model[true_model == 1000] = 0  # green
    true_model[true_model == 9000] = 1  # yellow

    fig3, axs = plt.subplots(2, 7, figsize=(14, 4))

    # Linha 1
    axs[0, 0].imshow(true_model.reshape(48, 48), vmin=-1, vmax=1)
    axs[0, 1].imshow(vae_decoder.predict(Result[3][0].T).mean(0), vmin=-1, vmax=1)
    axs[0, 2].imshow(vae_decoder.predict(Result[3][-1].T).mean(0), vmin=-1, vmax=1)
    axs[0, 3].imshow(vae_decoder.predict(Result[3][0].T).std(0), vmin=0, vmax=0.7)
    axs[0, 4].imshow(vae_decoder.predict(Result[3][-1].T).std(0), vmin=0, vmax=0.7)
    axs[0, 5].imshow(vae_decoder.predict(Result[3][0].T)[0], vmin=-1, vmax=1)
    axs[0, 6].imshow(vae_decoder.predict(Result[3][-1].T)[0], vmin=-1, vmax=1)


    for i in range(7):
        axs[1, i].axis('off')

    [axs[0, _].axis('off') for _ in range(7)]

    cols = ['True', 'Prior\n mean', 'Posterior\n mean', 'Prior\n std', 'Posterior\n std', 'Prior\n Member 1',
            'Posterior\n Member 1']
    for i in range(len(cols)):
        axs[0, i].set_title(cols[i], fontsize=8, pad=5)

    plt.tight_layout()


    fig3.savefig(f'maps_{name_case}.png', dpi=DPI, bbox_inches='tight')
    fig3.savefig(f'maps_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.show()

else:
    fig3, axs = plt.subplots(1, 5, figsize=(10, 2))

    axs[0].imshow(true_model.reshape(48, 48), vmin=0, vmax=10)
    axs[1].imshow(Result[3][0].mean(-1).reshape(48, 48), vmin=0, vmax=10)
    axs[2].imshow(Result[3][-1].mean(-1).reshape(48, 48), vmin=0, vmax=10)
    axs[3].imshow(Result[3][0].std(-1).reshape(48, 48), vmin=0, vmax=1.5)
    axs[4].imshow(Result[3][-1].std(-1).reshape(48, 48), vmin=0, vmax=1.5)

    [axs[_].axis('off') for _ in range(5)]

    cols = ['True', 'Prior mean', 'Posterior mean', 'Prior std', 'Posterior std']
    for i in range(len(cols)):
        axs[i].set_title(cols[i], fontsize=8)

    plt.tight_layout()


    fig3.savefig(f'maps_{name_case}.png', dpi=DPI, bbox_inches='tight')
    fig3.savefig(f'maps_{name_case}.svg', format='svg', bbox_inches='tight')
    plt.show()


print(f"- time_series_{name_case}.png e .svg")
print(f"- metrics_boxplot_{name_case}.png e .svg")
print(f"- maps_{name_case}.png e .svg")
print(f"DPI: {DPI}")