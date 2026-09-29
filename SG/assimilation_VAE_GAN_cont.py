import matplotlib.pyplot as plt
import numpy as np
import pickle
from main import run_assimilation, run_true_case
from processing import parameter_mapping, vectorization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization

N = 500
latent_dim = 512
ptype = 'gan_uii'
alpha = 0.2  # LeakyReLU alpha


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




encoder = build_encoder((48, 48, 1), latent_dim)


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

vae_encoder = build_encoder((48, 48, 1), latent_dim)

vae_encoder.load_weights("training_outputs/encoder_weights.h5")

#vae_encoder = tf.keras.models.load_model('training_outputs/encoder_model.h5', custom_objects={'sampling': sampling})


#GAN = tf.keras.models.load_model('gan_3facies_2d_80k.h5')
#vae_encoder = tf.keras.models.load_model('training_outputs/encoder_model.h5', custom_objects={'sampling': sampling})
vae_decoder = tf.keras.models.load_model('training_outputs/decoder_model.h5')
X_train = np.load('D:/PycharmProjects/GAN/simfiles/Perm_ensemble_cropped.npy')
X_train[X_train<1] = 1
X_train = np.log(X_train)
X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) -1
X_train = X_train[..., np.newaxis]
prior_ensemble = vae_encoder.predict(X_train[:N])[2].T
# true_model = np.random.normal(0, 1, (1, GAN.layers[0].output.shape[-1])).T
true_model = np.load('simfiles/Perm_ensemble_cropped.npy')[0:1]
true_model[true_model<1] = 1
true_model = vectorization(np.log(true_model), 'vec', (48, 48))

name_case = 'cont_32_nonloc'
n_iter = 32
tapering = None

D_true, D_measurements = run_true_case(parameter_mapping(true_model, type='log', additional=None))
# D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=ptype, additional=GAN))
Result = run_assimilation(prior_ensemble, true_model, D_measurements, n_iter, 12,
                          tapering= tapering, ptype=ptype, additional=vae_decoder, metric_space='latent')

with open(f'./outputs/Results_{name_case}.pickle', 'wb') as fp:  # save true measurements
    pickle.dump(Result, fp)

with open(f'./outputs/Results_{name_case}.pickle', 'rb') as fp:  # load
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


fig1.savefig(f'./outputs/properties_plot_{name_case}.png', dpi=300, bbox_inches='tight')
fig1.savefig(f'./outputs/properties_plot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig1)

fig2, axs = plt.subplots(1, 3, figsize=(8, 3))
axs[0].boxplot(Result[0], showfliers=False)
axs[1].boxplot(Result[1], showfliers=False)
axs[2].boxplot(Result[2], showfliers=False)
[axs[_].set_xlabel('iterations') for _ in range(3)]
ynames = ['DM', 'RMSE', 'Spread']
[axs[_].set_ylabel(ynames[_]) for _ in range(3)]
[axs[_].set_xticks(np.linspace(0 + 1, n_iter + 1, 4), np.linspace(0, n_iter, 4, dtype='int')) for _ in range(3)]
plt.tight_layout()


fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.png', dpi=300, bbox_inches='tight')
fig2.savefig(f'./outputs/metrics_boxplot_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig2)


fig3, axs = plt.subplots(2, 4, figsize=(9, 5))
for _1 in range(2):
    for _2 in range(4):
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1+3][:, _2, :], color=[0.2, 0.2, 0.2], alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1+3][:, _2, :], color='blue', alpha=0.35)
        axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1+3][:, _2], 'or', markersize=2)
for _2 in range(4):
    axs[0, _2].set_ylabel('WIR (m³/day)')
    axs[1, _2].set_ylabel('BHP (kgf/cm²)')
    # axs[2, _2].set_ylabel('BHP')
    [[axs[_1, _2].set_xlabel('Time (day)') for _1 in range(2)] for _2 in range(4)]
plt.tight_layout()
plt.show()





fig4, axs = plt.subplots(1, 7, figsize=(14, 2))
if vae_decoder:
    axs[0].imshow(true_model.reshape(48, 48), vmin=0, vmax=9)
    # axs[0].imshow(GAN.predict(true_model.T)[0], vmin=-1, vmax=1)
    axs[1].imshow(vae_decoder.predict(Result[3][0].T).mean(0), vmin=-1, vmax=1)
    axs[2].imshow(vae_decoder.predict(Result[3][-1].T).mean(0), vmin=-1, vmax=1)
    axs[3].imshow(vae_decoder.predict(Result[3][0].T).std(0), vmin=0, vmax=0.7)
    axs[4].imshow(vae_decoder.predict(Result[3][-1].T).std(0), vmin=0, vmax=0.7)
    axs[5].imshow(vae_decoder.predict(Result[3][0].T)[0], vmin=-1, vmax=1)
    axs[6].imshow(vae_decoder.predict(Result[3][-1].T)[0], vmin=-1, vmax=1)
    [axs[_].axis('off') for _ in range(7)]
    # Set titles directly on the axes
    cols = ['True', 'Prior\n mean', 'Posterior\n mean', 'Prior\n std', 'Posterior\n std', 'Prior\n Member 1',
            'Posterior\n Member 1']
    for i in range(len(cols)):
        axs[i].set_title(cols[i], fontsize=8, pad=5)  # Usando set_title diretamente
else:
    axs[0].imshow(true_model.reshape(48, 48), vmin=0, vmax=10)
    axs[1].imshow(Result[3][0].mean(-1).reshape(48, 48), vmin=0, vmax=10)
    axs[2].imshow(Result[3][-1].mean(-1).reshape(48, 48), vmin=0, vmax=10)
    axs[3].imshow(Result[3][0].std(-1).reshape(48, 48), vmin=0, vmax=1.5)
    axs[4].imshow(Result[3][-1].std(-1).reshape(48, 48), vmin=0, vmax=1.5)
    [axs[_].axis('off') for _ in range(5)]
    cols = ['True', 'Prior mean', 'Posterior mean', 'Prior std', 'Posterior std']
    for i in range(len(cols)):
        newaxis = fig4.add_axes(axs[i].get_position(), anchor='NW', zorder=1)
        newaxis.axis('off'), newaxis.set_title('{}'.format(cols[i]), fontsize=8)
plt.show()



fig4.savefig(f'./outputs/model_visualization_{name_case}.png', dpi=300, bbox_inches='tight')
fig4.savefig(f'./outputs/model_visualization_{name_case}.svg', format='svg', bbox_inches='tight')
plt.close(fig4)


print(f"- properties_plot_{name_case}.png/.svg")
print(f"- metrics_boxplot_{name_case}.png/.svg")
print(f"- model_visualization_{name_case}.png/.svg")