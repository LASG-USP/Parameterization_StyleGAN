import numpy as np

from processing2 import parameter_mapping, vectorization, create_measurements, process_measurements, process_ensemble
from coupling import run_ensemble_v2

from math_utils import normalized_data_mismatch, root_mean_squared_error, balanced_acc
from assimilation_methods import sinv, esmda
from loc_methods import compute_furrer_localization, compute_scaling_approximate

import os

import pickle

import matplotlib.pyplot as plt

np.random.seed(202410)

def run_true_case(m_true, meas_error=(0.1, 0.15, 5, 0.15, 5)):
#def run_true_case(m_true, meas_error=(4, 4, 2, 3, 2)):    # função para rodar o caso true
    run_ensemble_v2(m_true, 1, 'simfiles/48x48_toy_model.dat', 'simfiles/rwd_master.rwd')
    T_en, T = process_ensemble(1)                         # extract .rwo data
    Dobs_en = create_measurements(T_en, meas_error)
    return T_en, Dobs_en


def run_assimilation(prior, true, measurements, n_iterations, max_run, alpha=None,
                     tapering=None, tape_matrix=None,
                     ptype=None, additional=None, metric_space=None):  # função mestre da assimilação (ESMDA)
    ens_size = prior.shape[-1]
    if alpha is None:
        alpha = np.ones(n_iterations) * n_iterations


    ############
    # Remove os arquivos da CMG da pasta temp
    temp_folder = r'C:\Users\Marcio Sampaio\AppData\Local\Temp'
    dir_n = os.listdir(temp_folder)  # List all items (files and directories) in the Temp folder
    for item in dir_n:  # Loop through the items and delete them if they are files
        try:
            file_path = os.path.join(temp_folder, item)
            if os.path.isfile(file_path):  # Ensure it's a file
                os.remove(file_path)
                # print(f'Deleted file: {file_path}')
            elif os.path.isdir(file_path):
                print(f'Skipped directory: {file_path}')  # Handle directories as needed
        except Exception as ex:
            print(f'Error deleting {file_path}: {ex}')


    ##############
    # Start assimilation module
    obj = list()    # data mismatch
    rmse = list()
    spread = list()
    accuracy = list()

    # pack M and D
    Mpack = list()    # lista dos parâmetros
    Dpack = list()    # lista dos dados
    Denpack = list()  # lista dos dados no formato do simulador

    for i in range(n_iterations):
        print('iteration {}'.format(i))
        if i == 0:
            #d_obs, Cd = process_measurements(measurements, meas_error=(4, 4, 2, 3, 2))
            d_obs, Cd = process_measurements(measurements, meas_error=(0.1, 0.15, 5, 0.15, 5))
            # meas_error: erros de medição
            M = np.copy(prior)
            Mpack.append(M)
        else:
            M = np.copy(Ma)
            Mpack.append(M)


        run_ensemble_v2(parameter_mapping(M, mapping_type=ptype, additional=additional),
                        max_run, 'simfiles/48x48_toy_model.dat', 'simfiles/rwd_master.rwd')
        print('assimilating')
        D_en, D = process_ensemble(ens_size)  # extract .rwo data
        obj.append(normalized_data_mismatch(D, d_obs, Cd))  # compute data mismatch
        if metric_space == 'latent':
            rmse.append(root_mean_squared_error(parameter_mapping(M, mapping_type=ptype, additional=additional), true, axs=-1))  # compute model mismatch
            #spread.append(root_mean_squared_error(parameter_mapping(M, type=ptype, additional=additional), parameter_mapping(M, type=ptype, additional=additional).mean(-1)[:, np.newaxis], axs=-1))  # compute spread
            accuracy.append(balanced_acc(parameter_mapping(M, mapping_type=ptype, additional=additional), true))    # compute balanced accuracy
        else:
            rmse.append(root_mean_squared_error(M, true, axs=0))  # compute model mismatch
            accuracy.append(balanced_acc(M, true, axs=-1))    # compute balanced accuracy multiclass
        spread.append(root_mean_squared_error(M, M.mean(-1)[:, np.newaxis], axs=0))  # compute spread

        Dpack.append(D)
        Denpack.append(D_en)

        # calcula ganho de kalman (subspace inversion)
        # kgain = M @ (np.eye(ens_size) - np.ones((ens_size, ens_size)) * (1 / ens_size)) @ sinv(D, Cd, alpha[i], 0.99)
        # sinv -> inversão no subespaço
        if i == 0:
            if tapering is not None:
                if tape_matrix is not None:
                    tape = tape_matrix
                else:
                    if tapering == 'adaptive_furrer':
                        tape = compute_furrer_localization(M, ens_size, type='cross', y=D, ddof=1)
                        lmatrix = tape
                    elif tapering == 'scaling':
                        tape = compute_scaling_approximate(M, ens_size, type='cross', y=D, ddof=1, norm=Cd)
                        lmatrix = np.ones((M.shape[0], D.shape[0]))*tape
            else:
                tape = 1
                lmatrix = np.ones((M.shape[0], D.shape[0]))

        # kgain = tape * kgain

        #dp = np.random.multivariate_normal(np.zeros(d_obs.shape)[:, 0], alpha[i] * Cd, ens_size).T  # perturba medições

        #Ma = M + kgain @ (d_obs + dp - D)  #eq. de atualização
        Ma = esmda(M, D, d_obs, Cd, loc=lmatrix).subspace(alpha[i], 0.99, loc=lmatrix)
        
        # Quando for o VAE a equação de atualização deverá ser o decoder

    i += 1  # last forward step
    print('last forward step')
    M = Ma.copy()
    Mpack.append(M)
    run_ensemble_v2(parameter_mapping(M, mapping_type=ptype, additional=additional),
                    max_run, 'simfiles/48x48_toy_model.dat', 'simfiles/rwd_master.rwd')
    D_en, D = process_ensemble(ens_size)                  # extract .rwo data
    obj.append(normalized_data_mismatch(D, d_obs, Cd))    # compute data mismatch
    if metric_space == 'latent':
        rmse.append(root_mean_squared_error(parameter_mapping(M, mapping_type=ptype, additional=additional), true,
                                            axs=-1))      # compute model mismatch
        # spread.append(root_mean_squared_error(parameter_mapping(M, type=ptype, additional=additional),
                                              #parameter_mapping(M, type=ptype, additional=additional).mean(-1)[:,
                                              #np.newaxis], axs=-1))  # compute spread
        accuracy.append(balanced_acc(parameter_mapping(M, mapping_type=ptype, additional=additional), true))  # compute balanced accuracy
    else:
        rmse.append(root_mean_squared_error(M, true, axs=-1))  # compute model mismatch
        accuracy.append(balanced_acc(M, true, axs=-1))  # compute balanced accuracy
    spread.append(root_mean_squared_error(M, M.mean(-1)[:, np.newaxis], axs=-1))  # compute spread

    Dpack.append(D)
    Denpack.append(D_en)
    return obj, rmse, spread, Mpack, Dpack, Denpack, tape, accuracy



####################################### U-II sem parametrização/localização
# ens_size = 20
# id_true = 0
# ptype = 'log'
# full_ensemble = vectorization(np.log(np.maximum(np.load('simfiles/Perm_ensemble_cropped.npy'), 1)), 'vec', (48, 48))
# true_model = full_ensemble[:, id_true][:, np.newaxis]
# ensemble = np.delete(full_ensemble, id_true, axis=-1)
# prior_ensemble = full_ensemble[:, np.random.choice(ensemble.shape[0], ens_size, replace=False)]
#
# D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=ptype))  # roda modelo true e obtém medições
# Result = run_assimilation(prior_ensemble, true_model, D_measurements, 4, 12, ptype='log')
# with open('./outputs/Results.pickle', 'wb') as fp:  # save true measurements
#     pickle.dump(Result, fp)
#######################################  U-II com GAN
# ens_size = 20  # tamanho do conjunto
# ptype = 'gan_uii'
# import tensorflow as tf
# GAN = tf.keras.models.load_model('gan_uiiperm_15k.h5')
# # prior dos vetores latente
# prior_ensemble = np.random.normal(0, 1, (ens_size, GAN.layers[0].output.shape[-1])).T
# true_model = np.random.normal(0, 1, (1, GAN.layers[0].output.shape[-1])).T
#
# D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=ptype, additional=GAN))
# Result = run_assimilation(prior_ensemble, true_model, D_measurements, 8, 12,
#                           tapering=None, #'adaptive_furrer' para localizaççao
#                           ptype=ptype, additional=GAN)
# with open('./outputs/Results.pickle', 'wb') as fp:  # save true measurements
#     pickle.dump(Result, fp)
############################################ 3facies/GAN
# ens_size = 20
# ptype = 'gan_3f'
# import tensorflow as tf
# GAN = tf.keras.models.load_model('gan_3facies_80k.h5')
# prior_ensemble = np.random.normal(0, 1, (ens_size, GAN.layers[0].output.shape[-1])).T
# true_model = np.random.normal(0, 1, (1, GAN.layers[0].output.shape[-1])).T
#
# D_true, D_measurements = run_true_case(parameter_mapping(true_model, type=ptype, additional=GAN))
# Result = run_assimilation(prior_ensemble, true_model, D_measurements, 8, 12,
#                           tapering=None,
#                           ptype=ptype, additional=GAN)
# with open('./outputs/Results.pickle', 'wb') as fp:  # save true measurements
#     pickle.dump(Result, fp)
#
# ###########################################################################################################
# fig, axs = plt.subplots(1, 5)  # gera os mapas
# if GAN:
#     axs[0].imshow(GAN.predict(true_model.T)[0], vmin=-1, vmax=1)
#     axs[1].imshow(GAN.predict(Result[3][0].T).mean(0), vmin=-1, vmax=1)
#     axs[2].imshow(GAN.predict(Result[3][-1].T).mean(0), vmin=-1, vmax=1)
#     axs[3].imshow(GAN.predict(Result[3][0].T).std(0), vmin=0, vmax=0.7)
#     axs[4].imshow(GAN.predict(Result[3][-1].T).std(0), vmin=0, vmax=0.7)
#     [axs[_].axis('off') for _ in range(5)]
#     cols = ['True', 'Pr. mean', 'Po. mean', 'Pr. std', 'Po. std']
#     for i in range(len(cols)):
#         newaxis = fig.add_axes(axs[i].get_position(), anchor='NW', zorder=1)
#         newaxis.axis('off'), newaxis.set_title('{}'.format(cols[i]), fontsize=8)
# else:
#     axs[0].imshow(true_model.reshape(48, 48), vmin=0, vmax=10)
#     axs[1].imshow(Result[3][0].mean(-1).reshape(48, 48), vmin=0, vmax=10)
#     axs[2].imshow(Result[3][-1].mean(-1).reshape(48, 48), vmin=0, vmax=10)
#     axs[3].imshow(Result[3][0].std(-1).reshape(48, 48), vmin=0, vmax=1.5)
#     axs[4].imshow(Result[3][-1].std(-1).reshape(48, 48), vmin=0, vmax=1.5)
#     [axs[_].axis('off') for _ in range(5)]
#     cols = ['True', 'Pr. mean', 'Po. mean', 'Pr. std', 'Po. std']
#     for i in range(len(cols)):
#         newaxis = fig.add_axes(axs[i].get_position(), anchor='NW', zorder=1)
#         newaxis.axis('off'), newaxis.set_title('{}'.format(cols[i]), fontsize=8)
#
# fig, axs = plt.subplots(3, 9, figsize=(18, 5))
# for _1 in range(3):
#     for _2 in range(9):
#         axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][0][_1][:, _2, :], 'b', alpha=0.35)
#         axs[_1, _2].plot(np.arange(90, 3000, 90), Result[5][-1][_1][:, _2, :], 'g', alpha=0.35)
#         axs[_1, _2].plot(np.arange(90, 3000, 90), D_measurements[_1][:, _2], 'or', markersize=2)
# for _2 in range(9):
#     axs[0, _2].set_ylabel('OPR')
#     axs[1, _2].set_ylabel('WPR')
#     axs[2, _2].set_ylabel('BHP')
# plt.tight_layout()
#
# fig, axs = plt.subplots(1, 3, figsize=(8, 3))
# axs[0].boxplot(Result[0])
# axs[1].boxplot(Result[1])
# axs[2].boxplot(Result[2])
# [axs[_].set_xlabel('iteration') for _ in range(3)]
# ynames = ['DM', 'RMSE', 'Spread']
# [axs[_].set_ylabel(ynames[_]) for _ in range(3)]
# plt.tight_layout()