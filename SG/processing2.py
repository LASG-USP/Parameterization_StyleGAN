import numpy as np
import pickle
import tensorflow as tf


def read_rwo(file):
    raw = open(file)
    a = raw.read()
    b = a.split('')

    values = list()

    for i in range(1, b.__len__()):
        with open('observation_temp.txt', 'w') as f:
            f.write(b[i])
        values.append(np.genfromtxt('observation_temp.txt', skip_header=8, invalid_raise=False))

    return values


def generate_include(m, filename):
    with open(filename, 'w') as f:
        np.savetxt(f, m, fmt='%.4f')


def generate_include_full(values, id):
    generate_include(values, './cmgforward/perm_{}.inc'.format(id))

    rock_type = np.zeros(values.shape) + 2
    rock_type[values < 800] = 1
    generate_include(rock_type, 'cmgforward/rtype_{}.inc'.format(id))


def process_measurements(meas, meas_error=(0.1, 0.15, 5 / 3, 0.05, 5 / 3)):
    d_vec = list()
    cd = list()

    for mt in range(meas.__len__()):
        d_vec.append(meas[mt][:, 0:].flatten(order='F'))
        # d_vec.append(meas[mt][:, 1:].flatten(order='F'))

        if meas_error[mt] < 1:
            cd_temp = meas[mt][:, 0:].flatten(order='F') * meas_error[mt] / 3
            # cd_temp = meas[mt][:, 1:].flatten(order='F') * meas_error[mt] / 3
            cd_temp = (cd_temp < 1).choose(cd_temp, 1)
        else:
            cd_temp = np.ones(meas[mt][:, 0:].flatten(order='F').shape) * meas_error[mt]
            # cd_temp = np.ones(meas[mt][:, 1:].flatten(order='F').shape) * meas_error[mt]
        cd.append(cd_temp)

    d_vec = np.concatenate(d_vec)
    cd = np.concatenate(cd)

    return np.expand_dims(d_vec, -1), np.diag(cd ** 2)


def process_ensemble(N):
    output = list()
    for mt in range(5):  # five types of data
        output.append(list())

    try:
        for n in range(N):
            temp = read_rwo('./cmgforward/member_{}.rwo'.format(n))
            for mt in range(5):
                output[mt].append(temp[mt][:, 1:])
    except:
        print(n)

    for mt in range(5):  # five types of data
        output[mt] = np.array(output[mt])
        output[mt] = np.einsum('zxy->xyz', output[mt])
        output[mt] = np.delete(output[mt], 0, 0)

    output_vec = list()
    for mt in range(5):  # five types of data
        output_vec.append(np.reshape(output[mt], (np.prod(output[mt].shape[0:2]), output[mt].shape[-1]), order='F'))
    output_vec = np.concatenate(output_vec)

    return output, output_vec


def vectorization(x, mode='vec', shape=(41, 41)):
    if mode == 'vec':
        N = x.shape[0]
        y = np.reshape(x.T, (shape[0] * shape[1], N), order='F')
    if mode == 'unvec':
        N = x.shape[-1]
        y = x.reshape((shape[0], shape[1], N), order='F').T
    return y


def field_mapping(poro_en, perm_en, netg_en):
    poro_en_vec = vectorization(poro_en, mode='vec', dim=poro_en.shape[1:])
    perm_en_vec = vectorization(perm_en, mode='vec', dim=perm_en.shape[1:])
    perm_en_vec[perm_en_vec == 0] = 1
    netg_en_vec = vectorization(netg_en, mode='vec', dim=netg_en.shape[1:])
    M = np.concatenate((poro_en_vec, np.log(perm_en_vec), netg_en_vec))
    return M


def reverse_field_mapping(Ma, dim):
    poro_en_a, perm_en_a, netg_en_a = np.split(Ma, 3)

    poro_en_a = vectorization(poro_en_a, mode='unvec', dim=dim)
    perm_en_a = vectorization(perm_en_a, mode='unvec', dim=dim)
    netg_en_a = vectorization(netg_en_a, mode='unvec', dim=dim)

    poro_en_a[poro_en_a < 0.01] = 0.01
    poro_en_a[poro_en_a > 0.4] = 0.4

    perm_en_a[perm_en_a < 0] = 0
    perm_en_a[perm_en_a > 10] = 10

    netg_en_a[netg_en_a < 0.01] = 0.01
    netg_en_a[netg_en_a > 1] = 1

    return poro_en_a, np.exp(perm_en_a), netg_en_a


def get_ensemble(full_ensemble, ens_size):
    ntotal = full_ensemble.shape[-1]
    idx = np.random.choice(ntotal, ens_size, replace=False)
    ensemble = full_ensemble[:, idx]
    return ensemble


def create_measurements(true, meas_error):
    measurements = list()
    for mt in range(5):
        time_series = np.zeros(true[mt].shape)
        st_dev = np.zeros(true[mt].shape)

        for i1 in range(time_series.shape[0]):
            for i2 in range(time_series.shape[1]):
                if meas_error[mt] < 1:
                    st_dev[i1, i2] = (true[mt][i1, i2] * meas_error[mt] / 3)  # proportional measurement error
                    if st_dev[i1, i2] < 1:
                        st_dev[i1, i2] = 1
                else:
                    st_dev[i1, i2] = meas_error[mt]  # constant measurement error

        for i1 in range(time_series.shape[0]):
            for i2 in range(time_series.shape[1]):
                if true[mt][i1, i2] > 0:
                    time_series[i1, i2] = np.random.normal(true[mt][i1, i2], st_dev[i1, i2])
                    # time_series[i1, i2] = np.random.normal(true[mt][i1, i2], true[mt][i1, i2] * meas_error[mt] / 3)
                else:
                    time_series[i1, i2] = 0
        time_series[time_series < 0] = 0

        measurements.append(time_series)

        with open('./outputs/D_true.pickle', 'wb') as fp:  # save true measurements
            pickle.dump(true, fp)
        with open('./outputs/D_observations.pickle', 'wb') as fp:  # save true measurements
            pickle.dump(measurements, fp)

    return measurements


def parameter_mapping(M, mapping_type=None, additional=None):  # MUDEI: type -> mapping_type
    if mapping_type == 'log':  # MUDEI
        out = np.exp(M)

    elif mapping_type == 'gan_uii':  # MUDEI
        # Para modelos carregados com tf.saved_model.load() (SavedModel)
        try:
            if hasattr(additional, 'predict'):  # Modelo Keras
                out = additional.predict(M.T, verbose=0)[..., 0]
            else:  # Modelo SavedModel
                # Converter para tensor TensorFlow
                M_tensor = tf.convert_to_tensor(M.T, dtype=tf.float32)

                # DEBUG: Verificar o tipo do objeto
                print(f"DEBUG - Tipo de additional: {type(additional).__name__}")  # CORRIGIDO

                # Tentar diferentes formas de chamar o modelo
                if callable(additional):
                    output = additional(M_tensor)
                elif hasattr(additional, 'signatures'):
                    print(f"DEBUG - Assinaturas disponíveis: {list(additional.signatures.keys())}")
                    if 'serving_default' in additional.signatures:
                        output = additional.signatures['serving_default'](latent=M_tensor)
                        if isinstance(output, dict):
                            output = list(output.values())[0]
                    else:
                        # Usar a primeira assinatura disponível
                        first_sig = list(additional.signatures.keys())[0]
                        output = additional.signatures[first_sig](latent=M_tensor)
                        if isinstance(output, dict):
                            output = list(output.values())[0]
                else:
                    # Tentar acessar diretamente como um método
                    try:
                        output = additional.__call__(M_tensor)
                    except:
                        # Último recurso: tentar como atributo
                        output = additional(M_tensor)

                # Converter para numpy
                if hasattr(output, 'numpy'):
                    out = output.numpy()[..., 0]
                else:
                    out = output[..., 0]

            out = 8.8378 * (out + 1) / 2  # constante de normalização do UNISIM-II
            out = vectorization(out, 'vec', shape=(48, 48))
            out = np.exp(out)

        except Exception as e:
            print(f"Erro no mapeamento GAN UII: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: retornar entrada original
            out = M

    elif mapping_type == 'gan_3f':  # MUDEI
        # Para modelos carregados com tf.saved_model.load() (SavedModel)
        try:
            if hasattr(additional, 'predict'):  # Modelo Keras
                out = additional.predict(M.T, verbose=0)[..., 0]
            else:  # Modelo SavedModel
                # Converter para tensor TensorFlow
                M_tensor = tf.convert_to_tensor(M.T, dtype=tf.float32)

                # DEBUG: Verificar o tipo do objeto
                print(f"DEBUG - Tipo de additional: {type(additional).__name__}")  # CORRIGIDO

                # Tentar diferentes formas de chamar o modelo
                output = None

                # Método 1: Verificar se tem assinaturas
                if hasattr(additional, 'signatures'):
                    print(f"DEBUG - Assinaturas disponíveis: {list(additional.signatures.keys())}")

                    # Tentar diferentes nomes de assinatura
                    signature_names = ['serving_default', 'default', 'predict', 'call']
                    for sig_name in signature_names:
                        if sig_name in additional.signatures:
                            print(f"DEBUG - Usando assinatura: {sig_name}")
                            try:
                                # Tentar com diferentes nomes de parâmetro
                                try:
                                    output = additional.signatures[sig_name](latent=M_tensor)
                                except:
                                    try:
                                        output = additional.signatures[sig_name](input_1=M_tensor)
                                    except:
                                        try:
                                            output = additional.signatures[sig_name](inputs=M_tensor)
                                        except:
                                            output = additional.signatures[sig_name](M_tensor)

                                if isinstance(output, dict):
                                    output = list(output.values())[0]
                                break
                            except Exception as sig_err:
                                print(f"DEBUG - Erro com assinatura {sig_name}: {sig_err}")
                                continue

                    # Se não encontrou assinatura, usar a primeira disponível
                    if output is None and additional.signatures:
                        first_sig = list(additional.signatures.keys())[0]
                        print(f"DEBUG - Usando primeira assinatura disponível: {first_sig}")
                        try:
                            output = additional.signatures[first_sig](M_tensor)
                            if isinstance(output, dict):
                                output = list(output.values())[0]
                        except Exception as sig_err:
                            print(f"DEBUG - Erro com primeira assinatura: {sig_err}")

                # Método 2: Tentar __call__ diretamente
                if output is None:
                    try:
                        print(f"DEBUG - Tentando __call__")
                        output = additional.__call__(M_tensor)
                    except Exception as call_err:
                        print(f"DEBUG - Erro no __call__: {call_err}")

                # Método 3: Verificar se é um SavedModel com método direto
                if output is None:
                    try:
                        # SavedModels podem ter um método forward ou similar
                        if hasattr(additional, 'forward'):
                            output = additional.forward(M_tensor)
                        elif hasattr(additional, 'generate'):
                            output = additional.generate(M_tensor)
                    except Exception as method_err:
                        print(f"DEBUG - Erro em método específico: {method_err}")

                if output is None:
                    raise ValueError("Não foi possível chamar o modelo GAN")

                # Converter para numpy
                if hasattr(output, 'numpy'):
                    out = output.numpy()
                else:
                    out = output

                # Extrair primeiro canal se necessário
                if len(out.shape) > 2:
                    out = out[..., 0]

            # Processar a saída
            out = np.round(out)
            out[out == -1] = 100  # roxo
            out[out == 0] = 1000  # verde
            out[out == 1] = 9000  # amarelo
            out = vectorization(out, 'vec', shape=(48, 48))

        except Exception as e:
            print(f"Erro no mapeamento GAN 3F: {e}")
            import traceback
            traceback.print_exc()
            print(f"Shape de M: {M.shape}")
            print(f"Tipo de additional: {type(additional).__name__}")  # CORRIGIDO
            # Fallback: retornar entrada original
            out = M

    elif mapping_type == 'vae_3f':  # MUDEI
        # Assumindo que VAE também pode ser carregado como SavedModel
        try:
            if hasattr(additional, 'predict'):  # Modelo Keras
                out = additional.predict(M.T, verbose=0)[..., 0]
            else:  # Modelo SavedModel
                # Converter para tensor TensorFlow
                M_tensor = tf.convert_to_tensor(M.T, dtype=tf.float32)

                # Chamar o modelo SavedModel (mesma lógica do gan_3f)
                if hasattr(additional, 'signatures'):
                    if 'serving_default' in additional.signatures:
                        output = additional.signatures['serving_default'](latent=M_tensor)
                        if isinstance(output, dict):
                            output = list(output.values())[0]
                    else:
                        first_sig = list(additional.signatures.keys())[0]
                        output = additional.signatures[first_sig](latent=M_tensor)
                        if isinstance(output, dict):
                            output = list(output.values())[0]
                elif callable(additional):
                    output = additional(M_tensor)
                else:
                    output = additional.__call__(M_tensor)

                # Converter para numpy
                if hasattr(output, 'numpy'):
                    out = output.numpy()[..., 0]
                else:
                    out = output[..., 0]

            out = np.round(out)
            out[out == -1] = 100  # roxo
            out[out == 0] = 1000  # verde
            out[out == 1] = 9000  # amarelo
            out = vectorization(out, 'vec', shape=(48, 48))

        except Exception as e:
            print(f"Erro no mapeamento VAE 3F: {e}")
            # Fallback: retornar entrada original
            out = M

    else:
        out = M

    return out