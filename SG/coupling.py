import os
import shutil
from distutils.dir_util import copy_tree
import subprocess
import numpy as np
import time
import matplotlib.pyplot as plt
from subprocess import run as nsystem
import re

# from executor_script import run_member
from processing_old import generate_include_full
from concurrent.futures import as_completed, ThreadPoolExecutor


def createdat(fileinput, n):
    # generate .dat file from master
    master = open(fileinput, 'r')
    raw = master.read()
    outtext = raw.replace("\\", '{}'.format(n))
    out = open('./cmgforward/member_{}.dat'.format(n), 'w')
    out.write(outtext)


def write_simulation_times(fileinput, steps, dt, targets=None):
    total_steps = np.arange(dt, steps, dt)

    if targets is not None:
        STL = targets[0]+targets[1]
        STW = targets[3]

    with open(fileinput, "a") as file:
        for t in range(len(total_steps)):
            if targets is None and (t == 0):
                outtext = '*TIME 1\n'
                file.write(outtext)
                outtext = '*OPEN \'P1\' \'P2\' \'P3\' \'P4\' \'P5\' \'P6\' \'P7\' \'P8\' \'P9\'\n'
                file.write(outtext)
                outtext = '*OPEN \'I1\' \'I2\' \'I3\' \'I4\'\n'
                file.write(outtext)
            if targets is not None:
                outtext = '*TARGET STL \'P1\' \'P2\' \'P3\' \'P4\' \'P5\' \'P6\' \'P7\' \'P8\' \'P9\'\n'
                file.write(outtext)
                outtext = re.sub('[\[\]]', '',
                                 np.array2string(STL[t,1:], max_line_width=10000, precision=4, separator=' '))
                file.write('{}\n'.format(outtext))

                outtext = '*TARGET STW \'I1\' \'I2\' \'I3\' \'I4\'\n'
                file.write(outtext)
                outtext = re.sub('[\[\]]', '',
                                 np.array2string(STW[t, 1:], max_line_width=10000, precision=4, separator=' '))
                file.write('{}\n'.format(outtext))

            outtext = '*TIME {}\n'.format(total_steps[t])
            file.write(outtext)



        file.write('*STOP\n')


def createbat(fileinput, version, n):
    # generate .bat file from master
    master = open(fileinput, 'r')
    raw = master.read()
    outtext = raw.replace("$version$", '{}'.format(version))
    outtext = outtext.replace("$$", '{}'.format(n))
    out = open('./cmgforward/member_{}.bat'.format(n), 'w')
    out.write(outtext)


def createrwd(fileinput, n):
    # generate .rwd file from master
    master = open(fileinput, 'r')
    raw = master.read()
    outtext = raw.replace("\\", '{}'.format(n))
    out = open('./cmgforward/member_{}.rwd'.format(n), 'w')
    out.write(outtext)
    out.close()


def run_ensemble_v2(M, maxrun, datfile, rwdfile):
    def prepare_to_run(M, dat_master, rwd_master):
        N = M.shape[-1]
        dir_n = os.listdir('./cmgforward/')
        for item in dir_n:
            try:
                os.remove(os.path.join('./cmgforward/', item))
            except Exception as e:
                print(f"Failed to remove {item}: {e}")

        for n in range(N):
            generate_include_full(M[:, n], n)
            createdat(dat_master, n)  # create .dat file
            createrwd(rwd_master, n)
            write_simulation_times('cmgforward/member_{}.dat'.format(n), 3000, 90, targets=None)

    def run_member(identifier, version):
        datfile = f"member_{identifier}.dat"
        # exe_path = fr"C:/Program Files (x86)/CMG/IMEX/{version}.10/Win_x64/EXE/mx{version}10.exe"
        exe_path = fr"C:\Program Files\CMG\IMEX\{version}.20\Win_x64\EXE\mx{version}20.exe"
        command = [exe_path, "-f", datfile, "-dd", "-wait", "1"]
        # print(command)
        for attempt in range(30):
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                # Verifica se o processo terminou com sucesso
                if process.returncode == 0:
                    # print(f"Member {identifier} done.")
                    pass
                else:
                    print(f"Error in member: {identifier} with following error: {stderr.decode()}")
                if 'Abnormal Termination' in stdout.decode('utf-8'):
                    print(f"'Abnormal Termination' detected. Retrying... ({attempt + 1}/{5})")
                    time.sleep(0.01)  # Aguarda antes de tentar novamente
                else:
                    return
            except Exception as e:
                print(f"Unexpected error: {identifier}: {e}")
        return process.returncode

    def get_member_result(identifier, version):
        infile = f"member_{identifier}.rwd"
        outfile = f"member_{identifier}.rwo"
        # exe_path = fr"C:/Program Files (x86)/CMG/BR/{version}.10/Win_x64/EXE/report.exe"
        exe_path = fr"C:\Program Files\CMG\RESULTS\{version}.20\Win_x64\exe\report.exe"
        command = [exe_path, "-f", infile, "-o", outfile, "-q", "0"]
        # for attempt in range(5):
        #     try:
        #         # Executa o comando de forma assíncrona com subprocess.Popen
        #
        #         process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #         stdout, stderr = process.communicate()
        #         # Verifica se o processo terminou com sucesso
        #         if process.returncode == 0:
        #             print(f"Member {identifier} done.")
        #             pass
        #         else:
        #             print(f"Error in member: {identifier} with following error: {stderr.decode()}")
        #     except Exception as e:
        #         print(f"Unexpected error: {identifier}: {e}")
        while True:  # Keep trying until success
            try:
                # Executa o comando de forma assíncrona com subprocess.Popen
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()

                # Verifica se o processo terminou com sucesso
                if process.returncode == 0:
                    # print(f"Member {identifier} done.")
                    break  # Success, exit the loop
                else:
                    print(f"Error in member: {identifier} with following error: {stderr.decode()}")
                    time.sleep(5)  # Wait before retrying (you can adjust this delay)

            except Exception as e:
                print(f"Unexpected error for member {identifier}: {e}")
                time.sleep(5)  # Wait before retrying (in case of unexpected errors)


    def run_parallel_simulations(total_simulations, max_simulations_at_once, version):
        # Cria um ThreadPoolExecutor com o número máximo de simulações paralelas
        with ThreadPoolExecutor(max_workers=max_simulations_at_once) as executor:
            futures = []

            for identifier in range(total_simulations):
                futures.append(executor.submit(run_member, identifier, version))

            # Acompanha as simulações à medida que são concluídas
            for future in as_completed(futures):
                future.result()
        return futures

    def run_parallel_results(total_simulations, max_simulations_at_once, version):
        # Cria um ThreadPoolExecutor com o número máximo de simulações paralelas
        #with ThreadPoolExecutor(max_workers=max_simulations_at_once) as executor:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for identifier in range(total_simulations):
                futures.append(executor.submit(get_member_result, identifier, version))

            # Acompanha as simulações à medida que são concluídas
            for future in as_completed(futures):
                future.result(timeout=5)
        return futures

    prepare_to_run(M, dat_master=datfile, rwd_master=rwdfile)
    os.chdir('./cmgforward/')
    print('running forward models')
    run_parallel_simulations(M.shape[-1], maxrun, 2023)
    print('running results tool')
    run_parallel_results(M.shape[-1], maxrun, 2023)
    os.chdir('..')