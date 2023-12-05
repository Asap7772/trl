import os
import pprint
import shutil
import multiprocessing as mp

path_to_rm = '/iris/u/asap7772/trl/sft_best_preference_multi/'
keep_ub = 10000
num_processes = 8

all_checkpoints = os.listdir(path_to_rm)
filtered_checkpoints = filter(lambda x: x.startswith('checkpoint') and int(x.split('-')[-1]) > keep_ub, all_checkpoints)

def remove_checkpoint(checkpoint):
    path = os.path.join(path_to_rm, checkpoint)
    shutil.rmtree(path)

with mp.Pool(processes=min(num_processes, mp.cpu_count())) as pool:
    pool.map(remove_checkpoint, filtered_checkpoints)