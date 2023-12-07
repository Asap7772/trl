import os
import pprint
import shutil
import multiprocessing as mp
from functools import reduce

path_to_rm = '/iris/u/asap7772/trl/exp_checkpoints/'
keep_ub = 10
num_processes = 8

all_checkpoints = [x[0] for x in os.walk(path_to_rm)]
print('All checkpoints:', all_checkpoints)
filtered_checkpoints = filter(lambda x: 'epoch_' in x and int(x.split('epoch_')[-1]) > keep_ub, all_checkpoints)
print('Filtered checkpoints:', list(filtered_checkpoints))

def remove_checkpoint(checkpoint):
    path = os.path.join(path_to_rm, checkpoint)
    shutil.rmtree(path)

with mp.Pool(processes=min(num_processes, mp.cpu_count())) as pool:
    pool.map(remove_checkpoint, filtered_checkpoints)