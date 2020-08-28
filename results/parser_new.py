import pandas as pd
from pprint import pprint
import zlib
import pickle
import base64

fn = 'weak_x_9_cart'
data = open(fn, 'r').read()

encoded = [block.split('\n')[0] for block in data.split('data: ')[1:]]
for run in encoded:
    data = pickle.loads(zlib.decompress(base64.b64decode(run)))
    # Subtract the communication time from the apply time.
    for op in ['S', 'W', 'WT', 'WT_S_W']:
        for d in data:
            d[op]['time_applies'] -= d[op]['time_communication']

    df = pd.DataFrame(data)

    print('N', data[0]['N'], 'J_time', data[0]['args']['J_time'])
    print('M', data[0]['M'])
    print('Procs', data[0]['size'])
    print('total_time', data[0]['solve_time'])
    print('time / S_iters',
          data[0]['solve_time'] / data[0]['S']['num_applies'])
    print('iters', data[0]['iters'])
    print('S_iters', data[0]['S']['num_applies'])
    print('P_iters', data[0]['P']['num_applies'])
    print('W_time_apply {:.3f} {:.3f}'.format(
        min(d['W']['time_applies'] for d in data),
        max(d['W']['time_applies'] for d in data)))
    print('W_time_communucation {:.3f} {:.3f}'.format(
        min(d['W']['time_communication'] for d in data),
        max(d['W']['time_communication'] for d in data)))
    print('S_time_apply {:.3f} {:.3f}'.format(
        min(d['S']['time_applies'] for d in data),
        max(d['S']['time_applies'] for d in data)))
    print('S_time_communucation {:.3f} {:.3f}'.format(
        min(d['S']['time_communication'] for d in data),
        max(d['S']['time_communication'] for d in data)))
    print('WT_time_apply {:.3f} {:.3f}'.format(
        min(d['WT']['time_applies'] for d in data),
        max(d['WT']['time_applies'] for d in data)))
    print('WT_time_communucation {:.3f} {:.3f}'.format(
        min(d['WT']['time_communication'] for d in data),
        max(d['WT']['time_communication'] for d in data)))
    print('P_time_apply {:.3f} {:.3f}'.format(
        min(d['P']['time_applies'] for d in data),
        max(d['P']['time_applies'] for d in data)))
    print('')
