import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import zlib
import pickle
import base64

fn = 'weak_timing'
data = open(fn, 'r').read()

encoded = [block.split('\n')[0] for block in data.split('data: ')[1:]]
for run in encoded[-1:]:
    data = pickle.loads(zlib.decompress(base64.b64decode(run)))
    iters = data[0]['args']['iters']
    procs = data[0]['size']

    for op in ['P', 'S', 'W', 'WT']:
        times_op_iter = [[] for _ in range(iters)]
        times_comm_iter = [[] for _ in range(iters)]
        for d in data:
            if not d[op]['time_communication_iter']:
                d[op]['time_communication_iter'] = d[op]['time_applies_iter'][
                    1::2]
                d[op]['time_applies_iter'] = d[op]['time_applies_iter'][0::2]

            d[op]['time_applies'] -= d[op]['time_communication']
            d[op]['time_communication_iter'] = np.array(
                d[op]['time_communication_iter'])
            d[op]['time_applies_iter'] = np.array(
                d[op]['time_applies_iter']) - np.array(
                    d[op]['time_communication_iter'])

            for it in range(iters):
                times_op_iter[it].append(d[op]['time_applies_iter'][it])
                times_comm_iter[it].append(
                    d[op]['time_communication_iter'][it])

        # Plot data per iter.
        for it in range(10):
            plt.figure(it)
            plt.scatter(range(procs), times_op_iter[it], label=op + "_apply")
            plt.scatter(range(procs), times_comm_iter[it], label=op + "_comm")

        # Plot aggregated data.
        plt.figure(1337)
        plt.scatter(range(procs), [d[op]['time_applies'] for d in data],
                    label=op + "_apply")
        plt.scatter(range(procs), [d[op]['time_communication'] for d in data],
                    label=op + "_comm")

    for it in range(10):
        plt.figure(it)
        plt.legend()
        plt.yscale('log')
        plt.title('procs: {} iter {}'.format(procs, it + 1))

    plt.figure(1337)
    plt.legend()
    plt.yscale('log')
    plt.title('procs: {} aggregated'.format(procs))
    plt.show()

    # Subtract the communication time from the apply time.
    #for op in ['S', 'W', 'WT', 'WT_S_W']:
    #    for d in data:
    #        d[op]['time_applies'] -= d[op]['time_communication']

    #df = pd.DataFrame(data)

    #print('N', data[0]['N'], 'J_time', data[0]['args']['J_time'])
    #print('M', data[0]['M'])
    #print('Procs', data[0]['size'])
    #print('total_time', data[0]['solve_time'])
    #print('time / S_iters',
    #      data[0]['solve_time'] / data[0]['S']['num_applies'])
    #print('iters', data[0]['iters'])
    #print('S_iters', data[0]['S']['num_applies'])
    #print('P_iters', data[0]['P']['num_applies'])
    #print('W_time_apply {:.3f} {:.3f}'.format(
    #    min(d['W']['time_applies'] for d in data),
    #    max(d['W']['time_applies'] for d in data)))
    #print('W_time_communucation {:.3f} {:.3f}'.format(
    #    min(d['W']['time_communication'] for d in data),
    #    max(d['W']['time_communication'] for d in data)))
    #print('S_time_apply {:.3f} {:.3f}'.format(
    #    min(d['S']['time_applies'] for d in data),
    #    max(d['S']['time_applies'] for d in data)))
    #print('S_time_communucation {:.3f} {:.3f}'.format(
    #    min(d['S']['time_communication'] for d in data),
    #    max(d['S']['time_communication'] for d in data)))
    #print('WT_time_apply {:.3f} {:.3f}'.format(
    #    min(d['WT']['time_applies'] for d in data),
    #    max(d['WT']['time_applies'] for d in data)))
    #print('WT_time_communucation {:.3f} {:.3f}'.format(
    #    min(d['WT']['time_communication'] for d in data),
    #    max(d['WT']['time_communication'] for d in data)))
    #print('P_time_apply {:.3f} {:.3f}'.format(
    #    min(d['P']['time_applies'] for d in data),
    #    max(d['P']['time_applies'] for d in data)))
    #print('')
