import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import zlib
import pickle
import base64

fn = 'weak_timing_3'
data = open(fn, 'r').read()

encoded = [block.split('\n')[0] for block in data.split('data: ')[1:]]
for run in encoded:
    data = pickle.loads(zlib.decompress(base64.b64decode(run)))
    iters = data[0]['args']['iters']
    procs = data[0]['size']

    print('\nprocs: {}'.format(procs))
    print('N: {}\tJ_time: {}'.format(data[0]['N'], data[0]['args']['J_time']))
    print('M: {}\tJ_space: {}'.format(data[0]['M'],
                                      data[0]['args']['J_space']))
    print('total_time: {:.3f}'.format(data[0]['time_total']))
    for op in ['P', 'S', 'W', 'WT']:
        times_op_iter = [[] for _ in range(iters)]
        times_comm_iter = [[] for _ in range(iters)]
        print('total_time {}: {:.3f}'.format(op, data[0][op]['time_total']))
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
        plt.figure(procs)
        plt.scatter(range(procs), [d[op]['time_applies'] for d in data],
                    label=op + "_apply")
        plt.scatter(range(procs), [d[op]['time_communication'] for d in data],
                    label=op + "_comm")

    for it in range(10):
        plt.figure(it)
        plt.legend()
        plt.yscale('log')
        plt.title('procs: {} iter {}'.format(procs, it + 1))

    plt.figure(procs)
    plt.legend()
    plt.yscale('log')
    plt.title('procs: {} aggregated'.format(procs))
    plt.show()
