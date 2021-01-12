import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import zlib
import pickle
import base64

for fn in [
        'strong_3d_16_procs', 'strong_2d_16_procs', 'weak_2d_16_procs',
        'weak_3d_16_procs'
]:
    data = open(fn, 'r').read()

    encoded = [block.split('\n')[0] for block in data.split('data: ')[1:]]
    datas = []
    for run in encoded:
        data = pickle.loads(zlib.decompress(base64.b64decode(run)))
        procs = data[0]['size']
        N = data[0]['N']
        M = data[0]['M']
        iters = data[0]['iters']
        solve_time = data[0]['solve_time']
        datas.append({
            'procs': data[0]['size'],
            'N': data[0]['N'],
            'M': data[0]['M'],
            '#dofs': data[0]['N'] * data[0]['M'],
            'iters': data[0]['iters'],
            'solve-time': data[0]['solve_time']
        })
        #        plt.figure(procs)
        #        plt.scatter(range(procs), [d['solve_time'] for d in data],
        #                    label="solve_time")

        # Subtract the communication time from the apply time.
        for op in ['P', 'S', 'W', 'WT']:
            for d in data:
                d[op]['time_applies'] -= d[op]['time_communication']

            # Plot aggregated data.
            plt.figure(procs)
            plt.scatter(range(procs), [d[op]['time_applies'] for d in data],
                        label=op + "_apply")
            plt.scatter(range(procs),
                        [d[op]['time_communication'] for d in data],
                        label=op + "_comm")
        continue

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

        plt.figure(procs)
        plt.legend()
        plt.yscale('log')
        plt.title('procs: {} aggregated'.format(procs))

    df = pd.DataFrame(data=datas)
    df['solve-time-per-iter'] = df['solve-time'] / df['iters']
    df['total-cpu-time'] = df['solve-time'] * df['procs'] / 3600.0
    df.sort_values('procs', inplace=True)
    df = df.round(2)
    print(fn.replace('_', ' '))
    print(
        df.to_latex(columns=[
            'procs', 'N', 'M', '#dofs', 'iters', 'solve-time',
            'solve-time-per-iter', 'total-cpu-time'
        ],
                    index=False))
    #plt.show()
