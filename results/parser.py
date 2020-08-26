import pandas as pd

for fn in ["strong_11_9", "weak_x_7", "strong_3d_9_5"]:
    print(fn)
    data = open(fn, 'r').read()

    blocks = [
        'Creating mesh' + block for block in data.split('Creating mesh')
    ][1:]
    datas = []
    for block in blocks:
        data = {}
        data['procs'] = int(block.split('MPI tasks:  ')[1].split('\n')[0])
        data['N'] = int(block.split('N = ')[1].split('.')[0])
        data['M'] = int(block.split('M = ')[1].split('.')[0])
        data['iters'] = int(block.split('Completed in ')[1].split(' PCG')[0])
        data['total-time'] = float(
            block.split('solve time: ')[1].split('s.')[0])
        data['W'] = float(block.split('W:  ')[1].split('\t')[0])
        data['S'] = float(block.split('S:  ')[1].split('\t')[0])
        data['WT'] = float(block.split('WT: ')[1].split('\t')[0])
        data['P'] = float(block.split('P: ')[1].split('\t')[0])
        data['WTSW'] = float(block.split('WTSW: ')[1].split('\t')[0])
        datas.append(data)

    df = pd.DataFrame(data=datas)
    df['time-per-iter'] = df['total-time'] / df['iters']
    print(
        df.to_latex(columns=[
            'procs', 'N', 'M', 'iters', 'total-time', 'time-per-iter'
        ],
                    index=False))
