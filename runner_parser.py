import os
import stat
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "."

files_sorted_by_date = []
filepaths = [os.path.join(path, file) for file in os.listdir(path)]
file_statuses = [(os.stat(filepath), filepath) for filepath in filepaths]
files = ((status[stat.ST_CTIME], filepath)
         for status, filepath in file_statuses
         if stat.S_ISREG(status[stat.ST_MODE]))
for creation_time, filepath in sorted(files):
    creation_date = time.ctime(creation_time)
    filename = os.path.basename(filepath)
    files_sorted_by_date.append(creation_date + " " + filename)

latestfile = None
for fn in files_sorted_by_date[::-1]:
    fn = fn.split(" ")[-1]
    if fn.startswith('output_'):
        latestfile = fn
        break
assert latestfile is not None
latestfile = 'output_1600168236.5840044.txt'

runs = []
with open(latestfile) as f:
    lines = f.readlines()
    argsstarts = []
    lancstarts = []
    for i, line in enumerate(lines):
        if line.startswith("Arguments"):
            argsstarts.append(i)
        elif line.startswith("Lanczos"):
            lancstarts.append(i)

    important = [(lines[args] + ", " + lines[args + 1].replace('.', ','),
                  lines[lancs])
                 for (args, lancs) in zip(argsstarts, lancstarts)]

    for (args, lanczos) in important:
        arglist = [arg.split("=") for arg in args[21:-2].split(', ')]
        argdict = {arg[0].strip(): arg[1].strip() for arg in arglist}
        begin = 23 if 'NOT' in lanczos else 20
        lanclist = [lanc.split("=") for lanc in lanczos[begin:-2].split()]
        lancdict = {lanc[0].strip(): lanc[1].strip() for lanc in lanclist}
        print(lancdict)
        runs.append((argdict, lancdict))

df = pd.DataFrame([{**argdict, **lancdict} for (argdict, lancdict) in runs])
cols = df.columns
for c in cols:
    try:
        df[c] = pd.to_numeric(df[c])
    except:
        pass
df.drop_duplicates(
    subset=['precond', 'J_time', 'J_space', 'alpha', 'vcycles', 'smoothsteps'])
df.to_excel('pandas.xls')

#plt.figure(figsize=(2, 2))
#plt.xlabel(r"$\alpha$")
#plt.plot([
#    alpha for alpha, kappa in zip(df['alpha'], df['kappa'])
#    if alpha not in [0.02] and 0.1 <= alpha
#] + [1.0], [
#    kappa for alpha, kappa in zip(df['alpha'], df['kappa'])
#    if alpha not in [0.02] and 0.1 <= alpha
#] + [50.33396527889129], 'k-')
#plt.title(r"$\kappa_2({\bf K}_X \hat {\bf S})$")
#plt.xticks(np.arange(0, 1.2, step=0.2))
#plt.tight_layout()
#plt.show()
#1 / 0
df['time_per_it'] = df['time'] / df['its']

P = lambda y: df.pivot_table(index=y + ['vcycles'],
                             columns='smoothsteps',
                             values=['kappa', 'time_per_it'])
L = lambda x: x.to_latex(float_format=lambda x: "%.2f" % abs(x)
                         if x == x else '\Vhrulefill',
                         escape=False)
print(L(P([])))
#print(
#    L((P((df['precond'] == "'mg'")
#         & (df['alpha'] == 0.3), ['vcycles', 'smoothsteps']) - P(
#             (df['precond'] == "'direct'") & (df['alpha'] == 0.3), []))))
