from collections import defaultdict

runs = defaultdict(list)
with open('results/strong_11_9') as f:
    lines = f.readlines()
    starts = [0]
    for i, line in enumerate(lines):
        if len(line.strip()) == 0 and len(lines[i + 1].strip()) == 0:
            starts.append(i + 2)
    starts.append(len(lines))
    for i, start in enumerate(starts[:-1]):
        mylines = lines[start:starts[i + 1]]
        tasks = int(mylines[1].strip().split()[-1])
        wavelettransform = mylines[2].split("'")[-2]
        solvetime = float(mylines[12].split()[-1][:-3])
        Wtime = float(mylines[13].split()[-2])
        Stime = float(mylines[14].split()[-2])
        WTtime = float(mylines[15].split()[-2])
        Ptime = float(mylines[16].split()[-2])
        WTSWtime = float(mylines[18].split()[-2])
        runs[wavelettransform].append(
            (tasks, solvetime, Wtime, Stime, WTtime, Ptime, WTSWtime))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2)
for ax, (col, name) in zip(axes, zip([1, 6], ['solve', 'WT S W'])):
    ax.set_title(name)
    for wavelettransform in runs:
        myrun = runs[wavelettransform]
        ax.loglog([run[0] for run in myrun],
                  [myrun[0][col] / run[col] for run in myrun],
                  label=wavelettransform)
    ax.loglog([run[0] for run in myrun],
              [run[0] / myrun[0][0] for run in myrun],
              'k--',
              label="optimal")
    ax.set_ylabel('speedup wrt smallest')
    ax.set_xlabel('n tasks')
    ax.legend()
plt.show()
