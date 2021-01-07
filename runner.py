import os
import time

now = time.time()
for J_space in [2, 3, 4, 5, 6]:
    for J_time in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for alpha in [0.3]:  #, 0.5, 1.0]:
            for precond in ['direct']:  #, 'mg']:
                string = 'python3 mpi_heateq.py --kappa2 --no-solve --J_time=%d --J_space=%d --alpha=%f --precond=%s' % (
                    J_time, J_space, alpha, precond)
                if precond == 'direct':
                    os.system(string + " >> output_%s.txt" % now)
                elif precond == 'mg':
                    for vcycles in [1, 2, 3]:
                        for smoothsteps in [1, 2, 3]:
                            string += ' --vcycles=%d --smoothsteps=%d' % (
                                vcycles, smoothsteps)
                            os.system(string + " >> output_%s.txt" % now)
