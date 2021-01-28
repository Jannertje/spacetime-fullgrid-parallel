import argparse
import base64
import pickle
import sys
import zlib

import numpy as np
from mpi4py import MPI

from heateq_mpi import HeatEquationMPI, mem
from source.mpi_vector import KronVectorMPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time several components of parallel heatequation.")
    parser.add_argument('--problem',
                        default='square',
                        help='problem type (square, ns)')
    parser.add_argument('--J_time',
                        type=int,
                        default=7,
                        help='number of time refines')
    parser.add_argument('--J_space',
                        type=int,
                        default=7,
                        help='number of space refines')
    parser.add_argument('--smoothsteps',
                        type=int,
                        default=3,
                        help='number of smoothing steps')
    parser.add_argument('--vcycles',
                        type=int,
                        default=2,
                        help='number of vcycles')
    parser.add_argument('--wavelettransform',
                        default='original',
                        help='type of wavelettransform')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('--iters',
                        type=int,
                        default=10,
                        help='number of iterations per operator')

    args = parser.parse_args()
    J_time = args.J_time
    J_space = args.J_space

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    data = {'rank': rank, 'size': size}
    if size > 2**J_time + 1:
        print('Too many MPI processors!')
        sys.exit('1')

    heat_eq_mpi = HeatEquationMPI(J_space=J_space,
                                  J_time=J_time,
                                  problem=args.problem,
                                  smoothsteps=args.smoothsteps,
                                  vcycles=args.vcycles,
                                  alpha=args.alpha,
                                  wavelettransform=args.wavelettransform)
    if rank == 0:
        data['args'] = vars(args)
        data['N'] = heat_eq_mpi.N
        data['M'] = heat_eq_mpi.M
        print('\n\nCreating mesh with {} time refines and {} space refines.'.
              format(J_time, J_space))
        print('MPI tasks: ', size)
        print('Arguments:', args)
        print('N = {}. M = {}.'.format(heat_eq_mpi.N, heat_eq_mpi.M))
        print('Constructed bilinear forms in {} s.'.format(
            heat_eq_mpi.setup_time))
        print('Memory after construction: {}mb.'.format(mem()))

    data['mem_after_construction'] = mem()

    MPI.COMM_WORLD.Barrier()
    time_total = MPI.Wtime()

    # Time the four operors separately.
    vec = KronVectorMPI(heat_eq_mpi.dofs_distr)
    np.random.seed(128)
    vec.X_loc[:] = np.random.rand(*vec.X_loc.shape)
    for name, op in [('W', heat_eq_mpi.W), ('S', heat_eq_mpi.S),
                     ('WT', heat_eq_mpi.WT), ('P', heat_eq_mpi.P)]:
        time_applies_iter = []
        time_communication_iter = []
        time_total_op = MPI.Wtime()

        for _ in range(args.iters):
            t_a = op.time_applies
            t_c = op.time_communication

            # Apply the operator.
            vec._invalidate()
            op @ vec

            time_applies_iter.append(op.time_applies - t_a)
            time_communication_iter.append(op.time_communication - t_c)

            # Wait for all other ops to be done as well.
            MPI.COMM_WORLD.Barrier()

        data[name] = {
            'time_applies': op.time_applies,
            'time_communication': op.time_communication,
            'time_applies_iter': time_applies_iter,
            'time_communication_iter': time_communication_iter,
            'num_applies': op.num_applies,
            'time_total': MPI.Wtime() - time_total_op
        }

    MPI.COMM_WORLD.Barrier()
    data['time_total'] = MPI.Wtime() - time_total
    data['mem_after_timing'] = mem()

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('')
        print('Completed {} iters steps.'.format(args.iters))
        print('Total time: {}s.'.format(data['time_total']))
        heat_eq_mpi.print_time_per_apply()
        print('Memory after solve: {}mb.'.format(mem()))

    data = MPI.COMM_WORLD.gather(data, root=0)
    if rank == 0:
        print('\ndata: {}'.format(
            str(base64.b64encode(zlib.compress(pickle.dumps(data))), 'ascii')))
