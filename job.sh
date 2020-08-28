#!/bin/bash
#SBATCH -t 0:04:00
#SBATCH -p normal
#SBATCH -n 24


module load 2019
module load Python/3.7.5-foss-2018b

J_time=8
J_space=8
problem=square
set -e

srun python3 $HOME/spacetime/mpi_heateq_shared.py --J_time=$J_time --J_space=$J_space  --wavelettransform=composite --problem=$problem
