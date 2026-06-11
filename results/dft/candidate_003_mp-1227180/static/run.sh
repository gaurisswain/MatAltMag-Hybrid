#!/bin/bash
#SBATCH --job-name=mp-1227180_static
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --output=static.out
#SBATCH --error=static.err

# Adjust module names to match your cluster
module load vasp/6.3.0

cd $SLURM_SUBMIT_DIR
cp ../relax/CONTCAR POSCAR  # use relaxed structure

mpirun -np $SLURM_NTASKS vasp_std > vasp.log 2>&1
echo "VASP finished with exit code $?"
