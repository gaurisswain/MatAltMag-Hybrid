#!/bin/bash
#SBATCH --job-name=mp-1189260_bands
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --output=bands.out
#SBATCH --error=bands.err

# Adjust module names to match your cluster
module load vasp/6.3.0

cd $SLURM_SUBMIT_DIR
cp ../static/CHGCAR .       # use converged charge density
cp ../static/WAVECAR . 2>/dev/null || true

mpirun -np $SLURM_NTASKS vasp_std > vasp.log 2>&1
echo "VASP finished with exit code $?"
