# Parallel and Sequential Primality Testing

# How to Build

## Miller-Rabin
First run `make clean`. Then, run `make seq` or `make parallel` depending on which algorithm you want to run. To run on your machine, run `make run`.

## SPRP Testing

First run `module load cuda` if on TACC and `make clean`. Then, run `make seq` or `make parallel` depending on which algorithm you want to run. To run on your machine, run `make run`. If on TACC, run `sbatch seq` for sequential `sbatch parallel` for parallel.
