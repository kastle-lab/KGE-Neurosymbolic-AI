#!/bin/bash
sbatch pc_learn.slurm   
sleep 1.5
tail -f ~/pc_learn.out