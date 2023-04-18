#!/bin/bash
#BSUB -nnodes 9
#BSUB -W 2:00
#BSUB -q batch
#BSUB -o magma_pythia70m_out.%J
#BSUB -e magma_pythia70m_err.%J
#BSUB -J magma_pythia70m
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499

source /gpfs/alpine/csc499/proj-shared/env_setup/setup.sh

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/scratch/$(whoami)/csc499/cache/torch_extensions

# Write the hostfile for this job
cat $LSB_DJOB_HOSTFILE | sort | uniq | tail -n +2 | sed -e 's/$/ slots=6/' > /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts
export DLTS_HOSTFILE=/ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts

NNODES=$(wc -l < /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts)

export WANDB_DIR=/gpfs/alpine/scratch/$(whoami)/csc499/wandb
export WANDB_MODE=dryrun

if [ ! -e configs/summit_clipH_pythia70m_$NNODES.yml ]; then
	cp configs/summit_clipH_pythia70m_template.yml configs/summit_clipH_pythia70m_$NNODES.yml
	sed -i "s/{{NODES}}/$NNODES/g" configs/summit_clipH_pythia70m_$NNODES.yml
	sed -i "s/{{USER}}/$(whoami)/g" configs/summit_clipH_pythia70m_$NNODES.yml
fi

if [ ${1:-1} = "-l" ]
  then
    deepspeed -H /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts train.py --config summit_clipH_pythia70m_$NNODES.yml >  log_$NNODES\_$sec.log 2>&1 &
  else
    deepspeed -H /ccs/home/$(whoami)/scratch/hostfiles/$LSB_JOBID-hosts train.py --config summit_clipH_pythia70m_$NNODES.yml
fi

