#!/bin/bash
#SBATCH -n 1
#SBATCH -p JX-GPU-IB
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 20G
#SBATCH --gpus-per-task 1

#srun python -u trainer_slurm_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v2_3090_binary
#srun python -u trainer_slurm_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v5_3090_ent_rep_test
#srun python -u trainer_slurm_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v5_3090_ent_rep_test_no_noise_sent
#srun python -u trainer_slurm_fsdp_v1.py -cp conf/t5/wiki -cn merit_v1_large_pv9_v8_2_gen_3090
#srun python -u trainer_slurm_fsdp_v1.py -cp conf/t5/wiki -cn merit_v1_large_pv9_v8_2_gen_3090_v2_0
#python -m torch.distributed.launch --nproc_per_node 2 --nnodes 2  trainer_base_fsdp_v1.py -cp conf/roberta/wiki -cn merit_v1_pv9_v9_v3_3090

srun python -u trainer_slurm_fsdp_v1.py -cp conf/t5/ar-lsat -cn merit_v1_large_pv9_v8_2_gen_v2_2_1


HYDRA_FULL_ERROR=1 srun -N 1 -n 1 -c 4 --gres=gpu:4 --mem 50G -p "JX-GPU" python trainer_slurm_fsdp_v1.py -cp conf/mt5/xl -cn cls_bm25_table_dpr_en_col_en_aug_v1_0_slurm
