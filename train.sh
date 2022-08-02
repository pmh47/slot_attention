#!/usr/bin/bash

cd `dirname "$(readlink -f "$0")"`/.. # i.e. change to the directory above that containing this script
export PYTHONPATH=.
exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/arrow_seed-3_cvr-v3-2e-2 --dataset ood --data_path data/arrow --resolution 96 --num_slots 5 --max_num_frames 10 --seed 3 --cpt_var_reg_weight 2e-2
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/arrow_seed-2 --dataset ood --data_path data/arrow --resolution 96 --num_slots 5 --max_num_frames 10 --seed 2
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/arrow_seed-3_cvr-1e-1 --dataset ood --data_path data/arrow --resolution 96 --num_slots 5 --max_num_frames 10 --seed 3 --cpt_var_reg_weight 1.e-1
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/gqn_5-slot_seed-3_cvr-1e-2 --dataset ood --data_path data/gqn --resolution 80 --num_slots 5 --seed 3 --cpt_var_reg_weight 1.e-2

# For training on original clevr data (requires tensorflow_datasets installed):
#export TFDS_DATA_DIR=/workspace/tensorflow_datasets
#exec python slot_attention/object_discovery/train.py  --model_dir slot_attention/checkpoints/object_discovery
