#!/usr/bin/bash

GQN_DATA_PATH=/root/workspace/data/gqn
GQN_CKPT_PATH=/root/workspace/slot_attention/checkpoints/gqn_2022-04-13
GQN_RESULTS_FILENAME=./gqn_slot-attn_results.txt
GQN_NUM_SLOTS=5

ARROW_DATA_PATH=/root/workspace/data/arrow
ARROW_CKPT_PATH=/root/workspace/slot_attention/checkpoints/arrow_2022-04-19
ARROW_RESULTS_FILENAME=./arrow_slot-attn_results.txt
ARROW_NUM_SLOTS=5


GQN_OOD_SPLITS="OOD1 OOD5 OOD6 OOD7 OOD8 OOD9 OODfgbg OODposition OODviews OODviews2 test"
GQN_RESOLUTION=80

ARROW_OOD_SPLITS="OODviews OODposition OODseven OODsix OODfive OODone OODcomposition test"
ARROW_RESOLUTION=96

NUM_SCENES=100

set -e

if [ -f $GQN_RESULTS_FILENAME ] ; then
    rm $GQN_RESULTS_FILENAME
fi

if [ -f $ARROW_RESULTS_FILENAME ] ; then
    rm $ARROW_RESULTS_FILENAME
fi


for SPLIT in $GQN_OOD_SPLITS ; do
PYTHONPATH=.:src python slot_attention/object_discovery/evaluate.py \
    --data_path $GQN_DATA_PATH \
    --split "$SPLIT" \
    --ckpt_path $GQN_CKPT_PATH \
    --num_slots $GQN_NUM_SLOTS \
    --num_scenes "$NUM_SCENES" \
    --resolution $GQN_RESOLUTION \
    --metrics_filename /tmp/results.txt \
    --out_path slot_attention/eval_output/gqn/"$SPLIT"
echo "$SPLIT:" >> $GQN_RESULTS_FILENAME
cat /tmp/results.txt >> $GQN_RESULTS_FILENAME
done


for SPLIT in $ARROW_OOD_SPLITS ; do
PYTHONPATH=.:src python slot_attention/object_discovery/evaluate.py \
    --data_path $ARROW_DATA_PATH \
    --split "$SPLIT" \
    --ckpt_path $ARROW_CKPT_PATH \
    --num_slots $ARROW_NUM_SLOTS \
    --num_scenes "$NUM_SCENES" \
    --resolution $ARROW_RESOLUTION \
    --metrics_filename /tmp/results.txt \
    --out_path slot_attention/eval_output/arrow/"$SPLIT"
echo "$SPLIT:" >> $ARROW_RESULTS_FILENAME
cat /tmp/results.txt >> $ARROW_RESULTS_FILENAME
done

