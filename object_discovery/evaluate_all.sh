#!/usr/bin/bash

GQN_DATA_PATH=/workspace/o3d-nerf/data/gqn
GQN_SUFFIX=seed-1_cvr-1e-2
GQN_CKPT_PATH=/workspace/o3d-nerf/slot_attention/checkpoints/gqn_5-slot_${GQN_SUFFIX}
GQN_NUM_EVAL_SLOTS=7
GQN_RESULTS_FILENAME=/workspace/o3d-nerf/slot_attention/eval/gqn-results_${GQN_SUFFIX}_${GQN_NUM_EVAL_SLOTS}-slot-eval.txt
GQN_OUT_PATH=/workspace/o3d-nerf/slot_attention/eval/gqn-images_5-slot_${GQN_SUFFIX}_${GQN_NUM_EVAL_SLOTS}-slot-eval

ARROW_DATA_PATH=/workspace/o3d-nerf/data/arrow
ARROW_SUFFIX=seed-1
ARROW_CKPT_PATH=/workspace/o3d-nerf/slot_attention/checkpoints/arrow_${ARROW_SUFFIX}
ARROW_NUM_EVAL_SLOTS=7
ARROW_RESULTS_FILENAME=/workspace/o3d-nerf/slot_attention/eval/arrow-results_${ARROW_SUFFIX}_${ARROW_NUM_EVAL_SLOTS}-slot-eval.txt
ARROW_OUT_PATH=/workspace/o3d-nerf/slot_attention/eval/arrow-images_${ARROW_SUFFIX}_${ARROW_NUM_EVAL_SLOTS}-slot-eval


OUT_PATH=/workspace/o3d-nerf/slot_attention/eval

GQN_OOD_SPLITS="OOD1 OOD5 OOD6 OODfgbg OODposition OODviews OODviews2 test"
#GQN_OOD_SPLITS="OOD7 OOD8 OOD9"
GQN_RESOLUTION=80

ARROW_OOD_SPLITS="OODsix OODposition OODviews2"
#ARROW_OOD_SPLITS="OODone OODfive OODsix OODcomposition OODposition OODviews OODviews2 test"
#ARROW_OOD_SPLITS="OODseven"
ARROW_RESOLUTION=96

NUM_SCENES=100

set -e


#if [ -f $GQN_RESULTS_FILENAME ] ; then
#    rm $GQN_RESULTS_FILENAME
#fi
#for SPLIT in $GQN_OOD_SPLITS ; do
#PYTHONPATH=.:src python slot_attention/object_discovery/evaluate.py \
#    --data_path $GQN_DATA_PATH \
#    --split $SPLIT \
#    --ckpt_path $GQN_CKPT_PATH \
#    --num_slots $GQN_NUM_EVAL_SLOTS \
#    --resolution $GQN_RESOLUTION \
#    --metrics_filename /tmp/gqn-${SPLIT} \
#    --out_path $GQN_OUT_PATH/$SPLIT
#echo "$SPLIT" >> $GQN_RESULTS_FILENAME
#done
#for SPLIT in $GQN_OOD_SPLITS ; do
#cat /tmp/gqn-${SPLIT}.metrics.txt >> $GQN_RESULTS_FILENAME
#break
#done
#echo mean >> $GQN_RESULTS_FILENAME
#for SPLIT in $GQN_OOD_SPLITS ; do
#cat /tmp/gqn-${SPLIT}.mean.txt >> $GQN_RESULTS_FILENAME
#done
#echo std >> $GQN_RESULTS_FILENAME
#for SPLIT in $GQN_OOD_SPLITS ; do
#cat /tmp/gqn-${SPLIT}.std.txt >> $GQN_RESULTS_FILENAME
#done


if [ -f $ARROW_RESULTS_FILENAME ] ; then
    rm $ARROW_RESULTS_FILENAME
fi
for SPLIT in $ARROW_OOD_SPLITS ; do
PYTHONPATH=.:src python slot_attention/object_discovery/evaluate.py \
    --data_path $ARROW_DATA_PATH \
    --split "$SPLIT" \
    --ckpt_path $ARROW_CKPT_PATH \
    --num_slots $ARROW_NUM_EVAL_SLOTS \
    --num_scenes "$NUM_SCENES" \
    --resolution $ARROW_RESOLUTION \
    --metrics_filename /tmp/arrow-${SPLIT} \
    --out_path $ARROW_OUT_PATH/$SPLIT
echo "$SPLIT" >> $ARROW_RESULTS_FILENAME
done
echo ARROW_OOD_SPLITS: $ARROW_OOD_SPLITS
for SPLIT in $ARROW_OOD_SPLITS ; do
cat /tmp/arrow-${SPLIT}.metrics.txt >> $ARROW_RESULTS_FILENAME
break
done
echo mean >> $ARROW_RESULTS_FILENAME
for SPLIT in $ARROW_OOD_SPLITS ; do
cat /tmp/arrow-${SPLIT}.mean.txt >> $ARROW_RESULTS_FILENAME
done
echo std >> $ARROW_RESULTS_FILENAME
for SPLIT in $ARROW_OOD_SPLITS ; do
cat /tmp/arrow-${SPLIT}.std.txt >> $ARROW_RESULTS_FILENAME
done

