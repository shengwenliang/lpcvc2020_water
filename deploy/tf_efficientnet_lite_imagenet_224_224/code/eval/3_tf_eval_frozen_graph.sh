#!/bin/sh
#conda activate vitis-ai-tensorflow

EVAL_SCRIPT_PATH=file

python ${EVAL_SCRIPT_PATH}/efficientnet_eval.py \
       --input_frozen_graph ../../float/frozen.pb \
       --input_node images \
       --output_node logits \
       --eval_batches 5000 \
       --batch_size 10 \
       --eval_image_dir /workspace/CBIR/ILSVRC2012_img_val/ \
       --eval_image_list /workspace/CBIR/val.txt \
