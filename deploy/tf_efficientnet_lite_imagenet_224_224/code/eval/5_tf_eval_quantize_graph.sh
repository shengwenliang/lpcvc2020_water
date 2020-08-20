#!/bin/sh

EVAL_SCRIPT_PATH=file
EVAL_MODEL_PATH=quantize_output

python ${EVAL_SCRIPT_PATH}/efficientnet_eval.py \
       --input_frozen_graph quantize_output/quantize_eval_model.pb \
       --input_node images \
       --output_node logits \
       --eval_batches 5000 \
       --batch_size 10 \
       --eval_image_dir /workspace/CBIR/ILSVRC2012_img_val/ \
       --eval_image_list /workspace/CBIR/val.txt \

