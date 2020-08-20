#!/bin/sh

vai_q_tensorflow quantize --input_frozen_graph ../../float/frozen.pb \
			  --input_fn file.input_fn.calib_input \
			  --output_dir ./quantize_output \
	                  --input_nodes images \
			  --output_nodes logits \
			  --weight_bit 8 \
			  --input_shapes ?,224,224,3 \
			  --calib_iter 100 
