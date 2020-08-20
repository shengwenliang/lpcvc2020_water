#!/bin/sh

TARGET=Ultra96
BOARD=Ultra96
NET_NAME=efficientnet
DEPLOY_MODEL_PATH=quantize_output

sudo mkdir -p /opt/vitis_ai/compiler/arch/dpuv2/Ultra96
sudo cp -f Ultra96.json /opt/vitis_ai/compiler/arch/dpuv2/Ultra96/Ultra96.json
dlet -f dpu.hwh
sudo cp *.dcf /opt/vitis_ai/compiler/arch/dpuv2/${BOARD}/${BOARD}.dcf

ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${TARGET}/${TARGET}.json

vai_c_tensorflow --frozen_pb ${DEPLOY_MODEL_PATH}/deploy_model.pb \
                 --arch ${ARCH} \
		 --output_dir deploy_output/ \
		 --net_name tf_${NET_NAME} 


