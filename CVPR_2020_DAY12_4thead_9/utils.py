#  Copyright (C) 2020 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
from pynq_dpu import DpuOverlay
from ctypes import *
import cv2
import numpy as np
from dnndk import n2cube
import os
import threading
import time
import math
from pynq_dpu import dputils
        
KERNEL_CONV = "tf_efficientnet"
KERNEL_CONV_INPUT = "efficientnet_lite0_model_stem_conv2d_Conv2D"
KERNEL_FC_OUTPUT = "efficientnet_lite0_model_head_dense_MatMul"

image_folder = "/home/xilinx/holdout_images/"

IMAGE_SIZE = 224
CROP_PADDING = 32

BASE_DIR = "."
RESULT_DIR = BASE_DIR
RESULT_FILE = RESULT_DIR + '/image.list.result'

lock = threading.Lock()


class Processor:
    def __init__(self):
        pass

    # User should rewrite this function to run their data set and save output to result file.
    # Result file name should be "image.list.result" and be saved in the main directory
    
    def run_dpu_task(self, kernel, i, iter_cnt, listimage, result):
        start = 0
        count = 0
        task = n2cube.dpuCreateTask(kernel, 0)
        pre_time = 0
        calc_time = 0
        while count < iter_cnt:
            img_name = listimage[i].pop(0)
            path = os.path.join(image_folder, img_name)
            im = cv2.imread(path)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            padded_center_crop_size = int((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) * min(h, w))
            offset_height = ((h - padded_center_crop_size) + 1) // 2
            offset_width = ((w - padded_center_crop_size) + 1) // 2
            image_crop = im[offset_height: padded_center_crop_size + offset_height, offset_width: padded_center_crop_size + offset_width,:]
            image = cv2.resize(image_crop, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            im = cv2.normalize(image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            count = count + 1
            input_len = n2cube.dpuGetInputTensorSize(task, KERNEL_CONV_INPUT)
            n2cube.dpuSetInputTensorInHWCFP32(task, KERNEL_CONV_INPUT, im, input_len)
            
            n2cube.dpuRunTask(task)
            size = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)
            channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)
            conf = n2cube.dpuGetOutputTensorAddress(task, KERNEL_FC_OUTPUT)
            outputScale = n2cube.dpuGetOutputTensorScale(task, KERNEL_FC_OUTPUT)
            softmax = n2cube.dpuRunSoftmax(conf, channel, size//channel, outputScale)
            
            idx   = np.argpartition(softmax, -5)[-5:]
            top_5 = idx[np.argsort(-softmax[idx])]
            for val in top_5:
                result[i].append("{} {}".format(img_name, val))
        n2cube.dpuDestroyTask(task)
            
    
    def run(self):
        overlay = DpuOverlay("./bitstream/dpu.bit")
        overlay.load_model("./model/dpu_tf_efficientnet.elf")
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        threadnum = 4
        num_iterations = 0
        listimage = [[] * i for i in range(threadnum)]
        result = [[] * i for i in range(threadnum)]
        img_processed = [[] * i for i in range(threadnum)]
        
        cnt = 0
        thread = 0
        list_image = sorted([i for i in os.listdir(image_folder) if i.endswith("JPEG")])
        picture_num = 0
        picture_num = len(list_image)
        for i in list_image:
            listimage[thread].append(i)
            if cnt % math.ceil(picture_num/threadnum) == 0 and cnt != 0:
                thread = thread + 1
            cnt = cnt + 1
        
        n2cube.dpuOpen()
        kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
        threadAll = []
        for i in range(threadnum):
            t1 = threading.Thread(target=self.run_dpu_task, args=(kernel, i, len(listimage[i]), listimage, result))
            threadAll.append(t1)
        for x in threadAll:
            x.start()
        for x in threadAll:
            x.join()               

        with open(RESULT_FILE, 'w') as result_file:
            for item in result:
                for i in item:
                    result_file.write("%s\n" % i)
        
        rtn = n2cube.dpuDestroyKernel(kernel)
        n2cube.dpuClose()
        # Run all date set and write your outputs to result file.
        # Please see README and "classification_result.sample" to know the result file format.
        #time.sleep(10)

        return
