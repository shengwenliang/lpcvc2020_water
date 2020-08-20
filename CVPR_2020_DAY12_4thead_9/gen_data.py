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

import os
import json
import re


def gen_dict(name2class_file):
    fr = open(name2class_file, 'r')
    class2name_dict = json.load(fr)
    name2class_dict = {}
    for key in class2name_dict.keys():
        name2class_dict[class2name_dict[key][0]] = key
    return name2class_dict


def main():
    name2class_dict = gen_dict("imagenet_class_index.json")
    fwa = open('image.list.gt', "w")
    directory = './val'
    for xml2code_file in os.listdir(directory):
        with open(f'./val/{xml2code_file}', 'r') as f:
            text = f.read()
            x = re.search(r'<filename>(.*?)</filename>', text).group(1)
            y = re.search(r'<name>(.*?)</name>', text).group(1)
            fwa.writelines(f'{x}.JPEG {name2class_dict[y]}\n')


if __name__ == '__main__':
    main()
