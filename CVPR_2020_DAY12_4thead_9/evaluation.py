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


import sys

# argv[1] must ground truth
readfile = open(sys.argv[1], 'r')
readfile1 = open(sys.argv[2], 'r')

dic_val = {}

m = 0
for line in readfile:
    temp = line.strip('/').split()
    key = temp[0]
    value = int(temp[1])
    dic_val[key] = value
    m = m + 1

n = 0
for line1 in readfile1:
    temp = line1.strip('/').split()
    if temp[0] in dic_val and int(temp[1]) == dic_val[temp[0]]:
        # print int(temp[1]),  dic_val[temp[0]]
        n = n + 1

# print m
# print n
readfile1.close()
readfile2 = open(sys.argv[2], 'r')
rate = float(n) / float(m)
# print "accuracy of top-5: ", rate
print("accuracy of top-5: ", rate)

l = 0
a = 0
for line2 in readfile2:
    a = a + 1
    if a % 5 != 1: continue
    temp = line2.strip('/').split()
    if temp[0] in dic_val and int(temp[1]) == dic_val[temp[0]]:
        l = l + 1
rate1 = float(l) / float(m)
# print "accuracy of top-1: ", rate1
print("accuracy of top-1: ", rate1)
