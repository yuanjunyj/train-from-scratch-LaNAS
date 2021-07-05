# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

for (( c=0; c < 16; c++ ))
do
   echo "---------------------------------"
   echo $PWD
   cp -rf "clientX" "client$c"
   cd "client$c"
   echo $PWD
   nohup CUDA_VISBLE_DEVICES=$c python client.py &
   # nohup echo "$c" &
   touch $!".pid"
   cd ".."
   echo "$PWD"
done
