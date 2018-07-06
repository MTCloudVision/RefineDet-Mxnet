#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python -u train.py --network vgg16_reduced --train-path ./data/train.rec --val-path ./data/val.rec \
 --resume -1 --batch-size 32 --lr 0.001 --label-width 380 --num-class 20  --pretrained ./model/vgg16_reduced --epoch 0 \
 --prefix checkpoint/refinedet \
 --gpus 2,3  --data-shape 320 --log logs/refinedet_train_vgg_voc.log  \
 --class-names 'aeroplane, bicycle, bird, boat,bottle,bus,car,cat,chair,cow,diningtable,dog, horse, motorbike,person, pottedplant,sheep,sofa,train,tvmonitor'
