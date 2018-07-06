#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:31:48 2018

@author: Guo Shuangshuang
"""
import mxnet as mx
import numpy as np


class modify_label(mx.operator.CustomOp):
    def __init__(self, ):
        super(modify_label, self).__init__()


    def forward(self, is_train, req, in_data, out_data, aux):

        label = in_data[0]
        #labels = label.asnumpy()
        #print('label', labels[labels[:,:,0]==0])
        label_dif = mx.nd.slice_axis(label, axis=2, begin=5, end=6)
        label_cls = mx.nd.slice_axis(label, axis=2, begin=0, end=1)
        label_loc = mx.nd.slice_axis(label, axis=2, begin=1, end=5)
        temp = mx.nd.zeros_like(label_cls)
        condition = label_cls >= temp
        label_cls = mx.nd.where(condition=condition, x=temp, y=label_cls)
        label_arm = mx.nd.concat(label_cls, label_loc, label_dif, dim=2)

        #print('label_arm',label_arm[0, 0:10, :])
        self.assign(out_data[0], req[0], label_arm)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)



@mx.operator.register("modify_label")
class modify_labelProp(mx.operator.CustomOpProp):
    def __init__(self,):
        super(modify_labelProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['label']

    def list_outputs(self):
        return ['label_arm']

    def infer_shape(self, in_shape):
        label_shape = in_shape[0]

        out_shape = label_shape

        return [label_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return modify_label()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
