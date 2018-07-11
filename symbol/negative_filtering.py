#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:31:48 2018

@author: Guo Shuangshuang
"""
import mxnet as mx
import numpy as np


class negative_filtering(mx.operator.CustomOp):
    def __init__(self, ):
        super(negative_filtering, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        arm_cls_preds = in_data[0]
        odm_cls_target = in_data[1]
        odm_loc_target_mask = in_data[2]

        # apply filtering to odm_cls_target
        # arm_cls_preds_shape: (batch, 2, num_anchors)
        arm_cls_preds = mx.nd.softmax(data=arm_cls_preds)
        arm_cls_preds_classes = mx.nd.split(data=arm_cls_preds, axis=1, num_outputs=2)
        arm_cls_preds_bg = mx.nd.Reshape(data=arm_cls_preds_classes[0], shape=(0, -1))
        prob_temp = mx.nd.ones_like(arm_cls_preds_bg) * 0.99
        cond1 = arm_cls_preds_bg >= prob_temp
        temp1 = mx.nd.ones_like(odm_cls_target) * (-1)
        odm_cls_target_mask = mx.nd.where(condition=cond1, x=temp1, y=odm_cls_target)

        # apply filtering to odm_loc_target_mask
        # odm_loc_target_mask_shape: (batch, num_anchors, 4)

        arm_cls_preds_bg = mx.nd.Reshape(data=arm_cls_preds_bg, shape=(0, -1, 1))
        #arm_cls_preds_bg = mx.nd.concatenate([arm_cls_preds_bg]*4, axis=2)
        #odm_loc_target_mask = mx.nd.Reshape(data=odm_loc_target_mask, shape=(0, -1, 4))
        odm_loc_target_mask = mx.nd.reshape(data=odm_loc_target_mask, shape=(0, -1, 4))
        odm_loc_target_mask = odm_loc_target_mask[:, :, 0]
        odm_loc_target_mask = mx.nd.Reshape(data=odm_loc_target_mask, shape=(0, -1, 1))
        loc_temp = mx.nd.ones_like(odm_loc_target_mask) * 0.99
        cond2 = arm_cls_preds_bg >= loc_temp
        temp2 = mx.nd.zeros_like(odm_loc_target_mask)
        odm_loc_target_bg_mask = mx.nd.where(condition=cond2, x=temp2, y=odm_loc_target_mask)
        odm_loc_target_bg_mask = mx.nd.concatenate([odm_loc_target_bg_mask]*4, axis=2)
        odm_loc_target_bg_mask = mx.nd.Reshape(data=odm_loc_target_bg_mask, shape=(0, -1))
        #print(odm_loc_target_mask[0, 0:40])

        for ind, val in enumerate([odm_cls_target_mask, odm_loc_target_bg_mask]):
            self.assign(out_data[ind], req[ind], val)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("negative_filtering")
class negative_filteringProp(mx.operator.CustomOpProp):
    def __init__(self,):
        super(negative_filteringProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['arm_cls_preds', 'odm_cls_target', 'odm_loc_target_mask']

    def list_outputs(self):
        return ['odm_cls_target_mask', 'odm_loc_target_bg_mask']

    def infer_shape(self, in_shape):
        arm_cls_preds_shape = in_shape[0]
        odm_cls_target_shape = in_shape[1]
        odm_loc_target_mask_shape = in_shape[2]
        odm_cls_target_mask_shape = [odm_cls_target_shape[0], odm_cls_target_shape[1]]
        odm_loc_target_bg_mask_shape = [odm_loc_target_mask_shape[0], odm_loc_target_mask_shape[1]]

        return [arm_cls_preds_shape, odm_cls_target_shape, odm_loc_target_mask_shape], [odm_cls_target_mask_shape, odm_loc_target_bg_mask_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return negative_filtering()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
