"""
Created on Wed Jul  4 13:31:48 2018
Modified by: Guo Shuangshuang
"""
from symbol.common import multi_layer_feature, multibox_layer
import refine_anchor_generator
import modify_label
import negative_filtering
import mxnet as mx
def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)

def get_symbol_train(network, num_classes, from_layers, num_filters,
                     sizes, ratios, batch_size, gpus, normalizations=-1, steps=[], nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    times = batch_size / gpus
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers)
    arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = multibox_layer(layers, num_filters, num_classes, \
                sizes=sizes, ratios=ratios, normalization=normalizations, num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # modify arm label
    label_arm = mx.symbol.Custom(label=label, op_type='modify_label')

    arm_tmp = mx.contrib.symbol.MultiBoxTarget(*[arm_anchor_boxes, label_arm, arm_cls_preds], overlap_threshold=.5, \
                                            ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
                                            negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2), name="arm_multibox_target")
    arm_loc_target = arm_tmp[0]
    arm_loc_target_mask = arm_tmp[1]
    arm_cls_target = arm_tmp[2]

    # odm module
    odm_anchor_boxes = mx.symbol.Custom(arm_anchor_boxes=arm_anchor_boxes, arm_loc_preds=arm_loc_preds, arm_loc_mask=arm_loc_target_mask,\
                                        op_type='refine_anchor_generator')
    odm_anchor_boxes_bs = mx.sym.split(data=odm_anchor_boxes, axis=0, num_outputs=times)
    odm_loc_target = []
    odm_loc_target_mask = []
    odm_cls_target = []
    label_bs = mx.sym.split(data=label, axis=0, num_outputs=times)
    odm_cls_preds_bs = mx.sym.split(data=odm_cls_preds, axis=0, num_outputs=times)
    for i in range(times):
        odm_tmp = mx.contrib.symbol.MultiBoxTarget(*[odm_anchor_boxes_bs[i], label_bs[i], odm_cls_preds_bs[i]],\
                                    overlap_threshold=.5, ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0,\
                                    negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2), name="odm_multibox_target_{}".format(i))
        odm_loc_target.append(odm_tmp[0])
        odm_loc_target_mask.append(odm_tmp[1])
        odm_cls_target.append(odm_tmp[2])

    odm_loc_target = mx.symbol.concat(*odm_loc_target, num_args=len(odm_loc_target), dim=0)
    odm_loc_target_mask = mx.symbol.concat(*odm_loc_target_mask, num_args=len(odm_loc_target_mask), dim=0)
    odm_cls_target = mx.symbol.concat(*odm_cls_target, num_args=len(odm_cls_target), dim=0)

    group = mx.symbol.Custom(arm_cls_preds=arm_cls_preds, odm_cls_target=odm_cls_target, odm_loc_target_mask=odm_loc_target_mask,\
                             op_type='negative_filtering')
    odm_cls_target = group[0]
    odm_loc_target_mask = group[1]

    # monitoring training status
    arm_cls_prob = mx.symbol.SoftmaxOutput(data=arm_cls_preds, label=arm_cls_target, ignore_label=-1, use_ignore=True, \
                                           grad_scale=1.0, multi_output=True, normalization='valid', name="arm_cls_prob")
    arm_loc_loss_ = mx.symbol.smooth_l1(name="arm_loc_loss_", data=arm_loc_target_mask * (arm_loc_preds - arm_loc_target), scalar=1.0)
    arm_loc_loss = mx.symbol.MakeLoss(arm_loc_loss_, grad_scale=1.0, normalization='valid', name="arm_loc_loss")
    arm_cls_label = mx.symbol.MakeLoss(data=arm_cls_target, grad_scale=0, name="arm_cls_label")

    odm_cls_prob = mx.symbol.SoftmaxOutput(data=odm_cls_preds, label=odm_cls_target, ignore_label=-1, use_ignore=True, grad_scale=1.0, \
                                           multi_output=True, normalization='valid', name="odm_cls_prob")
    odm_loc_loss_ = mx.symbol.smooth_l1(name="odm_loc_loss_", data=odm_loc_target_mask * (odm_loc_preds - odm_loc_target), scalar=1.0)
    odm_loc_loss = mx.symbol.MakeLoss(odm_loc_loss_, grad_scale=1.0, normalization='valid', name="odm_loc_loss")
    odm_cls_label = mx.symbol.MakeLoss(data=odm_cls_target, grad_scale=0, name="odm_cls_label")

    odm_det = []
    odm_loc_preds_bs = mx.sym.split(data=odm_loc_preds, axis=0, num_outputs=times)
    odm_cls_prob_bs = mx.sym.split(data=odm_cls_prob, axis=0, num_outputs=times)
    for i in range(times):
        odm_det_tmp = mx.contrib.symbol.MultiBoxDetection(*[odm_cls_prob_bs[i], odm_loc_preds_bs[i], odm_anchor_boxes_bs[i]], \
                                    name="odm_detection_{}".format(i), nms_threshold=nms_thresh, force_suppress=force_suppress, \
                                                          variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
        odm_det.append(odm_det_tmp)
    odm_det = mx.symbol.concat(*odm_det, num_args=len(odm_det), dim=0)
    odm_det = mx.symbol.MakeLoss(data=odm_det, grad_scale=0, name="odm_det_out")

    # group output
    out = mx.symbol.Group([arm_cls_prob, arm_loc_loss, arm_cls_label, odm_cls_prob, odm_loc_loss, odm_cls_label, odm_det])
    return out

def get_symbol(network, num_classes, from_layers, num_filters,\
                     sizes, ratios, normalizations=-1, steps=[], nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    batch = 32
    times = batch / 1
    body = import_module(network).get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers)
    arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = multibox_layer(layers, num_filters, num_classes,
                sizes=sizes, ratios=ratios, normalization=normalizations, num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    arm_cls_prob = mx.symbol.SoftmaxActivation(data=arm_cls_preds, mode='channel', name='arm_cls_prob')
    arm_out = mx.contrib.symbol.MultiBoxDetection(*[arm_cls_prob, arm_loc_preds, arm_anchor_boxes], name="arm_detection", \
                            nms_threshold=nms_thresh, force_suppress=force_suppress, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)

    odm_anchor_boxes = mx.symbol.slice_axis(arm_out, axis=2, begin=2, end=6)
    odm_cls_prob = mx.symbol.SoftmaxActivation(data=odm_cls_preds, mode='channel', name='odm_cls_prob')
    odm_det = []
    odm_loc_preds_bs = mx.sym.split(data=odm_loc_preds, axis=0, num_outputs=times)
    odm_anchor_boxes_bs = mx.sym.split(data=odm_anchor_boxes, axis=0, num_outputs=times)
    odm_cls_prob_bs = mx.sym.split(data=odm_cls_prob, axis=0, num_outputs=times)
    for i in range(times):
        odm_det_tmp = mx.contrib.symbol.MultiBoxDetection(*[odm_cls_prob_bs[i], odm_loc_preds_bs[i], odm_anchor_boxes_bs[i]], \
                                              name="odm_detection", nms_threshold=nms_thresh, force_suppress=force_suppress, \
                                              variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
        odm_det.append(odm_det_tmp)
    odm_det = mx.symbol.Concat(*odm_det, num_args=len(odm_det), dim=0)
    odm_det = mx.symbol.Reshape(data=odm_det, shape=(times, -1, 6))
    return odm_det

