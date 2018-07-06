"""
Created on Wed Jul  4 13:31:48 2018
Modified by: Guo Shuangshuang
"""
import mxnet as mx
import numpy as np

def conv_act_layer(from_layer, name, num_filter, use_bn=False, kernel=(3,3),stride=(1,1), pad=(1,1), act_type="relu", num=1):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}_conv{}".format(name, num))
    if use_bn:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn{}".format(name, num))
    if act_type:
        conv = mx.symbol.Activation(data=conv, act_type=act_type, name="{}_{}{}".format(name, act_type, num))
    return conv

def deconv_act_layer(from_layer, name, num_filter, use_bn=False, kernel=(2,2), pad=(0,0), stride=(2,2), act_type="relu"):
    deconv = mx.symbol.Deconvolution(
        data=from_layer, kernel=kernel, pad=pad, stride=stride, num_filter=num_filter, name="{}_deconv".format(name))
    if use_bn:
        deconv = mx.symbol.BatchNorm(data=deconv, name="{}_bn".format(name))
    if act_type:
        deconv = mx.symbol.Activation(data=deconv, act_type=act_type, name="{}_{}".format(name, act_type))
    return deconv

def tcb_module(conv, deconv, num_filter, level = 1):
    deconv = deconv_act_layer(deconv, "tcb{}".format(level), num_filter, use_bn=False, kernel=(2,2), pad=(0,0), stride=(2,2), act_type=None)
    conv1 = conv_act_layer(conv, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type="relu", num=1)
    conv2 = conv_act_layer(conv1, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type=None, num=2)
    eltwise = mx.symbol.ElementWiseSum(*[deconv, conv2])
    relu = mx.symbol.Activation(data=eltwise, act_type="relu", name="tcb{}_elt_relu".format(level))
    conv3 = conv_act_layer(relu, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type="relu", num=3)
    return conv3

def tcb_module_last(conv, num_filter, level = 1):
    conv1 = conv_act_layer(conv, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type="relu", num=1)
    conv2 = conv_act_layer(conv1, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type="relu", num=2)
    conv3 = conv_act_layer(conv2, "tcb{}".format(level), num_filter, use_bn=False, kernel=(3, 3), stride=(1, 1), pad=(1, 1), act_type="relu", num=3)
    return conv3

def construct_refinedet(from_layers):
    out_layers = []
    layers = from_layers[::-1]
    for k, from_layer in enumerate(layers):
        if k == 0:
            out_layer = tcb_module_last(layers[k], 256, level=4-k)
        else:
            out_layer = tcb_module(layers[k], out_layer, 256, level=4-k)
        out_layers.append(out_layer)
    return out_layers[::-1]

def getpred(from_layers, num_classes, sizes, ratios, mode='arm', clip=False, interm_layer=0, steps=[]):
    loc_layers = []
    cls_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0
    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # Add intermediate layers.
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                             stride=(1,1), pad=(1,1), num_filter=interm_layer, name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) - 1 + len(ratio)

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_conv_bias".format(from_name), init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=num_loc_pred, name="{}_loc_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_conv_bias".format(from_name), init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
                        stride=(1,1), pad=(1,1), num_filter=num_cls_pred, name="{}_cls_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_layers.append(cls_pred)
        if mode == 'arm':
            # create anchor generation layer
            if steps:
                step = (steps[k], steps[k])
            else:
                step = '(-1.0, -1.0)'
            anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, clip=clip,\
                                                      name="{}_anchors".format(from_name), steps=step)
            anchors = mx.symbol.Flatten(data=anchors)
            anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_layers, num_args=len(loc_layers), dim=1, name="{}_multibox_loc".format(mode))
    cls_preds = mx.symbol.Concat(*cls_layers, num_args=len(cls_layers), dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="{}_multibox_cls".format(mode))
    if mode == 'arm':
        anchor_boxes = mx.symbol.Concat(*anchor_layers, num_args=len(anchor_layers), dim=1)
        anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="{}_multibox_anchors".format(mode))
        return [loc_preds, cls_preds, anchor_boxes]
    else:
        return [loc_preds, cls_preds]
def multi_layer_feature(body, from_layers):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
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
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    internals = body.get_internals()
    layers = []
    for k, from_layer in enumerate(from_layers):
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            layer = layers[-1]
            conv6_1 = conv_act_layer(layer, 'conv6_1', 256, use_bn=False, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu', num=1)
            conv6_2 = conv_act_layer(conv6_1, 'conv6_2', 512, use_bn=False, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type='relu', num=1)
            layers.append(conv6_2)
    return layers

def multibox_layer(layers, num_filters, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[], clip=False, interm_layer=0, steps=[]):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(layers)
    assert len(ratios) == len(layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(layers)
    assert len(normalization) == len(layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(layers), "provide steps for all layers or leave empty"

    for k, from_layer in enumerate(layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name), shape=(1, num_channels[k], 1, 1),
                init=mx.init.Constant(normalization[k]), attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
            layers[k] = from_layer

    odm_layers = construct_refinedet(layers)
    arm_loc, arm_cls, arm_anchor_boxes = getpred(layers, 1, sizes, ratios, mode='arm', clip=False, interm_layer=0, steps=steps)
    odm_loc, odm_cls = getpred(odm_layers, num_classes, sizes, ratios, mode='odm', clip=False, interm_layer=0, steps=steps)
    return [arm_loc, arm_cls, arm_anchor_boxes, odm_loc, odm_cls]

