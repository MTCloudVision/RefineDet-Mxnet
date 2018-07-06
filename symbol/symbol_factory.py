"""ets for various network configurations"""
import logging
from symbol import symbol_builder
import numpy as np

def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):
    """ Following the ssd arxiv paper, regarding the calculation of scales & ratios

    Parameters
    ----------
    min_scale : float
    max_scales: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list

    return
    ------
    sizes : list
        list of scale sizes per feature layer
    ratios : list
        list of anchor_ratios per feature layer
    """

    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales

def get_config(network, data_shape, batch_size=32, gpus=1, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    configs:
        from_layers :from which layer to continue to build additional prediction layers
        num_filters :numer of filter for each additional prediction layer
        topdown_layers :use which layer to construct top-down architecture
        use_perdict_module :which layer use perdict module
        use_tdm : if set to True, use TDM to build top-down architecture rather than Decovlutional layer
        use_res_unit : use residual unit in each additional prediction layers 
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu5_3', 'relu7', '']
            num_filters = [512, 512, 1024, 512]
            #strides = [-1, -1, 2, 2, 2, 2, 2]
            #pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[0.0625], [0.125], [0.25], [0.5]]
            #sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5], [1,2,.5], [1,2,.5]]
            normalizations = [10, 8, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in [8, 16, 32, 64]]
        else:
            from_layers = ['relu4_3', 'relu5_3', 'relu7', '']
            num_filters = [512, 512, 1024, 512]
            # strides = [-1, -1, 2, 2, 2, 2, 2]
            # pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[0.1], [0.2], [0.4], [0.8]]
            # sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
            # sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
            ratios = [[1, 2, .5], [1, 2, .5], [1, 2, .5], [1, 2, .5]]
            normalizations = [10, 8, -1, -1]
            steps = [] if data_shape != 320 else [x / 320.0 for x in [8, 16, 32, 64]]
        if not (data_shape == 320 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'inceptionv3':
        if data_shape >= 448:
            from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
            num_filters = [-1, -1, 512, 256, 256, 128]
            strides = [-1, -1, 2, 2, 2, 2]
            pads = [-1, -1, 1, 1, 1, 1]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        else:
            from_layers = ['ch_concat_mixed_2_chconcat', 'ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '']
            num_filters = [-1, -1, -1, 256, 256, 128]
            strides = [-1, -1, -1, 2, 2, 2]
            pads = [-1, -1, -1, 1, 1, 1]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '', '','']
        num_filters = [1024,2048,512,256,256,128]
        topdown_layers = [1, 1, 1, 1, -1, -1]
        use_perdict_module = [1, 1, 1, 1, -1, -1]
        use_tdm = False
        use_res_unit = False
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobilenet':
        from_layers = ['conv_12_relu', 'conv_14_relu', '', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                  [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet121':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 24, 16]
        num_stage = 4
        growth_rate = 32
        bottle_neck = True
        from_layers = ['DBstage3_concat24', 'DBstage4_concat16', '', '', '', '']
        num_filters = [-1, -1, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet-tiny':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 18, 12]
        num_stage = 4
        growth_rate = 16
        bottle_neck = True
        from_layers = ['DBstage2_concat12', 'DBstage3_concat18', '', '', '', '']
        num_filters = [-1, -1, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_symbol_train(network, data_shape, batch_size=32, gpus=1, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, batch_size, gpus, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol_train(**config)

def get_symbol(network, data_shape, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder.get_symbol(**config)

