from google.protobuf import text_format
import numpy as np
import caffe_parse.caffe_pb2 as caffe_pb2


def parse_caffemodel(file_path):
    """
    parses the trained .caffemodel file

    filepath: /path/to/trained-model.caffemodel

    returns: layers
    """
    f = open(file_path, 'rb')
    contents = f.read()

    net_param = caffe_pb2.NetParameter()
    net_param.ParseFromString(contents)

    layers = find_layers(net_param)
    return layers


def find_layers(net_param):
    if len(net_param.layers) > 0:
        return net_param.layers
    elif len(net_param.layer) > 0:
        return net_param.layer
    else:
        raise Exception("Couldn't find layers")


def main():
    param_dict = parse_caffemodel('xxx.caffemodel')


if __name__ == '__main__':
    main()
