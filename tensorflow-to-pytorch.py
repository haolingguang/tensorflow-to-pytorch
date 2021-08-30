import tensorflow as tf
import deepdish as dd
import argparse
import os
import numpy as np
import torch
from torchvision import models

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts ckpt weights to deepdish hdf5")
    parser.add_argument("--infile", type=str, default='./models/adv_inception_v3_rename.ckpt',
                        help="Path to the ckpt.")  # ***model.ckpt-22177***
    # parser.add_argument("outfile", type=str, nargs='?', default='',
                        # help="Output file (inferred if missing).")
    args = parser.parse_args()
    # if args.outfile == '':
    #     args.outfile = os.path.splitext(args.infile)[0] + '.h5'
    # outdir = os.path.dirname(args.outfile)
    # if not os.path.exists(outdir):
    #     os.makedirs(outdir)
    weights = read_ckpt(args.infile)
    # dd.io.save(args.outfile, weights)
    # pretarined_dict = dd.io.load('/models.h5')
    net= models.inception_v3(pretrained=True)
    model_dict = net.state_dict()
    new_pre_dict ={}
    for k,v in weights.items():
        new_pre_dict[k] =torch.Tensor(v)
    model_dict.update(new_pre_dict)
    net.load_state_dict(model_dict)
    torch.save(net,'./Models/adv_inception_v3.pth')

