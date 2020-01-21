#!/usr/bin/env python3

import argparse

if __name__ == '__main__':
    from models import perceptual_model as model
    from util import util
else:
    from .models import perceptual_model as model
    from .util import util


LPIPS_MODEL = None

def compute_lpips(
    pathImg0: str,
    pathImg1: str,
    imagePartOrigin=None,
    imagePartDim=(0,0),
    useGpu: bool=True ) -> float:

    ## Initialize the model if it is not yet setup
    global LPIPS_MODEL
    if LPIPS_MODEL is None:
        LPIPS_MODEL = model.PerceptualLoss(
                model='net-lin', net='alex', use_gpu=useGpu)

    # Load images
    img0 = util.load_tensor(pathImg0, imagePartOrigin, imagePartDim, useGpu)
    img1 = util.load_tensor(pathImg1, imagePartOrigin, imagePartDim, useGpu)

    # Compute distance
    return float(LPIPS_MODEL.forward(img0, img1))

###############################################################################
# script execution
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
    parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
    parser.add_argument(
            '--use_gpu', action='store_true', help='turn on flag to use GPU')

    opt = parser.parse_args()

    dist01 = compute_lpips(opt.path0, opt.path1, useGpu=opt.use_gpu)
    print('Distance: %.3f'%dist01)

