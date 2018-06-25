#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from optparse import OptionParser
import argparse
from scripts.invariants import InvariantsCalculator


def save(file_out, order, original, rotated_90, rotated_45, rotated_180):
    f = open(file_out, 'w')
    f.write()
    f.close()

def main(order, picname, file_out):
    pic = misc.imread(picname, flatten=1)
    print("ORIGINAL")
    print(pic)
    pic_rotated_90 = ndimage.rotate(pic, 90)
    pic_rotated_45 = ndimage.rotate(pic, 45, reshape=False)
    pic_rotated_180 = ndimage.rotate(pic, 180)

    img = np.asmatrix(pic)
    img_rotated_90 = np.asmatrix(pic_rotated_90)
    img_rotated_45 = np.asmatrix(pic_rotated_45)
    img_rotated_180 = np.asmatrix(pic_rotated_180)

    print("*******************")
    print(img)
    print("----------")
    print(img_rotated_90)
    print("----------")
    print(img_rotated_45)
    print("----------")
    print(img_rotated_180)
    print("*******************")

    print("*-*-*-*-*-*-*-*-*-*-*\n")
    print(img)
    invar = InvariantsCalculator()
    invar90 = InvariantsCalculator()
    invar45 = InvariantsCalculator()
    invar180 = InvariantsCalculator()
    invar.calculate_invariants(img)
    invar90.calculate_invariants(img_rotated_90)
    invar45.calculate_invariants(img_rotated_45)
    invar180.calculate_invariants(img_rotated_180)
    print("RESULTS")
    print(invar.getInvariants())
    print(invar90.getInvariants())
    print(invar45.getInvariants())
    print(invar180.getInvariants())

    invar.write_out(file_out)
    invar45.write_out(file_out)
    invar90.write_out(file_out)
    invar180.write_out(file_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run method with arguments to get moment invariants.")
    parser.add_argument("order", type = int, default = 3)
    parser.add_argument("picname")
    parser.add_argument("file_out")
    args = vars(parser.parse_args())
    main(**args)