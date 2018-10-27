#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from optparse import OptionParser
import argparse
from scripts.invariants import InvariantsCalculator


def main(order, picname, file_out):
    try:
        img = Image.open(picname)
        img_rotated_90 = img.rotate(90, expand = True)
        img_rotated_45 = img.rotate(45, expand = True)
        img_rotated_180 = img.rotate(180, expand = True)

        invar = InvariantsCalculator()
        invar90 = InvariantsCalculator()
        invar45 = InvariantsCalculator()
        invar180 = InvariantsCalculator()

        invar.calculateInvariants(np.asmatrix(np.array(img)))
        invar90.calculateInvariants(np.asmatrix(np.array(img_rotated_90)))
        invar45.calculateInvariants(np.asmatrix(np.array(img_rotated_45)))
        invar180.calculateInvariants(np.asmatrix(np.array(img_rotated_180)))

        print("RESULTS")
        print(invar.getInvariants())
        print(invar90.getInvariants())
        print(invar45.getInvariants())
        print(invar180.getInvariants())

        invar.writeOut(file_out)
        invar45.writeOut(file_out)
        invar90.writeOut(file_out)
        invar180.writeOut(file_out)
    except IOError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run method with arguments to get moment invariants.")
    parser.add_argument("order", type = int, default = 3)
    parser.add_argument("picname")
    parser.add_argument("file_out")
    args = vars(parser.parse_args())
    main(**args)