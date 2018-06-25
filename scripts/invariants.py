# -*- coding: utf-8 -*-

from scripts import *
import math
import numpy as np
from scipy.special import comb
from scipy import *


class InvariantsCalculator:
    """
    The class represents calculator of invariants.
    """

    def __init__(self, order = 3):
        self.invts = np.array([])
        self.order = order


    def getInvariants(self):
        return self.invts


    def toText(self):
        return ";".join(map(str, self.invts))


    def write_out(self, fname):
        f = open(fname, 'a')
        f.write(self.toText())
        f.write("\n")
        f.close()


    def calculate_invariants(self, data_matrix):
        """
        It calculates the invariants.
        
        :param data_matrix: numpy array
        """

        p0 = 1
        q0 = 2
        if self.order == 2:
            p0 = 0
            q0 = 2
        if p0 > q0:
            aux = p0
            p0 = q0
            q0 = aux
        m = self.__calculateMoments(data_matrix)
        c = self.__geometricMomentsToComplexMoments(m)
        c = np.round(c, 5)
        tmpx = np.linspace(0, self.order, self.order + 1)
        tmpy = np.linspace(0, self.order, self.order + 1)

        qm, pm = np.meshgrid(tmpx, tmpy)
        c = c / ((m[0, 0] ** ((qm + pm + 2) / 2.0)) * 1.0)
        c = c * ((pm + qm) / 2 + 1) * math.pi ** ((pm + qm) / 2.0)
        ident = q0 - p0
        ni = 0
        pwi = np.array([])
        if ident == 0:
            for r1 in range(2, self.order + 1, 2):
                p = round(r1 / 2.0, 0)
                tmp = c[p, p]
                self.invts = np.append(self.invts, tmp.real)
                pwi = np.append(pwi, 1)
                ni = ni + 1
        else:
            for r1 in range(max(2, ident), self.order + 1, ident):
                for p in range(int(round(float(r1) / 2.0, 0)), r1 + 1):
                    q = r1 - p
                    if (p - q) % ident == 0:
                        tmp = c[p, q] * c[p0, q0] ** ((p - q) / ident)
                        self.invts = np.append(self.invts, tmp.real)
                        pwi = np.append(pwi, (1 + (p - q) / ident))
                        ni = ni + 1
                        if (p > q) and (p != q0 or q != p0):
                            tmp2 = c[p, q] * c[p0, q0] ** ((p - q) / ident)
                            self.invts = np.append(self.invts, tmp2.imag)
                            pwi = np.append(pwi, 1 + (p - q) / ident)
                            ni = ni + 1
        self.invts = np.sign(self.invts) * (np.absolute(self.invts) ** (1 / pwi))


    def __calculateMoments(self, data_matrix):
        """
        It calculates geometric moments.
        
        :param data_matrix: numpy array
        :return: geometric moments: numpy array
        """
        M = np.zeros((self.order + 1, self.order + 1))
        (n1, n2) = np.shape(data_matrix)
        m00 = data_matrix.sum()
        w = np.linspace(1, n2, n2)
        v = np.linspace(1, n1, n1)
        if m00 != 0:
            tx = ((data_matrix * np.array([w]).T).sum()) / float(m00)
            ty = ((data_matrix.T * np.array([v]).T).sum()) / float(m00)
        else:
            tx = 0
            ty = 0
        a = w - tx
        c = v - ty
        for i in range(1, self.order + 1 + 1):
            for j in range(1, self.order + 2 - i + 1):
                p = i - 1
                q = j - 1
                A = np.power(a, p)
                C = np.power(c, q)
                oo = C * data_matrix * (np.array([A]).T)
                M[i - 1, j - 1] = oo
        if self.order > 0:
            M[0, 1] = 0
            M[1, 0] = 0
        return M


    def __geometricMomentsToComplexMoments(self, gm):
        """
        It uses the geometric momements to calculate the complex moments.
        
        :param gm: geometric moments: numpy array
        :return: complex moments: numpy array
        """

        c = np.zeros((self.order + 1, self.order + 1)).astype(complex)
        for p in range(0, self.order + 1):
            for q in range(0, self.order - p + 1):
                for k in range(0, p + 1):
                    pk = comb(p, k)
                    for w in range(0, q + 1):
                        qw = comb(q, w)
                        c[p, q] = c[p, q] + pk * qw * (-1) ** (q - w) * 1j ** (p + q - k - w) * gm.item(
                            (k + w, p + q - k - w))
        return c
