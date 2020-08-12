import time
from math import sqrt

import numpy as np
import scipy.sparse as sp


class Lanczos:
    # Class parameters
    MAXLANCZOS = 2000
    TOLBISEC = 0.000001
    TOL = 0.0001

    def bisec(self, k, ymax, zmin, tolBisec):
        #  precondition: 0 < (ymax-lmax[k-1])/ymax < TOLBISEC,
        #                0 < (lmin[k-1]-zmin)/zmin < TOLBISEC
        zmax = self.alpha[0] + abs(self.beta[0])
        ymin = self.alpha[0] - abs(self.beta[0])

        for l in range(1, k):
            zmax = max(
                zmax,
                self.alpha[l] + abs(self.beta[l - 1]) + abs(self.beta[l]))
            ymin = min(
                ymin,
                self.alpha[l] - abs(self.beta[l - 1]) - abs(self.beta[l]))

        zmax = max(zmax,
                   self.alpha[k] + abs(self.beta[k - 1]))  # upp for lmax[k]
        ymin = min(ymin,
                   self.alpha[k] - abs(self.beta[k - 1]))  # low for lmin[k]
        ymin = max(ymin,
                   0.0)  # ymin >= 0 because we are dealing with spd matrices

        pz = self.pol(k, zmax)
        while abs(zmax - ymax) > tolBisec * min(abs(zmax), abs(ymax)):
            x = (ymax + zmax) / 2.0
            px = self.pol(k, x)
            if np.signbit(px) != np.signbit(pz):  # lmax[k] in [x,zmax]
                ymax = x
            else:  # lmax[k]<x
                zmax = x
                pz = px

        py = self.pol(k, ymax)
        if np.signbit(pz) != np.signbit(
                py) and py != 0:  # lmax[k] in (ymax,zmax]
            ymax = zmax

        # postcondition: 0 < (ymax-lmax[k])/ymax < TOLBISEC

        py = self.pol(k, ymin)
        while abs(zmin - ymin) > tolBisec * min(abs(zmin), abs(ymin)):
            x = (ymin + zmin) / 2.0
            px = self.pol(k, x)

            if np.signbit(px) != np.signbit(py):  # lmin[k] in [ymin,x]
                zmin = x
            else:  # lmin[k]>x
                ymin = x
                py = px

        pz = self.pol(k, zmin)
        if np.signbit(pz) != np.signbit(
                py) and pz != 0:  # lmin[k] in [ymin,zmin)
            zmin = ymin

        # postcondition: 0 < (lmin[k-1]-zmin)/zmin < TOLBISEC
        return (ymax, zmin)

    def pol(self, k, x):
        r = 1
        p = self.alpha[0] - x
        for l in range(1, k + 1):
            q = p
            p = (self.alpha[l] - x) * p - self.beta[l - 1] * self.beta[l -
                                                                       1] * r
            r = q
        return p

    def __init__(self,
                 A,
                 P=None,
                 maxIterations=MAXLANCZOS,
                 tol=TOL,
                 tolBisec=TOLBISEC):
        # Initialize the instance variables
        self.alpha = np.zeros(maxIterations)
        self.beta = np.zeros(maxIterations - 1)
        self.converged = True
        self.dofs = A.shape[0]

        if P is None:
            P = sp.identity(A.shape[0])

        # Measure start time
        start = time.clock()

        # Initialize with random vector
        w = 2.0 * np.random.rand(A.shape[0]) - 1.0

        v = A.dot(w)  # v=Aw
        norm = sqrt(v.dot(w))
        v = v / norm
        w = w / norm

        v = P.dot(v)

        u = A.dot(v)
        self.alpha[0] = u.dot(w)  # alpha[0] = <v,w>

        lmax = lmin = self.alpha[0]

        k = 0
        while True:
            if k == maxIterations - 1:
                self.converged = False
                break

            v -= self.alpha[k] * w  # v=v-alpha_[k]w
            u = A.dot(v)
            self.beta[k] = sqrt(u.dot(v))  # beta[k]=||v||

            temp = w
            w = v / self.beta[k]
            v = -self.beta[k] * temp

            u = A.dot(w)
            u = P.dot(u)
            v += u

            k += 1
            u = A.dot(v)
            self.alpha[k] = u.dot(w)  # alpha_[++k]=<v,w> */

            lmaxold = lmax
            lminold = lmin
            (lmax, lmin) = self.bisec(k, lmax, lmin, tolBisec)

            if (lmax - lmaxold) < tol * lmaxold and (lminold -
                                                     lmin) < tol * lmin:
                break

        # final values
        self.iterations = k + 1
        self.time = time.clock() - start

        self.lmax = lmax
        self.lmin = lmin

        self.alpha = np.resize(self.alpha, k)
        self.beta = np.resize(self.beta, k - 1)

    def cond(self):
        return self.lmax / self.lmin

    def __str__(self):

        convText = "converged"
        if not self.converged:
            convText = "NOT " + convText
        return '{}\tdofs={}\tits={}\tlmax={}\tlmin={}\tkappa={}\ttime={} s'.format(
            convText, self.dofs, self.iterations, self.lmax, self.lmin,
            self.cond(), self.time)
