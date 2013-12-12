# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:19:29 2013

@author: roger
"""
from __future__ import division
import random
import math
import numpy as np


def onesidedRobert(a, alpha):
    """ Implement onesided truncated standard normal with lower bound a
    alpha: optimal parameter in Robert's paper
    """
    while True:
        z = random.expovariate(alpha) + a
        if a < alpha:
            ro = math.exp(-(alpha-z)**2/2)
        else:
            ro = math.exp(-(a-alpha)**2/2-(alpha-z)**2/2)
        u = random.uniform(0, 1)
        if u <= ro:
            return z
        else:
            continue

def twosidedRobert(a, b):
    """ Implement twosided truncated normal with mean 0 and sd 1 """
    while True:
        z = random.uniform(a, b)
        if b < 0:
            ro = math.exp((b**2-z**2)/2)
        elif a > 0:
            ro = math.exp((a**2-z**2)/2)
        else:
            ro = math.exp(-z**2/2)
        u = random.uniform(0, 1)
        if u <= ro:
            return z
        else:
            continue

def truncNorm(n, mu, sigma, a='-inf', b='inf'):
    """ Sample from truncated normal
    n: number of samples
    mu, sigma: normal distribution paras
    a, b: truncation point
    """
    result = []
    if a == '-inf' and b == 'inf':
        for i in xrange(n):
            result.append(random.gauss(mu, sigma))
        return result
    
    elif b == 'inf':
        mu_neg = (a - mu) / sigma
        alpha = (mu_neg + (mu_neg**2+4)**0.5) / 2
        for i in xrange(n):
            result.append(onesidedRobert(mu_neg, alpha)*sigma+mu)
        return result

    elif a == '-inf':
        mu_pos = (b - mu) / sigma
        alpha = (- mu_pos + (mu_pos**2+4)**0.5) / 2
        for i in xrange(n):
            result.append(-onesidedRobert(-mu_pos, alpha)*sigma+mu)
        return result

    elif a > b:
        a, b = b, a

    mu_neg = (a - mu) / sigma
    mu_pos = (b - mu) / sigma

    for i in xrange(n):
        result.append(twosidedRobert(mu_neg, mu_pos)*sigma+mu)
        
    return result

def truncNormVec(n, mu_vec, sigma, a='-inf', b='inf'):
    nu_vec
