# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:48:28 2013 

@author: roger

STA 250 - HW2 Problem 3, scatter plot
"""
import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

with open('q3_variance.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        x.append(int(row[0]))
        y.append(float(row[1]))
        
x = np.array(x)
y = np.array(y)

plt.plot(x,y)
plt.title('within-group variances')
plt.savefig('q3_variance.png')
plt.show()
