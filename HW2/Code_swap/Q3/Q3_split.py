# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:20:24 2013

read the hive result file and produce csv data file

@author: roger

"""
output = []
with open("variance/000000_0.txt","r") as f:
    for row in f:
        output.append (",".join(row.strip().split('\001')))
        
with open("q3_variance.csv","w") as w:
    w.write("\n".join(output))