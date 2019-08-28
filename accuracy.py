# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:30:54 2019

@author: rubya
"""
import pickle
accuracy = ([0, 31.451612903225808],
[1, 30.64516129032258],
[2, 71.7741935483871],
[3, 65.32258064516128],
[4, 64.51612903225806],
[5, 66.93548387096774],
[6, 75.80645161290323])

with open('accuracy.pkl', 'wb') as f:
    pickle.dump(accuracy,f)