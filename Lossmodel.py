# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:49:45 2022

@author: lasseba
"""

from lossmodels import KO
from lossmodels import Benner as br

def loss(cascade_data, lossmodel):
    
    if (lossmodel == 'KO'):
        Y, loss_dict = KO.KO(cascade_data)
    elif (lossmodel == 'Benner'):
        Y, loss_dict = br.Benner(cascade_data)
    return [Y, loss_dict]