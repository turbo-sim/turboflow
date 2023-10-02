# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:49:45 2022

@author: lasseba
"""

from . import loss_model_benner as br
from . import loss_model_kacker_okapuu as ko


# Define a list of available loss models
available_loss_models = ['ko', 'benner', 'isentropic']

def loss(cascade_data, lossmodel):
    """
    Calculate loss coefficient based on the selected loss model.

    Args:
        cascade_data (dict): Data for the cascade.
        lossmodel (str): The selected loss model.

    Returns:
        list: [Y, loss_dict], where Y is the loss coefficient and loss_dict is a dictionary of loss components.
    """
    if lossmodel in available_loss_models:
        if lossmodel == 'ko':
            Y, loss_dict = ko.calculate_loss_coefficient(cascade_data)
        elif lossmodel == 'benner':
            Y, loss_dict = br.calculate_loss_coefficient(cascade_data)
        elif lossmodel == 'isentropic':
            loss_dict = {"Profile" : 0,
                         "Secondary" : 0,
                         "Clearance" : 0,
                         "Total" : 0}
            Y = 0
            
        return [Y, loss_dict]
    else:
        raise ValueError(f"Invalid loss model. Available options: {', '.join(available_loss_models)}")

