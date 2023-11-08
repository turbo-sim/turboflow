# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:24:20 2023

@author: laboan
"""

import numpy as np
radius_reference = 'mean'


def caluclate_geometry(geometry):
    
    A_in = []
    A_out = []
    H_m = []
    r_ht_in_vec = []
    r_m_in_vec = []
    r_m_out_vec = []
    
    for r_in, r_out, H_in, H_out in geometry["n_cascades"]:
        
        
        if geometry["radius_reference"] == 'mean':
            
            r_ht_in = (r_in-H_in/2)(r_in + H_in/2)
            r_ht_out = (r_out-H_out/2)(r_out + H_out/2)
            r_m_in = r_in
            r_m_out = r_out
            
        elif geometry["radius_reference"] == 'hub':
            
            r_t_in = r_in+H_in
            r_t_out = r_out+H_out
            r_ht_in = r_in/r_t_in
            r_ht_out = r_out/r_t_out
            r_m_in = (r_t_in+r_in)/2
            r_m_out = (r_t_out+r_out)/2
            
        elif geometry["radius_reference"] == 'tip':
            
            r_h_in = r_in-H_in
            r_h_out = r_out-H_out
            r_ht_in = r_h_in/r_in
            r_ht_out = r_h_out/r_out
            r_m_in = (r_h_in+r_in)/2
            r_m_out = (r_h_out+r_out)/2        
        
        H_m.append((H_in+H_out)/2)             
        A_in.append(2*np.pi*H_in*r_in)
        A_out.append(2*np.pi*H_out*r_out)
        r_m_in_vec.append(r_m_in)
        r_m_out_vec.append(r_m_out)
        r_ht_in_vec.append(r_ht_in)
        
    geometry["r_ht_in"] = r_ht_in
    geometry["r_m_in"] = r_m_in
    geometry["r_m_out"] = r_m_out
        