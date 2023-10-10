import numpy as np
def polar2cart(theta, phi, r):
    return np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta) * np.ones_like(phi)
    ]).squeeze()

def cart2polar(x,y,z):
    r = np.linalg.norm([x,y,z],axis=0)
    z_div_r = np.divide(z, r)
    phi = np.arctan2(y,x)
    return np.stack([
        np.arccos(z_div_r),
        phi + np.less(phi,0) * 2 * np.pi, # map -pi .. pi to 0 .. 2pi
        r
    ])
    
def half2cplanes(LID,theta,phi):
    # concatenate half planes from LID to complete C-planes
    # this function assumes a lot about the structure of the angle grid pattern
    # needs to start at 0, increment in constant interval
    # check if phi-values include 180 degrees (pi): 2 phi-cuts form plane
    # return LVK composed of C-Planes: indexed with: [c,theta]
    phi_half_idx = np.where(phi.squeeze() == np.pi)[0][0]
    assert(phi_half_idx)
    # split LVK at phi_half_idx: (transpose to index phi first)
    r_cuts = LID.T[:phi_half_idx] # straight forward, includes theta = 0
    # left cuts are more complicated:
    # start at half_idx, limit to 2*half_idx to keep the same size as r_cuts
    # remove theta=0, since it is already in r_cuts
    l_cuts = LID.T[phi_half_idx : 2*phi_half_idx,1:]
    l_cuts = np.flip(l_cuts, axis=1)# needs to be flipped
    LID_c = np.hstack((l_cuts,r_cuts)) # concatenate
    # create new theta and phi versions for c-planes
    theta_c = np.vstack((-np.flip(theta[1:,:]), theta))
    phi_c = phi[:,:phi_half_idx]
    return LID_c, theta_c, phi_c



    
    
    
    