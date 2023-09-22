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
        np.arccos(z_div_r), # low acuracy!!
        phi + np.less(phi,0) * 2 * np.pi, # map -pi .. pi to 0 .. 2pi
        r
    ])

def cosn_LID(n:int):
    """returns a function with parameters (theta,phi) that corresponds to the LVK of a cos^n distributed illuminant

    Args:
        n (int): order of distribution
    """
    if(n < 1):
        n=1
    name = "cos{order}".format(order=n)
    def LID(theta,phi):
        intensity = 100 * np.power(np.cos(theta),n) + np.zeros_like(phi)
        return intensity.clip(min=0)
    LID.__name__ = name
    return LID