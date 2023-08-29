import sys
import IPython
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections
    import collections.abc
    setattr(collections, "MutableMapping", collections.abc.MutableMapping)
import numpy as np
import chaospy as cp
import uncertainpy as un
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py

from HalfplaneVisu import *


def polar2cart(theta, phi, r):
    return np.dstack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta) * np.ones_like(phi)
    ]).squeeze()

def cart2polar(x,y,z):
    r = np.linalg.norm([x,y,z],axis=0)
    z_div_r = np.divide(z, r)
    phi = np.arctan2(y,x)
    return np.dstack([
        np.arccos(z_div_r), # low acuracy!!
        phi + np.less(phi,0) * 2 * np.pi, # map -pi .. pi to 0 .. 2pi
        r
    ])


def cosn_LVK(n:int):
    """returns a function with parameters (theta,phi) that corresponds to the LVK of a cos^n distributed illuminant

    Args:
        n (int): order of distribution
    """
    if(n < 1):
        n=1
    name = "cos{order}".format(order=n)
    def LVK(theta,phi): # test for cos distribution, needs to use phi in order to keep dimension
        intensity = 100 * np.power(np.cos(theta),n) + np.zeros_like(phi)
        return intensity.clip(min=0)
    LVK.__name__ = name
    return LVK
    
def goniometer(x_LSP, y_LSP, z_LSP, r_detector, LVK, theta, phi):
    """Goniometer Equation for Point Source with shifted center of light

    Args:
        x_LSP (float):      x-Coordinate of the center of light in m
        y_LSP (float):      y-Coordinate of the center of light in m
        z_LSP (float):      z-Coordinate of the center of light in m
        r_detector (float): Distance from center of rotation to detector
        LVK (function):     analytic light intensity distribution: LVK(theta, phi) -> I (angles in radians)
        theta(list):    theta angles at which the LVK should be probed (radians)
        phi(list):      phi angles at which the LVK should be probed (radians)
    """
    # calculate measured LVK values for the given xyz offset
    # calculate distance between point source and detector for every detector position phi, theta
    detector_positions = polar2cart(theta,phi,r_detector)
    source_position = np.array([x_LSP,y_LSP,z_LSP])
    source_detector_vectors = detector_positions - source_position
    source_detector_distances = np.linalg.norm(source_detector_vectors,axis=-1)

    # calculate angles theta_m, phi_m: 
    # assume vector from source to detector comes from origin, convert to polar coordinate -> angles to sample LVK of source
    source_detector_xyz = np.moveaxis(source_detector_vectors,2,0) # shift dimensions to parse as x, y, z
    s_d_polar = cart2polar(*source_detector_xyz)
    # grab theta_m, phi_m values and add an empty dimension
    theta_m = s_d_polar[:,:,0]
    phi_m = s_d_polar[:,:,1]
    # sample LVK at theta_m and phi_m
    LVK_m = LVK(theta_m,phi_m)

    # calculate measured luminous intensity from I source and r_m (apply wrong distance)
    # Em = I/rm^2 * cos(alpha)
    # Im = Em * r^2 = (I/rm^2) * r^2 * cos(apha)

    # correct for different sensitivity of photometer at an angle. 
    # it's the angle between the incoming light and the optical axis of the detector
    # here: source_detector_vectors and origin_detector_vectors(detector_positions)
    # angle between vectors a,b: (a dot b) / |a|*|b|
    # cos(alpha) = (source_detector_vectors dot detector_positions) / source_detector_distances * r_detector
    dot = np.sum(np.multiply(source_detector_vectors,detector_positions),axis=-1)
    cos_a = dot / (source_detector_distances * r_detector)
    
    len_factor = np.square(r_detector) / np.square(source_detector_distances)
    
    I_m = LVK_m * len_factor * cos_a

    outs = {"LVK"  : LVK_m,
            "len"  : len_factor,
            "alpha": cos_a}
    
    return None, I_m, outs

def model_out_LVK(t, out_1, outs):
    return t, outs["LVK"]

def model_out_len(t, out_1, outs):
    return t, outs["len"]

def model_out_alpha(t, out_1, outs):
    return t, outs["alpha"]


if __name__ == '__main__':
    # Coordinate definition:
    #   x-axis: right hand orientation
    #   y-axis: up
    #   z-axis: towards detector
    # hard-coded C-Ebenen coordinate system, make it a parameter later on
    theta_max=30
    theta_deg = np.linspace(start=0,stop=theta_max,num=2*theta_max+1) # don't go to full 90°, errors become problematic
    phi_deg = np.linspace(start=0,stop=360,num=13)
    theta = np.deg2rad(theta_deg)[:,None]
    phi = np.deg2rad(phi_deg)[None,:]
    LVK = cosn_LVK(n=100) 
    #true_LVK = cos_LVK(theta,phi)
    true_LVK = LVK(theta,phi)
    # Create the distributions
    x_dist = cp.Uniform(-0.001, 0.001)
    y_dist = cp.Uniform(-0.001, 0.001)
    z_dist = cp.Uniform(-0.01, 0.01)
    #x_dist = cp.Normal(0.0, 0.001)
    #y_dist = cp.Normal(0.0, 0.001)
    #z_dist = cp.Normal(0.0, 0.01)
    # parameter dictionary
    parameters = {"x_LSP": x_dist, "y_LSP": y_dist,"z_LSP": z_dist,
                  "r_detector": 1, "LVK": LVK,
                  "theta": theta, "phi": phi}
    
    feature_list = [model_out_LVK, model_out_len, model_out_alpha] # get additional outputs
    
    model = un.Model(run=goniometer, labels=["Measured Intensity"])

    UQ = un.UncertaintyQuantification(model=model, parameters=parameters, features=feature_list, logger_level="debug")
    # concatenate model name, name of the used LVK-function,name of the output and set as name for output file
    method = 'mc'
    fname = UQ.model.name + "_" + method + "_" + parameters['LVK'].__name__
    UQ.quantify(method=method, plot=None, save=False, seed=334560, nr_mc_samples=2**14)
    # remove evaluations before saving
    UQ.data['goniometer'].pop('evaluations')
    UQ.data['model_out_LVK'].pop('evaluations')
    UQ.data['model_out_len'].pop('evaluations')
    UQ.data['model_out_alpha'].pop('evaluations')
    
    UQ.save(filename=fname, folder="uni_r1_xye-3_ze-2")
    
    IPython.embed()
    
    # Visualization Uncertainty
    if(False):
        # load the data file
        f = h5py.File('uni_r1_xye-3_ze-2/goniometer_mc_cos100.h5', 'r')
        dI = f['goniometer']
        dLVK = f['model_out_LVK']
        dlen = f['model_out_len']
        dalpha = f['model_out_alpha']
        # show LVK mean: colored LVK body, similar to TT viewer
        # convert Theta,Phi,I to X,Y,Z. Take I as color
        mean_I = np.array(dI['mean'])
        var_I = np.array(dI['variance'])
        std_I = np.sqrt(var_I)
        
        diff_LVK = mean_LVK - true_LVK
        diff_rel_LVK = diff_LVK / true_LVK
        plt.plot(theta.squeeze(),diff_rel_LVK[:,0])

        mean_I = np.array(dI['mean'])
        var_I = np.array(dI['variance'])
        std_I = np.sqrt(var_I)
        
        p5 = np.array(dI['percentile_5'])
        p95 = np.array(dI['percentile_95'])
        mean_I_c, theta_c, phi_c = half2cplanes(mean_I, theta, phi)
        std_I_c = half2cplanes(std,theta,phi)[0]
        p5_c = half2cplanes(p5,theta,phi)[0]
        p95_c = half2cplanes(p95,theta,phi)[0]
        
        fig = plt.figure(figsize=(12,8))
        ax_cart, ax_polar, ax_cart2 = create_axes_LID(fig, cart=True, polar=False, cart2=True,
                                                    theta_max=theta_max,theta_tick_dist=15,r_max = 115,r_min=-20)
        plot_cart(theta_c,mean_I_c[c],ax_cart, c='lightgreen', label=r'$C_0$-Plane of the mean LID')
        plot_cart(theta_c,p5_c[c],ax_cart, c='orange', label=r'95% interval')
        plot_cart(theta_c,p95_c[c],ax_cart, c='orange')
        ax_cart.legend(loc='upper left')
        plot_cart(theta_c,std_I_c[c],ax_cart2, c='darkblue', label=r'standard deviation') # error relative to $I_0$
        ax_cart2.legend(loc=("upper right"))
        # fix up parsite axes ax_cart2, map the ax_cart y zero location to a certain ax_cart2 value
        center_y_ax_cart2 = 0
        damp_y2 = 0.8 # damp so that the plots to ax_cart2 don't get too tight
        ylim = np.array(ax_cart.get_ylim())
        ratio_ylim = ylim[0] / ylim[1] # min / max
        ylim2 = np.array(ax_cart2.get_ylim())
        ylim2 -= center_y_ax_cart2 # substract value to center, now min max are centered around 0
        ylim2_min = ylim2[0]
        ylim2_max = ylim2[1]
        if(np.abs(ylim2_max) > np.abs(ylim2_min)): # if max lim is bigger than min lim
            ylim2_max = ylim2_max / damp_y2
            ylim2_min = ylim2_max * ratio_ylim
        else:
            ylim2_min = ylim2_min / damp_y2
            ylim2_max = ylim2_min / ratio_ylim
        ax_cart2.set_ylim(bottom = ylim2_min + center_y_ax_cart2, top = ylim2_max + center_y_ax_cart2)
        #ax_cart2.set_ylabel('%', size=14, labelpad=-38, y=1.05, rotation=0) # -38 for rel i0
        ax_cart2.set_ylabel('$I$ / $cd$', size=14, labelpad=-26, y=1.07, rotation=0)
        fig.savefig('uni_r1_xye-3_ze-2/uncertainty_cos100.svg')