import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.axes_grid1.parasite_axes import HostAxes
from IPython import get_ipython
#import mpl_toolkits.axisartist as AA

class HalfplaneAxes:
    def __init__(self,fig = None, cart = True, polar = False, cart2 = False,
                 theta_max=90, theta_tick_dist=15,r_max = 115, r_min=-15):
        # disable tight bbox from Ipython magic
        get_ipython().run_line_magic('config',"InlineBackend.print_figure_kwargs = {'bbox_inches':None}")
        
        # make sure there is a figure
        if(fig == None):
            fig = plt.figure(figsize=(12,8))
        self.fig = fig
        
        # make sure there is at least one axes. if neither polar nor cart is specified, default to cartesian plot
        if(polar == False):
            cart = True
        if(cart == False):
            polar = True
            
        self.cart = cart
        self.polar = polar
        self.cart2 = cart2
        
        # create the axes and store them in member variables
        (self.ax_cart,
         self.ax_polar,
         self.ax_cart2) = create_axes_CPlane(fig, cart, polar, cart2, theta_max, theta_tick_dist, r_max, r_min)
        plt.close(fig)
        # create a dictionary of lines plotted to the different axes: one for combined cart/polar and one for the auxilary axis
        # key is the label, value is a line-object for the aus plots and a dictionary of line objects (polar,cart) for the main plots
        self.plots = {}
        self.plots_aux = {}
        
    def get_axes_handles(self):
        return self.ax_cart, self.ax_polar, self.ax_cart2
    
    def plot(self, theta, r, label, **kwargs):
        theta = np.array(theta).squeeze()
        if(self.cart == True):
            if(self.polar == True):
                # split up into left polar and right cartesian plot
                # left side:
                pos_theta_indices = np.where(theta >= 0)
                x_cart = np.rad2deg(theta[pos_theta_indices])
                y_cart= r[pos_theta_indices]
                
                # right side:
                neg_theta_indices = np.where(theta <= 0)
                theta_polar = theta[neg_theta_indices]
                r_polar = r[neg_theta_indices]
                
                # test, if the label was already plotted:
                if label in self.plots:
                    # line data needs to be updated
                    self.plots[label]['cart'].set_data(x_cart,y_cart)
                    self.plots[label]['polar'].set_data(theta_polar, r_polar)
                else:
                    # new plots
                    line_cart = self.ax_cart.plot(x_cart, y_cart, label=label, **kwargs)
                    line_polar = self.ax_polar.plot(theta_polar, r_polar, label=label, **kwargs)
                    # add this plot to the dictionary
                    self.plots[label] = {'cart':line_cart[0], 'polar':line_polar[0]}
            else:
                # only plot cart
                x_cart = np.rad2deg(theta)
                y_cart = r
                if label in self.plots:
                    self.plots[label]['cart'].set_data(x_cart,y_cart)
                else:
                    line_cart = self.ax_cart.plot(x_cart, y_cart, label=label, **kwargs)
                    self.plots[label] = {'cart':line_cart[0], 'polar':None}
        else:
            # only polar
            if label in self.plots:
                self.plots[label]['polar'].set_data(theta,r)
            else:
                line_polar = self.ax_polar.plot(theta, r, label=label, **kwargs)
                self.plots[label] = {'cart':None, 'polar':line_polar[0]}
    
    def plot_aux(self, theta,aux, label, **kwargs):
        if(self.polar == True and self.cart == True):
            # only plot to the right side
            pos_theta_indices = np.where(theta >= 0)
            x_cart = np.rad2deg(theta[pos_theta_indices])
            y_cart= aux[pos_theta_indices]
        else:
            x_cart = np.rad2deg(theta)
            y_cart = aux
        
        if label in self.plots_aux:
            self.plots_aux[label].set_data(x_cart,y_cart)
        else:
            line_cart = self.ax_cart2.plot(x_cart, y_cart, label=label, **kwargs)
            self.plots_aux[label] = line_cart

def half2cplanes(LVK,theta,phi):
    # concatenate half planes from LID to complete C-planes
    # check if phi-values include 180 degrees (pi): 2 phi-cuts form plane
    # return LVK composed of C-Planes: indexed with: [c,theta]
    phi_half_idx = np.where(phi.squeeze() == np.pi)[0][0] # maybe is close?
    assert(phi_half_idx)
    # split LVK at phi_half_idx: (transpose to index phi first)
    r_cuts = LVK.T[:phi_half_idx] # straight forward, includes theta = 0
    # left cuts are more complicated:
    # start at half_idx, limit to 2*half_idx to keep the same size as r_cuts
    # remove theta=0, since it is already in r_cuts
    l_cuts = LVK.T[phi_half_idx : 2*phi_half_idx,1:]
    l_cuts = np.flip(l_cuts, axis=1)# needs to be flipped
    LVK_c = np.hstack((l_cuts,r_cuts)) # concatenate
    # create new theta and phi versions for c-planes
    theta_c = np.vstack((-np.flip(theta[1:,:]), theta))
    phi_c = phi[:,:phi_half_idx]
    return LVK_c, theta_c, phi_c

def create_axes_CPlane(fig, cart, polar, cart2, theta_max, theta_tick_dist,r_max, r_min):
        
    plot_limits =  [0.05, 0.05, 0.9, 0.9] # realtive to figure size
    
    # create cartesian axes
    ax_cart = fig.add_axes(plot_limits, axes_class=HostAxes)
    # create polar axes, is inset later
    ax_polar = fig.add_axes(plot_limits, polar=True)
    
    # prepare cartesian axes
    ax_cart.set_zorder(2)
    ax_cart.patch.set_alpha(0)
    # move bottom and left spines to center
    ax_cart.spines['bottom'].set_position('zero')
    ax_cart.spines['left'].set_position('zero')
    # Remove top and right spines
    ax_cart.spines['top'].set_visible(False)
    ax_cart.spines['right'].set_visible(False)
    # label axes
    ax_cart.set_xlabel(r'$\theta$ / °', size=14, labelpad=-24, x=1.03)
    ax_cart.set_ylabel('$I$ / $cd$', size=14, labelpad=-5, y=1.02, rotation=0)
    # arrow format
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    # setup y-ticks
    ymax = r_max
    y_ticks_dist = 20
    y_ticks_dist_minor = 10
    y_ticks = np.arange(y_ticks_dist, ymax+1, y_ticks_dist)
    y_ticks_minor = np.arange(0, ymax+1, y_ticks_dist_minor)
    ax_cart.set_yticks(ticks=y_ticks, labels=y_ticks, ha='left')
    ax_cart.set_yticks(y_ticks_minor, minor=True)
    ax_cart.tick_params(axis='y', which='both', direction='in', pad=-10)
    bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.7)
    plt.setp(ax_cart.get_yticklabels(), bbox=bbox)
    plt.setp(ax_cart.get_xticklabels(), bbox=bbox)
    ax_cart.plot((0), (1), marker='^', transform=ax_cart.get_xaxis_transform(), **arrow_fmt)
    
    xmax = theta_max + 2
    # prepare polar axes
    r_lim = y_ticks_minor[-1] # polar plot extends to last minor tick on y-axis 
    ax_polar.set_theta_direction(-1)
    ax_polar.set_theta_zero_location('N')
    theta_min = -90
    ax_polar.set_rmax(r_lim)
    ax_polar.patch.set_alpha(0)
    ax_polar.autoscale(enable=False, axis='both', tight=None)
    # setup theta ticks
    theta_ticks_dist = 10
    left_theta_ticks = np.deg2rad(np.arange(theta_min + theta_ticks_dist,0,theta_ticks_dist))
    # mirror ticks. too many if theta_max = 0, but doesn't hurt
    theta_ticks = np.hstack((left_theta_ticks, np.flip(-left_theta_ticks))) 
    ax_polar.set_xticks(theta_ticks)
    ax_polar.tick_params(pad=15, labelleft=False)
    
    # setup conditional part of cartesian and polar axes
    if(cart == True): # draw ticks
        ymin = r_min
        x_ticks_dist = theta_tick_dist
        if(x_ticks_dist // 5 > 1):
            x_ticks_dist_minor = 5
        else:
            x_ticks_dist_minor = 1
        ax_cart.plot((1), (0), marker='>', transform=ax_cart.get_yaxis_transform(), **arrow_fmt)
        aspect = xmax/ymax * 0.8
        if(polar == True):
            xmin = - aspect * r_lim # defined to fit the polar plot with aspect ratio of 1
            x_ticks = np.arange(x_ticks_dist, xmax+1, x_ticks_dist)
            x_ticks_minor = np.arange(0, xmax+1, x_ticks_dist_minor)
            theta_max = 0
            polar_visible = True
            #ax_polar.spines['polar'].set_visible(False) # remove outer spine
        else:
            xmin = -xmax
            right_ticks = np.arange(x_ticks_dist, xmax+1, x_ticks_dist)
            x_ticks = np.hstack((np.flip(-right_ticks), right_ticks)) # mirror ticks
            right_ticks_minor = np.arange(0, xmax+1, x_ticks_dist_minor)
            x_ticks_minor = np.hstack((np.flip(-right_ticks_minor), right_ticks_minor)) # mirror ticks minor
            theta_max = 90
            polar_visible = False
        ax_cart.set_xticks(x_ticks)
        ax_cart.set_xticks(x_ticks_minor, minor=True)
    else: # only polar
        ax_cart.set_xticks([])
        aspect = xmax / r_lim
        xmin = -xmax
        ymin = 0
        theta_max = 90
        polar_visible = True
    
    # finish cartesian axes
    ax_cart.set(xlim=(xmin, xmax), ylim=(ymin, ymax), autoscale_on=False, axisbelow=False)
    ax_cart.set_aspect(aspect)
    
    # finish polar axes
    ax_polar.set_thetamin(theta_min)
    ax_polar.set_thetamax(theta_max)
    ax_polar.set_visible(polar_visible)
    
    # locate the polar axes inside the cartesian axes
    polar_height = r_lim / (ymax - ymin) # proportion of y-axis used by the polar plot
    polar_height = 2 * polar_height # bbox of polar wedge is the same hight as full circle
    polar_width = aspect * r_lim / (xmax - xmin) # proportion of x-axus used by the polar plot
    if(theta_max == 90):
        polar_width = 2 * polar_width # double the width for -90..90 polar plot
    polar_left_pos = -aspect * r_lim
    neg_y_offest = -ymin / (ymax - ymin) # proportion of negative part of y-axis
    polar_bottom_pos = (- 0.25 * polar_height) + neg_y_offest
    ip = InsetPosition(ax_cart,[0 , polar_bottom_pos, polar_width, polar_height])
    ax_polar.set_axes_locator(ip)
    
    
    ax_cart2 = None
    # create second y-axis next to the cartesian axes
    if(cart2 == True):
        ax_cart2 = ax_cart.twinx()
        ax_cart2.spines['right'].set_position(('axes', 1.07))
        ax_cart2.plot((1.07), (1), marker='^', transform=ax_cart2.transAxes, **arrow_fmt)
        ax_cart2.tick_params(axis='x', which='minor', bottom=False) # strangely host minor ticks are visible otherwise 
        ax_cart2.get_yaxis().get_major_formatter().set_useOffset(False)
    return ax_cart, ax_polar, ax_cart2
