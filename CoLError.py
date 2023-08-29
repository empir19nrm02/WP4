import numpy as np
import matplotlib.pyplot as plt

from CoLUncertain import cosn_LVK, polar2cart, goniometer
from HalfplaneVisu import *

theta_max=89 # 30
cos_order = 1
theta_deg = np.linspace(start=0,stop=theta_max,num=2*theta_max+1)
phi_deg = np.linspace(start=0,stop=360,num=721)
theta = np.deg2rad(theta_deg)[:,None]
phi = np.deg2rad(phi_deg)[None,:]
LVK = cosn_LVK(n=cos_order)
true_LVK = LVK(theta,phi)

# run model
err = 0.001 # deviation LC
r = 1 # normalized goniometer distance
x_error_LVK, x_error_outs = goniometer(err, 0, 0, r, LVK, theta, phi)[1:]
y_error_LVK, y_error_outs = goniometer(0, -err, 0, r, LVK, theta, phi)[1:]
z_pos_error_LVK, z_pos_error_outs = goniometer(0, 0, err, r, LVK, theta, phi)[1:]
z_neg_error_LVK, z_neg_error_outs = goniometer(0, 0, -err, r, LVK, theta, phi)[1:]

# calculate error
# two options: relative to luminous intensity in each direction or
# relative to luminous intensity in Hauptabstrahlrichtung

relto =  true_LVK # for relative error
#relto = true_LVK[0,0] # absolute error
x_error =     100 * (x_error_LVK - true_LVK) / relto
y_error =     100 * (y_error_LVK - true_LVK) / relto
z_pos_error = 100 * (z_pos_error_LVK - true_LVK) / relto
z_neg_error = 100 * (z_neg_error_LVK - true_LVK) / relto

amp_factor = 100
x_amplified =     true_LVK + amp_factor * (x_error_LVK - true_LVK)
y_amplified =     true_LVK + amp_factor * (y_error_LVK - true_LVK)
z_pos_amplified = true_LVK + amp_factor * (z_pos_error_LVK - true_LVK)
z_neg_amplified = true_LVK + amp_factor * (z_neg_error_LVK - true_LVK)

# prepare the Model-Outputs
# Model comes from CoLUncertain.py 

outs = x_error_outs

lvk_out = np.ma.masked_equal(outs['LVK'],0)
lvk_factor = (outs['LVK'] / true_LVK)
lvk_factor_percent = (lvk_factor - 1) * 100
lvk_error = outs['LVK'] - true_LVK

len_factor = outs['len']
len_factor_percent = (len_factor - 1) * 100
len_error = (len_factor * true_LVK) - true_LVK

alpha_factor = outs['alpha']
alpha_factor_percent = (alpha_factor - 1) * 100
alpha_error = (alpha_factor * true_LVK) - true_LVK

ges_factor = lvk_factor * len_factor * alpha_factor
ges_factor_pcnt = (ges_factor - 1) * 100

# convert to c_plane structure for plotting
true_LVK_c, theta_c, phi_c = half2cplanes(true_LVK, theta, phi)
error_LVK_c = half2cplanes(x_error_LVK, theta, phi)[0]
amplified_c = half2cplanes(x_amplified, theta, phi)[0] # change
error_c = half2cplanes(x_error, theta, phi)[0] # change
lvk_factor_c = half2cplanes(lvk_factor, theta, phi)[0]
len_factor_c = half2cplanes(len_factor,theta,phi)[0]
alpha_factor_c = half2cplanes(alpha_factor,theta,phi)[0]
ges_factor_c = half2cplanes(ges_factor,theta,phi)[0]
ges_factor_pcnt_c = half2cplanes(ges_factor_pcnt,theta,phi)[0]
c = 0


fig = plt.figure(figsize=(12,8))
ax_cart, ax_polar, ax_cart2 = create_axes_LID(fig, cart=True, polar=True, cart2=True,
                                              theta_max=theta_max,theta_tick_dist=15,r_max = 115,r_min=-65)
plot_cart_polar(theta_c,true_LVK_c[c],ax_cart, ax_polar, c='lightgreen', label=r'$C_0$-Plane of the source LID')
plot_cart_polar(theta_c,amplified_c[c],ax_cart, ax_polar, c='red', label=r'$C_0$-Plane of measured LID, error amplified 10x')
ax_cart.legend(loc=(0.0,0.1)) # (0.0,0.1) for rel i0

plot_cart_polar(theta_c,error_c[c],ax_cart2, None, c='darkblue', label=r'error relative to $I_0$') # relative error
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
ax_cart2.set_ylabel('%', size=14, labelpad=-22, y=1.05, rotation=0) # -38 for rel i0

#fig.savefig('figures/Verformung_rel_i0_r1_e0.01/z_neg_error_plot_cos100.svg')


# polar plot
fig = plt.figure(figsize=(12,8))
ax_cart, ax_polar, ax_cart2 = create_axes_LID(fig, cart=False, polar=True, cart2=False,
                                              theta_max=theta_max,theta_tick_dist=15,r_max = 115,r_min=-20)
ax_polar.plot(theta_c,true_LVK_c[c], c='lightgreen', label=r'$C_0$-Plane of the source LID')
ax_polar.plot(theta_c,amplified_c[c], c='red', label=r'$C_0$-Plane of measured LID, error amplified 20x')
ax_polar.legend(loc=(0.0,0.1))

fig.savefig('figures/Verformung_rel_i_r1_e0.001/x_error_polar_plot_cos100.svg')


# for error parts
#plot_cart(theta_c,lvk_factor_c[c],ax_cart2, c='lightblue', label='deformation factor')
#plot_cart(theta_c,len_factor_c[c],ax_cart2, c='yellow', label='length factor')
#plot_cart(theta_c,alpha_factor_c[c],ax_cart2, c='orange', label='incidence angle factor')
#plot_cart(theta_c,error_c[c],ax_cart2, c='darkblue', label='relative error')

# 3D visualization

# lower resolution grid for mesh of source LVK
theta_deg_mesh = np.linspace(start=0,stop=90,num=13)
phi_deg_mesh = np.linspace(start=0,stop=360,num=37)
theta_mesh = np.deg2rad(theta_deg_mesh)[:,None]
phi_mesh = np.deg2rad(phi_deg_mesh)[None,:]
mesh_LVK = LVK(theta_mesh,phi_mesh)

from mayavi import mlab
# select Data for plotting
# mayavi needs cartesian data points
plot_mesh = mesh_LVK
cart_mesh_body = polar2cart(theta_mesh,phi_mesh,plot_mesh)
Xm, Ym, Zm = np.moveaxis(cart_mesh_body,2,0)

plot_LVK = z_pos_amplified
cart_LVK_body = polar2cart(theta,phi,plot_LVK)
X, Y, Z = np.moveaxis(cart_LVK_body,2,0)

color_LVK = z_pos_error
# visu with mayavi

#meshframe = mlab.mesh(X, Y, Z, representation = 'mesh', tube_radius='0.3', tube_sides='12' , color=(0.8,0.8,0.8))
fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1100, 1100))
mlab.clf()
renderer = fig.scene.renderer
window = renderer.render_window
window.alpha_bit_planes_on()
window.multi_samples = 0
renderer.use_depth_peeling = 1
renderer.maximum_number_of_peels = 20
renderer.occlusion_ratio = 0.002
# for x-error, now all errors
vmin = -3 # values for colormap
vmax = 3

wframe = mlab.mesh(Xm, Ym, Zm, representation = 'mesh', tube_radius='0.3', tube_sides='12' , color=(0,0,0))
mesh = mlab.mesh(X, Y, Z, scalars=color_LVK, colormap='coolwarm', transparent=True, opacity=0.9, vmin=-vmin, vmax=vmax)
bar = mlab.scalarbar(object=mesh,orientation='vertical', label_fmt='%.1f', nb_labels=7) #title=r'  % of I0'
bar.title_text_property.font_size=28
bar.title_text_property.font_family='courier'
bar.title_text_property.bold = False
bar.data_range = (vmin,vmax)
bar.scalar_bar.unconstrained_font_size = True
bar.label_text_property.font_size=20
bar.label_text_property.font_family='courier'
title = mlab.title(r'Relative Error in Luminous Intensity for z-Offset',size=0.35, height=0.9)
title.property.font_family = 'courier'
extent=(-60,60,-60,60,-10,110)
mlab.outline(wframe,extent=extent)
xlabel = mlab.text3d(text='X',x=-6,y=-75,z=-20,scale=6)
ylabel = mlab.text3d(text='Y',x=65,y=15,z=-15,scale=6)
zlabel = mlab.text3d(text='Z',x=-72,y=-60,z=45,scale=6)
#mlab.view(-80, 50) # (x,y-axis)
mlab.view(-85, 83) # (z-axis)
fig.scene.parallel_projection = True
fig.scene.renderer.active_camera.zoom(0.65)

mlab.show()