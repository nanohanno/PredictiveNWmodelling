#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate

# Parameters
ETA = 3.25          # droplet shape factor
LAMBDA = 2400 #1200     # diffusion length in nm
GR_VS_0 = 5.616   # linear factor for VS growth rate from fit to Tersoff model
GR_AX =  76        # GR_AXial growth rate in nm/min
RADIUS_0 = 15    # initial NW radius in nm
# Define range of V/III ratio
RATIO_MIN = 1.4
RATIO_MAX = 30
RATIO_INCREMENT = 0.2
RATIO_RANGE = np.arange(RATIO_MIN, RATIO_MAX, RATIO_INCREMENT)
# Define range of growth time in min
TIME_MIN=10
TIME_MAX=120
TIME_INCREMENT=1
TIME_RANGE = np.arange(TIME_MIN, TIME_MAX, TIME_INCREMENT)
# Define NW length array for solution (in nm)
X_MIN = GR_AX*5. # Calculate length at beginning of second growth step
X_VLS_STOP = GR_AX*TIME_MAX    # Calculate length at end of second growth step
X_INCREMENT = 20
X_RANGE_VLS = np.arange(X_MIN, X_VLS_STOP, X_INCREMENT)

def derivative_diameter_vls(y, x, PARAMETERS_VLS):  # Calculate the derivative of the diameter in Tersoffs model at a given heigth and length
            r = y     
            ETA, R, LAMBDA = PARAMETERS_VLS  
            R_eff = R*0.459
            derivs = 0.84/(ETA*(3+ETA**2))*(1/R_eff*(1+LAMBDA/((1+ETA**2)*r))-1)
            return derivs

def gr_vs(t,gr_vs_tot,GR_AX,LAMBDA,y):  # Calculate the VS growth rate at a given time and height
            y_total = GR_AX*t
            if y<y_total:
                gr_vs = gr_vs_tot*(1-math.exp(-(GR_AX*t-y)/LAMBDA))
            else:
                gr_vs = 0
            return gr_vs

def size_untapered(tapering_series_at_t, diameter_untapered, length_untapered, taper_untapered ,ratio_untapered , radius_total_top, TIME_RANGE, R_i):  # Calculate the diameter, length, flux ratio, and top diameter of the untapered NW of a time series
    i_taper_min = np.argmin([abs(x) for x in tapering_series_at_t])  # Index of time with lowest tapering
        
    if abs(tapering_series_at_t[i_taper_min]) < 0.02 : #assure that it is not only minimum but also untapered
        diameter_untapered.append(2*radius_total_top[i_taper_min])   #list of diameter at lowest tapering
        length_untapered.append(TIME_RANGE[i_taper_min]*GR_AX)
        taper_untapered.append(i_taper_min)    #list of indices with lowest tapering
        ratio_untapered.append(R_i)
    return diameter_untapered, length_untapered, taper_untapered, ratio_untapered



tapering_for_ratio_time = np.empty([0, len(TIME_RANGE)])
taper_untapered = []
diameter_untapered = []
length_untapered = []
ratio_untapered = []

for R_i in RATIO_RANGE:
    print('Completion: ' + repr(round(np.where(RATIO_RANGE == R_i)[0][0]/len(RATIO_RANGE)*100, 1))+ '%')
    tapering_series_at_t = []
    radius_total_top = []
    
    PARAMETERS_VLS = [ETA, R_i, LAMBDA]  
    radius_vls = integrate.odeint(derivative_diameter_vls, RADIUS_0, X_RANGE_VLS, args=(PARAMETERS_VLS,))  #solve ODE for VLS model
    
    for t_i in TIME_RANGE:
        radius_vs = []
        
        gr_vs_tot = GR_VS_0/R_i   # Calculate nominal growth rate from flux ratio
        x_total_stop = GR_AX*t_i   # Calculate length of NW
        x = np.arange(X_MIN, x_total_stop, X_INCREMENT)  # Define length axis values for NW
                
        for x_i in x:
            VS_int = integrate.quad(gr_vs,5,t_i,args=(gr_vs_tot,GR_AX,LAMBDA,x_i))
            radius_vs[len(radius_vs):] = [VS_int[0]]
        
        radius_total = radius_vls[:len(x),0] + radius_vs #calculate total radius
        radius_total_top.append(radius_total[len(x)-1])  #diameter at top of NW for different times
        
        taper_i = -1*(radius_total[len(x)-1]-radius_total[6])/x_total_stop*100 # taper at one point, compared to x=500nm
        tapering_series_at_t.append(taper_i)         #calculate taper values of time series
                
    diameter_untapered, length_untapered, taper_untapered, ratio_untapered = size_untapered(tapering_series_at_t, diameter_untapered, length_untapered, taper_untapered, ratio_untapered, radius_total_top, TIME_RANGE, R_i)
    
    tapering_for_ratio_time = np.append(tapering_for_ratio_time, [tapering_series_at_t], axis=0)

diameter_length_untapered = np.transpose(np.array([length_untapered, diameter_untapered]))



#####Output and File save

fname = "Map-time%d_%d_%d-ratio%d_%d_%d-lambda%d.csv" % (TIME_MIN,TIME_MAX,TIME_INCREMENT,RATIO_MIN,RATIO_MAX,RATIO_INCREMENT,LAMBDA)    
fname_size = "size-flux%d_%d_%d-ratio%d_%d_%d-lambda%d.csv" % (TIME_MIN,TIME_MAX,TIME_INCREMENT,RATIO_MIN,RATIO_MAX,RATIO_INCREMENT,LAMBDA)    

np.savetxt(fname, tapering_for_ratio_time, delimiter=",")
np.savetxt(fname_size, diameter_length_untapered, delimiter=",")

    
    
fig, (ax1,ax2) = plt.subplots(1,2)    
im=ax1.imshow(tapering_for_ratio_time,extent=[TIME_MIN,TIME_MAX,RATIO_MIN,RATIO_MAX],interpolation='none',cmap="bwr", vmin=np.amin(tapering_for_ratio_time), vmax=-np.amin(tapering_for_ratio_time), origin=['lower'])
plt.colorbar(im)
ax1.set_aspect('auto')
ax1.plot(TIME_RANGE[taper_untapered], ratio_untapered, 'g-', linewidth=3)

ax2.scatter(length_untapered,diameter_untapered)
ax2.set_xlabel('NW length')
ax2.set_ylabel('Diameter of straight NW')
plt.show()


