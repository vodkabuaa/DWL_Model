
"""
Bob's analysis of the  carbon cycle
"""
import numpy as np
import pandas as pd

years=range(2000,2110,10)
scenarios=['2.6','4.5','6','8.5']
scenarios_all=scenarios+['A2','B1','IS92a']

#---Data--------------------------------------------------

"""
From the "Climate System Scenario Tables," Table AII.2.1.c, p. 1410
Anthropogenic total CO2 emissions (PgC/yr)
"""
#flows=pd.read_csv('flows.csv',header=0,index_col=0)
flow_array=np.array([
[8.03, 8.03, 8.03, 8.03],
[9.70, 9.48, 9.32, 9.98],
[9.97, 10.20, 9.37, 12.28],
[8.00, 11.06, 9.57, 14.53],
[5.30, 11.46, 10.80, 17.33],
[3.50, 11.15, 12.52, 20.61],
[2.10, 9.60, 14.46, 23.83],
[0.81, 7.27, 16.29, 26.17],
[0.16, 4.65, 17.07, 27.60],
[-0.23, 4.22, 14.94, 28.44],
[-0.42, 4.13, 13.82, 28.77]])

# flow over the 10 year period, divide by 2.3 to change units 
fl10 = pd.DataFrame(flow_array*(10/2.13),index=years,columns=scenarios) 

"""
From the "Climate System Scenario Tables," Table AII.4.1, p. 1422
CO2 abundance (ppm)
"""
#abundance=pd.read_csv('abundance.csv',header=0,index_col=0)
abundance_array=np.array([
[368.9, 368.9, 368.9, 368.9],
[389.3, 389.1, 389.1, 389.3],
[412.1, 411.1, 409.4, 415.8],
[430.8, 435.0, 428.9, 448.8],
[440.2, 460.8, 450.7, 489.4],
[442.7, 486.5, 477.7, 540.5],
[441.7, 508.9, 510.6, 603.5],
[437.5, 524.3, 549.8, 677.1],
[431.6, 531.1, 594.3, 758.2],
[426.0, 533.7, 635.6, 844.8],
[420.9, 538.4, 669.7, 935.9]])

base=pd.DataFrame(abundance_array,index=years,columns=scenarios)


"""
From the "Climate System Scenario Tables," Table AII.6, p. 1433
Effective Radiative Forcing Table AII.6.1 
ERF from CO2 (W/m^2)
"""
forcing_array=np.array([
[1.51, 1.51, 1.51, 1.51, 1.50, 1.50, 1.50],
[1.80, 1.80, 1.80, 1.80, 1.78, 1.77, 1.78],
[2.11, 2.09, 2.07, 2.15, 2.16, 2.09, 2.13],
[2.34, 2.40, 2.32, 2.56, 2.55, 2.38, 2.48],
[2.46, 2.70, 2.58, 3.03, 2.99, 2.69, 2.83],
[2.49, 2.99, 2.90, 3.56, 3.42, 2.98, 3.18],
[2.48, 3.23, 3.25, 4.15, 3.88, 3.20, 3.53],
[2.43, 3.39, 3.65, 4.76, 4.36, 3.37, 3.89],
[2.35, 3.46, 4.06, 5.37, 4.86, 3.49, 4.25],
[2.28, 3.49, 4.42, 5.95, 5.39, 3.57, 4.64],
[2.22, 3.54, 4.70, 6.49, 5.95, 3.59, 5.04]])

forcing_all=pd.DataFrame(forcing_array,index=years,columns=scenarios_all) 
forcing=forcing_all.filter(items=scenarios)

# --- Analysis ------------------------------------------------------------

fl_mp=(fl10+fl10.shift(1))/2
ghg_new=base.shift(1)+fl_mp
sink=ghg_new-base #implied tonnes of CO2 absorbed over preceding decade
base_mp=(base+base.shift(1))/2
rate_mp=sink/base_mp

print "Estimation of the rate of absorption for four scenarios:"
print 55*'='+'\n'
print rate_mp

"""
Estimate the sink function
"""
from scipy.optimize import fmin as fmin

css=sink.cumsum()  # cumulative sink up to end of preceding decade

def absor(x):
    a,b,c,d=x
    lsc=(b+c*css) # land_sea_concentration
    #print ls
    sink_est= a*np.sign(base_mp-lsc)*np.abs(base_mp-lsc)**d  #10-year absorption rate
    return sink_est

def abs_err(x):
    eps=(sink-absor(x))/sink
    ssr=(eps**2).sum().sum()
    return ssr
    
startingVals=np.array([0.05,100.,0.1,1.])
a_est = fmin(abs_err, startingVals, ftol=1.0e-9,maxfun=5000)
print a_est
a,b,c,d=a_est
print '\n'+'fractional errors in estimation of sink:'+'\n'+51*'='
print '\nActual sink:\n'
print sink
print '\nEstimated Sink:\n'
print absor(a_est)
print '\nActual rate:\n'
print sink/base_mp
print '\nEstimated Rate:\n'
print absor(a_est)/base_mp
print '\nFractional Error (Sink-Sink_est/GHG):\n'
print (absor(a_est)-sink)/base_mp


"""
Now fit the atmospheric forcing function
"""
print '\n'+"Estimation of the Forcing Function:"+'\n'+34*'='+'\n'

def force3(x):
    a,b,c = x
    return a*np.sign(base-b)*(abs(base-b))**c

def f3(x):
    eps=forcing-force3(x)
    ssr=(eps**2).sum().sum()
    return ssr

from scipy.optimize import fmin as fmin
startingVals=np.array([0.05,280,0.75])
x_est = fmin(f3, startingVals, ftol=1.0e-9,maxfun=5000)
print x_est
[a,b,c]=x_est

print '\n'+'fractional errors in estimation of forcing function:'+'\n'+51*'='+'\n'
print (forcing-a*(base-b)**c)/forcing

def force3b(g,x):
    a,b,c = x
    return a*np.sign(g-b)*(abs(g-b))**c

xx=np.arange(210,990)

import matplotlib.pyplot as plt

plt.close('all')
plt.plot(xx,force3b(xx,x_est),'b-')
plt.xlabel('GHG Concentration (ppm)')
plt.ylabel("Forcing")
plt.plot(np.array(list(np.ndarray.flatten(base.values))),np.array(list(np.ndarray.flatten(forcing.values))),'r+')
plt.axhline(color='k',linestyle=':')
plt.title('Plot of %5.3f(GHG-%3i)^%4.2f' % tuple(x_est))
plt.savefig('forcing.pdf')


"""
Now fit the atmospheric forcing function, with the cutoff fixed at 280
"""
print '\n'+"Estimation of the Forcing Function:"+'\n'+34*'='+'\n'

def force2(x):
    a,c = x
    b=280.
    return a*np.sign(base-b)*(abs(base-b))**c

def f2(x):
    eps=forcing-force2(x)
    ssr=(eps**2).sum().sum()
    return ssr

from scipy.optimize import fmin as fmin
startingVals=np.array([0.05,0.75])
x2_est = fmin(f2, startingVals, ftol=1.0e-9,maxfun=5000)
print x2_est
[a,c]=x2_est

print '\n'+'fractional errors in estimation of forcing function:'+'\n'+51*'='+'\n'
print (forcing-a*(base-280)**c)/forcing

def force2b(g,x):
    a,c = x
    b=280
    return a*np.sign(g-b)*(abs(g-b))**c

xx=np.arange(210,990)

plt.close('all')
plt.plot(xx,force2b(xx,x2_est),'b-')
plt.xlabel('GHG Concentration (ppm)')
plt.ylabel("Forcing")
plt.plot(np.array(list(np.ndarray.flatten(base.values))),np.array(list(np.ndarray.flatten(forcing.values))),'r+')
plt.axhline(color='k',linestyle=':')
plt.title('Plot of %5.3f(GHG-280)^%4.2f' % tuple(x2_est))
plt.savefig('forcing2.pdf')
