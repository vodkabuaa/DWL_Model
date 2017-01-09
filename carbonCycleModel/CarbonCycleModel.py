# -*- coding: utf-8 -*-
"""
Bob's analysis of the  carbon cycle
Encapsulate by Weiyu
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import fmin as fmin
import matplotlib.pyplot as plt

class CarbonCycleModel():
    def __init__(self):
        ###set path
        path = os.getcwd()

        #---Data Preparation--------------------------------------------------
        """
        From the "Climate System Scenario Tables," Table AII.2.1.c, p. 1410
        Anthropogenic total CO2 emissions (PgC/yr)
        """
        flow_df = pd.read_csv(path + "\\IPCC_data\\flow.csv", index_col=0)
        # flow over the 10 year period, divide by 2.3 to change units
        fl10 = flow_df * 10 / 2.13

        """
        From the "Climate System Scenario Tables," Table AII.4.1, p. 1422
        CO2 abundance (ppm)
        """
        abundance_df = pd.read_csv(path + "\\IPCC_data\\abundance.csv", index_col=0)

        """
        From the "Climate System Scenario Tables," Table AII.6, p. 1433
        Effective Radiative Forcing Table AII.6.1
        ERF from CO2 (W/m^2)
        """
        forcing_all_df = pd.read_csv(path + "\\IPCC_data\\forcing.csv", index_col=0)
        forcing_df = forcing_all_df.iloc[:,0:4]

        # --------------------- Analysis --------------------#
        self.base = abundance_df
        fl_mp = (fl10 + fl10.shift(1)) / 2
        ghg_new = self.base.shift(1) + fl_mp
        self.sink = ghg_new - self.base #implied tonnes of CO2 absorbed over preceding decade
        self.base_mp = (self.base + self.base.shift(1)) / 2
        self.rate_mp = self.sink / self.base_mp
        self.css = self.sink.cumsum()
        self.forcing = forcing_df

        #----------Model Estimation----------------#
        self.sink_model = self.estimate_sink_model()
        self.param_forcing3_model = self.estimate_forcing3_model()
        self.param_forcing2_model = self.estimate_forcing2_model()

    #--------------Model Estimator--------------------# Fitting least Square

    def estimate_sink_model(self):
        """
        Estimate the sink model parameters
        """
        startingVals = np.array([0.05,100.,0.1,1.])
        param_sink_model = fmin(self.abs_err, startingVals, ftol=1.0e-9,maxfun=5000)
        return param_sink_model

    def estimate_forcing3_model(self):
        """
        Estimate the forcing model parameters
        """
        startingVals=np.array([0.05,280,0.75])
        param_forcing3_model = fmin(self.f3, startingVals, ftol=1.0e-9,maxfun=5000)
        return param_forcing3_model

    def estimate_forcing2_model(self):
        """
        Estimate the forcing2 model parameters
        """
        startingVals = np.array([0.05,0.75])
        param_forcing2_model = fmin(self.f2, startingVals, ftol=1.0e-9,maxfun=5000)
        return param_forcing2_model


    #---------------------- Sink Model----------------#
    def absor(self, x):
        """
        absorb function
        """
        a, b, c, d = x
        lsc=(b + c * self.css) # land_sea_concentration
        #print ls
        sink_est= a * np.sign(self.base_mp - lsc) * np.abs(self.base_mp - lsc)**d  #10-year absorption rate
        return sink_est

    def abs_err(self, x):
        eps = (self.sink - self.absor(x)) / self.sink
        ssr = (eps**2).sum().sum()
        return ssr

    #-----------------Forcing3 Model------------------#
    def force3(self, x):
        a,b,c = x
        return a * np.sign(self.base-b) * (abs(self.base-b))**c

    def f3(self, x):
        eps = self.forcing - self.force3(x)
        ssr = (eps**2).sum().sum()
        return ssr
    #use to plot
    def force3b(self, g, x):
        a,b,c = x
        return a*np.sign(g - b)*(abs(g - b))**c

    #-----------------Forcing2 Model------------------#
    def force2(self, x):
        a,c = x
        b = 280.
        return a * np.sign(self.base - b) * (abs(self.base - b))**c

    def f2(self, x):
        eps = self.forcing-self.force2(x)
        ssr = (eps**2).sum().sum()
        return ssr
    # use to plot
    def force2b(self, g, x):
        a, c = x
        b = 280
        return a * np.sign(g - b) * (abs(g - b))**c

    #---------------- Print analysis result for different model -------#
    def print_sink_analysis(self):
        print ('\n'+'fractional errors in estimation of sink:'+'\n'+51*'=')
        print ('\nActual sink:\n')
        print (self.sink)
        print ('\nEstimated Sink:\n')
        print (self.absor(self.sink_model))
        print ('\nActual rate:\n')
        print (self.sink / self.base_mp)
        print ('\nEstimated Rate:\n')
        print (self.absor(self.sink_model) / self.base_mp)
        print ('\nFractional Error (Sink-Sink_est/GHG):\n')
        print ((self.absor(self.sink_model) - self.sink) / self.base_mp)

    def print_forcing3_analysis(self):
        [a,b,c] = self.param_forcing3_model
        print ('\n'+'fractional errors in estimation of forcing function:'+'\n'+51*'='+'\n')
        print ((self.forcing - a * (self.base - b)**c) / self.forcing)

    def print_forcing2_analysis(self):
        [a,c] = self.param_forcing2_model
        print ('\n'+'fractional errors in estimation of forcing function:'+'\n'+51*'='+'\n')
        print ((self.forcing - a * (self.base - 280)**c) / self.forcing)

    #---------------Plot--------------------#
    def plot_forcing3(self):
        xx = np.arange(210,990)
        plt.close('all')
        para = self.param_forcing3_model
        plt.plot(xx, self.force3b(xx, self.param_forcing3_model),'b-')
        plt.xlabel('GHG Concentration (ppm)')
        plt.ylabel("Forcing")
        plt.plot(np.array(list(np.ndarray.flatten(self.base.values))),np.array(list(np.ndarray.flatten(self.forcing.values))),'r+')
        plt.axhline(color='k',linestyle=':')
        plt.title('Plot of %5.3f(GHG-%3i)^%4.2f' % tuple(self.param_forcing3_model))
        plt.savefig('forcing.pdf')

    def plot_forcing2(self):

        xx = np.arange(210,990)
        plt.close('all')

        plt.plot(xx, self.force2b(xx, self.param_forcing2_model),'b-')
        plt.xlabel('GHG Concentration (ppm)')
        plt.ylabel("Forcing")
        plt.plot(np.array(list(np.ndarray.flatten(self.base.values))),np.array(list(np.ndarray.flatten(self.forcing.values))),'r+')
        plt.axhline(color='k', linestyle=':')
        plt.title('Plot of %5.3f(GHG-280)^%4.2f' % tuple(self.param_forcing2_model))
        plt.savefig('forcing2.pdf')
