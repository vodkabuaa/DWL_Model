#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Shu Ye
"""
import numpy as np
from tree_model_class import *

class BusinessAsUsual:
    def __init__(self, bau_emit_time = [0, 30, 60], bau_emit_level = [52.0, 70.0, 81.4]):
        """
        emissions growth is assumed to slow down exogenously -- these assumptions
        represent an attempt to model emissions growth in a business-as-usual scenario
        that is in the absence of incentives
        """
        self.nperiods = tree.num_period
        self.decision_times = tree.decision_times
        self.emissions_per_period = np.zeros(self.nperiods)
        self.emissions_to_ghg = np.zeros(self.nperiods)
        self.bau_emit_time = bau_emit_time
        self.bau_emit_level = bau_emit_level

    def emission_by_time(self, time):
        """
        return the bau emissions at any time t
        """
        if time < self.bau_emit_time[1] :
            emissions = self.bau_emit_level[0] + time / (self.bau_emit_time[1] - self.bau_emit_time[0]) * \
                            (self.bau_emit_level[1] - self.bau_emit_level[0])
        elif time < self.bau_emit_time[2] :
            emissions = self.bau_emit_level[1] + (time - self.bau_emit_time[1]) / (self.bau_emit_time[2] - \
                            self.bau_emit_time[1]) * (self.bau_emit_level[2] - self.bau_emit_level[1])
        else :
            emissions = self.bau_emit_level[2]

        return emissions

    def bau_emissions_setup(self, tree):
        """
        create default business as usual emissions path
        the emissions rate in each period are assumed to be the average of the emissions at the beginning
        and at the end of the period
        """
        for n in range(0, self.nperiods):
            this_period_time = self.decision_times[n+1] - self.decision_times[n]
            self.emissions_per_period[n] = this_period_time * (self.emission_by_time(self.decision_times[n]) + \
                                           self.emission_by_time(self.decision_times[n+1])) / 2
        total_emissions = self.emissions_per_period.sum()

        #the total increase in ghg level of 600 (from 400 to 1000) in the bau path is allocated over time
        self.emissions_to_ghg = 600 * self.emissions_per_period / total_emissions
        tree.emissions_per_period = self.emissions_per_period
        tree.emissions_to_ghg = self.emissions_to_ghg
        return


if __name__ == "__main__":
    tree = TreeModel()
    bau_default_model = BusinessAsUsual()
    bau_default_model.bau_emissions_setup(tree)
    print ("tree.emissions_per_period", tree.emissions_per_period)
    print ("tree.emissions_to_ghg", tree.emissions_to_ghg)
