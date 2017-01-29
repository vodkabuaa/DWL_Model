#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class TreeModel:
    def __init__(self, decision_times=[ 0, 15, 45, 85, 185, 285, 385],\
                sub_interval = 5, prob_scale = 1):
        """
        Here we only consider the decision nodes and periods.
        Since the last period is not uncertain, it will not a
        complete num_decision_nodes
        """
        self.decision_times = decision_times
        self.sub_interval = sub_interval
        self.prob_scale = prob_scale


        self.num_period = len(decision_times) - 1
        self.num_decision_nodes = 2**self.num_period - 1
        self.num_final_state = 2**(self.num_period - 1)

        self.damage_by_state = np.zeros(self.num_decision_nodes)
        self.cost_by_state = np.zeros(self.num_decision_nodes)
        self.grad = np.zeros(self.num_decision_nodes)

        ### nodes probability
        self.final_states_prob = np.zeros(self.num_final_state)
        self.node_prob = np.zeros(self.num_decision_nodes)

        #### emissions
        self.emissions_per_period = np.zeros(self.num_period)
        self.emissions_to_ghg = np.zeros(self.num_period)


        ### Initialize the probability
        self.create_probs()

        self.create_subintevals()
        ### Initialize subintervals


    def get_pos(self, period, state):
        """
           We can use the relationship between the period, state and index of
           these ndarrays to get the node number by O1.
        """
        if state >= 2**period :
            print("error: Index out of boundary")
            return []
        pos = 2**period + state
        return pos - 1

    def create_subintevals(self):
        """
        The subintrevals is calculated in the following step:
        1. calculate the number of subintervals based on the given length of subinterval
        2. use a dictionary to record the sub intervals. The key is node pairs, such as(0,1), (0,2) or (1,3) and the
        value is a ndarray use to record the information of intervals.

        """
        edges = {}
        num_subinterval = [ int((y - x)/5) for x,y in zip(self.decision_times[:-1],self.decision_times[1:])]
        for period in range(0,len(num_subinterval)-1):
            for state in range(0, 2**period):
                pos = 2**period-1 + state
                nums = num_subinterval[period] + 2
                edges[(pos, 2*pos+1)] = np.zeros(nums)
                edges[(pos, 2*pos+2)] = np.zeros(nums)
        self.sub_intervals = edges

    def create_probs(self):
        '''
           Creates the probabilities of the final states and every nodes in the tree structure;
           Here are the probabilities of each of the final states in the Monte Carlo with S total simulations,
           the damages are ordered from highest to lowest
           in state 0 with probability probs[0] let N0 = probs[0] x S
           the damage coefficient is the average damage in the first N0 ordered simulation damages
           in state 1 with probability probs[1] let N1 = probs[1] X S
           the damage coefficient is the average damage in the next N1 ordered simulation damages

           In other words, the states represent the unknown degrees of fragility of the environment
           prob_scale determines the relative probabilites in the array probs, they can be made equal by
           setting prob_scale = 1.0;they increase if prob_scale > 1, which means the we can potentially
           more accurately investigate the impact of fatter tails.

           for example, with prob_scale = 1.0
           probs:  [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
                    0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

           with prob_scale = 1.5
           probs:  [0.022511, 0.033767, 0.041357, 0.047342, 0.052392, 0.056818, 0.0607906, 0.064415, 0.067764,
                    0.070887, 0.073820, 0.076592, 0.079224, 0.081734, 0.084136, 0.0864418]
        '''
        self.final_states_prob[0] = 1.
        sum_probs = 1.
        next_prob = 1.
        ##Calculate the probability for the final state
        for n in range(1, self.num_final_state):
            next_prob = next_prob * self.prob_scale**(1./n)
            self.final_states_prob[n] = next_prob
        self.final_states_prob /= np.sum(self.final_states_prob)


        self.node_prob[self.num_final_state-1:] = self.final_states_prob
        for period in range(0, self.num_period-1)[::-1]:
            for state in range(0, 2**period):
                pos = self.get_pos(period, state)
                self.node_prob[pos] = self.node_prob[2*pos+1] + self.node_prob[2*pos+2]
        return

def test_tree_model():
    print ("case1: Default")
    my_tree = TreeModel()
    print ("my_tree.node_prob")
    print (my_tree.node_prob)
    print ("my_tree.sub_intervals")
    print (my_tree.sub_intervals)

    print ("case2: scale prob, decision_times=[ 0, 15, 45, 85, 185, 285],\
    sub_interval = 5, prob_scale = 1.5")
    my_tree = TreeModel()
    print ("my_tree.node_prob")
    print (my_tree.node_prob)
    print ("my_tree.sub_intervals")
    print (my_tree.sub_intervals)



if __name__ == "__main__":
    test_tree_model()
