import numpy as np

class tree_model(object):
    ''' This file contains code used to create a tree object
        this tree is used in the utility optimization of the daniel,litterman,wagner climate risk model

        Functions used in the tree class

        Functions
        ---------
    '''
    '''   six period initialization    '''
    def __init__(self,tp1=15,analysis=1,final_states=32,nperiods=6, peak_temp_interval=30.,x_dim=63,
                 sub_interval_length=5,prob_scale=1.0,growth=.02,eis=0.9,ra=7.0,time_pref=.005,
                 decision_times = [ 0, 15, 45, 85, 185, 285, 385],
                 print_options = [ 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1 ] ):
#                 print_options = [ 1, 1, 1, 1, 1,  1, 1, 0, 1, 1, 1 ] ):
#        '''   five period initialization    '''
#    def __init__(self,tp1=15,analysis=1,final_states=16,nperiods=5,peak_temp_interval=30.,x_dim=31,
#                 sub_interval_length=5,prob_scale=1.0,growth=.02,eis=.9,ra=7.,time_pref=.005,
#                 decision_times = [ 0., 35., 85., 185., 285., 385.],
#                 print_options = [ 1, 1, 1, 0, 0,  0, 0, 0, 0, 1, 0 ] ):
#                 print_options = [ 1, 1, 1, 1, 1,    1, 1, 5, 1, 1, 1 ] ):

        ''' Initializes a tree structure for use in the dlw climate model

        Parameters
        ----------
        analysis : integer
            0 = no optimization,
            1 = optimiztionon of mitigation path,
            2 = risk decomposition,
            3 = marginal cost of waiting
            4 = deadweight loss#                 print_options = [ 1, 1, 1, 1, 1,    1, 1, 5, 1, 1 ] ):
 of waiting

        final_states : integer
            the number of final states at time T

        nperiods : integer
            the number of periods in the model (=T)

        peak_temp_interval : float
            the normalization of the peak_temp parameter period length, which specifies the probability of a tipping point(temperature) over a given time interval

        x_dim : integer
            the number of nodes in the tree where decisions are made = dimension of the vector x of optimal mitigations

        sub_interval_length : integer
            the number of eqully spaced times in the utility tree between decision tree periods

        prob_scale : float
            parameter that determines the probabilities of the nodes (1 -> equal prob) (>1 implies explore the tail with more states)

        growth : float
            exogenous growth rate of consumption

        eis : float
            elasticity of intertemporal substitution

        ra : float
            coefficient of risk aversion

        time_pref : float
            pure rate of time discount of future utility

        decision_times :  float vector
            times of the tree in which decisions are made on mitigation

        print_options :choose the options for printing:
            0 implies off,
            1 implies on, except for sub-interval detail, in which case n implies output for the first n utility nodes

            [ maximized utility, mitigation plan, social cost of carbon, decision nodes, bond yield
              delta consumption, SDF's and marginal utilities, sub-interval detail, GHG levels, SCC decomposition
              SCC decomposition intermediate calculations]
        '''
        ## useless in this class
        self.analysis = analysis
        print ("analysis =", analysis)
        ## the interval between decision times including many subintevals
        decision_times[1] = tp1
        print ("time_period_one", tp1)

        ##############?###### whhter is derivates
        self.final_states = final_states
        self.nperiods = nperiods
        self.peak_temp_interval = peak_temp_interval
        self.x_dim = x_dim
#
#        self.utility_breaks = utility_breaks
#
        self.sub_interval_length = sub_interval_length

        self.prob_scale = prob_scale
        self.growth = growth
        self.eis = eis
        self.ra = ra
        self.time_pref = time_pref
        self.decision_times = decision_times
        self.print_options = print_options

        self.create_node_map()
        self.create_node_mapping()
        self.create_next_node()

        self.full_tree = self.x_dim + self.final_states
        self.funcalls = 0

        self.decision_nodes = [ 1 ]
        self.decision_period_pointer = [ 0 ]
        self.total_time = self.decision_times[nperiods]

        self.utility_nperiods = int(self.total_time / self.sub_interval_length)+1
        print ('utility_nperiods', self.utility_nperiods)

        for p in range(1, self.nperiods):
            self.decision_nodes.append( 2**p )
            self.decision_period_pointer.append( self.decision_period_pointer[p-1] + self.decision_nodes[p-1] )

        u_time = 0.
        self.first_period_intervals = int(self.decision_times[1]/self.sub_interval_length)
        self.first_period_epsilon = 0.0
        self.utility_times = [ u_time ]
        self.decision_period = [ 1 ]
        self.information_period = [ 1 ]
        first_node = 0
        self.utility_period_pointer = [ first_node ]
        self.utility_period_nodes = [ 1 ]
        tree_period = 0
        self.utility_decision_period = [ tree_period ]
        self.breaks = [0]

        """
        Since the interval of decsion time is different, the sub interval length is the same. Therefore, different decision time period have
        different number of sub_interval_length.


        How to save the edge? This is really a problem!!!!!

        """
        for p in range(0, self.nperiods):
            u_period = self.sub_interval_length
            self.breaks.append( int( (self.decision_times[p+1]-self.decision_times[p])/self.sub_interval_length))

            for brk in range(0, self.breaks[p+1]):
                if brk == 0 :
                    nodes = self.decision_nodes[p]
                else :
                    nodes = self.decision_nodes[min(p+1, self.nperiods-1)]
                    self.decision_period.append( 0 )
                    self.information_period.append( 0 )
                    u_time += u_period
                    self.utility_times.append( u_time )
                first_node += nodes
                self.utility_period_pointer.append( first_node )
                self.utility_period_nodes.append( nodes )
                self.utility_decision_period.append( p )


            u_time += u_period
            self.utility_times.append(u_time)
            self.decision_period.append( 1 )
            if p < self.nperiods-2 :
                self.information_period.append( 1 )
            else :
                self.information_period.append( 0 )


        for p in range(0, self.utility_nperiods-2):
            self.utility_period_nodes[p] = self.utility_period_nodes[p+1]
            self.utility_decision_period[p] = self.utility_decision_period[p+1]

        if self.breaks[self.nperiods]==0:
            self.utility_period_nodes[self.utility_nperiods-2] = self.utility_period_nodes[self.utility_nperiods-3]*2
            self.utility_decision_period[self.utility_nperiods-2] = self.utility_decision_period[self.utility_nperiods-3]+1
            self.utility_period_pointer.append( self.utility_period_pointer[self.utility_nperiods-2] + self.utility_period_nodes[self.utility_nperiods-2] )

        self.utility_full_tree = self.utility_period_pointer[self.utility_nperiods-1]+self.final_states
        self.create_probs()
        self.allocate_data_structures()
        self.final_period_consumption_epsilon = 0.
        self.initial_consumption = 1.



        ##################### the business-as-usual model should be in another class.###############

        '''   emissions growth is assumed to slow down exogenously -- these assumptions
              represent an attempt to model emissions growth in a business-as-usual scenario
              that is in the absence of incentives
        '''
        self.bau_emit_time = [ 0, 30, 60 ]
        self.bau_emit_level = [ 52.0, 70.0, 81.4]
        self.bau_emissions_setup()

        print ("initializing tree", "\n number of periods =", self.nperiods, "\n number of nodes in tree =", x_dim,"\n")
        print (" probability scale paramter =", self.prob_scale)
        print (" probabilities of final states \n ", self.probs, "\n")

    def bau_of_t(self, time):
        '''     return the bau emissions at any time t   '''
        if time < self.bau_emit_time[1] :
            bau_emissions = self.bau_emit_level[0] + float(time) / (self.bau_emit_time[1] - self.bau_emit_time[0] ) * ( self.bau_emit_level[1]-self.bau_emit_level[0] )
        elif time < self.bau_emit_time[2] :
            bau_emissions = self.bau_emit_level[1] + (float(time) - self.bau_emit_time[1]) / (self.bau_emit_time[2] - self.bau_emit_time[1] ) * ( self.bau_emit_level[2]-self.bau_emit_level[1] )
        else :
            bau_emissions = self.bau_emit_level[2]
        return(bau_emissions)


    def bau_emissions_setup(self):
        '''     create default business as usual emissions path

                the emissions rate in each period are assumed to be the average of the emissions at the beginning
                and at the end of the period
        '''
        total_emissions = self.emissions_per_period[0]
        for n in range(0, self.nperiods):
            this_period_time = self.decision_times[n+1] - self.decision_times[n]
            self.emissions_per_period[n] = this_period_time * ( self.bau_of_t(self.decision_times[n]) + self.bau_of_t(self.decision_times[n+1]) ) / 2.
            total_emissions += self.emissions_per_period[n]
        '''     the total increase in ghg level of 600 (from 400 to 1000) in the bau path is allocated over time
        '''
        for n in range(0, self.nperiods):
            self.emissions_to_ghg[n] = ( 600 / total_emissions ) * self.emissions_per_period[n]
        return


#######################################

    def create_node_map(self):
        '''  sets up a tree structure
             create node_map
             the node map is also a set of pointers used by the utility function
             the node_map is a 2-dimensional integer array: [nperiods-1] x [final_states]
             which specifies for each final state the node from period p which points to that state
             for periods 1 through nperiods  (period 0 reaches all states)

             also creates period_map which provides the period associated with any particular node
             period_map is a x_dim length vector

  node_map[p][s]
           provides a pointer to the node in period p which points to state s

           for example, with nperiods = 5
  node_map: [[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
             [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
             [7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
             [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
        '''


        self.node_map = []
        self.period_map = [ 0 ]
        from_n = 1

        for p in range(0, self.nperiods-1):
            node_from = []
            for p_nodes in range(0, self.final_states):
                add_n = int(p_nodes/(2**(self.nperiods-2-p)))
                node_from.append( from_n+add_n )
            last_n = from_n+add_n+1
            for p_nodes in range(from_n, last_n):
                self.period_map.append( p+1 )
            from_n = last_n
            self.node_map.append( node_from )

        return


    def create_node_mapping(self):
        '''  sets up a tree structure
               create node_mapping -- nodes are points in the tree structure

               an example: the node numbers for a 3 period tree structure
                   (with 4 final states and seven decisions on mitigation to optimize over) is shown below

    period:   0    1     2

     nodes:                       final_states
                         3             0
                   1
                         4             1
              0
                         5             2
                   2
                         6             3

     the node mapping is a set of pointers used by the utility function
     the node_mapping is a 3-dimensional integer array: [nperiods-2] x [decision_nodes[p+1]] x [2]
          which specifies the final states reachable from each node in the tree
          for periods 1 through nperiods-1  (period 0 reaches all states, in the last period the state is known)

     node_mapping[p][j][0]
           provides a pointer to the first state in the partition reached by the j'th node in period p
     node_mapping[p][j][1]
           provides a pointer to the last state in the partition reached by the j'th node in period p

           for example, with nperiods = 5
  node_mapping: [[[0, 7], [8, 15]],
                 [[0, 3], [4, 7], [8, 11], [12, 15]],
                 [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]]
        '''

        self.node_mapping = []
        for p in range(1, self.nperiods-1):
            node_range = []
            first_node = 0
            for p_nodes in range(0,2**p):
                last_node = first_node + 2**(self.nperiods-1-p)-1
                node_range.append( [first_node, last_node] )
                first_node = last_node+1
            self.node_mapping.append(  node_range  )

        return

    def create_next_node(self):
        '''  sets up the pointers for use in a tree structure

        the next_node array is a set of pointers used by the utility function
          next_node is a 2-dimensional integer array: [x_dim - final_states] x [2]
          which specifies for each node in periods 0 through nperiods-2
          what are the node numbers for the first and last nodes pointed to in the next period
          for example, node 0 (in period 0) points to nodes 1 and 2 in period 1
                       node 1 (in period 1) points to nodes 3 and 4 in period 2
                       node 2 (in period 1) points to nodes 5 and 6 in period 2
                       node 3 (in period 2) points to nodes 7 and 8 in period 3

         next_node[n][0]
               provides a pointer to the first node in the next period pointed to from node n
         next_node[n][1]
               provides a pointer to the last node in the next period pointed to from node n

         note that in this particular tree structure each node always points to 2 nodes, but this mapping
           allows that constraint to be relaxed

               for example, with nperiods = 5
         next_node: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20],
                       [21, 22], [23, 24], [25, 26], [27, 28], [29, 30]]

           also computed in this loop is node_num
           node_num provides, for each node, the number of nodes pointed to in the next period
                 again in this construction node_num always = 2, but this can be relaxed

           node_num: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        '''
        self.next_node = []
        self.node_num = []
        first_node = 1
        for n in range(0, 2**(self.nperiods-1)-1):
            self.next_node.append( [ first_node, first_node+1 ])
            first_node += 2
            self.node_num.append( 2 )

        return


    def create_probs(self):
        '''  creates the probabilities of the final states in the tree structure

           here are the probabilities of each of the final states
             in the Monte Carlo with S total simulations, the damages are ordered from highest to lowest
             in state 0 with probability probs[0] let N0 = probs[0] x S
             the damage coefficient is the average damage in the first N0 ordered simulation damages
             in state 1 with probability probs[1] let N1 = probs[1] X S
             the damage coefficient is the average damage in the next N1 ordered simulation damages

               in other words, the states represent the unknown degrees of fragility of the environment

          prob_scale determines the relative probabilites in the array probs
          they can be made equal by setting prob_scale = 1.0
          they increase if prob_scale > 1, which means the we can potentially more accurately investigate
          the impact of fatter tails

        for example, with prob_scale = 1.0
      probs:  [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
               0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

        with prob_scale = 1.5
      probs:  [0.022511, 0.033767, 0.041357, 0.047342, 0.052392, 0.056818, 0.0607906, 0.064415, 0.067764,
               0.070887, 0.073820, 0.076592, 0.079224, 0.081734, 0.084136, 0.0864418]
        '''
        self.probs =  [1.]
        sum_probs = 1.
        next_prob = 1.
        for n in range(1, self.final_states):
            next_prob = next_prob * self.prob_scale**(1./n)
            self.probs.append( next_prob )
            sum_probs += next_prob
        for n in range(0, self.final_states):
            self.probs[n] = self.probs[n] / sum_probs

        self.node_probs = np.ones(self.x_dim)
        probs_ra = np.zeros([self.final_states] )
        for n in range(0,self.final_states):
            probs_ra[n] = self.probs[n]
        for n in range(0,self.final_states):
            self.probs[n] = probs_ra[n]
        for n in range(0, self.final_states):
            self.node_probs[self.decision_period_pointer[self.nperiods-1]+n] = self.probs[n]
        for p in range(1,self.nperiods-1):
            for n in range(0, self.decision_nodes[self.nperiods-1-p]):
                sum_probs = 0.
                for ns in range( self.node_mapping[self.nperiods-2-p][n][0], self.node_mapping[self.nperiods-2-p][n][1]+1):
                    sum_probs += self.probs[ns]
                self.node_probs[self.node_map[self.nperiods-2-p][self.node_mapping[self.nperiods-2-p][n][0]]] = sum_probs
        return


    def allocate_data_structures(self):
        '''   Creates data structures to store tree values
        '''

        self.ave_mitigation = np.zeros(self.full_tree)
        self.potential_consumption = np.zeros(self.nperiods+1)
        for p in range(0, self.nperiods+1):
            self.potential_consumption[p] = (1.0+self.growth)**self.decision_times[p]

        self.consumption_by_state = np.zeros(self.utility_full_tree)
        self.damage_by_state = np.zeros(self.x_dim)
        self.final_damage_by_state = np.zeros(self.final_states)
        self.final_total_derivative_term = np.zeros(self.final_states)
        self.cost_by_state = np.zeros(self.x_dim)
        self.marginal_utility_by_state = np.zeros([self.utility_full_tree,3])
        self.marginal_utility_in_tree = np.zeros([self.full_tree,3])
        self.sdf_in_tree = np.zeros(self.utility_full_tree)
        self.ghg_by_state = np.zeros(self.full_tree)
        self.emissions_per_period = np.zeros(self.nperiods)
        self.cum_emissions_in_scenarios = np.zeros([self.nperiods,3])
        self.GHG_levels_in_scenarios = np.zeros([self.nperiods+1,3])
        self.forcing_per_period = np.zeros([self.nperiods,3])
        self.cum_forcing_per_period = np.zeros([self.nperiods+1,4])
        self.cum_forcing_by_state = np.zeros([self.full_tree,3])
        self.emissions_to_ghg = np.zeros(self.nperiods)
        self.additional_emissions_by_state = np.zeros(self.full_tree)
        self.utility_by_state = np.zeros(self.utility_full_tree)
        self.continuation_utility = np.zeros(self.final_states)
        self.cert_equiv_utility = np.zeros(self.utility_full_tree)
        self.ce_term = np.zeros(self.utility_full_tree)
        self.period_consumption_epsilon = np.zeros(self.utility_nperiods+1)
        self.node_consumption_epsilon = np.zeros(self.utility_full_tree)
        self.discount_prices = np.zeros(self.utility_nperiods+1)
        self.net_expected_damages = np.zeros(self.utility_nperiods+1)
        self.risk_premium = np.zeros(self.utility_nperiods+1)
        self.marginal_damages = np.zeros(self.utility_full_tree)

        ''' variables starting with the d_ are derivatives with respect to mitigation, x[n] '''
        self.d_consumption_by_state = np.zeros(self.utility_full_tree)
        self.d_cost_by_state = np.zeros([self.first_period_intervals,2])
        self.d_damage = np.zeros(self.utility_full_tree)
        self.d_final_damage_by_state = np.zeros([self.final_states, self.x_dim])
        self.d_utility_of_final_state = np.zeros([self.final_states, self.x_dim])
        self.d_utility_by_state = np.zeros([self.utility_full_tree, self.x_dim])
        self.d_cons_by_state = np.zeros([self.utility_full_tree, self.x_dim])

        self.grad = np.zeros(self.x_dim)

        return

    def ghg_levels(self,x):
        ''' calculates the ghg levels for each node in the tree
           additional_emissions_by_state(t,n) = [potential emissions(t)] * [1. - mitigation(t,n)]
           ghg_levels_by_state(t,n) = glg_levels_by_state(t-1,n) + additional_emissions_by_state(t-1,n)
        '''

        for p in range(0, self.nperiods):
            first_node = self.decision_period_pointer[p]
            for n in range(0, self.decision_nodes[p]):
                self.additional_emissions_by_state[first_node+n] = (1.0-x[first_node+n])* self.emissions_to_ghg[p]

        self.ghg_by_state[0] = 400.

        for p in range(1, self.nperiods):
            first_node = self.decision_period_pointer[p]
            previous_first_node = self.decision_period_pointer[p-1]
            for n in range(0, self.decision_nodes[p]):
                previous_node = previous_first_node+int(n/2)
                self.ghg_by_state[first_node+n] = self.ghg_by_state[previous_node] + self.additional_emissions_by_state[previous_node]

        first_node = self.decision_period_pointer[self.nperiods-1]+self.final_states
        previous_first_node = self.decision_period_pointer[self.nperiods-1]
        for n in range(0, self.final_states):
            previous_node = previous_first_node + n
            self.ghg_by_state[first_node+n] = self.ghg_by_state[previous_node] + self.additional_emissions_by_state[previous_node]

        return
