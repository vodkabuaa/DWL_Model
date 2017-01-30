import numpy as np
import random
import dlw_utility as fm

class optimize_plan(object):
    '''Includes functions and parameters to control the optimization of the climate model
    '''
    def __init__(self,my_tree,derivative_check=0,alt_input=0,randomize=0.,
                 filename='\\Users\\Bob Litterman\\Dropbox\\EZ Climate calibration paper\\dlw code\\monteparams6.txt'):
        '''Initializes the optimization class

        Parameters
        ----------
        my_tree : tree object
            Definition of the tree structure

        derivative_check : integerif = 1, then print analytic and numerical derivatives else check whether they match and if not give error message

        alt_input : integer
            if = 1, then read input guess from an alternative input file

        filename : string
            if alt_input = 1, then read input guess from an alternative input file 'filename'
        randomize : float
            if greater than 0, then introduces a random term to the initial guess

        '''
        self.my_tree = my_tree
        self.derivative_check = derivative_check
        self.alt_input = alt_input
        self.randomize = randomize
        self.filename = filename

    def get_initial_guess(self):
        '''    read initial mitigation guess from a file
                six values per row
        '''
        guess = np.zeros(self.my_tree.x_dim)+.5
        print ("getting mitigation guess from the file 'params.txt'")
        f = open('bestparams.txt', 'r')
        rows = int(self.my_tree.x_dim / 6)-1
        rest = rows*6
        for ip in range(0, (rows-1)*6+1, 6):
            guess[ip], guess[ip+1], guess[ip+2], guess[ip+3], guess[ip+4], guess[ip+5] = [float(x) for x in f.readline().split()]

        guess[rest], guess[rest+1], guess[rest+2], guess[rest+3], guess[rest+4], guess[rest+5], guess[rest+6]  = [float(x) for x in f.readline().split()]
        f.close()
        self.guess = guess

    def get_initial_guess6(self):
        '''    read initial mitigation guess from a file
                six values per row
        '''
        guess = np.zeros(self.my_tree.x_dim)+.5
        if self.alt_input == 0 :
            print ("getting mitigation guess from the file 'bestparams6.txt'")
            f = open('bestparams6.txt', 'r')
        else :
            print ("getting mitigation guess from the file ", self.filename)
            f = open(self.filename, 'r')
        rows = int(self.my_tree.x_dim / 6)-1
        rest = rows*6
        for ip in range(0, (rows-1)*6+1, 6):
            guess[ip], guess[ip+1], guess[ip+2], guess[ip+3], guess[ip+4], guess[ip+5] = [float(x) for x in f.readline().split()]

        guess[rest], guess[rest+1], guess[rest+2], guess[rest+3], guess[rest+4], guess[rest+5], guess[rest+6], guess[rest+7], guess[rest+8]  = [float(x) for x in f.readline().split()]
        if self.randomize > 0 :
            for ip in range(0, self.my_tree.x_dim):
                guess[ip] = max(0, guess[ip]*(1.0+random.normalvariate(0., self.randomize)))
        f.close()
        self.guess = guess

    def put_optimal_plan(self, plan):
        '''    write optimal mitigation plan to a file
                six values per row
        '''
        print ("putting optimal mitigation plan on the file 'bestparams.txt'")
        f = open('bestparams.txt', 'w')
        rows = int(self.my_tree.x_dim / 6)-1
        rest = rows*6
        for ip in range(0, (rows-1)*6+1, 6):
            f.writelines( str(plan[ip]) + "\t" + str(plan[ip+1]) + "\t" + str(plan[ip+2]) + "\t" + str(plan[ip+3]) + "\t" + str(plan[ip+4]) + "\t" + str(plan[ip+5]) + "\n")

        f.writelines( str(plan[rest]) + "\t" + str(plan[rest+1]) + "\t" + str(plan[rest+2]) + "\t" + str(plan[rest+3]) + "\t" + str(plan[rest+4]) + "\t" + str(plan[rest+5]) + "\t" + str(plan[rest+6]) + "\n" )
        f.close()

    def put_optimal_plan6(self, plan):
        '''    write optimal mitigation plan to a file
                six values per row
        '''
        print ("putting optimal mitigation plan on the file 'bestparams6.txt'")
        f = open('bestparams6.txt', 'w')
        rows = int(self.my_tree.x_dim / 6)-1
        rest = rows*6
        for ip in range(0, (rows-1)*6+1, 6):
            f.writelines( str(plan[ip]) + "\t" + str(plan[ip+1]) + "\t" + str(plan[ip+2]) + "\t" + str(plan[ip+3]) + "\t" + str(plan[ip+4]) + "\t" + str(plan[ip+5]) + "\n")

        f.writelines( str(plan[rest]) + "\t" + str(plan[rest+1]) + "\t" + str(plan[rest+2]) + "\t" + str(plan[rest+3]) + "\t" + str(plan[rest+4]) + "\t" + str(plan[rest+5]) + "\t" + str(plan[rest+6]) + "\t" + str(plan[rest+7]) + "\t" + str(plan[rest+8]) + "\n" )
        f.close()

    def set_constraints(self,constrain=-1,node_0=0.41419732,node_1=.55, node_2=.35):
        '''  set upper and lower boundary constraints on mitigation
        '''
        self.constrain = constrain

        xlbarray = np.zeros(self.my_tree.x_dim)
        xubarray = np.ones(self.my_tree.x_dim) + .25
        for n in range(1, self.my_tree.x_dim):
            xubarray[n] *= (1. + 4.*n/self.my_tree.x_dim)
        '''     first period mitigation can be constrained here
        '''
        if constrain <0:
          xlbarray[0] = node_0
          xubarray[0] = node_0
        if constrain >= 1:
          xlbarray[0] = node_0
          xubarray[0] = node_0
        if constrain >= 2:
          xlbarray[1] = node_1
          xubarray[1] = node_1
        if constrain >= 3:
          xlbarray[2] = node_2
          xubarray[2] = node_2

        self.xbounds = [[ xlbarray[0], xubarray[0] ]]
        for bnd in range(0, self.my_tree.x_dim-1):
            self.xbounds.append( [ xlbarray[bnd+1], xubarray[bnd+1] ] )

        return

    def create_output(self, best_mitigation_plan, best_fit, my_damage_model, my_cost_model, my_tree):
        '''   writes the output of the optimization to the console
        Parameters
        ----------

        best_mitigation_plan : float
            vector of optimal degrees of mitigation

        best_fit : float
            the value of the utility function at the best_mitigation_plan

        my_damage_model : damage_class object
            the damage model used in the optimization

        my_cost_model : cost_class object
            the cost model used in the optimization

        '''
        if my_tree.analysis >= 1 :
            if my_tree.print_options[0] == 1:
                print ('Print_Option[0] Maximized_utility_=', -best_fit)
        if my_tree.analysis >= 1 :
            if my_tree.print_options[1] == 1:
                print ('Print_Option[1] Optimized_mitigation_plan_=')
                lines = int(my_tree.x_dim/6)-1
                for line in range(0, lines) :
                    for i in range(0,5) :
                        print (best_mitigation_plan[line*6+i])
                    print (best_mitigation_plan[line*6+5])
                for i in range(lines*6,my_tree.x_dim-1) :
                        print (best_mitigation_plan[i])
                print (best_mitigation_plan[my_tree.x_dim-1])

        price = my_cost_model.price_by_state( best_mitigation_plan[0],0.,0.)

        if my_tree.analysis >= 1 :
            if my_tree.print_options[2] == 1:
                print ('Print_Option[2] Social_Cost_of_Carbon_=', price)

        if my_tree.print_options[3] == 1:
            emissions_to_bau = my_tree.emissions_to_ghg[my_tree.nperiods-1] / my_tree.emissions_per_period[my_tree.nperiods-1]
            bau_path = 400
            for p in range(0, my_tree.nperiods):
                ave_price = 0.
                first_node = my_tree.decision_period_pointer[p]
                this_period_time = my_tree.decision_times[p+1] - my_tree.decision_times[p]
                for n in range(0, my_tree.decision_nodes[p]):
                    average_mitigation = best_mitigation_plan[0] * my_tree.decision_times[1]
                    for pp in range(1, p):
                        j = int(n / 2**(p-pp))
                        average_mitigation += best_mitigation_plan[ my_tree.node_map[pp-1][ my_tree.node_mapping[pp-1][j][0] ]] * (my_tree.decision_times[pp+1]-my_tree.decision_times[pp])
                    price = my_cost_model.g * my_cost_model.a * best_mitigation_plan[first_node+n]**(my_cost_model.a-1.)
                    if p == 0 :
                        consump = 1.
                        average_mitigation = 0.
                    else :
                        consump = self.my_tree.potential_consumption[p]
                        average_mitigation = average_mitigation / my_tree.decision_times[p]
                        price = my_cost_model.price_by_state( best_mitigation_plan[first_node+n], average_mitigation, my_tree.decision_times[p] )
                    ave_price += my_tree.node_probs[first_node+n] * price
                    average_mitigation = my_tree.ave_mitigation[first_node+n]
                    average_emissions = my_tree.additional_emissions_by_state[first_node+n] / (this_period_time*emissions_to_bau)
                    print ('Print_Option[3] Period', p, 'time', int(2015+my_tree.decision_times[p]), \
                            'node', first_node+n, 'has_prob', my_tree.node_probs[first_node+n], 'Emission_mitigation_of',\
                            best_mitigation_plan[first_node+n], 'Price', price, 'Consumption', \
                            consump*(1.-my_tree.damage_by_state[first_node+n])*(1.-my_tree.cost_by_state[first_node+n]),\
                            'Average_mitigation', average_mitigation, 'Cost', my_tree.cost_by_state[first_node+n], 'Damage',\
                            my_tree.damage_by_state[first_node+n],' GHG_level_in_state', my_tree.ghg_by_state[first_node+n], \
                            'Average_annual_emissions_in_state', average_emissions)
                bau_path += emissions_to_bau * my_tree.emissions_per_period[p]
                print ('Print_Option[3] Period', p, 'time', int(2015+my_tree.decision_times[p]),\
                        'Average_price', ave_price, 'BAU_average_annual_emissions_in_period',\
                        my_tree.emissions_per_period[p]/this_period_time, 'end_of_period_bau_ghg_level', bau_path)

            print ('Print_Option[3] Final_period_consumption_and_damage')
            for state in range(0, my_tree.final_states):
                print ('Print_Option[3] Period', my_tree.nperiods, 'time', int(2015+my_tree.decision_times[my_tree.nperiods]),\
                 'final_state', state, 'consumption', my_tree.potential_consumption[my_tree.nperiods]*(1.-my_tree.final_damage_by_state[state]), \
                 'forward_damage', my_tree.final_damage_by_state[state])
        ''' use root finder and the function "find_term_structure" to find the bond price (and yield) that reflects the value of a fixed $1 payment in all nodes at time np '''
        from scipy.optimize import brentq
        np = my_tree.utility_nperiods-2
        utility = fm.utility_function( best_mitigation_plan, self.my_tree, my_damage_model, my_cost_model )
        res = brentq( self.find_term_structure, 0., .9999, args=( my_tree, my_damage_model, self, my_cost_model, np))
        res = max( .00000000001, res)
        my_tree.discount_prices[np] = res
        years_to_maturity = my_tree.utility_times[np]

        if my_tree.print_options[4] == 1:
            print ('Print_Option[4] Zero_coupon_bond_maturing_at_time', 2015+5*np, 'has_price_=',\
                    res, 'and_yield_=', 100. * (1./(res**(1./years_to_maturity))-1.))
        '''
            output for the decomposition of SCC into expected damage and risk premium
        '''
        utility = fm.utility_function( best_mitigation_plan, self.my_tree, my_damage_model, my_cost_model )
        base_grad = fm.analytic_utility_gradient(best_mitigation_plan, self.my_tree, my_damage_model, my_cost_model )
        d_cost_sum = 0.0
        discounted_expected_damages = 0.0
        net_discounted_expected_damages = 0.0
        risk_premium = 0.0

        if my_tree.analysis == 2 :
            consumption_cost = my_tree.d_consumption_by_state[0]
            if my_tree.print_options[5] == 1:
                print ('Print_Option[5] Period_0_delta_consumption', consumption_cost)
            if my_tree.print_options[6] == 1:
                print ('Print_Option[6] Period_0_marginal_utility_wrt_c(0)', my_tree.marginal_utility_by_state[0][0],\
                        'Period_0_marginal_utility_wrt_c(node1)_up_node', my_tree.marginal_utility_by_state[0][1],\
                        'Period_0_marginal_utility_wrt_c(node2)_down_node',my_tree.marginal_utility_by_state[0][2])
            '''
                in this loop calculate and print the expected damage and the risk premium term at each point in time
            '''
            my_tree.sdf_in_tree[0] = 1.0

            for time_period in range(1, self.my_tree.utility_nperiods):
                '''  for a given time_period in the utility_tree, tree_node points to the first node in the period of the last decision in the decision tree'''
                tree_node = my_tree.decision_period_pointer[ min( my_tree.nperiods-1, my_tree.utility_decision_period[time_period-1]+1) ]
                ''' first_node points to the first node in period time_period of the utility tree '''
                first_node = my_tree.utility_period_pointer[time_period]
                expected_damages = 0.0
                expected_sdf = 0.0
                cross_product_sdf_damages = 0.0
                '''
                    Now loop over all the nodes in the utility tree in period time_period
                    find the expected damages, the expected Stochastic Discount Factor, and the Covariance term
                '''
                for period_node in range(0, my_tree.utility_period_nodes[time_period]):
                    '''
                        d_consumption_by_state holds the change in consumption per unit change in emissions mitigation
                        in all periods except the first this "damage_in_node" equals the damage in the node caused by additional emissions
                        note however, in the first period this change also reflects the change in cost, the d_cost term, from reducing the mitigation
                        calculate the expected damage by probability weighting the damage in each node
                    '''
                    damage_in_node = (my_tree.d_consumption_by_state[first_node+period_node])
                    expected_damages += damage_in_node * my_tree.node_probs[tree_node+period_node]
                    if time_period <= my_tree.print_options[7] :
                        ''' if this is a risk decomposition and utility sub-interval output is desired '''
                        print ('Print_Option[7] Period', time_period, 'node', first_node+period_node, 'delta_consumption',\
                         my_tree.d_consumption_by_state[first_node+period_node], ' consumption_level',\
                          my_tree.consumption_by_state[first_node+period_node])
                    ''' from_node is the node from the previous period that leads to period_node '''
                    if my_tree.information_period[time_period-1] == 1 :
                        from_node = my_tree.utility_period_pointer[time_period-1] + int(period_node/2)
                    else :
                        from_node = my_tree.utility_period_pointer[time_period-1] + period_node

                    if my_tree.information_period[time_period-1] == 1 :
                        '''
                            information_period = 1 when there is a branch in the tree
                            first if (even values of period_node) considers upper branch nodes, second if considers lower branch nodes
                        '''
                        if int(period_node/2)*2 == period_node :
                            ''' total_prob is the sum of the probabilities in the up and down nodes reached from "from_node" '''
                            total_prob = my_tree.node_probs[tree_node+period_node] + my_tree.node_probs[tree_node+period_node+1]
                            ''' sdf is the stochastic discount factor required to discount consumption in period_node back to from_node '''
                            sdf = (total_prob/my_tree.node_probs[tree_node+period_node]) * my_tree.marginal_utility_by_state[from_node][1]/my_tree.marginal_utility_by_state[from_node][0]
                            if time_period <= my_tree.print_options[7] :
                                print ('Print_Option[7] Branch_from_node', from_node, 'MU_wrt_c(t)', my_tree.marginal_utility_by_state[from_node][0])
                                print ('Print_Option[7] Node:', tree_node+period_node,'SDF',sdf,'MU_wrt_c(t+1)_in_node', my_tree.marginal_utility_by_state[from_node][1])
                                print ('Print_Option[7] Probs:_up_node', tree_node+period_node, 'down_node', tree_node+period_node+1, 'probs', my_tree.node_probs[tree_node+period_node], my_tree.node_probs[tree_node+period_node+1])
                        else:
                            total_prob = my_tree.node_probs[tree_node+period_node] + my_tree.node_probs[tree_node+period_node-1]
                            sdf = (total_prob/my_tree.node_probs[tree_node+period_node]) * my_tree.marginal_utility_by_state[from_node][2]/my_tree.marginal_utility_by_state[from_node][0]
                            if time_period <= my_tree.print_options[7] :
                                print ('Print_Option[7] Branch_from_node', from_node, 'MU_wrt_c(t)', my_tree.marginal_utility_by_state[from_node][0])
                                print ('Print_Option[7] Node:', tree_node+period_node, 'SDF', sdf, 'MU_wrt_c(t+1)_in_node', my_tree.marginal_utility_by_state[from_node][2], 'MU_wrt_c(t)', my_tree.marginal_utility_by_state[from_node][0])
                                print ('Print_Option[7] Probs:_down_node', tree_node+period_node, 'up_node', tree_node+period_node-1, 'probs', my_tree.node_probs[tree_node+period_node], my_tree.node_probs[tree_node+period_node-1])
                    else:
                        ''' if no branching occurs this period then the probability of reaching period_node is 1 '''
                        sdf = my_tree.marginal_utility_by_state[from_node][1]/my_tree.marginal_utility_by_state[from_node][0]
                        mu_next = my_tree.marginal_utility_by_state[from_node][1]
                        ''' in the final period the marginal utility is with respect to the steady-state continuation value '''
                        if time_period == my_tree.utility_nperiods-1 :
                            sdf = my_tree.final_total_derivative_term[period_node]/my_tree.marginal_utility_by_state[from_node][0]
                            mu_next = my_tree.final_total_derivative_term[period_node]
                        if time_period <= my_tree.print_options[7] :
                            print ('Print_Option[7] No_branch_from_node', from_node, 'SDF', sdf, 'MU_wrt_c(t)', my_tree.marginal_utility_by_state[from_node][0], 'MU_wrt_c(t+1)_in_next_node', mu_next)
                    ''' sdf_in_tree is the discount factor used to present value consumption in node period_node (at time time_period) '''
                    my_tree.sdf_in_tree[first_node+period_node] = my_tree.sdf_in_tree[ from_node ] * sdf
                    ''' the expected_sdf is the probability weighted discount factors in the nodes at time time_period '''
                    expected_sdf += my_tree.sdf_in_tree[first_node+period_node] * my_tree.node_probs[tree_node+period_node]
                    ''' cross_product_sdf_damages is the expected cross_product of sdf's and damages at time time_period '''
                    cross_product_sdf_damages += my_tree.sdf_in_tree[first_node+period_node] * damage_in_node * my_tree.node_probs[tree_node+period_node]
                ''' store the expected sdf, which is the present value at time 0 of $1 received at time time_period'''
                my_tree.discount_prices[time_period] = expected_sdf
                ''' cov_term is the cross_product minus the product of the expected SDF and the expected damage at time time_period '''
                cov_term = cross_product_sdf_damages - expected_sdf * expected_damages

                ''' now calculated the components of net discounted damage and the risk premium which arise in period time_period per $ spent on mitigation at time 0 (consumption_cost) '''
                ''' if sub_interval time_period occurs during the first decision period we need to account for the impact of the cost of increased marginal mitigation
                    on consumption during this interval in order to be left with only the marginal damage impact on consumption '''
                if my_tree.utility_decision_period[ time_period] == 0 :
                    net_discounted_damage = -(expected_damages+my_tree.d_cost_by_state[time_period,1])*expected_sdf/consumption_cost
                    if my_tree.print_options[10] == 1 :
                        print ('Print_Option[10] Period', time_period, 'expected_damages', -expected_damages/consumption_cost, 'discount_price', expected_sdf, 'cross_product', -cross_product_sdf_damages/consumption_cost, 'cov_term', -cov_term/consumption_cost, 'net_discounted_damage', net_discounted_damage,'d_cost', -my_tree.d_cost_by_state[time_period,1]/consumption_cost)
                    ''' sum the present value of the costs of mitigation throughout the first decision period per $ spent on mitigation at time 0 '''
                    d_cost_sum += -my_tree.d_cost_by_state[time_period,1] * expected_sdf / consumption_cost
                else :
                    net_discounted_damage = -expected_damages*expected_sdf/consumption_cost
                    if my_tree.print_options[10] == 1 :
                        print ('Print_Option[10] Period', time_period, 'expected_damages', -expected_damages/consumption_cost, 'discount_price', expected_sdf, 'cross_product', -cross_product_sdf_damages/consumption_cost, 'cov_term', -cov_term/consumption_cost, 'net_discounted_damage', net_discounted_damage)
                my_tree.net_expected_damages[time_period] = net_discounted_damage
                my_tree.risk_premium[time_period] = -cov_term/consumption_cost
                ''' sum (over time) the components of the present value of expected damages and the risk premium'''
                discounted_expected_damages += -expected_damages * expected_sdf / consumption_cost
                ''' the net discounted damages nets out the cost associated with increasing mitigation throughout the first period '''
                net_discounted_expected_damages += net_discounted_damage
                risk_premium += -cov_term / consumption_cost

        ''' if desired, print out the SDF's and Marginal Utilities for the decision tree nodes'''
        for p in range(0, self.my_tree.nperiods):
            first_node = self.my_tree.decision_period_pointer[p]
            for n in range(0, self.my_tree.decision_nodes[p]):
                ''' no branching in the last period '''
                if p == self.my_tree.nperiods-1 :
                    if my_tree.print_options[6] == 1 :
                        print ('Print_Option[6] Period', p,'time', int(2015+my_tree.decision_times[p]), 'node', first_node+n, 'SDF=', my_tree.marginal_utility_in_tree[first_node+n,1] / my_tree.marginal_utility_in_tree[ first_node+n, 0 ])
                        print ('Print_Option[6] Marginal_utility(c(t)) ', my_tree.marginal_utility_in_tree[first_node+n, 0], 'Marginal_utility(c(t+1))', self.my_tree.marginal_utility_in_tree[first_node+n, 1])
                else :
                    if my_tree.print_options[6] == 1 :
                        to_node = self.my_tree.next_node[first_node+n][0]
                        prob_up = self.my_tree.node_probs[to_node]
                        prob_down = self.my_tree.node_probs[to_node+1]
                        total_prob = prob_up + prob_down
                        print ('Print_Option[6] Period', p, 'time', int(2015+my_tree.decision_times[p]), 'node', first_node+n, 'SDF_up=  ', (total_prob/prob_up) * self.my_tree.marginal_utility_in_tree[first_node+n,1] / self.my_tree.marginal_utility_in_tree[ first_node+n, 0 ])
                        print ('Print_Option[6] SDF_down=', (total_prob/prob_down) * self.my_tree.marginal_utility_in_tree[first_node+n,2] / self.my_tree.marginal_utility_in_tree[ first_node+n, 0 ], 'MU ', self.my_tree.marginal_utility_in_tree[first_node+n, 0], 'Marginal_utility(c(t+1))_up ', self.my_tree.marginal_utility_in_tree[first_node+n, 1])
                        print ('Print_Option[6] Marginal_utility(c(t+1))_down ', self.my_tree.marginal_utility_in_tree[first_node+n, 2])

        ''' if desired, print out the levels of GHG in each node '''
        if my_tree.print_options[8] == 1 :
            for p in range(0, my_tree.nperiods):
                first_node = my_tree.decision_period_pointer[p]
                for n in range(0, my_tree.decision_nodes[p]):
                    print ('Print_Option[8] Period', p, 'time', int(2015+my_tree.decision_times[p]), 'node', first_node+n, 'GHG_level=', my_tree.ghg_by_state[first_node+n])
            first_node = my_tree.x_dim
            for n in range(0, my_tree.final_states):
                print ('Print_Option[8] Period', my_tree.nperiods, 'time', int(2015+my_tree.decision_times[my_tree.nperiods]), 'node', first_node+n, 'GHG_level= ', my_tree.ghg_by_state[first_node+n])
        if my_tree.analysis == 2 :
            total = net_discounted_expected_damages + risk_premium
            '''  decompose the Social Cost of Carbon into the expected damages component versus the risk premium '''
            print ('Social_cost_of_carbon', my_cost_model.price_by_state( best_mitigation_plan[0],0.,0.))
            print ('Discounted_expected_damages', (net_discounted_expected_damages/total) * my_cost_model.price_by_state( best_mitigation_plan[0],0.,0.))
            print ('Risk_premium', (risk_premium/total) * my_cost_model.price_by_state( best_mitigation_plan[0],0.,0.))

            ''' print the decomposition of expected damages and risk premium over time  '''
            if my_tree.print_options[9] == 1:
                damage_scale = my_cost_model.price_by_state( best_mitigation_plan[0],0.,0.)/(net_discounted_expected_damages+risk_premium)
                for np in range(1, my_tree.utility_nperiods):
                    my_tree.net_expected_damages[np] *= damage_scale
                    my_tree.risk_premium[np] *= damage_scale
                    print ('Print_Option[9] Period', np, 'Year', 2015+np*my_tree.sub_interval_length, 'Net_discounted_expected_damage', my_tree.net_expected_damages[np], 'Risk_premium', my_tree.risk_premium[np])
        ''' if desired, print out the yield curve '''
        if my_tree.print_options[4] == 1 :
            if my_tree.analysis == 2 :
                for np in range(1, my_tree.utility_nperiods-1):
                    years_to_maturity = self.my_tree.utility_times[ np ]
                    print ('Print_Option[4] Period', np, 'years-to-maturity', years_to_maturity, 'price of bond', self.my_tree.discount_prices[np], ' yield ', 100. * (1./(self.my_tree.discount_prices[np]**(1./years_to_maturity))-1.))
                ''' find the yield on a perpetuity that begins paying at the time of the steady state continuation term '''
                np = my_tree.utility_nperiods-1
                years_to_maturity = self.my_tree.utility_times[ np ]
                perp_yield = brentq( self.perpetuity_yield, 0.1, 10., args=( np*5, self.my_tree.discount_prices[np]))
                print ('Print_Option[4] Period', my_tree.utility_nperiods-1, 'years-to-maturity', years_to_maturity, 'price of bond', self.my_tree.discount_prices[np], ' yield ', perp_yield)
        return

    '''
       function to call from optimizer to find the risk free zero coupon bond yield
    '''
    def find_ir(self, price, *var_args):
        '''
          Function called by a zero root finder which is used
          to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
          the purpose of this function is to find the interest rate embedded in the EZ Utility model

          first calculate the utility with a final payment
        '''
        my_tree = var_args[0]
        my_damage_model = var_args[1]
        my_optimization = var_args[2]
        my_cost_model = var_args[3]
        final_payment = 0.01

        self.my_tree.final_period_consumption_epsilon = final_payment
        utility_with_final_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        self.my_tree.final_period_consumption_epsilon = 0.
        '''
          then calculate the utility with an initial payment equal to the final payment discounted to today at the target interest_rate
        '''
        self.my_tree.first_period_epsilon = final_payment * price
        utility_with_initial_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        self.my_tree.first_period_epsilon = 0.0
        distance = (utility_with_final_payment - utility_with_initial_payment)

        return(distance)

    '''
       function to call from optimizer to find the break-even consumption to equalize utility from a constrained optimization
    '''
    def find_bec(self, delta_con, *var_args):
        '''
          Function called by a zero root finder which is used
          to find a value for consumption that equalizes utility at time 0 in two different solutions

          first calculate the utility with a base case
        '''
        my_tree = var_args[0]
        my_damage_model = var_args[1]
        my_optimization = var_args[2]
        my_cost_model = var_args[3]
        base_case = my_optimization.guess

        '''
        base_case_utility = fm.utility_function( base_case, my_tree, my_damage_model, my_cost_model )
          then calculate the utility with the alternative case

        alternative_case_utility = fm.utility_function( alternative_case, my_tree, my_damage_model, my_cost_model )

        distance = (alternative_case_utility - base_case_utility)
        '''
        my_tree.first_period_epsilon = 0.0

        base_utility = fm.utility_function(base_case, my_tree, my_damage_model, my_cost_model )

        my_tree.first_period_epsilon = delta_con

        new_utility = fm.utility_function(base_case, my_tree, my_damage_model, my_cost_model )

        my_tree.first_period_epsilon = 0.0

        distance = (new_utility-base_utility)-my_optimization.constraint_cost

        return(distance)

    def find_term_structure(self, price, *var_args):
        '''
          Function called by a zero root finder which is used
          to find the price of a bond that creates equal utility at time 0 as adding .01 to the value of consumption in the final period
          the purpose of this function is to find the interest rate embedded in the EZ Utility model

          first calculate the utility with a final payment
        '''
        my_tree = var_args[0]
        my_damage_model = var_args[1]
        my_optimization = var_args[2]
        my_cost_model = var_args[3]
        time_period = var_args[4]
        payment = 0.01

#        self.my_tree.period_consumption_epsilon[time_period] = 0.
#        utility_with_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        self.my_tree.period_consumption_epsilon[time_period] = payment
        utility_with_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        self.my_tree.period_consumption_epsilon[time_period] = 0.
#        utility_without_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        '''
          then calculate the utility with an initial payment equal to the final payment discounted to today at the target interest_rate
        '''
        self.my_tree.first_period_epsilon = payment * price
        utility_with_initial_payment = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
        self.my_tree.first_period_epsilon = 0.0
        distance = (utility_with_payment - utility_with_initial_payment)

        return(distance)

    def perpetuity_yield(self, perp_yield, *var_args):
        '''
          Function called by a zero root finder which is used
          to find the yield of a perpetuity starting at year start_date

        '''
        start_date = var_args[0]
        price = var_args[1]
        distance = price - (100. / (perp_yield+100.))**start_date * (perp_yield + 100)/perp_yield

        return(distance)

    def create_consumption(self, best_mitigation_plan, best_fit, my_damage_model, my_cost_model, my_tree):
        '''   writes the output of consumption to the console
        Parameters
        ----------

        best_mitigation_plan : float
            vector of optimal degrees of mitigation

        best_fit : float
            the value of the utility function at the best_mitigation_plan

        my_damage_model : damage_class object
            the damage model used in the optimization

        my_cost_model : cost_class object
            the cost model used in the optimization

        '''
        print (' best fit ', best_fit)
        print (' starting output', best_mitigation_plan)
        '''
            in this loop calculate and print the consumption at each node in the tree
        '''
        print (my_tree.consumption_by_state)
        for time_period in range(1, self.my_tree.utility_nperiods):
            tree_node = my_tree.decision_period_pointer[ min( my_tree.nperiods-1, my_tree.utility_decision_period[time_period-1]+1) ]
            first_node = my_tree.utility_period_pointer[time_period]
            for period_node in range(0, my_tree.utility_period_nodes[time_period]):
                print ('time',time_period, 'first_node', first_node, 'period_node', period_node, 'consumption', my_tree.consumption_by_state[first_node+period_node])
