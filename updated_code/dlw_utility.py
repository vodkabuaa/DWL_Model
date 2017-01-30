'''
   Python function code for dlw climate model
   Functions to calculate and optimize utility function
'''
import math
import numpy as np
from numba import jit

def utility_function(x,*var_args):
    '''
       first step: calculate the final period utility conditional on the state
       (states index fragility of the environment)
       future_damages_by_state[n] gives the final period present value of future damages in state n,
       these future damages depend on the given choices of emissions reductions in prior periods, x[0]...
       as well as emissions reductions in state n, x[n]
    '''
    my_tree = var_args[0]
    my_damage_model = var_args[1]
    my_cost_model = var_args[2]

    period = my_tree.nperiods

    ''' r is the parameter rho from the dlw paper
        a is alpha in the dlw paper
        b is beta in the dlw paper  (for continuation use an annual discounting period)
    '''
    period_length = my_tree.utility_times[1] - my_tree.utility_times[0]
    r = ( 1.0 - 1.0 / my_tree.eis)
    a = ( 1.0 - my_tree.ra)
    b = (1.0 - my_tree.time_pref)**period_length

    first_node = my_tree.x_dim
    utility_periods = my_tree.utility_nperiods-2
    first_utility_node = my_tree.utility_period_pointer[utility_periods] + my_tree.utility_period_nodes[utility_periods]
    for n in range(0, my_tree.final_states):
        #print("calculate average mitigation")
        my_tree.ave_mitigation[first_node+n] = my_damage_model.average_mitigation(x,first_node+n)
        #print("calculate damage_function")
        my_tree.final_damage_by_state[n] = my_damage_model.damage_function(x, first_node+n)
        '''
           we assume growth continues from the final_state forward, in which case EZ continuation utiity converges to the value continuation
        '''
        growth_term = (1. + my_tree.growth)

        continuation = (1. / ( 1. - b * growth_term**r ))**(1./r)
        '''
           utility_by_state in the final period:  a function of potential consumption, reduced by damages( x )
        '''
        my_tree.consumption_by_state[first_utility_node+n] = my_tree.potential_consumption[period] * (1. - my_tree.final_damage_by_state[n])
        my_tree.utility_by_state[first_utility_node+n] = (1. - b)**(1./r) * my_tree.consumption_by_state[first_utility_node+n] * continuation
#        print 'util calc', continuation, my_tree.consumption_by_state[first_utility_node+n],my_tree.utility_by_state[first_utility_node+n]
    '''
        calculate utility at time nperiods-2
        note:  no uncertainty at this time -- the value of the final state is known, the final mitigation is chosen with full information
    '''
    for back in range( 0, my_tree.utility_nperiods-1 ):
        u_period = utility_periods - back
        first_node = my_tree.utility_period_pointer[u_period]
        for n in range(0, my_tree.utility_period_nodes[u_period]):
            my_tree.utility_by_state[first_node+n] = utility_by_node( my_tree, my_damage_model, my_cost_model, u_period, n, x )
            marginal_utility_by_node( my_tree, my_damage_model, my_cost_model, u_period, n, x )
            my_tree.utility_by_state[first_node+n] = utility_by_node( my_tree, my_damage_model, my_cost_model, u_period, n, x )
#            if back>0 : #  for earlier periods add payment at t "epsilon" times marginal utility to utility at t
            my_tree.utility_by_state[first_node+n] += my_tree.period_consumption_epsilon[my_tree.utility_nperiods-back-2] * my_tree.marginal_utility_by_state[first_node+n,0]
            my_tree.utility_by_state[first_node+n] += my_tree.node_consumption_epsilon[first_node+n] * my_tree.marginal_utility_by_state[first_node+n,0]

            '''
                calculation of zero-coupon bond price requires finding price such that utility( cons + price ) = discounted final_state_utility(consumption + epsilon)
                in general epsilon = 0, but for finding bond price epsilon = $1
            '''
            if back==0 :
                my_tree.utility_by_state[first_node+n] += my_tree.final_period_consumption_epsilon * my_tree.marginal_utility_by_state[first_node+n,1]
            if back==0 :    # for final period use marginal utility at t of c(t+1) to calculate additional utility of payment at t+1
                my_tree.utility_by_state[first_node+n] += my_tree.period_consumption_epsilon[my_tree.utility_nperiods-1] * my_tree.marginal_utility_by_state[first_node+n,1]
            '''
               now for periods 1,...,nperiods-2 (working backwards) calculate the utility using info known at that time
               what is known is that the true state is in a given partition of the final states
            '''
    '''
        create a final certainty equivalent sum over the utility in states at time 1
    '''
    util = -my_tree.utility_by_state[0]

    my_tree.funcalls += 1
    '''
        calculate the ghg levels in each node
    '''
    #print("""ghg levels!!!!!""")
    import time
    #t1 = time.time()
    my_tree.ghg_levels( x )
    #print("""ghg levels end!!!!!""")
    #t2 = time.time()
    #print("time cost is:", t2 - t1 )

    return util

def utility_by_node( tree, damage_model, cost_model, period, period_node, x ):
    node = tree.utility_period_pointer[period] + period_node
    period_length = tree.utility_times[period+1] - tree.utility_times[period]
    tree_period = tree.utility_decision_period[period]
    if period!=tree_period and tree.decision_nodes[tree_period]!=tree.utility_period_nodes[period]:
        tree_node = tree.decision_period_pointer[tree_period]+int(period_node/2)
    else:
        tree_node = tree.decision_period_pointer[tree_period]+period_node
    r = ( 1.0 - 1.0 / tree.eis)
    a = ( 1.0 - tree.ra)
    b = (1.0 - tree.time_pref)**period_length
    average_mitigation = damage_model.average_mitigation(x, tree_node)
    tree.ave_mitigation[tree_node] = average_mitigation
    '''
           damages_by_state stores climate damages at each node in the tree
    '''
    tree.damage_by_state[tree_node] = damage_model.damage_function(x, tree_node)
    mitigation = x[ tree_node ]
    tree.cost_by_state[tree_node] = cost_model.cost_by_state( mitigation, average_mitigation, tree_node )
    if tree.information_period[period]==0 :
        '''
           no branching implies certainty equivalent utility at time period depends only on the utility next period given information known today
        '''
        tree.cert_equiv_utility[node] = tree.utility_by_state[tree.utility_period_pointer[period]+tree.utility_period_nodes[period]+period_node]
    else :
        '''
            the nodes with branching require calculation of expected utility**a
        '''
        sum_probs = 0.
        ave_util = 0.
        '''
                   this loops over the partition of states reached from node first_node+n
        '''
        next_utility_node = tree.utility_period_pointer[period+1] + 2*period_node
        for ns in range( tree.next_node[tree_node][0], tree.next_node[tree_node][1]+1 ):
            sum_probs += tree.node_probs[ns]
            ave_util += tree.utility_by_state[next_utility_node]**a * tree.node_probs[ns]
            next_utility_node += 1
        ave_util = ave_util/sum_probs
        '''
                   the certainty equivalent utility is the ability weighted sum of next period utility over the partition reachable from state n**(1/a)
        '''
        tree.cert_equiv_utility[node] =  ave_util**(1./a)
    '''
           consumption = potential consumption minus damages[n] and minus state dependent costs[n]
           consumption by state = the consumption at the beginning of the period
           the average consumption for the period = (cons_of_x(t+1)/cons_of_x(t)-1.) / ( ln[ (cons_of_x(t+1)/cons_of_x(t))^(1/period) ] * period )
    '''
    cons_at_t =  tree.potential_consumption[tree_period] * ( 1.0-tree.damage_by_state[tree_node])*(1.-tree.cost_by_state[tree_node])
    if tree.decision_period[period]==1 :
        '''
            if consumption is calculated at a decision period use it
        '''
        if node==0 :
            cons_at_t += tree.first_period_epsilon
        tree.consumption_by_state[node] = cons_at_t
    else:
        '''
            else use interpolated consumption
        '''
        next_utility_node = tree.utility_period_pointer[period]+tree.utility_period_nodes[period]+period_node
        cons_of_x_plus_1 = tree.consumption_by_state[next_utility_node]
        if tree.utility_decision_period[period+1] != tree_period :
            next_tree_node =  tree.decision_period_pointer[ tree.utility_decision_period[period+1] ]+ period_node
            cons_of_x_plus_1 = cons_of_x_plus_1 * (1.-tree.cost_by_state[tree_node])/(1.-tree.cost_by_state[next_tree_node])
        if tree_period == 0 :
            interval = tree.utility_times[period+1]
            segment = tree.utility_times[period]
        else:
            interval = tree.utility_times[period+1] - tree.decision_times[tree_period]
            segment = tree.utility_times[period] - tree.decision_times[tree_period]

        interp_cons_at_t = interval_consumption( cons_of_x_plus_1, cons_at_t, segment/interval)
        tree.consumption_by_state[node] = interp_cons_at_t
        cons_at_t = interp_cons_at_t
    '''
           utility(t) is a function of consumption(t) plus certainty equivalent utility from period t+1
    '''
    tree.ce_term[node] = b * tree.cert_equiv_utility[node]**r

    utility = ( ( 1. - b )*cons_at_t**r + b*tree.cert_equiv_utility[node]**r )**( 1./r )

    return utility


def marginal_utility_by_node( tree, damage_model, cost_model, period, period_node, x ):
    node = tree.utility_period_pointer[period] + period_node
    period_length = tree.utility_times[period+1] - tree.utility_times[period]
    tree_period = tree.utility_decision_period[period]
    if period!=tree_period and tree.decision_nodes[tree_period]!=tree.utility_period_nodes[period]:
        tree_node = tree.decision_period_pointer[tree_period]+int(period_node/2)
    else:
        tree_node = tree.decision_period_pointer[tree_period]+period_node
    r = ( 1.0 - 1.0 / tree.eis)
    a = ( 1.0 - tree.ra)
    b = (1.0 - tree.time_pref)**period_length
    cons_of_x = tree.consumption_by_state[node]
    growth_term = (1. + tree.growth)

    '''
           calculate and save marginal utilities -- used to compute the stochastic discount factors
    '''
    tree.marginal_utility_by_state[node, 0] = mu_0( cons_of_x, b, r, a, tree.ce_term[node] )
    if tree.decision_period[period] == 1 :
        tree_node = tree.decision_period_pointer[tree.utility_decision_period[period]]+period_node
        tree.marginal_utility_in_tree[tree_node,0] = tree.marginal_utility_by_state[node,0]

    if period == tree.utility_nperiods-2 :
        '''
           final period certainty equivalent is (total utility)**r less the component contributed by consumption
        '''
        next_node = tree.utility_period_pointer[period] + tree.utility_period_nodes[period] + period_node
        cons_at_t_plus_1 = tree.consumption_by_state[next_node]
        tree.ce_term[next_node] = tree.utility_by_state[next_node]**r - ( 1.0 - b )*cons_at_t_plus_1**r
        tree.marginal_utility_by_state[next_node, 0] = (1.0 - b ) * (tree.utility_by_state[node]/tree.consumption_by_state[next_node])**(1-r)

        next_term =  b * (1.0 - b ) / ( 1.0 - b * growth_term**r )
        tree.marginal_utility_by_state[node, 1] = tree.utility_by_state[node]**(1-r) * next_term * tree.consumption_by_state[next_node]**(r-1)
        u_term = ( (1.0 - b) * cons_of_x**r + next_term * cons_at_t_plus_1**r )
#        print 'u_term', u_term**(1.0/r), tree.utility_by_state[node], cons_of_x, cons_at_t_plus_1
#        tree.final_total_derivative_term[period_node] = next_term * cons_at_t_plus_1**(r-1) * u_term**(1.0/r - 1.0)
        tree.final_total_derivative_term[period_node] = next_term * cons_at_t_plus_1**(r-1) * tree.utility_by_state[node]**(1.0 - r)

        if tree.decision_period[period] == 1 :
            tree_node = tree.decision_period_pointer[tree.utility_decision_period[period]]+period_node
            tree.marginal_utility_in_tree[tree_node,1] = tree.marginal_utility_by_state[node,1]
    else :
        if period==0:
            next_up = 1
            next_down = 2
            prob = tree.node_probs[next_up]
            tree.marginal_utility_by_state[node,1] = mu_1( tree.consumption_by_state[next_up], b, r, a, cons_of_x, prob, tree.consumption_by_state[next_down], tree.ce_term[next_up], tree.ce_term[next_down])
            tree.marginal_utility_by_state[node,2] = mu_1( tree.consumption_by_state[next_down], b, r, a, cons_of_x, 1.-prob, tree.consumption_by_state[next_up], tree.ce_term[next_down], tree.ce_term[next_up])
            if tree.decision_period[period] == 1 :
                tree_node = tree.decision_period_pointer[tree.utility_decision_period[period]]+period_node
                tree.marginal_utility_in_tree[tree_node,1] = tree.marginal_utility_by_state[node,1]
                tree.marginal_utility_in_tree[tree_node,2] = tree.marginal_utility_by_state[node,2]
        else:
            if tree.information_period[period]==1 :
                next_up = tree.utility_period_pointer[period+1]+2*period_node
                next_down = tree.utility_period_pointer[period+1]+2*period_node+1
                next_tree_node_up = tree.decision_period_pointer[tree.utility_decision_period[period]+1]+2*period_node
                prob_up = tree.node_probs[next_tree_node_up]
                prob_down = tree.node_probs[next_tree_node_up+1]
                total_prob = prob_up+prob_down
                prob= prob_up/total_prob
                tree.marginal_utility_by_state[node,1] = mu_1( tree.consumption_by_state[next_up], b, r, a, cons_of_x, prob, tree.consumption_by_state[next_down], tree.ce_term[next_up], tree.ce_term[next_down])
                prob = prob_down/total_prob
                tree.marginal_utility_by_state[node,2] = mu_1( tree.consumption_by_state[next_down], b, r, a, cons_of_x, prob, tree.consumption_by_state[next_up], tree.ce_term[next_down], tree.ce_term[next_up])
                if tree.decision_period[period] == 1 :
                    tree_node = tree.decision_period_pointer[tree.utility_decision_period[period]]+period_node
                    tree.marginal_utility_in_tree[tree_node,1] = tree.marginal_utility_by_state[node,1]
                    tree.marginal_utility_in_tree[tree_node,2] = tree.marginal_utility_by_state[node,2]
            else:
                next_node = tree.utility_period_pointer[period+1] + period_node
                tree.marginal_utility_by_state[node,1] = mu_2(tree.consumption_by_state[next_node], b, r, a, cons_of_x, tree.ce_term[next_node])
                if tree.decision_period[period] == 1 :
                    tree_node = tree.decision_period_pointer[tree.utility_decision_period[period]]+period_node
                    tree.marginal_utility_in_tree[tree_node,1] = tree.marginal_utility_by_state[node,1]
    return

def mu_0( x, b, r, a, cefd ):
    '''        marginal utility with respect to time t consumption function
               d/dx of ( (1.0-b)*x^r + cefd ) )^(1/r)
               where cefd is cert_equiv = b*( E(U^a)^r/a ) discouted average of next period utility to the alpha power
    '''
    t1 = (1. - b)*x**(r-1.)
    t2 = ( cefd - (b-1)*x**r)**((1./r)-1.)
    mu = t1 * t2
    return mu

def mu_1( x, b, r, a, c0, p, c2, cefd1, cefd2 ):
    '''
       marginal utility of time t utility function with respect to consumption next period
       d/dx of ((1.0-b)*c0^r + b*( p*((1-b)*x^r + cefd1 )^(a/r) + (1-p)*((1-b)*c2^r + cefd2)^(a/r) )^(r/a) )^(1/r)
       where c0 is time t consumption, x is consumption and cefd1 is the certainty equiv utiity in the forward state for which the
       derivative is being taken, and c2 and cefd2 are the consumption and cert_equiv utility for the other state
    '''
    t1 = (1. - b) * b * p * x**(r-1)
    t2 = ( cefd1 - (b - 1.) * x**r )**(a/r-1)
    t3 = ( p * ( cefd1 - b*x**r + x**r )**(a/r) + (1-p) * ( cefd2 - ( b - 1. ) * c2**r )**(a/r) )**((r/a)-1.)
    t4 = ( p * ( cefd1 - b * x**r + x**r )**(a/r) + (1-p) * (cefd2 - b * c2**r + c2**r)**(a/r) )
    t5 = ( b * t4**(r/a) - (b-1) * c0**r )**((1.0/r)-1.)
    mu = (t1 * t2 * t3 * t5 )
    return mu

def mu_2( x, b, r, a, c0, cefd):
    '''
           marginal utility of time t consumption function with respect to last period consumption
           d/dx of ((1.0-b)*c0^r + b*( (1-b)*x^r + cefd) )^(r/a) )^(1/r)
    '''
    t1 = (1. - b) * b * x**(r-1)
    t2 = ( (1. - b) * c0**r - ( b - 1.) * b * x**r + b * cefd )**((1./r)-1.)
    mu = (t1 * t2 )
    return mu

def numerical_utility_gradient(x,*var_args):
    '''  numerical derivative
    '''
    my_tree = var_args[0]
    delta = .0000001
    for n in range(0, my_tree.x_dim):
        base_utility = utility_function(x,*var_args)
        x[n] += delta
        new_utility = utility_function(x,*var_args)
        x[n] -= delta
        my_tree.grad[n] = (new_utility - base_utility)/delta

    return(my_tree.grad)

def d_cert_equiv_utility( my_tree, a, utility_period, period_node, j):
    tree_period = my_tree.utility_decision_period[utility_period]
    utility_node = my_tree.utility_period_pointer[utility_period]+period_node
    tree_node = my_tree.decision_period_pointer[tree_period]+period_node
    next_node = my_tree.utility_period_pointer[utility_period] + my_tree.utility_period_nodes[utility_period]
    if my_tree.information_period[utility_period]==1 :
        ave_d_ceu = 0.
        sum_probs = 0.
        next_node  += 2*period_node
        for ns in range( my_tree.next_node[tree_node][0], my_tree.next_node[tree_node][1]+1):
            sum_probs += my_tree.node_probs[ns]
            ave_d_ceu += my_tree.node_probs[ns] * a * my_tree.utility_by_state[next_node]**(a-1) * my_tree.d_utility_by_state[next_node][j]
            next_node += 1
        ave_d_ceu = ave_d_ceu / sum_probs
        return( ave_d_ceu)
    else:
        next_node += period_node
        if utility_period==my_tree.utility_nperiods-2 :
            d_utility = my_tree.d_utility_of_final_state[period_node][j]
        else:
            d_utility = my_tree.d_utility_by_state[next_node][j]
        d_ceu = d_utility
    return( d_ceu )

def interval_consumption( ctp1, ct, t):
#    fraction = 1.0
#    t *= fraction
    consumption_growth = ( ctp1 / ct )**t
    consumption = consumption_growth*ct
    return( consumption)

def d_interval_consumption( ctp1, d_ctp1, ct, d_ct, t):
#    fraction = 1.0
#    t *= fraction
    term1 = ct**(-t) * ctp1**(t-1)
    term2 = t * ct * d_ctp1 - (t-1) * ctp1 * d_ct
    d_ave = term1 * term2
    return( d_ave)

def d_consumption( tree, damage_model, cost_model, utility_period, period_node, x, j):

    utility_node = tree.utility_period_pointer[utility_period]+period_node
    tree_period = tree.utility_decision_period[utility_period]
    if utility_period!=tree_period and tree.decision_nodes[tree_period]!=tree.utility_period_nodes[utility_period]:
        tree_node = tree.decision_period_pointer[tree_period]+int(period_node/2)
    else:
        tree_node = tree.decision_period_pointer[tree_period]+period_node
    d_cbs = cost_model.d_cost_by_state( damage_model, x[tree_node], tree.ave_mitigation[tree_node], tree_node, j )
    d_dbs = damage_model.d_damage_by_state(x, tree_node, j)
    d_cons = -tree.potential_consumption[tree_period] * ( d_dbs*(1.-tree.cost_by_state[tree_node]) + d_cbs*(1.-tree.damage_by_state[tree_node]) )
    d_dmgcons = -tree.potential_consumption[tree_period] * ( d_dbs*(1.-tree.cost_by_state[tree_node]) )
    if tree.decision_period[utility_period]==1 :
        '''
            if consumption is calculated at a decision period save the result
        '''
        tree.d_cons_by_state[utility_node][j] = d_cons
        if j==0 : tree.d_damage[utility_node] = d_dmgcons
        return( d_cons )
    else:
        '''
            else use interpolated consumption and calculate the derviative of interpolated consumption
        '''
        next_utility_node = utility_node + tree.utility_period_nodes[utility_period]
        cons_of_x_plus_1 = tree.consumption_by_state[next_utility_node]
        d_cons_p1 = tree.d_cons_by_state[next_utility_node][j]
        if j == 0 : d_dmgcons_p1 = tree.d_damage[next_utility_node]
        if tree.utility_decision_period[utility_period+1] != tree_period :
            next_tree_node =  tree.decision_period_pointer[ tree.utility_decision_period[utility_period+1] ]+ period_node
            cons_of_x_plus_1 = cons_of_x_plus_1 * (1.-tree.cost_by_state[tree_node])/(1.-tree.cost_by_state[next_tree_node])
            d_dbs = damage_model.d_damage_by_state(x, next_tree_node, j)
            d_cons_p1 = -tree.potential_consumption[ tree.utility_decision_period[utility_period+1] ] * ( d_dbs*(1.-tree.cost_by_state[tree_node]) + d_cbs*(1.-tree.damage_by_state[next_tree_node]))
            if j == 0 : d_dmgcons_p1 = -tree.potential_consumption[ tree.utility_decision_period[utility_period+1] ] * ( d_dbs*(1.-tree.cost_by_state[tree_node]) )
        if tree_period == 0 :
            interval = tree.utility_times[utility_period+1]
            segment = tree.utility_times[utility_period]
        else:
            interval = tree.utility_times[utility_period+1] - tree.decision_times[tree_period]
            segment = tree.utility_times[utility_period] - tree.decision_times[tree_period]
        cons_at_t = tree.potential_consumption[tree_period]*(1.-tree.cost_by_state[tree_node])*(1.-tree.damage_by_state[tree_node])
        d_inter_cons = d_interval_consumption( cons_of_x_plus_1, d_cons_p1, cons_at_t, d_cons, segment/interval )
        tree.d_cons_by_state[utility_node][j] = d_inter_cons
        if j == 0 :
            tree.d_damage[utility_node] = d_interval_consumption( cons_of_x_plus_1, d_dmgcons_p1, cons_at_t, d_dmgcons, segment/interval )

    return(d_inter_cons)
@jit()
def analytic_utility_gradient(x,*var_args):
    '''
       analytic derivatives are computed in this function
       all variables and arrays with a leading d_ are derivatives
    '''
    my_tree = var_args[0]
    my_damage_model = var_args[1]
    my_cost_model = var_args[2]

    period = my_tree.utility_nperiods-2
    tree_period = my_tree.utility_decision_period[period]+1

    '''
        r = rho in the dlw paper
        a = alpha in the dlw paper
        b = beta in the dlw paper (for continuation use an annual period)
    '''
    period_length = my_tree.utility_times[1] - my_tree.utility_times[0]
    r = ( 1. - 1./my_tree.eis)
    a = ( 1. - my_tree.ra)
    b = ( 1. - my_tree.time_pref)**period_length

    '''
        calculate final period utility and derivative
    '''
    growth_term = (1. + my_tree.growth)
    first_node = my_tree.utility_period_pointer[period]+my_tree.final_states
    first_tree_node = my_tree.x_dim

    import time
    t1 = time.time()
    for n in range(0, my_tree.final_states):
        my_tree.final_damage_by_state[n] = my_damage_model.damage_function(x, first_tree_node+n)
        for j in range(0, my_tree.x_dim):
            my_tree.d_final_damage_by_state[n][j] = my_damage_model.d_damage_by_state(x, first_tree_node+n, j )
        continuation = ( 1. / (1. - b*growth_term**r) )**(1./r)
        cons_of_x = (my_tree.potential_consumption[tree_period] * (1. - my_tree.final_damage_by_state[n]))
        my_tree.utility_by_state[first_node+n] = (1.-b)**(1./r) * cons_of_x * continuation
        for j in range(0, my_tree.x_dim):
            my_tree.d_utility_of_final_state[n][j] = -((1.-b)**(1./r) * continuation * my_tree.potential_consumption[tree_period]*my_damage_model.d_damage_by_state(x, first_tree_node+n, j))
            my_tree.d_cons_by_state[first_node+n][j] = -my_tree.potential_consumption[tree_period]*my_tree.d_final_damage_by_state[n][j]
        my_tree.marginal_damages[first_node+n] = my_tree.d_cons_by_state[first_node+n][0]
    t2 = time.time()
    print("calculate final period utility and derivative time:", t2 - t1)

    '''
        calculate previous period utility and derivative
    '''
    utility_periods = my_tree.utility_nperiods-2
    '''
                calculate the derivative of consumption in period at node n with respect to x(j)
    '''
    '''
            now for periods 1,...,nperiods-2 calculate the utility and derivative using info known at that time
            what is known is that the true state is in a partition of the final states
    '''


    t1 = time.time()
    for back in range( 0, my_tree.utility_nperiods-2 ):
        utility_period = utility_periods - back
        period_length = my_tree.utility_times[utility_period+1] - my_tree.utility_times[utility_period]
        b = ( 1. - my_tree.time_pref)**period_length
        first_node = my_tree.utility_period_pointer[utility_period]

        for n in range(0, my_tree.utility_period_nodes[utility_period]):
            '''
                calculate the derivative of utility with respect to x[j]
            '''
            t3= time.time()
            for j in range(0,my_tree.x_dim):
                term1 = (1./r) * ( (1.-b)* my_tree.consumption_by_state[first_node+n]**r + b * my_tree.cert_equiv_utility[first_node+n]**r )**(1./r - 1.)
                term2 = ( (1.-b) * r * my_tree.consumption_by_state[first_node+n]**(r-1.0))


                term3 = d_consumption( my_tree, my_damage_model, my_cost_model, utility_period, n, x, j)



                if j==0 :
                    my_tree.marginal_damages[first_node+n] = term3
                if(my_tree.information_period[utility_period]==0):
                    term4 = b * r * my_tree.cert_equiv_utility[first_node+n]**(r-1)
                else:
                    term4 = b * (r/a) * my_tree.cert_equiv_utility[first_node+n]**(r-a)
                term5 = d_cert_equiv_utility(  my_tree, a, utility_period, n, j )
                my_tree.d_utility_by_state[first_node+n][ j] = term1 * (term2*term3 + term4*term5)
            t4 = time.time()
            print("d_consumption cost: ", t4-t3)
    t2 = time.time()

    print ("calculate the derivative of consumption time ", t2 - t1)

    '''
       create a final sum over the partition at time 0
    '''
    period_length = my_tree.utility_times[1]
    b = ( 1. - my_tree.time_pref)**period_length
    n = 0
    t1 = time.time()
    for j in range(0, my_tree.x_dim):
        term1 = (1./r) * ( (1.-b)* my_tree.consumption_by_state[n]**r + b * my_tree.cert_equiv_utility[n]**r )**(1./r - 1.)
        term2 = ( (1.-b) * r * my_tree.consumption_by_state[n]**(r-1.))
        term3 = -my_cost_model.d_cost_by_state( my_damage_model, x[n], 0.0, n, j )
        if j==0 : my_tree.marginal_damages[0] = term3
        term4 = b * (r/a) * my_tree.cert_equiv_utility[n]**(r-a)
        term5 = d_cert_equiv_utility(  my_tree, a, n, n, j )
        my_tree.d_utility_by_state[n][j] = term1 * (term2*term3 + term4*term5)
        my_tree.grad[j] = -my_tree.d_utility_by_state[0][j]
    t2 = time.time()
    print(" create a final sum over the partition time: ", t2 - t1)

    return my_tree.grad
