'''
   dlw_run is a python script to find the optimal mitigation plan
   conditional on a set of damage coefficients in the daniel,litterman,wagner climate modeling paper
   for more information, contact Bob Litterman at blitterman@gmail.com

   additional scripts called by dlw_run --
     dlw_tree_class:          defines a tree object which is the basis of the utility calculation
     dlw_damage_class:        runs a monte carlo simulation to estimate damages and form the basis of a damage function
     dlw_cost_class:          approximates the McKinsey cost curve and incorporates technological change to create a cost function
     dlw_optimize_class:      controls the optimization, for example initializes mitigation, sets constraints, writes output
     dlw_utility:             evaluates the Epstein-Zin utility function, marginal utility, and analytic derivatives wrt mitigation
'''
print ('running dlw_run from C:/Users/blitt_000/Dropbox/EZ Climate calibration paper/dlw code/dlw_run.py')
import dlw_utility as fm
import math
import scipy
import sys
from numba import jit

'''
  when running in batch mode parameters are passed in sys.argv[i],
    where i = 1, 2, 3...
    this section of code sets up the argment list
    code using batch parameters is highlighted by a comment
'''
print ('These arguments set in batch mode')
#print 'growth rate = ', sys.argv[1]
print ('period_1_years =', sys.argv[1])
#print 'analysis =', sys.argv[2]
#print 'force_simul =', sys.argv[3]
'''
  initialize the tree class
'''
from dlw_tree_class import tree_model
my_tree = tree_model(tp1=int(sys.argv[1]))
#my_tree = tree_model()
print ('tree nodes', my_tree.decision_period_pointer)
print ('horizon times', my_tree.decision_times)
print ('utility periods', my_tree.utility_nperiods)
print ('utility full_tree', my_tree.utility_full_tree)
print ('u_times', my_tree.utility_times)
print ('decision_period?', my_tree.decision_period)
print ('branch_period?', my_tree.information_period)
print ('u_first_node', my_tree.utility_period_pointer)
print ('u_nodes', my_tree.utility_period_nodes)
print ('u_tree_period', my_tree.utility_decision_period)

'''
  initialize the damage class
'''
from dlw_damage_class import damage_model
my_damage_model = damage_model(my_tree=my_tree)
#my_damage_model = damage_model(my_tree=my_tree)
my_damage_model.damage_function_initialization()

'''
  initialize the cost class
'''
from dlw_cost_class import cost_model
my_cost_model = cost_model(tree=my_tree)

print ('economic growth', my_tree.growth, 'risk aversion', my_tree.ra, 'elasticity of intertemporal substitution', my_tree.eis)

'''
  initialize the tree and damage function interpolation coefficients
'''
my_damage_model.initialize_tree()
my_damage_model.dfc = my_damage_model.damage_function_interpolation()

'''
  get the initial parameters and output the parameters and initial fit
'''
from dlw_optimize_class import optimize_plan
#my_optimization = optimize_plan(my_tree=my_tree,randomize=float(sys.argv[2]),alt_input=int(sys.argv[3]))
my_optimization = optimize_plan(my_tree=my_tree)
if my_tree.nperiods <= 5 :
  my_optimization.get_initial_guess()
else :
  my_optimization.get_initial_guess6()

print ("initial guess", my_optimization.guess)

base = fm.utility_function( my_optimization.guess, my_tree, my_damage_model, my_cost_model )
print ('initial parameter fit', base)

'''
  numerical derivative  check of the gradient
'''
base_grad = fm.analytic_utility_gradient(my_optimization.guess, my_tree, my_damage_model, my_cost_model )
delta = .00001
guess = my_optimization.guess
for p in range(0,my_tree.x_dim):
  guess[p] += delta
  base_plus_p = fm.utility_function( guess, my_tree, my_damage_model, my_cost_model )
  num_deriv = (base_plus_p-base)/delta
  if abs((base_grad[p]-num_deriv)/num_deriv) > .05 :
    print ('CHECK GRADIENT: ','p = ', p, 'derivative calculation = ', base_grad[p], 'numerical derivative = ', num_deriv)
  if my_optimization.derivative_check == 1 :
    print ('p', p, 'derivative =', base_grad[p], 'numerical derivative', num_deriv)
  guess[p] -= delta

from scipy.optimize import fmin_l_bfgs_b

'''
   if my_tree.analysis == 1 or 2 then find the unconstrained optimal mitigation plan
   use scipy minimization function fmin_l_bfgs_b to maximize the utility function
   with respect to choices of mitigation
'''
"""------------------------------------------------------------------------------------"""
print ("Begin optimization!!!!!")


if my_tree.analysis == 1 or my_tree.analysis == 2 :
  my_optimization.set_constraints(constrain=0)
  res = fmin_l_bfgs_b( fm.utility_function, guess,fprime=fm.analytic_utility_gradient,factr=1.,pgtol=1.0e-5,bounds=(my_optimization.xbounds),maxfun=600,args=([my_tree, my_damage_model, my_cost_model]))
  bestfit = res[1]
  print ('best fit', bestfit)
  bestparams = res[0]
  print ('best parameters', bestparams)
  retparam = res[2]
  print ('gradient', retparam['grad'])
  print ('function calls', retparam['funcalls'])
else :
  bestfit = base
  bestparams = guess

'''
   if analysis = 1, then the only step is optimization: now print output to the terminal
'''
if my_tree.analysis == 1:
  my_optimization.create_output(bestparams, bestfit, my_damage_model, my_cost_model, my_tree)
if my_tree.nperiods <= 5 :
  my_optimization.put_optimal_plan(bestparams)
else :
  my_optimization.put_optimal_plan6(bestparams)
'''
   if my_tree.analysis = 2 then find the decomposition of the social cost of carbon
   into its risk premium and expected damage components and their decomposition over time
   first step is to save the initial optimized consumption values for all nodes in the utility tree ( in my_tree.d_consumption_by_state[] )
   and the cost component of consumption in decision period 0 (in my_tree.d_cost_by_state), which reflect the costs by node of decision period 0 mitigation
   this is done so that in the next we can increment the mitigation at time 0 and to calculate the marginal changes in consumption and cost
'''
if my_tree.analysis == 2:
  for node in range(0, my_tree.utility_full_tree) :
    my_tree.d_consumption_by_state[node] = my_tree.consumption_by_state[node]

  for sub_period in range(0, my_tree.first_period_intervals) :
    potential_consumption = (1.+my_tree.growth)**(my_tree.sub_interval_length * sub_period)
    my_tree.d_cost_by_state[sub_period,0] = potential_consumption * my_tree.cost_by_state[0]

  delta_x = .01
  bestparams[0] += delta_x
  '''
      next increment time 0 mitigation and run an optimization in which mitigation at time 0 is constained to = previous optimal mitigation + delta_x
  '''
  my_optimization.set_constraints(constrain=-1, node_0 = bestparams[0])
  guess = bestparams
  res = fmin_l_bfgs_b( fm.utility_function,guess,fprime=fm.analytic_utility_gradient,factr=1.,pgtol=1.0e-5,bounds=(my_optimization.xbounds),maxfun=600,args=([my_tree, my_damage_model, my_cost_model]))
  '''
     now calculate the changes in consumption and the mitigation cost component of consumption per unit change in mitigation in the new optimal plan
  '''
  for node in range(0, my_tree.utility_full_tree) :
    my_tree.d_consumption_by_state[node] = (my_tree.consumption_by_state[node]-my_tree.d_consumption_by_state[node])/delta_x

  for sub_period in range(0, my_tree.first_period_intervals) :
    potential_consumption = (1.+my_tree.growth)**(my_tree.sub_interval_length * sub_period)
    my_tree.d_cost_by_state[sub_period,1] = ( potential_consumption * my_tree.cost_by_state[0] - my_tree.d_cost_by_state[sub_period,0] )/delta_x
  bestparams[0] -= delta_x
  base = fm.utility_function( bestparams, my_tree, my_damage_model, my_cost_model )
  '''
     create the output, including the decomposition of SCC into the time paths of the net present value contributions from expected damage and risk premium components
  '''
  my_optimization.create_output(bestparams, base, my_damage_model, my_cost_model, my_tree)

  '''
     this section of code addresses the question: what is the cost of waiting to start mitigation untilt the end of the first period
     if my_tree.analysis == 3, then calculate the marginal cost of carbon when mitigation is delayed for the first period
     if my_tree.analysis == 4, then calculate the total consumption-equivalent cost of delay (relative to an optimal plan) for the first period
  '''
if my_tree.analysis >= 3:
  '''
        analysis == 3 calculates the marginal value of reducing emissions by 1 ton
        when the initial endowment has already been incremented by the payment equal to "lump_sum"
        base_x is the starting level of mitigation
        delta_x is used to calculate a numerical derivative of utility as a function of a marginal percentage
          increase in emissions reduction today while reoptimizing emissions reductions at all decision nodes in
          the future
        increment is the amount to increase base_x in each run
        the ipass loop increments base_x and repeats the analysis
  '''
  base_x = 0.0
  delta_x = .01
  increment = .025
  for ipass in range(0, 1):
    '''
      first step is to calculate the optimal plan when mitigation is constrained to base_x for the first period
    '''
    lump_sum = .0
    my_tree.first_period_epsilon = lump_sum
    my_optimization.set_constraints(constrain=1, node_0 = base_x, node_1 = base_x, node_2 = base_x)
    res = fmin_l_bfgs_b( fm.utility_function,guess,fprime=fm.analytic_utility_gradient,factr=1.,pgtol=1.0e-5,bounds=(my_optimization.xbounds),maxfun=600,args=([my_tree, my_damage_model, my_cost_model]))
    '''
      save the parameters for the run with mitigation = base_x in baseparams
      save the utility value in basefit
      save the cost of emissions reductions when mitigation = base_x in marginal_cost
      use the optimal mitigation parameters as the starting point for an optiization with first period
        emissions reduction constrained to be base_x + delta_x
    '''
    baseparams = res[0]
    basefit = res[1]
    marginal_cost = my_cost_model.price_by_state( baseparams[0],0.,0.)
    newparams = baseparams
    '''
       create output for the optimal plan with current mitigation constrained to equal base_x (the optimal plan subject to no action in period 0)
    '''
    my_optimization.create_output(baseparams, basefit, my_damage_model, my_cost_model, my_tree)
    '''
       next calculate the optimal plan
       when my_tree.analysis = 3 the "optimal" plan is constrained so that current mitigation is equal to
         base_x + delta_x
       when my_tree.analysis = 4 the current mitigation is indeed from the unconstrained optimal plan
    '''
    if my_tree.analysis == 3 :
      print ('base_x', base_x, 'delta_x', delta_x)
      newparams[0] += delta_x
      my_tree.first_period_epsilon = lump_sum
      my_optimization.set_constraints(constrain=1, node_0 = newparams[0], node_1 = newparams[1], node_2 = newparams[2])
      res = fmin_l_bfgs_b( fm.utility_function,newparams,fprime=fm.analytic_utility_gradient,factr=1.,pgtol=1.0e-5,bounds=(my_optimization.xbounds),maxfun=600,args=([my_tree, my_damage_model, my_cost_model]))
    else :
      my_optimization.set_constraints(constrain=0)
      res = fmin_l_bfgs_b( fm.utility_function,guess,fprime=fm.analytic_utility_gradient,factr=1.,pgtol=1.0e-5,bounds=(my_optimization.xbounds),maxfun=600,args=([my_tree, my_damage_model, my_cost_model]))
    '''
      save the parameters for the run with mitigation = base_x + delta_x in newparams
      save the utility value in newfit
    '''

    newparams = res[0]
    newfit = res[1]

    '''
     create output for the new optimal plan with either a marginal, or a fully optimal mitigation at time 0
    '''
    my_optimization.create_output(newparams, newfit, my_damage_model, my_cost_model, my_tree)

    '''
     delta_util_x is the change in utility from base_x to base_x + delta_x
    '''
    delta_util_x = newfit - basefit

    print ('delta_util_x', delta_util_x)

    if my_tree.analysis == 3:
      '''
        calculate the change in utility for a marginal percentage change in consumption starting from
          the mitigation plan with time 0 mitigation = base_x
        the utility_funtion calculates utility for a given mitigation plan, but if my_tree.first_period_epsilon
          is non-zero, adds the value of first_period_epsilon to first period consumption without changing any
          other consumption value
        the marginal impact on utility of a marginal increase in consumption is saved in delta_util_c
      '''
      delta_con = .01
      my_tree.first_period_epsilon = lump_sum
      my_tree.first_period_epsilon += delta_con
      baseparams[0]= base_x
      print ('epsilon', my_tree.first_period_epsilon)
      util_given_delta_con = fm.utility_function( baseparams, my_tree, my_damage_model, my_cost_model )
      delta_util_c = util_given_delta_con - basefit
      my_tree.first_period_epsilon = 0.0
      print ('basefit', basefit, 'newfit', newfit, 'util_given_delta', util_given_delta_con)
      print ('delta_con', delta_con)
      print ('delta_util_c', delta_util_c)
    else:
      delta_x = newparams[0]
      print ('delta_x', delta_x)
      from scipy.optimize import brentq
      '''
       my_optimization.constraint_cost is the utility cost of constraining first period mitigation to zero
      '''
      my_optimization.constraint_cost = newfit-basefit
      for miti in range(0, my_tree.x_dim):
        my_optimization.guess[miti] = baseparams[miti]
      '''
       when my_tree.analysis = 4, calculate deadweight loss by using the root finder routine to find a change in consumption
       that increases the zero mitigation constrained optimization utility to equal that of the unconstrained plan plus a lump sum
       that is, find delta_con such that [ delta_utility - my_optimization.constraint_cost] = 0
       where delta_utility = util(constrained plan, consumption[today] + delta_con)-util(unconstrained plan, consumption[today])
      '''
      delta_con = brentq( my_optimization.find_bec, -.1, .99, args=( my_tree, my_damage_model, my_optimization, my_cost_model))
      print ('delta consumption to match unconstrained optimal plan', delta_con)
    if my_tree.analysis == 3:
      print ('Marginal cost of emissions reduction at x = ', base_x, 'is', marginal_cost)
      '''
        my_cost_model.consperton0 is consumption in $ today per ton of emissions
        so the marginal benefit is the slope of the utility function wrt x / slope of the utility function wrt c * ($ consumption / ton of emissions)
      '''
      print ('Marginal benefit emissions reduction is', (delta_util_x / delta_util_c ) * delta_con * my_cost_model.consperton0 / delta_x)
      base_x += increment
    else :
      print ('delta_consumption_billions', delta_con * my_cost_model.consperton0 * my_tree.bau_emit_level[0])
      print ('delta_emissions_gigatons', delta_x * my_tree.bau_emit_level[0])
      deadweight = delta_con * my_cost_model.consperton0 / delta_x
      print ('Deadweight $ loss of consumption per year per ton of mitigation of not pricing carbon in period 0', deadweight)
'''
  calculate the marginal utility at time 0 of state contingent increases in consumption at each node
'''
print (' marginal utility at time 0 of state contingent increases in consumption at each node ')
delta_con = .01
base_util = fm.utility_function( bestparams, my_tree, my_damage_model, my_cost_model )
print (' base utility ', base_util)
for time_period in range(0, my_tree.utility_nperiods-1):
  is_tree_node = my_tree.decision_period[ time_period]
  tree_period = my_tree.utility_decision_period[time_period]
  tree_node = my_tree.decision_period_pointer[ tree_period ]
  if is_tree_node == 1:
    first_u_node = my_tree.utility_period_pointer[time_period]
    for period_node in range(0, my_tree.utility_period_nodes[time_period]):
      my_tree.node_consumption_epsilon[first_u_node+period_node] = delta_con
      new_util = fm.utility_function( bestparams, my_tree, my_damage_model, my_cost_model )
      my_tree.node_consumption_epsilon[first_u_node+period_node] = 0.0
      marginal_utility = (base_util - new_util) / delta_con
      print (' period ', tree_period, ' year ', 2015+my_tree.utility_times[time_period], ' node ', tree_node+period_node, ' utility', new_util, ' marginal_utility ', marginal_utility)
'''
   done
'''
