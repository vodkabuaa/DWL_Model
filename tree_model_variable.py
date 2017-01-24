"""
x_dim   the number of nodes in the tree where decisions are made = dimension of the vector x of optimal mitigations
"""

        self.damage_by_state = np.zeros(self.x_dim)
        self.cost_by_state = np.zeros(self.x_dim)
        self.d_final_damage_by_state = np.zeros([self.final_states, self.x_dim])
        self.d_utility_of_final_state = np.zeros([self.final_states, self.x_dim])
        self.d_utility_by_state = np.zeros([self.utility_full_tree, self.x_dim])
        self.d_cons_by_state = np.zeros([self.utility_full_tree, self.x_dim])
        self.grad = np.zeros(self.x_dim)


"""
full_tree self.x_dim + self.final_states
"""

        self.ave_mitigation = np.zeros(self.full_tree)
        self.marginal_utility_in_tree = np.zeros([self.full_tree,3])
        self.ghg_by_state = np.zeros(self.full_tree)
        self.cum_forcing_by_state = np.zeros([self.full_tree,3])
        self.additional_emissions_by_state = np.zeros(self.full_tree)
"""
utility_full_tree
"""
        self.consumption_by_state = np.zeros(self.utility_full_tree)
        self.marginal_utility_by_state = np.zeros([self.utility_full_tree,3])
        self.sdf_in_tree = np.zeros(self.utility_full_tree)
        self.utility_by_state = np.zeros(self.utility_full_tree)
        self.cert_equiv_utility = np.zeros(self.utility_full_tree)
        self.ce_term = np.zeros(self.utility_full_tree)
        self.node_consumption_epsilon = np.zeros(self.utility_full_tree)
        self.marginal_damages = np.zeros(self.utility_full_tree)
        self.d_consumption_by_state = np.zeros(self.utility_full_tree)
        self.d_damage = np.zeros(self.utility_full_tree)
        self.d_utility_by_state = np.zeros([self.utility_full_tree, self.x_dim])
        self.d_cons_by_state = np.zeros([self.utility_full_tree, self.x_dim])


"""
utility_nperiods
"""
        self.period_consumption_epsilon = np.zeros(self.utility_nperiods+1)
        self.discount_prices = np.zeros(self.utility_nperiods+1)
        self.net_expected_damages = np.zeros(self.utility_nperiods+1)
        self.risk_premium = np.zeros(self.utility_nperiods+1)



"""
final_states  the number of final states at time T
"""
        self.final_damage_by_state = np.zeros(self.final_states)
        self.final_total_derivative_term = np.zeros(self.final_states)
        self.continuation_utility = np.zeros(self.final_states)
        self.d_final_damage_by_state = np.zeros([self.final_states, self.x_dim])
        self.d_utility_of_final_state = np.zeros([self.final_states, self.x_dim])
"""
nperiods the number of periods in the model (=T)
"""

        self.potential_consumption = np.zeros(self.nperiods+1)
        self.emissions_per_period = np.zeros(self.nperiods)
        self.cum_emissions_in_scenarios = np.zeros([self.nperiods,3])
        self.GHG_levels_in_scenarios = np.zeros([self.nperiods+1,3])
        self.forcing_per_period = np.zeros([self.nperiods,3])
        self.cum_forcing_per_period = np.zeros([self.nperiods+1,4])
        self.emissions_to_ghg = np.zeros(self.nperiods)
