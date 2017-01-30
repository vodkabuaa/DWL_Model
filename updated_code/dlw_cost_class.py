import numpy as np

class cost_model(object):
    '''Includes functions to evaluate the cost curve for a climate model
    '''
    def __init__(self,tree,g=92.08,a=3.413,join=2000.,maxPrice=2500.,teconst=1.5,tescale=0.0,consat0=30460.):
        '''Initializes a climate parameter model
        
        Parameters
        ----------

        tree : object of the tree class
            provides the tree structure of the climate model
            
        g : float
            Initial scale of the cost function
        
        a : float
            Curvature of the cost function
        
        join : float
            Price at which the cost curve is extended
        
        maxPrice : float
            Price at which carbon dioxide can be removed from atmosphere in unlimited scale
        
        teconst : float
            Determines the degree of exogenous technological improvement over time
            1 implies 1% per year lower cost
            For example, if teconst = 0.0, then emissions reduction of 50% in year 30 costs the same as 50% reduction today
            but if teconst = 1, then a 50% emissions reduction in year 30 costs only 74% (=.99^30) as much as today
        
        tescale : float
            Determines the sensitivity of technological change to previous mitigation, 0 implies none
            If tescale = 1, then the per year rate of improvement of technological change = teconst+tescale*(average mitigation)
            where average mitigation is the average mitigation per year leading up to this point in time
            For example, if teconst = 1.0 and average mitigation is 50% per year up to year 30,
            then with tescale = 2 the rate of technological change is 2% per year
            and the cost of emissions reduction of 50% in year 30 is only 54.5% (=.98^30) as much as today
        
        consat0 : float
            Initial consumption
            Default value based on 2010 values:
                30460 billions current US$ consumption
        '''
        self.tree = tree
        self.g = g
        self.a = a
        self.join = join
        self.maxPrice = maxPrice
        self.teconst = teconst
        self.tescale = tescale
        self.cbs_level = (self.join / (self.g * self.a))**(1./(self.a-1.))
        self.cbs_deriv = self.cbs_level / (self.join * (self.a-1.))
        self.cbs_b = self.cbs_deriv * (self.maxPrice - self.join)/self.cbs_level
        self.cbs_k = self.cbs_level * (self.maxPrice - self.join)**self.cbs_b
        '''
            consperton0 = consat0 / bau_emit_level[0] = 30460 billions / 52 billion metric tons CO2 equivalent emissions
        '''
        self.consperton0 = consat0 / tree.bau_emit_level[0]
        self.cost_gradient = np.zeros([self.tree.x_dim, self.tree.x_dim])
        print ('Exogenous technological change =', teconst,'Endogenous technological change =', tescale)

    def cost_by_state( self, mitigation, average_mitigation, node):
        '''Calculates the mitigation cost by state
        
        Parameters
        ----------
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point
        
        node : integer
            node in tree for which mitigation cost is calculated
        
        Returns
        -------
        cbs : float
            Cost by state
        '''
        if( node == 0 ):
            tc_years = 0
        else:
            tc_years = self.tree.decision_times[ self.tree.period_map[node] ]
        te_term = ( 1. - ((self.teconst + self.tescale * average_mitigation)/100))**tc_years
        if mitigation < self.cbs_level :
            cbs = self.g * mitigation**self.a * te_term / self.consperton0
        else :
            base_cbs = self.g * self.cbs_level**self.a
            extension = ((mitigation - self.cbs_level)*self.maxPrice
                         - self.cbs_b * mitigation * (self.cbs_k/mitigation)**(1.0/self.cbs_b)/(self.cbs_b-1.)
                         + self.cbs_b * self.cbs_level * (self.cbs_k/self.cbs_level)**(1.0/self.cbs_b)/(self.cbs_b-1.))
            cbs = (base_cbs + extension) * te_term / self.consperton0

        return cbs
    
    def d_cost_by_state( self, my_damage_model, mitigation, average_mitigation, emit_node, x_node):
        '''Calculates the analytic derivative of cost by state
        
        Parameters
        ----------
        my_damage_model : damage class object
            allows calculation of derivative of average_mitigation wrt x
            
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point

        emit_node : integer
            the node for which to calculate derivative of cost

        x_node : integer
            the node whose mitigation effort the derivative is being taken with respect to

        Returns
        -------
        cost_gradient : float
            analytic derivative of Cost by state function wrt emissions mitigation
        '''
        emit_period = self.tree.period_map[emit_node]
        if( emit_period == 0 ):
            tc_years = 0
        else :
            tc_years = self.tree.decision_times[ emit_period ]
        if emit_node == x_node:
            cost_gradient = self.dd_own_cost_by_state(mitigation, average_mitigation, tc_years)
        else :
            x_period = self.tree.period_map[x_node]
            if x_period == 0 :
                return(self.dd_am_cost_by_state(my_damage_model, mitigation, average_mitigation, tc_years, emit_node, x_node ))
            if emit_period == self.tree.nperiods-1 :
                first_state = emit_node - self.tree.decision_period_pointer[emit_period]
                last_state = first_state
            else :
                first_state = self.tree.node_mapping[ emit_period-1 ][emit_node - self.tree.decision_period_pointer[emit_period]][0]
                last_state = self.tree.node_mapping[ emit_period-1 ][emit_node - self.tree.decision_period_pointer[emit_period]][1]
            if x_period >= emit_period :
                return(0.)
            if self.tree.node_mapping[ x_period-1 ][x_node - self.tree.decision_period_pointer[x_period]][0] > last_state :
                return(0.)
            if self.tree.node_mapping[ x_period-1 ][x_node - self.tree.decision_period_pointer[x_period]][1] < first_state :
                return(0.)
            cost_gradient = self.dd_am_cost_by_state(my_damage_model, mitigation, average_mitigation, tc_years, emit_node, x_node )

        return cost_gradient
    
    def nd_cost_by_state( self, mitigation, average_mitigation, emit_node, x_node ):
        '''Calculates and returns the numerical derivative of cost by state
        
        Parameters
        ----------
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point
        
        emit_node : integer
            the node for which to calculate the derivative of cost

        x_node : integer
            the node whose mtigation effort the derivative is being taken with respect to
        
        Returns
        -------
        cost_gradient : float
            numerical gradient of the Cost by state function wrt emissions mitigation
        '''
        base_cost = self.cost_by_state(mitigation, average_mitigation, emit_node)
        delta_mitigation = .0001
        emit_period = self.tree.period_map[emit_node]
        if( emit_period == 0 ):
            weight = 1.
        else:
            weight = self.tree.period_length / self.tree.decision_times[emit_period] 
        if emit_node == x_node :
            new_cost = self.cost_by_state(mitigation+delta_mitigation, average_mitigation, emit_node)
        else :
            x_period = self.tree.period_map[x_node]
            if x_period == 0 :
                new_cost = self.cost_by_state(mitigation, average_mitigation+weight*delta_mitigation, emit_node)
                return( (new_cost - base_cost) / delta_mitigation )
            if emit_period == 4 :
                first_state = emit_node - self.tree.decision_period_pointer[emit_period]
                last_state = first_state
            else :
                first_state = self.tree.node_mapping[ emit_period-1 ][emit_node - self.tree.decision_period_pointer[emit_period]][0]
                last_state = self.tree.node_mapping[ emit_period-1 ][emit_node - self.tree.decision_period_pointer[emit_period]][1]
            if x_period >= emit_period :
                return(0.)
            if self.tree.node_mapping[ x_period-1 ][x_node - self.tree.decision_period_pointer[x_period]][0] > last_state :
                return(0.)
            if self.tree.node_mapping[ x_period-1 ][x_node - self.tree.decision_period_pointer[x_period]][1] < first_state :
                return(0.)
            
            new_cost = self.cost_by_state(mitigation, average_mitigation+weight*delta_mitigation, emit_node)

        cost_gradient = (new_cost - base_cost) / delta_mitigation

        return cost_gradient
    
    def dd_am_cost_by_state( self, my_damage_model, mitigation, average_mitigation, tc_years, node, j ):
        '''Calculates the derivative of the cost_by_state function with respect to mitigation in previous periods
        Affected by induced technological innovations
        Depends on that period's weight in the average mitigation
        
        Parameters
        ----------
        my_damage_model : damage class object
            allows calculation of derivative of average_mitigation wrt x
            
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point
        
        tc_years : float
            Years of technological change so far

        node : integer
            node in tree where the derivative of cost is

        j : integer
            node in the tree with respect to which mitigation is taken 
        
        Returns
        -------
        dd_cbs : float
            Derivative of the cost by state with respect to mitigation
            
        '''
        te_term1 = tc_years * ( 1. - ((self.teconst + self.tescale * average_mitigation)/100))**(tc_years-1.0)
        dd_term = -te_term1 * self.tescale * my_damage_model.d_average_mitigation(node, j) / 100.0
        if mitigation < self.cbs_level :
            dd_cbs = ( self.g * mitigation**self.a * dd_term ) / self.consperton0
        else :
            base_cbs = self.g * self.cbs_level**self.a
            extension = ((mitigation - self.cbs_level)*self.maxPrice
                         - self.cbs_b * mitigation * (self.cbs_k/mitigation)**(1./self.cbs_b)/(self.cbs_b-1.)
                         + self.cbs_b * self.cbs_level * (self.cbs_k/self.cbs_level)**(1.0/self.cbs_b)/(self.cbs_b-1.))
            dd_cbs = ( (base_cbs + extension) * dd_term ) / self.consperton0
        return dd_cbs

    def dd_own_cost_by_state( self, mitigation, average_mitigation, tc_years ):
        '''Calculates the derivative of the cost_by_state function with respect to mitigation in the current period
        
        Parameters
        ----------
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point
        
        tc_years : float
            Years of technological change so far
        
        Returns
        -------
        dd_cbs : float
            Derivative of the cost by state with respect to mitigation

        '''
        te_term = ( 1. - ((self.teconst + self.tescale * average_mitigation)/100))**tc_years
        if mitigation < self.cbs_level :
            dd_cbs = self.g * self.a * mitigation**(self.a-1.0)* te_term / self.consperton0
        else:
            dd_cbs =  (self.maxPrice - ( self.cbs_k / mitigation )**(1.0/self.cbs_b)) * te_term / self.consperton0
        return dd_cbs
    
    def price_by_state( self, mitigation, average_mitigation, tc_years ):
        '''Inverse of the cost function, gives emissions price for any given degree of mitigation, average_mitigation, and horizon
        
        Parameters
        ----------
        mitigation : float
            Current mitigation value
        
        average_mitigation : float
            Average mitigation per year up to this point
        
        tc_years : float
            Years of technological change so far
        
        Returns
        -------
        price : float
            Emission price per ton CO2 equivalent
        '''
        te_term = ( 1. - ((self.teconst + self.tescale * average_mitigation)/100))**tc_years
        if mitigation < self.cbs_level :
            price = self.g * self.a * mitigation**(self.a-1.) * te_term
            return price
        else :
            price = (self.maxPrice - (self.cbs_k/mitigation)**(1./self.cbs_b)) * te_term
            return price
