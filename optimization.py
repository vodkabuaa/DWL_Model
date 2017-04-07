from __future__ import division, print_function
import numpy as np
import multiprocessing

class GenericAlgorithm(object):
	"""Optimization algorithm for the DLW model.
	Args:
		pop_amount (int): Number of individuals in the population.
		num_feature (int): The number of elements in each individual, i.e. number of nodes in tree-model.
		num_generations (int): Number of generations of the populations to be evaluated.
		bound (float): Amount to reduce the
		cx_prob (float): Probability of mating.
		mut_prob (float): Probability of mutation.
		utility (obj 'Utility'): Utility object containing the valuation function.
		constraints (ndarray): 1D-array of size (ind_size)
	TODO: Create and individual class.
	"""
	def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound, num_feature, utility,
				 fixed_values=None, fixed_indicies=None, print_progress=False):
		self.num_feature = num_feature
		self.pop_amount = pop_amount
		self.num_gen = num_generations
		self.cx_prob = cx_prob
		self.mut_prob = mut_prob
		self.u = utility
		self.bound = bound
		self.fixed_values = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress

	def _generate_population(self, size):
		"""Return 1D-array of random value in the given bound as the initial population.

		Returns:
			ndarray: Array of random value in the given bound with the shape of ('pop_amount', 'num_feature').
		"""
		#pop = np.random.random([self.pop_amount, self.num_feature]).cumsum(axis=1)*0.1
		pop = np.random.random([size, self.num_feature])*self.bound
		if self.fixed_values is not None:
			for ind in pop:
				ind[self.fixed_indicies] = self.fixed_values
		return pop

	def _evaluate(self, indvidual):
		"""Returns the utility of given individual.

		Parameters
			indvidual (ndarray or list): The shape of 'pop' define as 1 times of num_feature.

		Returns:
			ndarray: Array with utility at time zero.
		"""
		return self.u.utility(indvidual)

	def _select(self, pop, rate):
		"""Returns a 1D-array of selected individuals.

		Parameters:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
			rate (float): The probability of an individual can be selected among population

		Returns:
			ndarray: Selected individuals.
		"""
		index = np.random.choice(self.pop_amount, int(rate*self.pop_amount), replace=False)
		return pop[index,:]

	def _random_index(self, individuals, size):
		"""Generate a random index of individuals of size 'size'.
		Args:
			individuals (ndarray or list): 2D-array of individuals.
			size (int): The number of indices to generate.

		Returns:
			ndarray: 1D-array of indices.
		"""
		inds_size = len(individuals)
		return np.random.choice(inds_size, size)

	def _selection_tournament(self, pop, k, tournsize, fitness):
		"""Select 'k' individuals from the input 'individuals' using 'k'
		tournaments of 'tournsize' individuals.

		Args:
			individuals (ndarray or list): 2D-array of individuals to select from.
			k (int): The number of individuals to select.
			tournsize (int): The number of individuals participating in each tournament.

		Returns:
			ndarray: Selected individuals.

		"""
		chosen = []
		for i in xrange(k):
			index = self._random_index(pop, tournsize)
			aspirants = pop[index]
			aspirants_fitness = fitness[index]
			chosen_index = np.where(aspirants_fitness == np.max(aspirants_fitness))[0]
			if len(chosen_index) != 0:
				chosen_index = chosen_index[0]
			chosen.append(aspirants[chosen_index])
		return np.array(chosen)

	def _two_point_cross_over(self, pop):
		"""Performs a two-point cross-over of the population.

		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			if np.random.random() <= self.cx_prob:
				cxpoint1 = np.random.randint(1, self.num_feature)
				cxpoint2 = np.random.randint(1, self.num_feature - 1)
				if cxpoint2 >= cxpoint1:
					cxpoint2 += 1
				else: # Swap the two cx points
					cxpoint1, cxpoint2 = cxpoint2, cxpoint1
				child1[cxpoint1:cxpoint2], child2[cxpoint1:cxpoint2] \
				= child2[cxpoint1:cxpoint2].copy(), child1[cxpoint1:cxpoint2].copy()
				if self.fixed_values is not None:
					child1[self.fixed_indicies] = self.fixed_values
					child2[self.fixed_indicies] = self.fixed_values

	def _uniform_cross_over(self, pop, ind_prob):
		"""Performs a uniform cross-over of the population.

		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
			ind_prob (float): Probability of feature cross-over.

		"""
		child_group1 = pop[::2]
		child_group2 = pop[1::2]
		for child1, child2 in zip(child_group1, child_group2):
			size = min(len(child1), len(child2))
			for i in range(size):
				if np.random.random() < ind_prob:
					child1[i], child2[i] = child2[i], child1[i]

	def _mutate(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability
		of 'self.mut_prob' of beeing selected and every element in this
		individual has a probability 'ind_prob' of beeing mutated. The mutated
		value is a random number.
		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
			ind_prob (float): Probability of feature mutation.
			scale (float): The scaling of the random generated number for mutation.
		"""
		pop_tmp = np.copy(pop)
		mutate_index = np.random.choice(self.pop_amount, int(self.mut_prob * self.pop_amount), replace=False)
		for i in mutate_index:
			feature_index = np.random.choice(self.num_feature, int(ind_prob * self.num_feature), replace=False)
			for j in feature_index:
				if self.fixed_indicies is not None and j in self.fixed_indicies:
					continue
				else:
					pop[i][j] = max(0.0, pop[i][j]+(np.random.random()-0.5)*scale)

	def _uniform_mutation(self, pop, ind_prob, scale=2.0):
		"""Mutates individual's elements. The individual has a probability
		of 'self.mut_prob' of beeing selected and every element in this
		individual has a probability 'ind_prob' of beeing mutated. The mutated
		value is the current value plus a scaled uniform [-0.5,0.5] random value.
		Args:
			pop (ndarray): Population given by 2D-array with shape ('pop_amount', 'num_feature').
			ind_prob (float): Probability of feature mutation.
			scale (float): The scaling of the random generated number for mutation.
		"""
		pop_len = len(pop)
		mutate_index = np.random.choice(pop_len, int(self.mut_prob * pop_len), replace=False)
		for i in mutate_index:
			prob = np.random.random(self.num_feature)
			inc = (np.random.random(self.num_feature) - 0.5)*scale
			pop[i] += (prob > (1.0-ind_prob)).astype(int)*inc
			pop[i] = np.maximum(0.0, pop[i])
			if self.fixed_values is not None:
				pop[i][self.fixed_indicies] = self.fixed_values

	def _show_evolution(self, fits, pop):
		"""Print statistics of the evolution of the population."""
		length = len(pop)
		mean = fits.mean()
		std = fits.std()
		min_val = fits.min()
		max_val = fits.max()
		print (" Min {} \n Max {} \n Avg {}".format(min_val, max_val, mean))
		print (" Std {} \n Population Size {}".format(std, length))
		#print (" Best Individual: ", pop[np.argmax(fits)])

	def _survive(self, pop_tmp, fitness_tmp):
		"""
		"""
		index_fits  = np.argsort(fitness_tmp)[::-1]
		fitness = fitness_tmp[index_fits]
		pop = pop_tmp[index_fits]
		num_survive = int(0.8*self.pop_amount)
		survive_pop = np.copy(pop[:num_survive])
		survive_fitness = np.copy(fitness[:num_survive])
		return np.copy(survive_pop), np.copy(survive_fitness)

	def run(self):
		"""Start the evolution process.
		The evolution steps:
			1. Select the individuals to perform cross-over and mutation.
			2. Cross over among the selected candidate.
			3. Mutate result as offspring.
			4. Combine the result of offspring and parent together. And selected the top
			   80 percent of original population amount.
			5. Random Generate 20 percent of original population amount new individuals
			   and combine the above new population.
		"""
		print("----------------Genetic Evolution Starting----------------")
		pop = self._generate_population(self.pop_amount)
		pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
		fitness = pool.map(self._evaluate, pop) # how do we know pop[i] belongs to fitness[i]?
		fitness = np.array([val[0] for val in fitness])
		for g in range(0, self.num_gen):
			print ("-- Generation {} --".format(g+1))
			pop_select = self._select(np.copy(pop), rate=1)
			#pop_select = self._selection_tournament(pop, len(pop), 4, fitness)
			#self._two_point_cross_over(pop_select)
			self._uniform_cross_over(pop_select, 0.50)
			#do the check for mutation other here and save the indices
			self._uniform_mutation(pop_select, 0.20, np.exp(-float(g)/self.num_gen)**2)
			self._mutate(pop_select, 0.05)

			fitness_select = pool.map(self._evaluate, pop_select)
			fitness_select = np.array([val[0] for val in fitness_select])

			pop_tmp = np.append(pop, pop_select, axis=0)
			fitness_tmp = np.append(fitness, fitness_select, axis=0)

			pop_survive, fitness_survive = self._survive(pop_tmp, fitness_tmp)

			#pop_new = np.random.random([self.pop_amount - len(pop_survive), self.num_feature])*self.bound
			pop_new = self._generate_population(self.pop_amount - len(pop_survive))
			fitness_new = pool.map(self._evaluate, pop_new)
			fitness_new = np.array([val[0] for val in fitness_new])

			pop = np.append(pop_survive, pop_new, axis=0)
			fitness = np.append(fitness_survive, fitness_new, axis=0)
			if self.print_progress:
				self._show_evolution(fitness, pop)

		fitness = pool.map(self._evaluate, pop)
		fitness = np.array([val[0] for val in fitness])
		return pop, fitness



class GradientSearch(object) :
	"""
    reference the algorithm in http://cs231n.github.io/neural-networks-3/
	"""

	def __init__(self, learning_rate, var_nums, utility, accuracy=1e-06, iterations=100,
				 step=0.00001, fixed_values=None, fixed_indicies=None, print_progress=False, scale_alpha=0.01):
		self.alpha = learning_rate
		self.u = utility
		self.var_nums = var_nums
		self.step = step
		self.accuracy = accuracy
		self.iterations = iterations
		self.fixed_values  = fixed_values
		self.fixed_indicies = fixed_indicies
		self.print_progress = print_progress
		self.scale_alpha = scale_alpha
		if scale_alpha is None:
			self.scale_alpha = np.exp(np.linspace(0.0, 3.0, var_nums))

	def _initial_values(self, size):
		m = np.random.random(size) * 2
		return m

	def _ada_grad(self, cum_grad):
		epsilon = 1e-8
		ita = 0.001
		return 1/np.sqrt(cum_grad + epsilon) * ita


	def _rms_prop_revise(self, history_grad, grad_t, accumlate_dgrad):
		beta1 = 0.9
		beta2 = 0.999
		ita = 0.001
		eps = 1e-8
		def accelerate_factor(k, accumlate_dgrad):
			print("accumlate_dgrad: ", accumlate_dgrad)
			return (k / (1 + np.exp(np.abs(accumlate_dgrad))) - k / 2)

		acc_factor = accelerate_factor(5, accumlate_dgrad)
		print("acc_factor: ", acc_factor)
		E_g2 = np.mean(np.power(history_grad, 2), axis =0)
		E_g2 = 0.9 * E_g2 + 0.1 * np.power(grad_t, 2)
		return ita / np.sqrt(E_g2 + eps) * grad_t * (1)

	def _accelerate_scale(self, accelerator, history_grad, grad):
		sign_vector = np.sign(history_grad[-1] * grad)
		scale_vector = np.ones(self.var_nums) * ( 1 + 0.001)
		scale_vector[sign_vector < 0] = ( 1 - 0.001)
		print("accelerator: ", accelerator)
		return accelerator * scale_vector

	def _rms_prop(self, history_grad, grad_t):
		"""
		RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton in Lecture 6e
		of his Coursera Class.
		http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
		RMSprop and Adadelta have both been developed independently around the same time stemming
		from the need to resolve Adagrad's radically diminishing learning rates.
		"""

		beta1 = 0.9
		beta2 = 0.999
		ita = 0.001
		eps = 1e-8
		E_g2 = np.mean(np.power(history_grad, 2), axis =0)
		E_g2 = 0.9 * E_g2 + 0.1 * np.power(grad_t, 2)
		return ita / np.sqrt(E_g2 + eps) * grad_t

	def _adam(self, t, history_grad, grad_t):
		"""
		http://sebastianruder.com/optimizing-gradient-descent/index.html#fnref:15
		Adaptive Moment Estimation (Adam) [15] is another method that computes adaptive learning rates for each parameter.
		In addition to storing an exponentially decaying average of past squared gradients vtvt like Adadelta and
		RMSprop, Adam also keeps an exponentially decaying average of past gradients mtmt, similar to momentum
		"""
		beta1 = 0.9
		beta2 = 0.9
		ita = 0.002
		eps = 1e-8
		m_t = np.mean(history_grad, axis = 0)
		v_t = np.mean(np.power(history_grad, 2), axis =0)
		m_t = beta1 * m_t + (1 - beta1) * grad_t
		v_t = beta2 * v_t + (1 - beta2) * np.power(grad_t, 2)
		m_t = m_t / (1 - beta1**t)
		v_t = v_t / (1 - beta2**t)
		print("history_grad", history_grad)
		print("m_t", m_t)
		print("v_t", v_t)
		return ita * m_t / (np.sqrt(v_t) + eps)

	def _dynamic_alpha(self, x_increase, grad_increase, grad_size):
		if np.all(grad_increase == 0):
			return np.zeros(grad_size)
		cons = np.abs(np.dot(x_increase, grad_increase) /  np.square(grad_increase).sum())
		return cons*self.scale_alpha

	def gradient_descent(self, initial_point, return_last=False):
		"""
		Annealing the learning rate. Step decay: Reduce the learning rate by some factor every few epochs.
		Typical values might be reducing the learning rate by a half every 5 epochs,
		"""
		learning_rate = self.alpha
		num_decision_nodes = initial_point.shape[0]
		x_hist = np.zeros((self.iterations+1, num_decision_nodes))
		u_hist = np.zeros(self.iterations+1)
		u_hist[0] = self.u.utility(initial_point)
		x_hist[0] = initial_point
		prev_grad = 0.0

		cum_grad = np.array(np.zeros(self.var_nums))
		history_grad = np.array(np.zeros(self.var_nums))
		accumlate_dgrad = np.array(np.zeros(self.var_nums))
		adam_rate = np.array(np.zeros(self.var_nums))
		rms_prop_rate = np.array(np.zeros(self.var_nums))
		accelerator = np.ones(self.var_nums)

		for i in range(self.iterations):
			grad = self.u.numerical_gradient(x_hist[i], fixed_indicies=self.fixed_indicies)
			cum_grad += np.power(grad,2)
			if i != 0:
				#learning_rate = self._dynamic_alpha(x_hist[i]-x_hist[i-1], grad-prev_grad, len(grad))
				#learning_rate = self._ada_grad(cum_grad)
				adam_rate = self._adam(i+1, history_grad, grad)
				accelerator = self._accelerate_scale(self, accelerator, history_grad, grad)
				#accumlate_dgrad += (grad - history_grad[-1]) /(grad)
				#print("accumlate_dgrad------:", accumlate_dgrad)
				#rms_prop_rate = self._rms_prop_revise(history_grad, grad, accumlate_dgrad)
				#rms_prop_rate = self._rms_prop(history_grad, grad)
			#new_x = x_hist[i] + grad * learning_rate

			new_x = x_hist[i] + adam_rate * accelerator
			#new_x = x_hist[i] + rms_prop_rate
			history_grad = np.vstack([history_grad, grad])
			print("grade: ", grad)
			if self.fixed_values is not None:
				new_x[self.fixed_indicies] = self.fixed_values

			current = self.u.utility(new_x)[0]
			x_hist[i+1] = new_x
			u_hist[i+1] = current
			prev_grad = grad.copy()
			#if i > 50:
			#	x_diff = np.abs(x_hist[i+1] - x_hist[i]).sum()
			#	u_diff = np.abs(u_hist[i+1] - u_hist[i])
			#	if x_diff < self.accuracy or u_diff < self.accuracy:
			#		print("Broke iteration..")
			#		break
			if self.print_progress:
				print("-- Interation {} -- \n Current Utility: {}".format(i+1, current))
				print(new_x)

		if return_last:
			return x_hist[i+1], u_hist[i+1]
		best_index = np.argmax(u_hist)
		return x_hist[best_index], u_hist[best_index]

	def run(self, topk=4, initial_point_list=None, size=None):
		"""Initiate the gradient search algorithm.
		Args:
			m (ndarray or list): 1D numpy array of size (num_decision_nodes).
			alpha (float): Step size in gradient descent.
			num_iter (int): Number of iterations to run.
		Returns:
			ndarray: The history of parameter vector, 2D numpy array of size (num_iter+1, num_decision_nodes)
		"""
		print("----------------Gradient Search Starting----------------")
		if initial_point_list is None:
			if size is None:
				raise ValueError("Need size of the initial point array")
			initial_point_list = np.array(self._initial_values(size))

		if topk > len(initial_point_list):
			raise ValueError("topk {} > number of initial points {}".format(topk, len(initial_point_list)))

		candidate_points = initial_point_list[:topk]
		mitigations = []
		utilities = np.zeros(topk)
		for cp, count in zip(candidate_points, range(topk)):
			if not isinstance(cp, np.ndarray):
				cp = np.array(cp)
			print("Starting process {} of Gradient Descent".format(count+1))
			m, u = self.gradient_descent(cp)
			mitigations.append(m)
			utilities[count] = u
		best_index = np.argmax(utilities)

		return mitigations[best_index], utilities[best_index]


class NodeMaximum(object):
	@staticmethod
	def _min_func(x, m, i, utility):
		m_copy = m.copy()
		m_copy[i] = x
		return -utility.utility(m_copy)[0]

	@classmethod
	def run(cls, m, utility):
		from scipy.optimize import fmin
		print("Utility before {}".format(utility.utility(m)[0]))
		print("Starting maximizing node wise..")
		for i in range(len(m))[::-1]:
			m[i] = fmin(cls._min_func, x0=m[i], args=(m, i, utility), disp=False)
		print("Done!")
		print("Utility after {}".format(utility.utility(m)[0]))
		return m
