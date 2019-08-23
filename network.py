from random import sample, random
import math

def sign(x):
	return 1 if x>0 else -1 if x<0 else 0

def logistic(x):
    return 1 / (1 + pow(math.e, -x))

class Network:
	def __init__(self, discount_factor, discount_decay, learning_rate, decay_rate):
		self.discount_factor=discount_factor
		self.discount_decay=discount_decay
		self.learning_rate=learning_rate
		self.decay_rate=decay_rate
		self.input_indices=[]
		self.hidden_indices=[]
		self.output_indices=[]
		self.weights={}
		self.outputs=[]
		self.totals=[]
		self.discounts=[]
		self.thresholds=[]
		self.last_outputs=[]

	def create_nodes(self, input_count, hidden_count, output_count):
		node_count=input_count+hidden_count+output_count
		for i in range(node_count):
			self.outputs.append(0)
			self.totals.append(0)
			self.discounts.append(0)
			self.thresholds.append(random())
			self.weights[i]={}
			if i < input_count:
				self.input_indices.append(i)
			elif i < input_count+hidden_count:
				self.hidden_indices.append(i)
			else:self.output_indices.append(i)

	def create_projection(self, sources, targets, connectivity=1.0):
		for i in targets:
			options=list(sources)
			if i in options:
				options.remove(i)
			count=int(len(options)*connectivity)
			if count > 0:
				for j in sample(sources, count):
					self.weights[i][j]=random()

	def compute_discount(self, i):
		di=self.discounts[i]
		yi=self.last_outputs[i]
		di+=yi*self.discount_factor-sign(di)*self.discount_decay
		if di < 0:di=0
		self.discounts[i]=di

	def compute_output(self, i):
		xi=0
		di=self.discounts[i]
		hi=self.thresholds[i]
		for j in self.weights[i]:
			wij=self.weights[i][j]
			yj=self.last_outputs[j]
			xi+=wij*yj
		yi=logistic(xi-di-hi)
		self.totals[i]=xi
		self.outputs[i]=yi

	def update_node(self, i):
		hi=self.thresholds[i]
		xi=self.totals[i]
		yi=self.outputs[i]
		for j in self.weights[i]:
			yj=self.last_outputs[j]
			wij=self.weights[i][j]
			wij+=yi*yj*self.learning_rate-sign(wij)*self.decay_rate
			if abs(wij) > 4:wij=4*sign(wij)
			self.weights[i][j]=wij

		hi+=(xi-hi)*self.learning_rate
		self.thresholds[i]=hi

	def compute_discounts(self):
		for i in self.hidden_indices+self.output_indices:
			self.compute_discount(i)

	def compute_outputs(self):
		for i in self.hidden_indices+self.output_indices:
			self.compute_output(i)

	def update_nodes(self):
		for i in self.hidden_indices+self.output_indices:
			self.update_node(i)

	def set_inputs(self, X):
		for i in range(len(self.input_indices)):
			index=self.input_indices
			value=X[i]
			self.outputs[i]=value

	def get_outputs(self):
		return [self.outputs[i] for i in self.output_indices]

	def update(self, inputs):
		self.last_outputs=self.outputs
		self.set_inputs(inputs)
		self.compute_discounts()
		self.compute_outputs()
		self.update_nodes()
		return self.get_outputs()
