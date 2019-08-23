# Continuous Spiking Neural Network

A continuous spiking neural network is a spiking neural network where the neurons produce continuous outputs. The refractory period after a spike is emulated using a continuous variable called a discount, which is based on the output of a neuron. A discount is subtracted from the input of a neuron making it more difficult to activate if it has recently produced a high output. The discount decays over time until it reaches zero, at which point it no longer affects the neuron.

[Visualization](https://www.youtube.com/watch?v=2ukK5qQspEA&feature=youtu.be)

## Example

    # Create network
	network=Network(
		discount_factor=0.2,
		discount_decay=0. ,
		learning_rate=0.1,
		decay_rate=0.01)

	# Create nodes
	network.create_nodes(
		input_count=6,
		hidden_count=15,
		output_count=5)

	# Create input-to-hidden links
	network.create_projection(
		sources=network.input_indices,
		targets=network.hidden_indices,
		connectivity=0.25)

	# Create hidden-to-hidden links
	network.create_projection(
		sources=network.hidden_indices,
		targets=network.hidden_indices,
		connectivity=0.1)

	# Create hidden-to-output links
	network.create_projection(
		sources=network.hidden_indices,
		targets=network.output_indices,
		connectivity=0.25)

	sample=[randrange(2) for i in range(len(network.input_indices))]
	output=network.update(sample)
	print(output)
