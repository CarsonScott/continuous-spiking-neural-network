# Continuous Spiking Neural Network

A continuous spiking neural network is a spiking neural network where the neurons produce continuous outputs. The refractory period after a spike is emulated using a continuous variable called a discount, which is based on the output of a neuron. A discount is subtracted from the input of a neuron making it more difficult to activate if it has recently produced a high output. The discount decays over time until it reaches zero, at which point it no longer affects the neuron.

[Visualization](https://www.youtube.com/watch?v=2ukK5qQspEA&feature=youtu.be)
