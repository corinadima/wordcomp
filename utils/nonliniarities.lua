require 'nn'

-- Wrapper module for nonlinearities

nonliniarities = {}

function nonliniarities:tanhNonlinearity()
	return nn.Tanh()
end

function nonliniarities:hardTanhNonlinearity()
	return nn.HardTanh()
end

function nonliniarities:sigmoidNonlinearity()
	return nn.Sigmoid()
end

function nonliniarities:reLUNonlinearity()
	return nn.ReLU()
end

function nonliniarities:noneNonlinearity()
	return nil
end

return nonliniarities