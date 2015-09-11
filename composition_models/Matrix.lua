require 'torch'
require 'nn'

local compose_utils = require 'utils.compose_utils'

--
-- Matrix composition model
--
-- 

local Matrix, parent = torch.class('torch.Matrix', 'torch.CompositionModel')

function Matrix:__init(inputs, outputs)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
end

function Matrix:architecture(nonlinearity)
	print("# Matrix; vector a and b are concatenated and composed through the global matrix W (size 2nxn);")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	local mlp = nn.Sequential()
	mlp:add(nn.Reshape(self.inputs))

	mlp:add(nn.Linear(self.inputs, self.outputs))

	-- note: nonlinearity after the linear layer outperforms 
	-- nonlinearity before the linear layer
	if (nonlinearity ~= nil) then
		mlp:add(nonlinearity)
	end

	print("==> Network configuration")
	print(mlp)
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

function Matrix:data(trainSet, devSet, cmhEmbeddings)
	local trainDataset = compose_utils:createCMH2TensorDataset(trainSet, cmhEmbeddings)
	local devDataset = compose_utils:createCMH2TensorDataset(devSet, cmhEmbeddings)

	return trainDataset, devDataset
end
