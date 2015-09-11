require 'torch'
require 'nn'

local compose_utils = require 'utils.compose_utils'

--
-- Full lexical composition model
--
-- 


local Fulllex, parent = torch.class('torch.Fulllex', 'torch.CompositionModel')

function Fulllex:__init(inputs, outputs, mhEmbeddingsSize, withNoise)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.mhEmbeddings = mhEmbeddings
	self.mhSize = mhEmbeddingsSize
	self.withNoise = withNoise
end

function Fulllex:parallelLayer()

	local seq = nn.Sequential()
	local p = nn.ParallelTable()

	local c = nn.ConcatTable()
	local s1 = nn.Sequential()
	local lt = nn.LookupTable(self.mhSize, self.inputs * self.inputs)

	local e = 0.0001
	local ltInit = torch.Tensor(self.mhSize, self.inputs * self.inputs)
	for i = 1, self.mhSize do
		ltInit[i] = torch.diag(torch.ones(self.inputs)) 
		if (self.withNoise == true) then
			ltInit[i] = ltInit[i] + (torch.rand(self.inputs, self.inputs) * e)
		end
	end

	if (self.withNoise == false) then
		lt.bias = torch.zeros(self.mhSize)
	end

	lt.weight = ltInit

	s1:add(lt)
	s1:add(nn.Reshape(self.inputs, self.inputs))
	c:add(s1)

	local pseq = nn.Sequential()
	pseq:add(c)
	pseq:add(nn.JoinTable(2))
	p:add(pseq)

	local s2 = nn.Sequential()
	local ident = nn.Identity()
	s2:add(ident)
	s2:add(nn.Reshape(self.inputs, 1))
	p:add(s2)

	seq:add(p)
	seq:add(nn.MM())
	seq:add(nn.Reshape(self.inputs * 2))

	return seq
end

function Fulllex:architecture(nonlinearity)
	print("# Fulllex; vector a and b are each multiplied by a word matrix (B and A), then concatenated and composed through the global matrix W (size 2nxn);")
	print("# B and A initialization: weights start from eye(no_inputs) + noise")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	local mlp = nn.Sequential() 
	local p = self:parallelLayer()
	mlp:add(p)

	mlp:add(nn.Linear(self.inputs * 2, self.outputs))

	if (nonlinearity ~= nil) then
		mlp:add(nonlinearity)
	end

	print("==> Network configuration")
	print(mlp)
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

function Fulllex:data(trainSet, devSet, cmhEmbeddings)
	local trainDataset, devDataset
	trainDataset = compose_utils:createCMHMVDataset(trainSet, cmhEmbeddings)
	devDataset = compose_utils:createCMHMVDataset(devSet, cmhEmbeddings)

	return trainDataset, devDataset
end
