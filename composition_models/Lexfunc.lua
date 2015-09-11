require 'torch'
require 'nn'

local compose_utils = require 'utils.compose_utils'
--
-- Lexical function composition model
--
-- 
local Lexfunc, parent = torch.class('torch.Lexfunc', 'torch.CompositionModel')

function Lexfunc:__init(inputs, mhEmbeddingsSize, withNoise)
	parent.__init(self)
	self.inputs = inputs
	self.mhSize = mhEmbeddingsSize
	self.withNoise = withNoise
end

function Lexfunc:architecture(nonlinearity)
	print("# Lexfunc; p = Av, A learned, starts from I^nxn + noise, v given (embedding)")
	print("# inputs " .. self.inputs)

	local mlp = nn.Sequential()
	local p = nn.ParallelTable()
	local seq = nn.Sequential()
	local m_lt = nn.LookupTable(self.mhSize, self.inputs * self.inputs)

	local ltInit = torch.Tensor(self.mhSize, self.inputs * self.inputs)
	local e = 0.0001
	for i = 1, self.mhSize do
		ltInit[i] = torch.eye(self.inputs, self.inputs) 
		if (self.withNoise == true) then
			ltInit[i] = ltInit[i] +  (torch.rand(self.inputs, self.inputs) * e)
		end
	end
	m_lt.weight = ltInit
	seq:add(m_lt)
	seq:add(nn.Reshape(self.inputs, self.inputs))

	p:add(seq)

	local ident = nn.Identity()
	p:add(ident)

	mlp:add(p)
	mlp:add(nn.MM())

	print("==> Network configuration")
	print(mlp)
	print(mlp:getParameters():size())

	return mlp
end

function Lexfunc:data(trainSet, devSet, cmhEmbeddings)
	local trainDataset = compose_utils:createCMHMV1Dataset(trainSet, cmhEmbeddings)
	local devDataset = compose_utils:createCMHMV1Dataset(devSet, cmhEmbeddings)

	return trainDataset, devDataset
end
