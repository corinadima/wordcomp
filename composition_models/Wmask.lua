require 'torch'
require 'nn'

local compose_utils = require 'utils.compose_utils'

--
-- Wmask composition model
--
-- 

local Wmask, parent = torch.class('torch.Wmask', 'torch.CompositionModel')

function Wmask:__init(inputs, outputs, mhSize, withDroput)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.mhSize = mhSize
	self.withDroput = withDroput
end

function Wmask:architecture(nonlinearity)
	print("# Wmask; vectors a and b are first masked, then concatenated and composed through the global matrix W (size 2nxn);")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	local mlp = nn.Sequential()

	local p = nn.ParallelTable()
	local s1 = nn.Sequential()
	local p1 = nn.ParallelTable()
	local ident_p1 = nn.Identity()
	p1:add(ident_p1)
	local m_lt = nn.LookupTable(self.mhSize, self.inputs)
	m_lt.weight = torch.ones(self.mhSize, self.inputs)
	p1:add(m_lt)
	s1:add(p1)
	s1:add(nn.CMulTable())

	local s2 = nn.Sequential()
	local p2 = nn.ParallelTable()
	local ident_p2 = nn.Identity()
	p2:add(ident_p2)
	local c_lt = nn.LookupTable(self.mhSize, self.inputs)
	c_lt.weight = torch.ones(self.mhSize, self.inputs)
	p2:add(c_lt)
	s2:add(p2)
	s2:add(nn.CMulTable())

	p:add(s1)
	p:add(s2)

	mlp:add(p)

	mlp:add(nn.JoinTable(1))
	mlp:add(nn.Reshape(2*self.inputs))

	if (self.withDroput) then
		mlp:add(nn.Dropout(0.15))
	end

	mlp:add(nn.Linear(2*self.inputs, self.outputs))

	if (nonlinearity ~= nil) then
		mlp:add(nonlinearity)
	end
	
	print("==> Network configuration")
	print(mlp)
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

function Wmask:data(trainSet, devSet, cmhEmbeddings)
	local trainDataset = compose_utils:createCMHMaskDataset(trainSet, cmhEmbeddings)
	local devDataset = compose_utils:createCMHMaskDataset(devSet, cmhEmbeddings)

	return trainDataset, devDataset
end
