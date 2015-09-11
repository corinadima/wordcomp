require 'torch'
require 'nn'

local compose_utils = require 'utils.compose_utils'

--
-- Additive mask composition model
--
-- 

local Addmask, parent = torch.class('torch.Addmask', 'torch.CompositionModel')

function Addmask:__init(inputs, outputs, mhSize)
	parent.__init(self)
	self.inputs = inputs
	self.outputs = outputs
	self.mhSize = mhSize
end

function Addmask:architecture(nonlinearity)
	print("# Addmask; vectors a and b are first masked (component-wise multiplication between the initial vector and the mask vector), "..
	 "then concatenated and composed through the global matrix W (size 2nxn);")
	print("# inputs " .. self.inputs .. ", outputs " .. self.outputs)

	local mlp = nn.Sequential()

	local p = nn.ParallelTable()
	local s1 = nn.Sequential()
	local p1 = nn.ParallelTable()
	local ident_p1 = nn.Identity()
	p1:add(ident_p1)
	-- modifier masks;
	local m_lt = nn.LookupTable(self.mhSize, self.inputs)
	-- initialize modifier masks with ones
	m_lt.weight = torch.ones(self.mhSize, self.inputs)
	p1:add(m_lt)
	s1:add(p1)
	s1:add(nn.CMulTable())

	local s2 = nn.Sequential()
	local p2 = nn.ParallelTable()
	local ident_p2 = nn.Identity()
	p2:add(ident_p2)
	-- head masks	
	local h_lt = nn.LookupTable(self.mhSize, self.inputs)
	-- initialize head masks with ones
	h_lt.weight = torch.ones(self.mhSize, self.inputs)
	p2:add(h_lt)
	s2:add(p2)
	s2:add(nn.CMulTable())

	p:add(s1)
	p:add(s2)

	mlp:add(p)
	mlp:add(nn.CAddTable())

	print("==> Network configuration")
	print(mlp)
	print("==> Parameters size")
	print(mlp:getParameters():size())

	return mlp
end

function Addmask:data(trainSet, devSet, cmhEmbeddings)
	local trainDataset = compose_utils:createCMHMaskDataset(trainSet, cmhEmbeddings)
	local devDataset = compose_utils:createCMHMaskDataset(devSet, cmhEmbeddings)

	return trainDataset, devDataset
end
