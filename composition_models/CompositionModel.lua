require 'torch'

local CompositionModel = torch.class('torch.CompositionModel')

function CompositionModel:__init()
	self.isTrainable = true
end

function CompositionModel:architecture()
end

function CompositionModel:architecture(nonlinearity)
end

function CompositionModel:data()
end