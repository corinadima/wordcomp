require 'paths'
dataLoader = require 'utils.DataLoader'

local compose_utils = {}

function compose_utils:loadDatasets(split, datasetDir, minNum)
	print('==> loading datasets...')
	local trainSet, devSet
	if (split == 'dev') then
		trainSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "train.txt"), " ")
		devSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "dev.txt"), " ")
	elseif (split == 'test') then
		trainSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "train.txt"), " ")
		devSet = dataLoader.loadSimpleDataset(paths.concat("data", datasetDir, "test.txt"), " ")
	else 
		print("Error: unknown dataset split type.")
	end
	print('==> dataset loaded, train size:', trainSet:size(), ' dev size', devSet:size())

	return trainSet, devSet 
end	

function compose_utils:loadCMHEmbeddings(datasetDir, embeddingsId, size, normalization)
	print('==> loading embeddings of size ' .. size .. '...')
	local norm = ''
	if (normalization ~= 'none') then
		norm = normalization
	end
	local cmhEmbeddings = dataLoader.loadSimpleDataset(paths.concat('data', datasetDir, 'embeddings', embeddingsId, embeddingsId .. '.' .. size .. 'd_cmh.emb'))
	print('==> embeddings loaded, size:', cmhEmbeddings:size())

	return cmhEmbeddings
end

function compose_utils:loadCMHDictionaries(datasetDir, minNum)
	print('==> loading dictionaries...')
	local cmhDictionary = dataLoader.loadDictionary(paths.concat('data', datasetDir, 'cmh_dict.txt'))
	print('==> dictionaries loaded.')

	return cmhDictionary
end

-- for GlobalW_ab
function compose_utils:createCMH2TensorDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local input = torch.zeros(2, sz)
		input[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		input[2] = cmhEmbeddings[tensorData[i][2]]:clone()


		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {input, output}
	end
	return dataset
end

-- for GlobalW_BaAb
function compose_utils:createCMHMVDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local vectors = torch.zeros(2, sz, 1) -- normal ordering for vectors, first the modifier, than the head
		vectors[1] = cmhEmbeddings[tensorData[i][1]]:clone()
		vectors[2] = cmhEmbeddings[tensorData[i][2]]:clone()

		local matrixIndices = torch.zeros(2,1)

		matrixIndices[1] = tensorData[i][2] -- switch indices for matrices, we want to compute Ba; Ab
		matrixIndices[2] = tensorData[i][1]

		local dualInput = {matrixIndices, vectors} -- first matrix indices, then vectors

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

-- for LexicalFunction
function compose_utils:createCMHMV1Dataset(tensorData, cmhEmbeddings)
	local dataset = {}

	-- matrix index for the modifier, vector for the head

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local modifierIndex = torch.zeros(1)
		modifierIndex[1] = tensorData[i][1]

		local hSz = cmhEmbeddings[tensorData[i][2]]:size()

		local headVector = torch.Tensor(1, hSz[1], 1)
		headVector[1] = cmhEmbeddings[tensorData[i][2]]:clone()

		local dualInput = {modifierIndex, headVector} -- first matrix index - for modifier, then vector - for head

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

-- for GlobalW_mask_ab, Additive_mask_ab
function compose_utils:createCMHMaskDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do

		local c1 = {cmhEmbeddings[tensorData[i][1]]:clone(), torch.Tensor({tensorData[i][1]})}
		local c2 = {cmhEmbeddings[tensorData[i][2]]:clone(), torch.Tensor({tensorData[i][2]})}

		local dualInput = {c1, c2}

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

-- for FullAdditive
function compose_utils:createCMHFullAddDataset(tensorData, cmhEmbeddings)
	local dataset = {}

	local sz = cmhEmbeddings:size()[2]
		
	function dataset:size() return tensorData:size()[1] end
	function dataset:findEntry(compoundIndex)
		for i = 1, tensorData:size()[1] do
			if (tensorData[i][3] == compoundIndex) then
				return i
			end
		end		
		return nil		
	end
	for i = 1, tensorData:size()[1] do
		local c1 = {torch.Tensor({1}), cmhEmbeddings[tensorData[i][1]]:clone()}
		local c2 = {torch.Tensor({1}), cmhEmbeddings[tensorData[i][2]]:clone()}

		local dualInput = {c1, c2}

		local outputIndex = tensorData[i][3]
		local output =  cmhEmbeddings[outputIndex]:clone();
		dataset[i] = {dualInput, output}
	end
	return dataset
end

return compose_utils
