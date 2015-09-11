local vector_utils = require 'utils.vector_utils'
local stats = require 'utils.stats'

test_utils = {}

function test_utils:evaluateModelRankCMH(mlp, indexSet, dataset, cmhEmbeddings, cmhDictionary)
	local sz = indexSet:size()[1]
	print("Evaluating on a dataset of size " .. sz)
	local ranks = {}
	local rankedCompounds = {}

	-- ranked compound abstraction
	local rankedCompound = {}
	function rankedCompound.new(cidx, rank)
		local self = {}
		self.cidx = cidx
		self.rank = rank
		self.compound = cmhDictionary[cidx]
		return self
	end
	function rankedCompound.ascending(c1, c2)
		return c1.rank < c2.rank
	end
	function rankedCompound.descending(c1, c2)
		return c1.rank > c2.rank
	end

	for i = 1, sz do
		if (i % 100 == 0) then
			print(i)
		end
		local cidx = indexSet[i][3]
		if (i < 100) then
			print("Target word: ", cmhDictionary[cidx], cidx)
		end

		local input =  dataset[i][1]
		local target = dataset[i][2]

		-- test sample
		local pred = mlp:forward(input)

		local k = 10
		if (i < 100) then
			local k_neighbours = vector_utils.neighbours(pred, cmhEmbeddings, nil, k)
			print("Predicted neighbours:")
			for key,value in ipairs(k_neighbours) do
				print(key .. " " .. value.index .. " " .. cmhDictionary[value.index])
			end

			print("Correct neighbours:")
			local k_neighbours = vector_utils.neighbours(target, cmhEmbeddings, nil, k)
			for key,value in ipairs(k_neighbours) do
				print(key .. " " .. value.index .. " " .. cmhDictionary[value.index])
			end
		end

		local rank = vector_utils.rank(pred, cidx, cmhEmbeddings, nil)
	    table.insert(rankedCompounds, rankedCompound.new(cidx, rank))
		table.insert(ranks, rank)		
	end

	table.sort(rankedCompounds, rankedCompound.ascending)

	local q1, median, q3, sortedRanks = stats.quartiles(ranks)

	return q1, median, q3, sortedRanks, rankedCompounds
end

function test_utils:avgCosine(mlp, dataset)
	local sz = dataset:size()
	local cosineDist = nn.CosineDistance()
	print("Evaluating on a dataset of size " .. sz)
	local cosineTotal = 0
	for i = 1, sz do
		local input =  dataset[i][1]
		local target = dataset[i][2]

		-- test sample
		local pred = mlp:forward(input)
		local pred_target_cosine = cosineDist:forward({pred, target})
		cosineTotal = cosineTotal + pred_target_cosine[1]
	end
	local avgCos = cosineTotal/sz
	return avgCos
end

return test_utils