require 'optim'

-- class for training a composition model
-- code based on the Torch supervised learning tutorial (https://github.com/torch/tutorials/tree/master/2_supervised)

local Composer = torch.class('nn.Composer')

function Composer:__init(m, config)
	self.module = m
	
	if (config.criterion == 'abs') then
		self.criterion = nn.AbsCriterion()
	elseif (config.criterion == 'mse') then
		self.criterion = nn.MSECriterion()
	else
		error("Unknown criterion.")
	end

	self.config = config
	print(self.config)
end

function Composer:train(trainDataset, devDataset)
	local parameters, gradParameters = self.module:getParameters()	
	self.bestModel = nil
	self.bestError = 2^20
	print("# Composer: training")

	function doTrain(module, criterion, config, trainDataset)
		module:training()

		epoch = epoch or 1
		print("# Epoch ", epoch)
		local trainError = 0

		-- shuffe indices; this means that each pass sees the training examples in a different order 
		local shuffledIndices = torch.randperm(trainDataset:size(), 'torch.LongTensor')

		-- go through the dataset
		for t = 1, trainDataset:size(), config.batchSize do

			-- create mini-batch
			local inputs = {}
			local targets = {}
			for i = t, math.min(t + config.batchSize - 1, trainDataset:size()) do
				-- load new sample
				local example = trainDataset[shuffledIndices[i]]
				local input = example[1]
				local target = example[2]

				table.insert(inputs, input)
				table.insert(targets, target)
			end

			-- create clojure to evaluate function and its derivative on the mini-batch
			local feval = function(x)
				-- just in case:
				collectgarbage()			

				-- get new parameters 
				if x ~= parameters then
					parameters:copy(x)
				end

				-- reset gradients
				gradParameters:zero()

				-- f is the average of all criterions (f is the average error on the mini-batch)
				local f = 0

				-- evaluate on mini-batch
				for i = 1, #inputs do
					-- estimate f for the current input
					local prediction = module:forward(inputs[i])

					-- compute the error between the prediction and the (correct) target using the criterion
					local err = criterion:forward(prediction, targets[i])

					-- add the error to f
					f = f + err

					-- estimate df/dW
					local df_do = criterion:backward(prediction, targets[i])
					module:backward(inputs[i], df_do)
				end   			

				-- normalize gradients and f(X)
				gradParameters:div(#inputs)
				f = f/#inputs
				trainError = trainError + f

				return f, gradParameters
			end	

			-- optimize the current mini-batch
			if config.optimizer == 'adagrad' then
				optim.adagrad(feval, parameters, config.adagrad_config)
			else 
				error('unknown optimization method')
			end
		end

		-- train error 
		trainError = trainError / math.floor(trainDataset:size()/config.batchSize)

		-- next epoch
		epoch = epoch + 1

		return trainError
	end

	function doTest(module, criterion, dataset)
		module:evaluate()
		local testError = 0
		for t = 1, dataset:size() do
			local input = dataset[t][1]
			local target = dataset[t][2]

			-- test sample
			local pred = module:forward(input)

			-- compute error
			err = criterion:forward(pred, target)
			testError = testError + err
		end

		-- average error over the dataset
		testError = testError/dataset:size()

		return testError
	end

	while true do
		trainErr = doTrain(self.module, self.criterion, self.config, trainDataset)
		testErr = doTest(self.module, self.criterion, devDataset)
		print('Train error:\t', string.format("%.6f", trainErr))
		print('Test error:\t', string.format("%.6f", testErr))
		print('Best error:\t', string.format("%.6f", self.bestError))

		-- early stopping when the error on the test set ceases to decrease
		if (self.config.earlyStopping == true) then
			if (testErr < self.bestError) then
				self.bestError = testErr
				self.bestModel = self.module:clone()
				self.extraIndex = 1
			else
				if (self.extraIndex < self.config.extraEpochs) then
					self.extraIndex = self.extraIndex + 1
				else
					print("# Composer: stopping - you have reached the maximum number of epochs after the best model")
					print("# Composer: best error: " .. self.bestError)
					self.module = self.bestModel:clone()
					break					 
				end
			end
		end

	end
end
