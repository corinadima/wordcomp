require 'torch'
require 'nn'
require 'optim'
require 'paths'
require 'xlua'

require 'composer.lua'

local lua_utils = require 'utils.lua_utils'
local compose_utils = require 'utils.compose_utils'
local test_utils = require 'utils.test_utils'
local nonliniarities = require 'utils.nonliniarities'

require 'composition_models.CompositionModel'
require 'composition_models.Lexfunc'
require 'composition_models.Fulllex'
require 'composition_models.Matrix'
require 'composition_models.Addmask'
require 'composition_models.Wmask'

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- command-line options
cmd = torch.CmdLine()
cmd:text()
cmd:text('Wordcomp: compositionality modelling')
cmd:text()
cmd:text('Options:')
cmd:argument('-model', 'compositionality model to train (see README for details): Wmask|Addmask|Matrix|Fulllex|Lexfunc')
cmd:option('-nonlinearity', 'tanh', 'nonlinearity to use, if needed by the architecture: none|tanh|sigmoid|reLU')
cmd:option('-split', 'dev', 'dataset split: dev|test')
cmd:option('-dim', 50, 'embeddings set, chosen via dimensionality: 50|100|200|300')
cmd:option('-dataset', 'german_compounds_composition_dataset', 'dataset to use: german_compounds_composition_dataset')
cmd:option('-mhSize', 8580, 'number of modifiers and heads in the dataset: 8580')
cmd:option('-embeddings', 'glove_decow14ax_10B.1M.l2norm_axis01', 'embeddings to use: glove_decow14ax_10B.1M.l2norm_axis01')
cmd:option('-threads', '1', 'number of threads to use')
cmd:option('-batchSize', 100, 'mini-batch size (number between 1 and the size of the training data')
cmd:option('-outputDir', 'models', 'output directory to store the trained models')
cmd:text()

opt = cmd:parse(arg)
---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- config
local config = {
	rundir = paths.concat(opt.outputDir, opt.split, opt.dataset, opt.embeddings, opt.dim .. 'd'),
	batchSize = opt.batchSize,
	optimizer = 'adagrad',
	criterion = 'mse',
	adagrad_config = {
		learningRate = 0.1
	},
	earlyStopping = true,
	extraEpochs = 5,
	manualSeed = 1
}

local tf=os.date('%Y-%m-%d_%H-%M',os.time())

-- fix seed, for repeatable experiments
torch.manualSeed(config.manualSeed)

config.configname = opt.model .. '_' .. opt.nonlinearity .. '_' .. config.optimizer .. "_batch" .. config.batchSize .. "_" .. config.criterion

local saveName = paths.concat(config.rundir, "model_" .. config.configname .. "_" .. tf)
xlua.log(saveName .. ".log")

torch.setnumthreads(opt.threads)
print("==> Using " .. torch.getnumthreads() .. " threads")
print("==> options: ", opt)

print("==> config", config)
print("==> optimizer_config: ", config.optimizer_config)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- load data
local trainSet, devSet = compose_utils:loadDatasets(opt.split, opt.dataset, opt.minNum)
local cmhEmbeddings = compose_utils:loadCMHEmbeddings(opt.dataset, opt.embeddings, opt.dim)
local cmhDictionary = compose_utils:loadCMHDictionaries(opt.dataset, opt.minNum)

local sz = cmhEmbeddings:size()[2]

local trainDataset = nil
local devDataset = nil

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- composition models
local composition_models = {}
local nl = {}

composition_models['Wmask'] = torch.Wmask(sz, sz, opt.mhSize, false)
composition_models['Addmask'] = torch.Addmask(sz, sz, opt.mhSize)
composition_models['Fulllex'] = torch.Fulllex(sz, sz, opt.mhSize, true)
composition_models['Matrix'] = torch.Matrix(sz * 2, sz)
composition_models['Lexfunc'] = torch.Lexfunc(sz, opt.mhSize, true)

---------------------------------------------------------------------------
---------------------------------------------------------------------------
-- nonlinearities
nl['tanh'] = nonliniarities:tanhNonlinearity()
nl['hardTanh'] = nonliniarities:hardTanhNonlinearity()
nl['sigmoid'] = nonliniarities:sigmoidNonlinearity()
nl['reLU'] = nonliniarities:reLUNonlinearity()
nl['none'] = nonliniarities:noneNonlinearity()
---------------------------------------------------------------------------
---------------------------------------------------------------------------

local composition_model = composition_models[opt.model]
local nonlinearity = nl[opt.nonlinearity]
local mlp = composition_model:architecture(nonlinearity)
local trainDataset, devDataset = composition_model:data(trainSet, devSet, cmhEmbeddings)

-- train composer
local composer = nn.Composer(mlp, config)
local timer = torch.Timer()


if (composition_model.isTrainable == true) then
	composer:train(trainDataset, devDataset)
	print("==> Training ended");
end

torch.save(saveName .. ".bin", mlp);

print("==> Model saved under " .. saveName .. ".bin");

print("==> Testing model")

local q1, median, q3, sortedRanks, rankedCompounds = test_utils:evaluateModelRankCMH(mlp, devSet, devDataset, cmhEmbeddings, cmhDictionary)
print(string.format("Q1: %.2f\nMedian: %.2f\nQ3: %.2f\n", q1, median, q3))

local ranksOutputFile = saveName .. "_ranks.txt"
local f = io.open(ranksOutputFile, "w")
for i = 1, #sortedRanks do
	f:write(sortedRanks[i] .. "\n")
end
f:close()
print("Sorted ranks saved under " .. ranksOutputFile)

local rankedCompoundsOutputFile = saveName .. "_rankedCompounds.txt"
local f = io.open(rankedCompoundsOutputFile, "w")
for i = 1, #rankedCompounds do
	f:write(rankedCompounds[i].compound .. " " .. rankedCompounds[i].rank .. "\n")
end
f:close()
print("Sorted compounds saved under " .. rankedCompoundsOutputFile)


local avgCos = test_utils:avgCosine(mlp, devDataset)
print("Average cosine distance from predicted to learned: %.2f" % avgCos)

print('Time elapsed (real): ' .. lua_utils:secondsToClock(timer:time().real))
print('Time elapsed (user): ' .. lua_utils:secondsToClock(timer:time().user))
print('Time elapsed (sys): ' .. lua_utils:secondsToClock(timer:time().sys))

print("==> Model saved under " .. saveName .. ".bin");
