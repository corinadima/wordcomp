require 'torch'

-- data loading utilities

local DataLoader = {}

function DataLoader.loadDictionary( fileName )
  local dict = {}
  io.input(fileName)
  local idx = 1
  for line in io.lines() do
    table.insert(dict, lua_utils:trim(line))
  end

  function  dict:getIndex(string)
    local index = -1
    local i = 1
    while i <= #dict do
      if (dict[i] == string) then
        index = i
        break
      end
      i = i + 1
    end
    return index
  end

  return dict
end

function DataLoader.loadSimpleDataset(fileName, separator)
  local dataset = {}
  local dt_index = 0

  function dataset:size()
    return dt_index
  end

  local field_delim = separator or ' '

  io.input(fileName)
  local lines = io.lines()
  local loaded_tensor = nil

  local load_limit = limit or -1 
  for line in io.lines() do
  	local sp = lua_utils:split(lua_utils:trim(line), field_delim)
  	local tensor_size = #sp
  	local loaded_tensor = torch.DoubleTensor(tensor_size):zero()
  	local index = 1
  	while index <= #sp do
  		loaded_tensor[index] = sp[index]
  		index = index + 1
  	end

  	dt_index = dt_index + 1;
    table.insert(dataset, loaded_tensor)
  end

  local datasetTensor = torch.Tensor(dataset:size(), dataset[1]:size()[1])
  for i = 1, dataset:size() do
    datasetTensor[i] = dataset[i]
  end

  return datasetTensor
end

return DataLoader
