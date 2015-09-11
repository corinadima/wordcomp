require 'torch'
require 'nn'

vector_utils = {}

-- neighbour abstraction
local neighbour = {}
function neighbour.new(c, t, idx)
  local self = {}
  self.cosine = c
  self.tensor = t
  self.index = idx
  return self
end
function neighbour.ascending(n1, n2)
  return n1.cosine < n2.cosine
end
function neighbour.descending(n1, n2)
  return n1.cosine > n2.cosine
end

function vector_utils.neighbours(vector, vector_space, metric, k)
-- cosine distance: 1 is exactly the same, 0 is independent 
-- in-between values indicate intermediate similarity or dissimilarity
  local metric = metric or nn.CosineDistance()
  
  -- compute neares neighbours with respect to dataset
  local neighbours = {}
  for i = 1, vector_space:size()[1] do
    compared_vector = vector_space[i]
    local distance = metric:forward({vector, compared_vector})
    table.insert(neighbours, neighbour.new(distance[1], compared_vector, i))
  end

  table.sort(neighbours, neighbour.descending)

  -- select the top k ones
  k_nearest = {}
  idx = 1
  for key,value in ipairs(neighbours) do
    if (idx > k) then
      break
    end
    table.insert(k_nearest,value) 
    idx = idx + 1
  end

  return k_nearest

end

function vector_utils.rank(vector, cidx, vector_space, metric)
  local metric = metric or nn.CosineDistance()
  
  -- -- neighbour abstraction
  -- local neighbour = {}
  -- function neighbour.new(c, t, idx)
  --   local self = {}
  --   self.cosine = c
  --   self.tensor = t
  --   self.index = idx
  --   return self
  -- end
  -- function neighbour.ascending(n1, n2)
  --   return n1.cosine < n2.cosine
  -- end
  -- function neighbour.descending(n1, n2)
  --   return n1.cosine > n2.cosine
  -- end

  -- compute neares neighbours with respect to dataset
  local neighbours = {}
  for i = 1, vector_space:size()[1] do
    compared_vector = vector_space[i]
    local distance = metric:forward({vector, compared_vector})
    table.insert(neighbours, neighbour.new(distance[1], compared_vector, i))
  end

  table.sort(neighbours, neighbour.descending)

  for key,value in ipairs(neighbours) do
    if (key > 1000) then
      return 1000
    end
    if (cidx == value.index) then
      return key
    end
  end
end

return vector_utils