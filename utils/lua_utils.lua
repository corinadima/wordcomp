require 'torch'

lua_utils = {}

function lua_utils:secondsToClock(sSeconds)
	local nSeconds = tonumber(sSeconds)
	if nSeconds == 0 then
	--return nil;
		return "00:00:00";
	else
		nHours = string.format("%02.f", math.floor(nSeconds/3600));
		nMins = string.format("%02.f", math.floor(nSeconds/60 - (nHours*60)));
		nSecs = string.format("%02.f", math.floor(nSeconds - nHours*3600 - nMins *60));
		return nHours..":"..nMins..":"..nSecs .. ' (hh:mm:ss)'
	end
end

-- remove trailing and leading whitespace from string.
-- http://en.wikipedia.org/wiki/Trim_(8programming)
function lua_utils:trim(s)
  -- from PiL2 20.4
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- splits the given input string according to the provided pattern, 
-- and returns a list with the splits
function lua_utils:split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
   table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

-- returns the actual number of items in a table
-- from http://stackoverflow.com/questions/2705793/how-to-get-number-of-entries-in-a-lua-table
function lua_utils:tableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

return lua_utils

