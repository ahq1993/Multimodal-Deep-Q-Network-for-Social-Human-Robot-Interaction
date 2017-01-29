require 'image'
require 'cunn'
require 'nn'
local trans = torch.class('TransitionTable')


function trans:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions or 4
    self.histLen = args.histLen or 8
    self.maxSize = args.maxSize or 30000  --- replay memory_size
    self.bufferSize = args.bufferSize or 3000
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.gpu = args.gpu
    self.buf_ind = 1
	 self.batch_ind_y = 1
	 self.batch_ind_d = 1
    self.histIndices = {}
	 self.numEntries = 0
    self.insertIndex = 0
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = 1 --self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    end
    self.s = torch.Tensor(self.maxSize, self.histLen, self.stateDim,self.stateDim):fill(0)
	 self.d = torch.Tensor(self.maxSize, self.histLen, self.stateDim,self.stateDim):fill(0)
    self.a = torch.Tensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.Tensor(self.maxSize):fill(0)
 

   
    self.buf_a      = torch.Tensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.Tensor(self.bufferSize):fill(0)
    self.buf_s      = torch.Tensor(self.bufferSize,self.histLen, self.stateDim,self.stateDim):fill(0)
	 self.buf_d      = torch.Tensor(self.bufferSize,self.histLen, self.stateDim,self.stateDim):fill(0)
    self.buf_s2     = torch.Tensor(self.bufferSize,self.histLen, self.stateDim,self.stateDim):fill(0)
	 self.buf_d2     = torch.Tensor(self.bufferSize,self.histLen, self.stateDim,self.stateDim):fill(0)
	 collectgarbage()
end

function trans:size()
   return self.numEntries
end

function trans:fill_buffer() 
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s,d, a, r, s2,d2, term = self:sample_one(1)
		  
        self.buf_s[buf_ind]:copy(s)
		  self.buf_d[buf_ind]:copy(d)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
		  self.buf_d2[buf_ind]:copy(d2)
        self.buf_term[buf_ind] = term
		  collectgarbage()
    end
    

	---preprocessing

	local mean = {}
	local std = {}

	collectgarbage()

    
end

function trans:sample_one() ---checx
	
    assert(self.numEntries > 1)
    local index
    local valid = false
    while not valid do
        -- start at 2 because of previous action
        index = torch.random(2, self.numEntries)
		  
        if self.t[index] == 0 then
            valid = true
        end
    end

    return self:get(index)
end

function trans:get(index) 
    local s = self.s[index]
	 
    local s2 = self.s[index+1]
	 local d = self.d[index]
    local d2 = self.d[index+1]
    return s,d, self.a[index], self.r[index], s2,d2, self.t[index+1]
end


function trans:sample_y(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)
    if self.batch_ind_y==1 or self.batch_ind_y + batch_size - 1 > self.bufferSize then

        self:fill_buffer()
    end

	 local s=torch.Tensor(batch_size,self.histLen,self.stateDim,self.stateDim)
	 local s2=torch.Tensor(batch_size,self.histLen,self.stateDim,self.stateDim)
	 
	 
	 if self.batch_ind_y>=self.bufferSize then
		self.batch_ind_y=1
	 end
    	

    local index = self.batch_ind_y
    
    self.batch_ind_y = self.batch_ind_y+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2, self.buf_a, self.buf_r, self.buf_term
    
	 local j=1
	 --image.display(buf_s[index])
	 for i=index,index+batch_size-1 do
		  s[j]=buf_s[i]
		  s2[j]=buf_s2[i]
		  j=j+1
	 end
	 s=s:cuda()
	 s2=s2:cuda()
		collectgarbage()
    return s, buf_a[range], buf_r[range],s2,buf_term[range]
end


function trans:sample_d(batch_size)--checx
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)
    if self.batch_ind_d==1 or self.batch_ind_d + batch_size - 1 > self.bufferSize then

        self:fill_buffer()
    end

	 local d=torch.Tensor(batch_size,self.histLen,self.stateDim,self.stateDim)
	 local d2=torch.Tensor(batch_size,self.histLen,self.stateDim,self.stateDim)
	 
	 if self.batch_ind_d>=self.bufferSize then
		self.batch_ind_d=1
	 end
    	

    local index = self.batch_ind_d
    
    self.batch_ind_d = self.batch_ind_d+batch_size
    local range = {{index, index+batch_size-1}}

    local  buf_d, buf_d2, buf_a, buf_r, buf_term = self.buf_d, self.buf_d2, self.buf_a, self.buf_r, self.buf_term
    
	 local j=1
	 for i=index,index+batch_size-1 do
		  d[j]=buf_d[i]
		  d2[j]=buf_d2[i]
		  j=j+1
	 end
	 d=d:cuda()
	 d2=d2:cuda()
	collectgarbage()
    return d, buf_a[range], buf_r[range],d2,buf_term[range]
end

function trans:add_recent_state(s, term)   


    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
	 
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end

function trans:get_recent_state(s, term) 
    
 return recent_s[1]
end

function trans:add_recent_action(a) 
    

    table.insert(self.recent_a, a)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
    end
end

function trans:add(s,d, a, r, term) 
    assert(s, 'State cannot be nil')
	 assert(d, 'Depth cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end
	
    -- Overwrite (s,d,a,r,t) at insertIndex
    self.s[self.insertIndex] = s
	 self.d[self.insertIndex] = d
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
	  if self.insertIndex == 1 then
    	image.display(self.s[self.insertIndex])
	 end
    if term==1 then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end
		collectgarbage()
end


