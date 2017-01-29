require 'gmodel'
require 'TransitionTable'
require 'optim'
require 'environment'
require 'image'
require 'paths'

local nql = torch.class('NeuralQLearner')


function nql:__init(args)
    self.state_dim  = 198 -- State dimensionality 84x84.
    self.actions    = {'1','2','3','4'}
    self.n_actions  = #self.actions
    
    --- epsilon annealing
    self.ep_start   = 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = 0.1
    self.ep_endt    = 30000

    ---- learning rate annealing
    self.lr_start       = 0.00025 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = self.lr
    self.lr_endt        = 100000   ---replay memory size
    self.wc             = 0  -- L2 weight cost.
    self.minibatch_size = 25
    self.valid_size     = 500

    --- Q-learning parameters
    self.discount       = 0.99 --Discount factor.
	 
    self.t_steps= args.tsteps
	 self.numSteps = 0
    -- Number of points to replay per learning step.
    self.n_replay       = 1
    -- Number of steps after which learning starts.
    self.learn_start    = 3000
     -- Size of the transition table.
    self.replay_memory  = 30000--10000

    self.hist_len       = 8
    self.clip_delta     = 1
    self.target_q       = 4
    self.bestq          = 0

    self.gpu            = 1

    self.ncols          = 1  -- number of color channels in input
    self.input_dims     = {8, 198, 198}
    self.histType       = "linear"  -- history type to use
    self.histSpacing    = 1
    
    self.bufferSize     =  2000
	
    self.episode=args.epi-1
	collectgarbage()
		
	 local modelA='results/ep'..self.episode..'/modelA_gpu.net'
	 local tmodelA='results/ep'..self.episode..'/tmodelA_gpu.net'
	 local modelB='results/ep'..self.episode..'/modelB_gpu.net'
	 local tmodelB='results/ep'..self.episode..'/tmodelB_gpu.net'
    if paths.filep(modelA) and paths.filep(modelB)  then
		print("Loading model")
		-- y-channel	
		self.network_A=torch.load(modelA)
		self.target_network_A=torch.load(tmodelA)
		--depth channel
		self.network_B=torch.load(modelB)
		self.target_network_B=torch.load(tmodelB)
	 else
		print("new model")
    	self.network_A,self.network_B =create_network()
		self.target_network_A=self.network_A:clone()
		self.target_network_B=self.network_B:clone()   
		 
	 end
collectgarbage()
	
----------------------------------------
if self.target_q and self.episode % self.target_q == 0 then
		  print ("cloning")
        self.target_network_A = self.network_A:clone()
		  self.target_network_B = self.network_B:clone()
end

    if self.gpu and self.gpu >= 0 then
        self.network_A:cuda()
		  self.network_B:cuda()
		  self.target_network_A:cuda()
		  self.target_network_B:cuda()
    else
        self.network_A:float()
		  self.network_B:float()
		  self.target_network_A:float()
		  self.target_network_B:float()
    end

   

   -- if self.gpu and self.gpu >= 0 then
    --     torch.setdefaulttensortype('torch.CudaTensor')       
    --else
		  torch.setdefaulttensortype('torch.FloatTensor')
    --end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing,bufferSize = self.bufferSize}

    self.transitions =TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
	 self.lastDepth = nil
    self.lastAction = nil
    self.lastTerminal=nil
    self.wc =0
	 self.wA, self.dwA = self.network_A:getParameters()
    self.dwA:zero()
	 self.deltasA = self.dwA:clone():fill(0)

    self.tmpA= self.dwA:clone():fill(0)
    self.gA  = self.dwA:clone():fill(0)
    self.gA2 = self.dwA:clone():fill(0)

	 self.wB, self.dwB = self.network_B:getParameters()
    self.dwB:zero()
	 self.deltasB = self.dwB:clone():fill(0)
    self.tmpB= self.dwB:clone():fill(0)
    self.gB  = self.dwB:clone():fill(0)
    self.gB2 = self.dwB:clone():fill(0)

    
end



function nql:load_data()
	print("loading")
	
	for i=1,14 do
			print(i)
			local dirname_rgb='dataset/RGB/ep'..i
			local dirname_dep='dataset/Depth/ep'..i


			k=0
			for file in paths.iterfiles(dirname_rgb) do
				k=k+1
			end
		   k=k/8
			while k%4 ~=0 do
				k=k-1
			end
			print("K")
			print(k)
	
		
			local images=torch.Tensor(k,self.hist_len,self.state_dim,self.state_dim)
			local depths=torch.Tensor(k,self.hist_len,self.state_dim,self.state_dim)	
	
			images,depths=get_data(i,k)	
			print ("loading done")
			local aset = {'1','2','3','4'}
	
			local rewards=torch.load('files/reward_history.dat')
			local actions=torch.load('files/action_history.dat')
			local ep_rewards=torch.load('files/ep_rewards.dat')
			collectgarbage()
				for step=1,k do
					local terminal =0
					
					if rewards[i][step]>3 then
						self.transitions:add(images[step],depths[step],actions[i][step],1,terminal)
					elseif rewards[i][step]<0 then
						self.transitions:add(images[step],depths[step],actions[i][step],-0.1,terminal)
					else

						self.transitions:add(images[step],depths[step],actions[i][step],0,terminal)
					end
		
				end		

			collectgarbage()
	
	end
end 
	
function nql:train()
	--self:load_data()
   local q_all_A,q_all_B, q_s,q_d, q2_max_s, q2_max_d, q2_s, q2_d,delta_s,delta_d
	self.network_A:training()
	self.network_B:training()
	local target_q_net_A,target_q_net_B
    if self.target_q then
        target_q_net_A = self.target_network_A
		  target_q_net_B = self.target_network_B
    else
        target_q_net_A = self.network_A
		  target_q_net_B = self.network_B
    end
   target_q_net_A:evaluate()
	target_q_net_B:evaluate()
	local q_max_avg_s=0
	local q_max_avg_d=0
	local q1_max_avg_s=0
	local q1_max_avg_d=0
	local td_err_s=0
	local td_err_d=0
	k=1
------------------------------------Training Y-channel network--------------------------------------------------	
	for j=1,self.bufferSize, self.minibatch_size do
    local s, a, r, s2, term = self.transitions:sample_y(self.minibatch_size)
		local win=nil
	 
		
			self.dwA:zero()

			for i=1,self.minibatch_size do
				local input1=s[i]
				input1=input1:cuda()
			   local input2=s2[i]
				input2=input2:cuda()
    	      q_all_A = self.network_A:forward(input1)
		 		q_s=q_all_A[a[i]] -- action index integer 
		 		q1_max_s=q_all_A:float():max()
			   local  out= target_q_net_A:forward(input2)
				q2_max_s=out:float():max()
				q_max_avg_s=q_max_avg_s+q2_max_s
				q1_max_avg_s=q1_max_avg_s+q1_max_s
				 -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    			q2_s = q2_max_s*self.discount
	 			delta_s = r[i]
   		   delta_s=delta_s+q2_s-q_s
				td_err_s=td_err_s+torch.abs(delta_s)
				if self.clip_delta then
				   if delta_s > self.clip_delta then delta_s=self.clip_delta end
					if delta_s < -self.clip_delta then delta_s=-self.clip_delta end
					
    			end

    			local df_do = torch.zeros(self.n_actions):float()
	 			 
			   df_do[a[i]] = delta_s
			   if self.gpu and self.gpu >= 0 then
            	df_do = df_do:cuda()		
 
				end
				self.network_A:backward(input1,df_do)
				
			 end

	 self.dwA:add(-self.wc, self.wA)


	  -- use gradients
    self.gA:mul(0.95):add(0.05, self.dwA)
    self.tmpA:cmul(self.dwA, self.dwA)
    self.gA2:mul(0.95):add(0.05, self.tmpA)
    self.tmpA:cmul(self.gA, self.gA)
    self.tmpA:mul(-1)
    self.tmpA:add(self.gA2)
    self.tmpA:add(0.01)
    self.tmpA:sqrt()

    -- accumulate update
    self.deltasA:mul(0):addcdiv(self.lr, self.dwA, self.tmpA)
    self.wA:add(self.deltasA)
				
	collectgarbage()
	 end

-------------------------------------Training Depth Channel-----------------------------------------
for j=1,self.bufferSize, self.minibatch_size do
    local d, a, r,d2, term = self.transitions:sample_d(self.minibatch_size)

	   local win=nil
	
			self.dwB:zero()
			local f=0
			for i=1,self.minibatch_size do
			   
				local input1=d[i]
				input1=input1:cuda()
			   local input2=d2[i]
				input2=input2:cuda()
    	      q_all_B = self.network_B:forward(input1)
				q1_max_d=q_all_B:float():max()
		 		q_d=q_all_B[a[i]] -- action index integer 
		 		
				--win=image.display({image=d[i],win=win})
				--compute error
			   local  out= target_q_net_B:forward(input2)
				q2_max_d=out:float():max()		 
				q_max_avg_d=q_max_avg_d+q2_max_d
				q1_max_avg_d=q1_max_avg_d+q1_max_d
				 -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    			q2_d = q2_max_d*self.discount   
    			delta_d = r[i] 
    			delta_d=delta_d+q2_d-q_d
				td_err_d=td_err_d+torch.abs(delta_d)

				if self.clip_delta then
				   if delta_d > self.clip_delta then delta_d=self.clip_delta end
					if delta_d < -self.clip_delta then delta_d=-self.clip_delta end
				  
    			end

    			local df_do = torch.zeros(self.n_actions):float()
	 			df_do[a[i]] = delta_d
				if self.gpu and self.gpu >= 0 then
            	df_do = df_do:cuda()					
				end
				self.network_B:backward(input1,df_do)
				
			 end

	 self.dwB:add(-self.wc, self.wB)

    -- compute linearly annealed learning rate

	  -- use gradients
    self.gB:mul(0.95):add(0.05, self.dwB)
    self.tmpB:cmul(self.dwB, self.dwB)
    self.gB2:mul(0.95):add(0.05, self.tmpB)
    self.tmpB:cmul(self.gB, self.gB)
    self.tmpB:mul(-1)
    self.tmpB:add(self.gB2)
    self.tmpB:add(0.01)
    self.tmpB:sqrt()

    -- accumulate update
    self.deltasB:mul(0):addcdiv(self.lr, self.dwB, self.tmpB)
    self.wB:add(self.deltasB)
	


	 collectgarbage()
	 end




td_err_d=td_err_d/self.bufferSize
td_err_s=td_err_s/self.bufferSize
q_max_avg_s=q_max_avg_s/self.bufferSize
q_max_avg_d=q_max_avg_d/self.bufferSize

q1_max_avg_s=q1_max_avg_s/self.bufferSize
q1_max_avg_d=q1_max_avg_d/self.bufferSize


return q_max_avg_s,q_max_avg_d,td_err_s,td_err_d,q1_max_avg_s,q1_max_avg_d	
end



