require 'torch'
require 'environment'
require 'image'
require 'NeuralQLearner'
require 'paths'

local t_steps=2000



local gpu=1

--- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    torch.manualSeed(torch.initialSeed())
    local firstRandInt = torch.random()
    if gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
    end


local win=nil
collectgarbage()
local episode=torch.load('files/episode.dat')
args={epi=episode, tsteps=t_steps}
local agent=NeuralQLearner(args)
collectgarbage()
print("hi2")
local win=nil


function main()

 local q1_max_s_ep=torch.load('files/q1_max_s_ep.dat')
 local q1_max_d_ep=torch.load('files/q1_max_d_ep.dat')
 
local q_max_s_ep=torch.load('files/q_max_s_ep.dat')
 local q_max_d_ep=torch.load('files/q_max_d_ep.dat')
 local td_s_ep=torch.load('files/td_err_s_ep.dat')
 local td_d_ep=torch.load('files/td_err_d_ep.dat')
 local target_net=4
 local q_s_replay={}
 local q_d_replay={}
 local q1_s_replay={}
 local q1_d_replay={}
 local t_s={}
 local t_d={}

 local q_max_avg_s=0
 local q_max_avg_d=0
 local q1_max_avg_s=0
 local q1_max_avg_d=0
 local td_err_s=0
 local td_err_d=0
 --- load data
agent:load_data()
 --- training 
 collectgarbage()
for j=1,50 do
 local q_s_replay={}
 local q_d_replay={}
 local q1_s_replay={}
 local q1_d_replay={}
 local t_s={}
 local t_d={}
 for i=1,10 do
	print("epoch="..i.."/10")
 	q_max_avg_s,q_max_avg_d,td_err_s,td_err_d,q1_max_avg_s,q1_max_avg_d=agent:train()
	
	table.insert(q_s_replay,q_max_avg_s)
   table.insert(q_d_replay,q_max_avg_d)
	
	table.insert(q1_s_replay,q1_max_avg_s)
   table.insert(q1_d_replay,q1_max_avg_d)
	
	table.insert(t_s,td_err_s)
	table.insert(t_d,td_err_d)
	collectgarbage()	
end

	agent.target_network_A=agent.network_A:clone()
	agent.target_network_B=agent.network_B:clone()
	table.insert(q1_max_s_ep,q1_s_replay)
 	table.insert(q1_max_d_ep,q1_d_replay)

 	table.insert(q_max_s_ep,q_s_replay)
 	table.insert(q_max_d_ep,q_d_replay)
 	table.insert(td_s_ep,t_s)
 	table.insert(td_d_ep,t_d)
 
end
	

 
 local modelA=agent.network_A:clone()
 local modelB=agent.network_B:clone()
 if episode%target_net==1 then
	agent.target_network_A=agent.network_A:clone()
	agent.target_network_B=agent.network_B:clone()
 end

 local tmodelA=agent.target_network_A:clone()
 local tmodelB=agent.target_network_B:clone()
 local model_dir='results/ep'..episode
 paths.mkdir(model_dir)
 local save_modelA_gpu=model_dir..'/modelA_gpu.net'
 local save_tmodelA_gpu=model_dir..'/tmodelA_gpu.net'
 local save_modelA_cpu=model_dir..'/modelA_cpu.net'
 local save_tmodelA_cpu=model_dir..'/tmodelA_cpu.net'

 local save_modelB_gpu=model_dir..'/modelB_gpu.net'
 local save_tmodelB_gpu=model_dir..'/tmodelB_gpu.net'
 local save_modelB_cpu=model_dir..'/modelB_cpu.net'
 local save_tmodelB_cpu=model_dir..'/tmodelB_cpu.net'

 torch.save(save_modelA_gpu,modelA)
 torch.save(save_tmodelA_gpu,tmodelA)
 torch.save(save_modelB_gpu,modelB)
 torch.save(save_tmodelB_gpu,tmodelB)

 modelA=modelA:float()
 tmodelA=tmodelA:float()
 modelB=modelB:float()
 tmodelB=tmodelB:float()
 
 torch.save(save_modelA_cpu,modelA)
 torch.save(save_tmodelA_cpu,tmodelA)
 torch.save(save_modelB_cpu,modelB)
 torch.save(save_tmodelB_cpu,tmodelB)

 
 torch.save('files/q1_max_s_ep.dat',q1_max_s_ep)
 torch.save('files/q1_max_d_ep.dat',q1_max_d_ep)
 torch.save('files/q_max_s_ep.dat',q_max_s_ep)
 torch.save('files/q_max_d_ep.dat',q_max_d_ep)
 torch.save('files/td_err_s_ep.dat',td_s_ep)
 torch.save('files/td_err_d_ep.dat',td_d_ep)
 collectgarbage()

end
collectgarbage()
main()
