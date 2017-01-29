require 'nn'
require 'torch'
require 'environment'
require 'image'
require 'NeuralQLearner'
require 'os'
local t_steps=1000

local gpu=-1

torch.manualSeed(torch.initialSeed())  
local win=nil

local episode=torch.load('files/episode.dat')
args={epi=episode}
local agent=NeuralQLearner(args)
local win=nil
local dirname_rgb='dataset/RGB/ep'..episode
local dirname_dep='dataset/Depth/ep'..episode
local dirname_model='results/ep'..episode
paths.mkdir(dirname_rgb)
paths.mkdir(dirname_dep)
paths.mkdir(dirname_model)
local screen, depth, reward, terminal = perform_action('-')


function generate_data(episode)
	--make new directory for new episode	
	local aset = {'1','2','3','4'}
	local actions={}
	local rewards={}
	local total_reward=0
	local reward_history=torch.load('files/reward_history.dat')
	local action_history=torch.load('files/action_history.dat')
	local ep_rewards=torch.load('files/ep_rewards.dat')
	
	
	local step=1
			while step <t_steps do
				print("Step="..step)
				local action_index=0
				numSteps=0*t_steps+step
				if reward>15 then
					action_index = agent:perceive(1, screen,depth, terminal, false, numSteps,step,0.1)
				else
					action_index = agent:perceive(0, screen,depth, terminal, false, numSteps,step,0.1)
            end
				step=step+1		
				if action_index == nil then
						action_index=2
				end
				if not terminal then 
					screen,depth,reward,terminal=perform_action(aset[action_index])
				else  
					screen,depth, reward, terminal = perform_action('-')
				end

				if step >= t_steps then
					terminal=1
				end
				table.insert(rewards,reward)
				table.insert(actions,action_index)
				total_reward=total_reward+reward
				torch.save('recent_rewards.dat',rewards)
				torch.save('recent_actions.dat',actions)
				
			end
	table.insert(reward_history,rewards)
	table.insert(action_history,actions)
	table.insert(ep_rewards,total_reward)
	
	 
	torch.save('files/ep_rewards.dat',ep_rewards)
	torch.save('files/reward_history.dat',reward_history)
	torch.save('files/action_history.dat',action_history)

end 

function main()

 generate_data(episode)
 --- training 
 episode=episode+1
 print (episode)
 torch.save('files/episode.txt',episode,'ascii')
 torch.save('files/episode.dat',episode)
 

end
main()
