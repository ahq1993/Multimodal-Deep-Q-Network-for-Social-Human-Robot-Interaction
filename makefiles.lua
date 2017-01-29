local e={}
episode=1
torch.save('files/reward_history.dat',e)
torch.save('files/action_history.dat',e)
torch.save('files/ep_rewards.dat',e)  
torch.save('files/episode.dat',episode)
torch.save('files/episode.txt',episode,'ascii')
