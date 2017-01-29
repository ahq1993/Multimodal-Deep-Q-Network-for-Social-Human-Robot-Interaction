require 'torch'
require 'image'

r_len=8 --recording time in sec
raw_frame_height= 320   -- 640 --- height and width of captured frame  -- 320
raw_frame_width=  240    -- 480 --- height and width of captured frame   -- 240
proc_frame_size=198 --
state_size=8
frame_per_sec=1
step=1

function pre_process(step)
	
	local images=torch.Tensor(r_len,1,raw_frame_width,raw_frame_height)
	local depths=torch.Tensor(r_len,1,raw_frame_width,raw_frame_height)
	episode=torch.load('files/episode.dat')
	dirname_rgb='dataset/RGB/ep'..episode
	dirname_dep='dataset/Depth/ep'..episode
	
	for i=1,r_len do
			local filename=dirname_rgb..'/image_'..step..'_'..i..'.png'
			local filename2=dirname_dep..'/depth_'..step..'_'..i..'.png'
    		images[i] =image.load(filename)
			depths[i] =image.load(filename2)
			
	end
     
	local proc_image=torch.Tensor(state_size,proc_frame_size,proc_frame_size)
	local proc_depth=torch.Tensor(state_size,proc_frame_size,proc_frame_size)
	for i=1, state_size do
		local x =images[i]
		local d=depths[i]
		local y=image.scale(x,proc_frame_size,proc_frame_size,'bilinear')
		local d2=image.scale(d,proc_frame_size,proc_frame_size,'bilinear')
		proc_image[i]=y[1]
		proc_depth[i]=d2[1]
	end
	
	return proc_image,proc_depth	

end 

function send_data_to_pepper(data)
	local socket = require("socket")
	local port = 12375         --- The port address on which 'robot_listen.py' is listenting
	local host='192.168.1.217' --- The IP address on which 'robot_listen.py' is listenting
	local client =socket.connect(host, port)
	client:send(data)
	flag=true
	while flag do
		local data2, emsg, partial=client:receive()
		print (data2)
		if data2 then
			client:close()
			return tonumber(data2)
		end
		break
	end  
	print("Connected with the server")
	client:close()
	return 0

end 

function perform_action(action)
   
	---out to pepper and get new state,reward, terminal
	--PERFORM action
	
	local r,term
	r=send_data_to_pepper(action)
	local s,d=pre_process(step)
	step=step+1
	term=false
   return s,d,r,term
end 

