require 'torch'
require 'image'
require 'cunn'
r_len=8 --recording time in sec
raw_frame_height=320 --- height and width of captured frame
raw_frame_width=240 --- height and width of captured frame
proc_frame_size=198 --
state_size=8
frame_per_sec=1
step=1

function get_data(episode,tsteps)
	local images=torch.Tensor(tsteps,state_size,proc_frame_size,proc_frame_size)
	local depths=torch.Tensor(tsteps,state_size,proc_frame_size,proc_frame_size)	
		dirname_rgb='dataset/RGB/ep'..episode
		dirname_dep='dataset/Depth/ep'..episode
		
		for step=1,tsteps do

			
			local im=torch.Tensor(r_len,1,raw_frame_width,raw_frame_height)
			local dep=torch.Tensor(r_len,1,raw_frame_width,raw_frame_height)	
			for i=1,r_len do
					local filename=dirname_rgb..'/image_'..step..'_'..i..'.png'
					local filename2=dirname_dep..'/depth_'..step..'_'..i..'.png'
			 		im[i] =image.load(filename)
					dep[i] =image.load(filename2)
			end
			  
			local proc_im=torch.Tensor(state_size,proc_frame_size,proc_frame_size)
			local proc_dep=torch.Tensor(state_size,proc_frame_size,proc_frame_size)
			for i=1, state_size do
				local d=dep[i]
				local y=image.scale(im[i],proc_frame_size,proc_frame_size,'bilinear')
				local d2=image.scale(d,proc_frame_size,proc_frame_size,'bilinear')
				proc_im[i]=y[1]
				proc_dep[i]=d2[1]
				
			end
			
			images[step]=proc_im
			depths[step]=proc_dep
			if step==50 then
				image.display(proc_im)
				image.display(proc_dep)
			end
				collectgarbage()
   			
		end

	
	return images,depths	

end 


