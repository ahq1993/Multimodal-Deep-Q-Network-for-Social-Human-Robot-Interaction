# Choregraphe simplified export in Python.
robotIp = "192.168.1.109" # The IP and port address on which 'robot_actions.py' is listenting
port=9559

from naoqi import ALProxy
import time
import random 
import thread
import socket
from multiprocessing import Value,Queue



def touch_sensor(opt):
	host='192.168.1.217' # ip of rasberry
	port=12395
	s=socket.socket()
	s.connect((host,port))
	s.send(opt)	   
	l=s.recv(1024)
	s.close()
	print 'reward='+str(l)
	return str(l)

def wait(opt,step):
    
	motion=ALProxy("ALMotion",robotIp,port)
	names =['HeadYaw','HeadPitch']
	times=[[0.7],[0.7]]

	if opt==1:
		print 'I am in 1'
		motion.angleInterpolation(names,[0.0,-0.16],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)
		
	elif opt==2:
		print 'I am in 2'
		motion.angleInterpolation(names,[0.2,-0.1],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)
	elif opt==3:
		print 'I am in 3'
		motion.angleInterpolation(names,[0.2,-0.1],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)
	elif opt==4:
		print 'I am in 4'
		motion.angleInterpolation(names,[-0.4,-0.1],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)
	elif opt==5:
		print 'I am in 5'
		motion.angleInterpolation(names,[0.0,-0.26179],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)
	elif opt==6:
		print 'I am in 6'
		motion.angleInterpolation(names,[0.0,-0.26179],times,True)
		time.sleep(3)
		motion.setAngles(names,[0.0,-0.26179],0.2)

	return str(0)		
		

def hello(step):
	names = list()
	times = list()
	keys = list()

	names.append("LElbowRoll")
	times.append([1, 1.5, 2, 2.5])
	keys.append([-1.02102, -0.537561, -1.02102, -0.537561])

	names.append("LElbowYaw")
	times.append([1, 2.5])
	keys.append([-0.66497, -0.66497])

	names.append("LHand")
	times.append([2.5])
	keys.append([0.66])

	names.append("LShoulderPitch")
	times.append([1, 2.5])
	keys.append([-0.707571, -0.707571])

	names.append("LShoulderRoll")
	times.append([1, 2.5])
	keys.append([0.558505, 0.558505])

	names.append("LWristYaw")
	times.append([1, 2.5])
	keys.append([-0.0191986, -0.0191986])
	names2=["LElbowRoll","LElbowYaw","LHand","LShoulderPitch","LShoulderRoll","LWristYaw"]
	angles=[-0.479966,-0.561996,0.66,1.30202,0.195477, -0.637045]
	motion = ALProxy("ALMotion", robotIp, 9559)
	motion.setExternalCollisionProtectionEnabled("Arms", False)
	tts = ALProxy("ALTextToSpeech",robotIp, port)
	tts.setParameter("speed", 100)
	tts.setLanguage("English")
	motion.angleInterpolation(names, keys, times, True)
	tts.say("Hello")
	motion.setAngles(names2,angles,0.3)
	
	return str(0)
	    
    

def shake_hand(step):
	names = list()
	times = list()
	keys = list()
	r=0
	names.append("RHand")
	times.append([2])
	keys.append([0.98])



	names.append("RShoulderPitch")
	times.append([2])
	keys.append([-0.2058])
	
	
	names2=["RElbowRoll","RElbowYaw","RHand","RShoulderPitch","RShoulderRoll","RWristYaw"]
	angles=[0.479966,0.561996,0.66,1.30202,-0.195477, 0.637045]
	
	names3=["RHand"]
	angles2=[0.5]
	
	motion = ALProxy("ALMotion", robotIp, 9559)
	motion.setExternalCollisionProtectionEnabled("Arms", False)
	tts = ALProxy("ALTextToSpeech",robotIp, port)
	tts.setParameter("speed", 60)
	tts.setLanguage("English")
	motion.setExternalCollisionProtectionEnabled("Arms", False)
	motion.angleInterpolation(names, keys, times, True)
	r=touch_sensor(str(1))
	print r
	if int(r)>4:
		tts.say("Nice to meet you")
		thread.start_new_thread(touch_sensor,(str(2),))
		motion.setAngles(names3,angles2,0.4)
		time.sleep(2)
	motion.setAngles(names2,angles,0.4)
	return r




def perform_actions(opt,data2,step): 
	data=str(data2)

	if data=='-':
		print "I am doing nothing"
		return str(0)
	
	if data=='1': #wait
		r=wait(opt,step)
		return r
	elif data=='2': # look towards human 
		time.sleep(3)		
		return str(0)
	elif data=='3': #hello
		r=hello(step)
		return r
	elif data=='4': #shake
		r=shake_hand(step)
		return r

	

def main(data2,step):

	list_wait=[1,2,3,4,5]
	random.shuffle(list_wait)
	opt=random.sample(list_wait,1)  
	opt1=opt[0]
	data=str(data2)
	r=perform_actions(opt1,data,step)
	return r



