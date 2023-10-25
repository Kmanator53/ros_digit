#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler as DH
import cv2
from cv_bridge import CvBridge
import sys


def talker(digits):
#	d.show_view()
#	d.disconnect()
	pub = [rospy.Publisher(f'DIGIT_findings_{d.serial}', Image, queue_size=10) for d in digits]
	rospy.init_node('tactitian', anonymous=True)
	rate = rospy.Rate(10) # 30hz
	
	bridge = CvBridge()
#		
	true = True	
	while not rospy.is_shutdown() and true:
		for i,d in enumerate(digits):
			frame = d.get_frame()
			cv2.imshow(f"Current frame {d.serial}", frame)
			keypress = cv2.waitKey(50)
			if keypress==27:
				true = False
				break
			
			image = bridge.cv2_to_imgmsg(frame,encoding="rgb8")
			image.header.frame_id = d.serial
			pub[i].publish(image)
		
#		rate.sleep()
		
	d.disconnect()


def collect_digits():
	#Searches for Digits plugged into the machine, returns a list of instances of the Digit class object connected to the devices
	#Gathering the serial numbers of the connected Digit devices
	digits = DH.list_digits()
	serials = [digit['serial'] for digit in digits]
	serials = list(dict.fromkeys(serials))

	#creating an empty list for storing the instances of the Digit handlers
	d = []
	#instantiating and connecting to each of the detected digits
	for i,serial in enumerate(serials):
		d.append(Digit(serial))
		d[i].connect()
		d[i].set_intensity(10)
	return d


if __name__ == '__main__':
	
	digits = collect_digits()
	
	#To prevent errors, exit if no digits are connected at this stage
	if len(digits)==0:
		print("no digits found, exiting")
		sys.exit()
	

	try:
		talker(digits)
       
	except rospy.ROSInterruptException:
		pass
