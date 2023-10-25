import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from digit_interface import Digit
from digit_interface.digit_handler import DigitHandler as DH
import cv2
from cv_bridge import CvBridge
import sys
import cv2
import numpy as np
from time import time



def talker(vidstream, loop = False):
    try:
        #	d.show_view()
        #	d.disconnect()
        pub = rospy.Publisher('DIGIT_findings_PRERECORD', Image, queue_size=10)
        rospy.init_node('tactitian', anonymous=True)
        rate = rospy.Rate(10) # 30hz
        cap = cv2.VideoCapture(vidstream)
        bridge = CvBridge()
        #		
        # delay = int(1000/cap.get(cv2.CAP_PROP_FPS))
        delay = 30
        ret = True	
        while not rospy.is_shutdown() and ret:
            end, frame = cap.read()
            if loop and not end:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                end, frame = cap.read()
            elif not end:
                ret = False
                continue
            cv2.imshow(f"Current frame ", frame)
            keypress = cv2.waitKey(delay)
            if keypress==27:
                ret = False
                continue

            image = bridge.cv2_to_imgmsg(frame,encoding="bgr8")
            image.header.frame_id = "video"
            pub.publish(image)

    finally:
        if isinstance(cap, cv2.VideoCapture):
            cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "loop":

        talker("/home/kristopher/Documents/AI_learning_work/digit_videos/edge/edge_1689050799.7146409.avi"
               if sys.argv[1] == "" else sys.argv[1], True)
    else:
        talker("/home/kristopher/Documents/AI_learning_work/digit_videos/edge/edge_1689050799.7146409.avi"
               if sys.argv[1] == "" else sys.argv[1],)
    