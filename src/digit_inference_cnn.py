import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler as DH
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import roslib.packages as roslib

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 160 * 120, num_classes)  # Adjust the input size as needed
        self.flatten = nn.Flatten()
        # print("Model initialized")

    def forward(self, x):
        # x = self.flatten(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


# class SimplerCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SimplerCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(32 * 160 * 120, num_classes)  # Adjust the input size as needed
#         self.flatten = nn.Flatten()
#         # print("Model initialized")

#     def forward(self, x):
#         # x = self.flatten(x)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         return x



def preprocess_image(frame):
    transform = transforms.ToTensor()

    return transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).view((1,1,320,240))



#The publication to the DIGIT_contact_status topic, should publish contact or no contact depending on what the digit sees
def initialize_talker():
    pub = rospy.Publisher("DIGIT_contact_status", String, queue_size=10)

    return pub


def callback(data, args):
    bridge = CvBridge()
    model = args[0]
    pub = args[1]
    frame = bridge.imgmsg_to_cv2(data)
    predicted = predictor(frame,model)
    if predicted == 0:
        message = "No Contact"
        rospy.loginfo(rospy.get_caller_id() + '\tNo Contact')
    else:
        message = "Contact"
        rospy.loginfo(rospy.get_caller_id() + '\tContact')
    pub.publish(message)



def predictor(frame, model):
    with torch.no_grad():  
        
        test_image = preprocess_image(frame)
        output = model(test_image)
        _, predicted = torch.max(output, 1)
        return predicted.item()


def listener():

    rospy.init_node('digit_inferer',anonymous=True)

    pub = initialize_talker()


    # model = torch.load("cnnsimpletestGENERAL2.pth")
    # model = torch.load("cnnsimpletestD20549only2.pth")

    # print("made it here first")

    # model = torch.load("/home/kristopher/catkin_ws/src/ros_digit/scripts/cnnsimpletestD20548only2.pth")

    path = roslib.get_pkg_dir('ros_digit') + '/src/'
    
    model = SimpleCNN()
    model.load_state_dict(state_dict=torch.load(path + "cnnsimpletestGENERAL3.pth"))

    # digit_inference_cnn.py
    # model = torch.load("/path/to/your/model.pth", map_location=torch.device('cpu'), custom_objects={'SimpleCNN': SimpleCNN})

    model.eval()



    rospy.Subscriber('DIGIT_findings_D20549',Image,callback,(model,pub))
    # rospy.Subscriber('DIGIT_findings_PRERECORD',Image,callback)
    rospy.spin()



                
if __name__ == '__main__':
    # try:
    #     rospy.get_master()
    #     listener()
    # except:
    #     import rospkg

    #     # get an instance of RosPack with the default search paths
    #     rospack = rospkg.RosPack()

    #     # get the file path for rospy_tutorials
    #     print(rospack.get_path('ros_digit'))

    rospy.get_master()
    listener()