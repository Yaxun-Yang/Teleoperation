import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

import numpy as np

from sensor_msgs.msg import JointState
from leap_hand.srv import LeapPosition, LeapPosVelEff
import time

from rclpy.task import Future
import scripts.leap_hand_utils.leap_hand_utils as lhu

def get_end_point_sim_zero():
    pose = lhu.sim_ones_to_LEAPhand(np.zeros(16))
    return pose
def get_end_point_setting_value():
    pose = np.array([3.1538644, 3.6002529, 5.0513988, 3.7337093, 3.0664277, 3.7643888, 4.5850687, 4.092661, 2.9329712, 3.9177868, 4.144816, 4.2874765, 4.862719, 3.19068, 3.6063888, 4.5221753])
    return pose
def get_home_pose():
    home_pose = lhu.allegro_to_LEAPhand(np.zeros(16))
    home_pose[12] = 4.27
    return home_pose


class Robotiq(Node):

    def __init__(self, **kwargs):
        super().__init__('robotiq_action_client')
        # self.cli = self.create_client(LeapPosition, '/leap_position')
        self.cli = self.create_client(LeapPosVelEff, '/leap_pos_vel_eff')
        ##Note if you need to read multiple values this is faster than calling each service individually for the motors
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = LeapPosVelEff.Request()
        # self.pub_hand_one = self.create_publisher(JointState, '/cmd_ones', 10) 
        # self.pub_hand_allegro = self.create_publisher(JointState, '/cmd_allegro', 10)
        self.pub_hand_leap = self.create_publisher(JointState, '/cmd_leap', 10)
        self.sleeping_time = 0.03 # sleeping for every small increasement
        self.start_point = get_home_pose()
        self.end_point = get_end_point_setting_value()
        self.inc, self.inc_map, self.iteration = self.calculate_commands(self.start_point, self.end_point)
        self.x = []
        self.iteration_map = np.linspace(1, 0, int(self.iteration))
        self.previous_width = 1.0

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def calculate_commands(self, start_point, end_point):
        print("start_point(current point): ", start_point)

        print("desire goal point: ", end_point)
        diff = end_point - start_point


        abs_diff = np.abs(diff)

        iteration = np.ceil(np.max(abs_diff)/0.025)
        print("desire iteration:", iteration ,"desire time(s): ",  iteration * self.sleeping_time)
        inc=  diff / iteration
        print("inc:", inc)

        #build increasement map
        inc_map = np.zeros((16, int(iteration)))
        for i in range(16):
            inc_map[i] = np.linspace(start_point[i], end_point[i], int(iteration))
        inc_map = inc_map.T

        return inc, inc_map, iteration

    def execute_commands(self, start_point, end_point):
        print("start_point(current point): ", start_point)

        print("desire goal point: ", end_point)
        diff = end_point - start_point


        abs_diff = np.abs(diff)

        iteration = np.ceil(np.max(abs_diff)/0.025)
        print("desire iteration:", iteration ,"desire time(s): ",  iteration * self.sleeping_time)
        inc=  diff / iteration
        print("inc:", inc)

        vaild_id  =  np.full(16, False)

        vaild_id [:] = True
        # vaild_id [1] = True

        # do not move the side motors
        vaild_id[0] = False
        vaild_id[4] = False
        vaild_id[8] = False
        print("valid_ids: ", vaild_id)

        num = 0
        total_num = 0
        # 0 for break, 1 for start position to end position, 2 for end position to start position
        
        x = start_point.copy()
        while True:
            total_num = total_num + 1
            # response = self.send_request()
            # print(response)  ##Receive 
            time.sleep(self.sleeping_time)
            x_state = JointState()
            x_state.position = list(x)
            # print(x_state)
            self.pub_hand_leap.publish(x_state)
            if num < iteration:
                x[vaild_id] = x[vaild_id] + inc[vaild_id]
                num = num + 1
            else:
                break
           
        print("finished moving the gripper with iteration: ", total_num)
        


    def apply_commands_clean(self, width:float, speed:float=0.1, force:float=0.1):
        # consider triggering the process to start the gripper

        # To-Do : check if the gripper is in the home pose otherwise raise an error

        # print("the input width value", width)

        # # do different actions based on the width value, 0 for close, 1 for open
        # if width == 0:
        #     end_point = get_end_point_setting_value()
        # elif width == 1:
        #     end_point = get_home_pose()
        # else:
        #     print("The width value is not valid, please input 0 for close, 1 for open")
        #     return
        
        # start_point = np.array(self.send_request().position)

        # self.execute_commands(start_point, end_point)

        vaild_id  =  np.full(16, False)

        vaild_id [:] = True
        # vaild_id [1] = True

        # do not move the side motors
        vaild_id[0] = False
        vaild_id[4] = False
        vaild_id[8] = False
        # print("valid_ids: ", vaild_id)

        inc = None
        if  len(self.x) == 0:
            self.x = np.array(self.send_request().position)
        if np.abs(width)<0.2 :
            return 
        elif  np.abs(width + 1) < 0.8 and all(self.x[id] < self.end_point[id] or not vaild_id[id] for id in range(16)):
            inc = self.inc
            # print('incresing')
        elif np.abs(width -1) < 0.8 and all(self.x[id] > self.start_point[id] or not vaild_id[id] for id in range(16)):
            inc = -1 * self.inc
            # print('decresing')
        else:
            print("out of limit!!!!!!!!!")
            if np.abs(width+1)<0.8:
                for i in range(16):
                    if self.x[i] >= self.end_point[i] and vaild_id[i]:
                        print("The motor ", i, " is out of limit", self.x[i], " > ", self.end_point[i])
            elif np.abs(width -1) < 0.8:
                for i in range(16):
                    if self.x[i] <= self.start_point[i] and vaild_id[i]:
                        print("The motor ", i, " is out of limit", self.x[i], " < ", self.start_point[i])
            else:
                print("The width value ", width ," is not valid, please input 0 for close, 1 for open")
            return
        
        

        self.x[vaild_id] = self.x[vaild_id] + inc[vaild_id]

        x_state = JointState()
        x_state.position = list(self.x)

        # print(x_state)
        self.pub_hand_leap.publish(x_state)
        
    def apply_commands(self, width:float, speed:float=0.1, force:float=0.1):
        
        if np.abs(width - self.previous_width) < 0.01:
            return
        else :
            # print("The width value is changed from ", self.previous_width, " to ", width)
            self.previous_width = width
        if width < 0 or width > 1:
            print(f"The width value {width} is not valid, please input 0 for close, 1 for open")

        # 0 for close, 1 for open

        start_point = np.array(self.send_request().position)

        x = start_point.copy()

        distance = width - self.iteration_map
        target_index = np.argmin(np.abs(distance))


        end_point = self.inc_map[target_index]

        self.execute_commands(start_point, end_point)





    def get_sensors(self):
        result = self.send_request()
        current_state =  np.array([result.position])
         # Calculate distances between current_state and each state in inc_map
        distances = np.linalg.norm(self.inc_map - current_state, axis=1)
    
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)
        # start point as 1, end point as 0
        closest_index = self.iteration - closest_index

        return np.array([closest_index/ self.iteration])

    def connect(self):
        pass
        # self._action_client.wait_for_server()

    def close(self):
        self.reset()
        rclpy.shutdown()

    def okay(self):
        return True

    def reset(self, width=0.1, **kwargs):
        # being home pose from current pose, calculate the value carefully
        # self.apply_commands(width=0.1)
        print("Resetting gripper")
        start_point = np.array(self.send_request().position)

        x = start_point.copy()

        end_point = get_home_pose()

        diff=end_point - start_point
        print("start_point: ", start_point, "end_point: ", end_point)
        print("diff: ", diff)
        abs_diff = np.abs(diff)

        iteration = np.ceil(np.max(abs_diff)/0.025)
        print("desire iteration:", iteration ,"desire time(s): ",  iteration * self.sleeping_time)
        inc=  diff / iteration
        print("inc:", inc)
        

        

        vaild_id  =  np.full(16, False)

        vaild_id [:] = True
        # vaild_id [1] = True

        # do not move the side motors
        # vaild_id[0] = False
        # vaild_id[4] = False
        # vaild_id[8] = False
        # print("valid_ids: ", vaild_id)

        num = 0
        total_num = 0
        # 0 for break, 1 for start position to end position, 2 for end position to start position

        flag = 1
        while True:
            total_num = total_num + 1
            response = self.send_request()
            # print(response)  ##Receive 
            time.sleep(self.sleeping_time)
            x_state = JointState()
            x_state.position = list(x)
            # print(x_state)
            self.pub_hand_leap.publish(x_state)
            if flag == 1: 
                if num < iteration:
                    x[vaild_id] = x[vaild_id] + inc[vaild_id]
                    num = num + 1
                else:
                    flag = 0
            elif flag == 2:
                if num > 0:
                    x[vaild_id] = x[vaild_id] - inc[vaild_id]
                    num = num - 1
                else:
                    flag = 0
            else:
                break
        self.x = get_home_pose()
        print("finished resetting with iteration: ", total_num)
