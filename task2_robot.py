#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: kaiv
"""

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
import copy
import math
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty

import random
import sys
import numpy as np
import abc
from DQN_tf import DeepQLearning
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, BoundedArraySpec, tensor_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
import math
import random
import matplotlib
import matplotlib.pyplot as plt


class gazebo_env(py_environment.PyEnvironment):

    #
    actionValues = [
        (0.15, 0),
        (0.15, math.pi / 5),
        (0.15, -math.pi / 5),
        (0.15, math.pi / 4),
        (0.15, -math.pi / 4),
        (0.7, 0),
    ]

    regions = [
        {"label": "left", "distances": [0, 1], "max": 2},
        {"label": "forward", "distances": [0, 0.3, 0.75, 1], "max": 4},
        {"label": "forward-right", "distances": [0, 1], "max": 2},
        {"label": "right", "distances": [0, 0.2, 0.5, 0.75, 1], "max": 5},
    ]

    # Constructor
    def __init__(self):

        """Global variables"""
        rospy.init_node("follow")
        self.left = 0
        self.right = 0
        self.front = 0
        self.fright = 0
        # self.robotState = np.zeros((4,), dtype=np.int32)  # Initial robot state
        self.trap = 0
        self.step_count = 0
        self.pose_candidates = []

        self.forward_convg_values = []
        self.left_convg_values = []
        self.right_convg_values = []
        self.step_numbers = []

        self.laser_subscriber = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback
        )
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        # self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        # self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.rate = rospy.Rate(5)

        self._action_spec = BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name="action_spec"
        )

        self._observation_spec = BoundedArraySpec(
            shape=(4,), dtype=np.int32, minimum=0, name="observation_spec"
        )

        self._state = np.zeros((4,), dtype=np.int32)
        self._episode_ended = False
        self._current_time_step = self.reset()
        self.action_counter = 0
        print("created py environment")
        self.same = False

        self.action_list = set()

    def same_action(self, action):

        act = action.tolist()

        # As long as its not forward
        if not act == 0:

            self.action_list.add(act)

            # The actions are different
            if len(self.action_list) > 1:
                self.action_list = set()
                self.action_counter = 0

            # They are the same

            else:
                self.action_counter = self.action_counter + 1

            if self.action_counter > 200:
                self.action_counter = 0
                return True

        return False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        # self._state = self.return_state()
        return self._observation_spec

    def _reset(self):
        """Resets the gazebo simulation and gets the initial state of the robot"""
        self._episode_ended = False
        # self.reset_robot()
        # self._state = self.robotState
        return ts.restart(self._state)

    def _current_time_step(self):
        return self._current_time_step

    def _step(self, action):
        self.step_count = self.step_count + 1
        """Reset to initial state if episode ended"""
        if self._episode_ended == True:
            return self._reset()

        # if self.same_action(action) == True:
        #     print("SAME ACTION: ", action)

        #     self._episode_ended = True

        # prevPose = self.getPose()

        """Move and get new state"""
        self.move(action)
        rospy.sleep(0.1)
        # self._state = self.robotState
        newState = self._state

        """Pause physics"""

        # newPose = self.getPose()
        reward = self.getReward(newState)

        """If the robot is trapped end the episode and reset"""
        # if self.isTrapped(prevPose, newPose):
        #     self._episode_ended = True
        # self.reset_robot()

        """ If end of episode return termination else return transition """
        if self._episode_ended == True:
            return ts.termination(newState, reward)
        else:
            return ts.transition(newState, reward, discount=1.0)

    """ Callback for the Lidar """

    def laser_callback(self, msg):

        """Make copies of the data"""
        self.data = copy.deepcopy(msg)
        self.header = msg.header
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.time_increment = msg.time_increment
        self.range_min = msg.range_min
        self.range_max = msg.range_max
        self.ranges = self.check_ranges(msg.ranges,self.range_min)

        """Get the laser scan values at each region """
        # self.ranges_sliced =  np.replace()
        self.right = min(min(self.ranges[255:285]), 5)
        # self.ranges_ = self.ranges[90:] + self.ranges[:90]
        # self.right = min(min(self.ranges[240:300]), 5)
        self.fright = min(min(self.ranges[300:330]), 5)
        self.front = min(min(self.ranges[0:15] + self.ranges[345:359]), 5)
        # self.front = min(min(self.ranges[0:30] + self.ranges[330:]), 5)
        self.left = min(min(self.ranges[75:105]), 5)
        # self.left = min(min(self.ranges[30:90]), 5)

        """ Determine the robots state and update everytime the callback fires """
        self._state = self.getState()
        rospy.Rate(5)

    """ Getter for the global varibale for robot state """

    # def return_state(self):
    #     return self.robotState
    
    def check_ranges(self, ranges, min):

        ranges = list(ranges)
        for i in range(0, len(ranges)):

            if ranges[i] < min or ranges[i] == 0:
                ranges[i] = math.inf


        return tuple(ranges)

    def goForwardFast(self):

        vel = Twist()
        vel.linear.x = self.actionValues[0][0]
        self.pub_vel.publish(vel)

    # Currently not in use , will experiment with the values in Part 2
    def goForwardSlow(self):

        vel = Twist()
        vel.linear.x = self.actionValues[0][0]
        self.pub_vel.publish(vel)

    def goLeft(self):

        vel = Twist()
        vel.linear.x = self.actionValues[1][0]
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.z = self.actionValues[1][1]
        self.pub_vel.publish(vel)

    def goRight(self):

        vel = Twist()
        vel.linear.x = self.actionValues[2][0]
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.z = self.actionValues[2][1]
        self.pub_vel.publish(vel)

    def goSharpLeft(self):

        vel = Twist()
        vel.linear.x = self.actionValues[3][0]
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.z = self.actionValues[3][1]
        self.pub_vel.publish(vel)

    def goSharpRight(self):

        vel = Twist()
        vel.linear.x = self.actionValues[4][0]
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.z = self.actionValues[4][1]
        self.pub_vel.publish(vel)

    def escape(self):

        vel = Twist()
        vel.linear.x = self.actionValues[5][0]
        self.pub_vel.publish(vel)

    """Gets the robots x,y position """

    # def getPose(self):

    #     rospy.wait_for_service("/gazebo/get_model_state")
    #     try:
    #         msg = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    #         output = msg("turtlebot3_burger", "map")
    #         x = round(output.pose.position.x, 4)
    #         y = round(output.pose.position.y, 4)
    #         return (x, y)
    #     except (rospy.ServiceException) as e:
    #         rospy.logerr("'/gazebo/get_model_state' call failed")

    """ Teleport the robot to a given pose """

    # def teleport(self):

    #     index = random.randint(0, 4)
    #     state_msg = ModelState()
    #     state_msg.model_name = "turtlebot3_burger"
    #     self.pose_candidates = self.getCandidates()
    #     pose = self.pose_candidates[index]
    #     state_msg.pose.position.x = pose.position.x
    #     state_msg.pose.position.y = pose.position.y
    #     state_msg.pose.position.z = pose.orientation.z
    #     state_msg.pose.orientation.x = pose.orientation.x
    #     state_msg.pose.orientation.y = pose.orientation.y
    #     state_msg.pose.orientation.z = pose.orientation.z
    #     state_msg.pose.orientation.w = pose.orientation.w
    #     rospy.wait_for_service("/gazebo/set_model_state")
    #     set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    #     resp = set_state(state_msg)

    def teleport(self):

        state_msg = ModelState()
        state_msg.model_name = "turtlebot3_burger"
        state_msg.pose.position.x = -2
        state_msg.pose.position.y = -2
        state_msg.pose.position.z = 0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1
        rospy.wait_for_service("/gazebo/set_model_state")
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        resp = set_state(state_msg)

    """ Resets the Gazebo simulation """

    def resetSimulation(self):
        reset_simulation = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        reset_simulation()

    """ Resets the simulation and moves robot to start position """

    # def reset_robot(self):
    #     self.resetSimulation
    #     self.teleport()

    # def consecutiveActionCheck(self):

    def isTrapped(self, prevPose, newPose):
        """Checks if the robot is trapped by checking the eucledian distance"""
        if self.eucDist(prevPose, newPose) < 0.005:
            self.trap = self.trap + 1

            """ If its in the same spot for 10 checks , its trapped """
            if self.trap > 10:
                self.trap = 0
                return True

            else:
                return False

    def eucDist(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
        )

    """ Help determine the current state by looking at the Laser scan data """

    def getState(self):

        # Initialise empty state
        state = []
        for i in range(0, 4):
            state.append(0)

        # Left
        if self.left >= (self.regions[0]["distances"][0]) and self.left < (
            self.regions[0]["distances"][1]
        ):
            state[0] = 0
        else:
            state[0] = 1

        # Forward
        if self.front >= (self.regions[1]["distances"][0]) and self.front < (
            self.regions[1]["distances"][1]
        ):
            state[1] = 0
        elif self.front >= (self.regions[1]["distances"][1]) and self.front < (
            self.regions[1]["distances"][2]
        ):
            state[1] = 1
        elif self.front >= (self.regions[1]["distances"][2]) and self.front < (
            self.regions[1]["distances"][3]
        ):
            state[1] = 2
        else:
            state[1] = 3

        # Forward-Right
        if self.fright >= (self.regions[2]["distances"][0]) and self.fright < (
            self.regions[2]["distances"][1]
        ):
            state[2] = 0
        else:
            state[2] = 1

        # Right
        if self.right >= (self.regions[3]["distances"][0]) and self.right < (
            self.regions[3]["distances"][1]
        ):
            state[3] = 0
        elif self.right >= (self.regions[3]["distances"][1]) and self.right < (
            self.regions[3]["distances"][2]
        ):
            state[3] = 1
        elif self.right >= (self.regions[3]["distances"][2]) and self.right < (
            self.regions[3]["distances"][3]
        ):
            state[3] = 2
        elif self.right >= (self.regions[3]["distances"][3]) and self.right < (
            self.regions[3]["distances"][4]
        ):
            state[3] = 3
        else:
            state[3] = 4

        state_array = np.array(state, dtype=np.int32)
        return state_array

    """ Executes based on the current action """

    # def find_convg(self, step, state, action):

    #     # self.forward_convg_values = []
    #     # self.left_convg_values = []
    #     # self.right_convg_values = []
    #     # self.step_numbers  = []

    #     forward_good_count = 0.0
    #     left_good_count = 0.0
    #     right_good_count = 0.0

    #     # Convergence Check

    #     # Left
    #     if state[0] < 0.35 and (0.35 < state[2] < 0.75):
    #         total_count_left = total_count_left + 1.0
    #         if action == 2 or action == 4:
    #             left_good_count = left_good_count + 1.0

    #     # Right
    #     if state[0] > 0.3 and state[1] > 1:
    #         total_count_right = total_count_right + 1.0
    #         if action == 1 or action == 3:
    #             right_good_count = right_good_count + 1.0

    #     # Forward
    #     if (state[0] > 0.2 and state[1] > 0.1) or (
    #         state[0] > 0.2 and (0.35 < state[2] < 0.75)
    #     ):
    #         total_count_forward = total_count_forward + 1.0
    #         if action == 0:
    #             forward_good_count = forward_good_count + 1.0

    #     if step % 100 == 0:
    #         forward_convg = forward_good_count / total_count_forward
    #         self.forward_convg_values.append(forward_convg)

    #         left_convg = left_good_count / total_count_left
    #         self.left_convg_values.append(left_convg)

    #         right_convg = right_good_count / total_count_right
    #         self.right_convg_values.append(right_convg)

    #         self.step_numbers.append(step)

    # def get_convergence_plots(self):

    #     x = self.step_numbers
    #     y = self.left_convg_values

    #     plt.plot(x, y)
    #     plt.title("Left convg")
    #     plt.xlabel("episodes")
    #     plt.ylabel("convergence values")
    #     plt.savefig("Left.pdf")

    #     x = self.step_numbers
    #     y = self.right_convg_values

    #     plt.plot(x, y)
    #     plt.title("Right convg")
    #     plt.xlabel("episodes")
    #     plt.ylabel("convergence values")
    #     plt.savefig("Right.pdf")

    #     x = self.step_numbers
    #     y = self.forward_convg_values

    #     plt.plot(x, y)
    #     plt.title("Forward convg")
    #     plt.xlabel("episodes")
    #     plt.ylabel("convergence values")
    #     plt.savefig("Forward.pdf")

    def move(self, a):

        if a == 0:
            self.goForwardFast()

        # if a == 1:
        #     self.goForwardSlow()

        if a == 1:
            self.goLeft()

        if a == 2:
            self.goRight()

        if a == 3:
            self.goSharpLeft()

        if a == 4:
            self.goSharpRight()

        if a == 5:
            self.escape()

    """ Determines the reward for a given state """

    def getReward(self, state):

        reward = 0

        L = state[0]
        F = state[1]
        FR = state[2]
        R = state[3]

        tooClose = 0
        close = 1
        med = 2
        far = 3
        tooFar = 4
        # Negative reward
        # if R == tooFar:
        #     reward = -1.5

        if R == tooClose or R == tooFar or F == tooClose or L == tooClose:
            reward = -1.0
        # Positive reward for staying near the wall
        if R == close or R == med:
            reward = 0.5

        return float(reward)


if __name__ == "__main__":

    try:

        gz = gazebo_env()
        dq = DeepQLearning(gz)
        dq.Learn()

    except rospy.ROSInterruptException:
        pass

# rospy.spin()
