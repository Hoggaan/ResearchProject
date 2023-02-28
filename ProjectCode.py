import rospy
import numpy as np
import subprocess
import os
import time
import math

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class MultiRobotEnv:
    def __init__(self,launchfile, num_robots=4, num_goals=4):
        self.num_robots = num_robots
        self.num_goals = num_goals
        self.robot_positions = np.zeros((num_robots, 3)) # Possitions[x,y,z]
        self.robot_orientations = np.zeros((num_robots, 3)) # Orientation[roll,pitch,yaw]
        self.robot_laser_data = np.zeros((num_robots, 360))
        self.goals = np.random.randint(0, 4, size=(3, 2))
        self.start_time = rospy.Time.now()
        self.last_linear_velocities = np.zeros((num_robots,))
        self.last_angular_velocities = np.zeros((num_robots,))

        self.o_t_e = np.zeros((3, num_goals, 2))
        self.o_t_o = np.zeros((len(self.num_robots) - 1, 3, len(self.goals), 2))
        self.laser_scanner = np.zeros((3, 360))

        self.last_distances = self.distances()
        self.available_goals = self.goals

        # # Laser scan parameters
        # self.num_laser_readings = 360
        # self.max_laser_range = 10.0  # meters
        # self.min_laser_range = 0.1   # meters
        # self.laser_threshold = 0.2   # meters
        

        # Launch the Gazebo simulation
        # Run ross master - roscore
        port = '11311'
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        #Launch the simulation with the given launchfile
        rospy.init_node('hoggaan', anonymous=True)
        if launchfile.startswith('/'):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Wait for the Gazebo service to be available
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Set the initial positions of the robots in the Gazebo simulation
        for i in range(self.num_robots):
            model_state = ModelState()
            model_state.model_name = "robot_{}".format(i)
            model_state.pose.position.x = self.robot_positions[i][0]
            model_state.pose.position.y = self.robot_positions[i][1]
            model_state.pose.position.z = self.robot_positions[i][2]
            model_state.pose.orientation.x = 0.0
            model_state.pose.orientation.y = 0.0
            model_state.pose.orientation.z = 0.0
            model_state.pose.orientation.w = 1.0
            self.set_model_state(model_state)
        
        # Subscribe to the laser and odometry topics for each robot
        self.laser_subs = []
        self.odom_subs = []
        for i in range(self.num_robots):
            laser_sub = rospy.Subscriber("/robot_{}/scan".format(i), LaserScan, self.laser_callback, i)
            odom_sub = rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.odom_callback, i)
            self.laser_subs.append(laser_sub)
            self.odom_subs.append(odom_sub)

    def laser_callback(self, msg, robot_index):
        """Callback method for processing laser scan data for a specific robot."""
        self.robot_laser_data[robot_index] = msg.ranges

    def odom_callback(self, msg, robot_index):
        """Callback method for processing odometry data for a specific robot."""
        self.robot_positions[robot_index][0] = msg.pose.pose.position.x
        self.robot_positions[robot_index][1] = msg.pose.pose.position.y
        self.robot_positions[robot_index][2] = msg.pose.pose.position.z
        self.last_linear_velocities[robot_index] = msg.twist.twist.linear.x
        self.last_angular_velocities[robot_index] = msg.twist.twist.angular.z

        # Extract the orientation quaternion from the message
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        #Convert the quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        self.robot_orientations[robot_index][0] = roll
        self.robot_orientations[robot_index][1] = pitch
        self.robot_orientations[robot_index][2] = yaw
    
    def step(self, actions):
        # Ensure that the correct number of actions have been provided.
        assert len(actions) == self.num_robots
        
        # Publish actions for each robot
        for i in range(self.num_robots):
            pub = rospy.Publisher("/robot_{}/cmd_vel".format(i), Twist, queue_size=10)
            twist = Twist()
            twist.linear.x = actions[i][0]
            twist.angular.z = actions[i][1]
            pub.publish(twist)

        # Wait for physics to unpause and then pause again
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed! ")

        time.sleep(0.1)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed! ")
        
        # Calculate observations, rewards, and dones for all robots
        observations = []
        rewards = []
        dones = []
        for i in range(self.num_robots):
            observation = self.calculate_observation(i)
            reward, done = self.calculate_reward(i)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
        
        return observations, rewards, dones, {}


    def reset(self):
        # Resets the state of the environment and returns 
        # an Initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed! ")
        
        # Set the initial positions of the robots in the Gazebo simulation
        for i in range(self.num_robots):
            self.model_states[i].pose.position.x = 0.0
            self.model_states[i].pose.position.y = 0.0
            self.model_states[i].pose.position.z = 0.0
            self.model_states[i].pose.orientation.x = 0.0
            self.model_states[i].pose.orientation.y = 0.0
            self.model_states[i].pose.orientation.z = 0.0
            self.model_states[i].pose.orientation.w = 1.0

        self.last_distances = self.distances()
        self.start_time = rospy.Time.now()
        new_observation = self.calculate_observations()
        return new_observation
    
    # A method that calculates all the distances between the robots and the goals.
    def distances(self):
        distances = np.zeros((self.num_robots, self.num_goals))
        for i in range(self.num_robots):
            for j in range(self.num_goals):
                distances[i][j] = np.linalg.norm(self.robot_positions[i,:2] - self.goals[j])
        return distances

    def calculate_reward(self, robot_index):
        """Compute the reward for the current timestep of the simulation."""
        reward = 0
        done = False
        
        # Check for collisions between the specified robot and other robots
        for i in range(self.num_robots):
            if i == robot_index:
                continue
            # If the distance between the two robots is less than a threshold, penalize
            if np.linalg.norm(self.robot_positions[robot_index] - self.robot_positions[i]) < 0.2:
                reward -= 100
            
        # Check for collisions between the specified robot and obstacles
        # For each laser range reading for the current robot
        for j, range_reading in enumerate(self.robot_laser_data[robot_index]):
            # If the range reading is less than a certain threshold, consider it a collision
            if range_reading < 0.3:
                # Add a negative reward for the collision
                reward -= 100
                    
        # Check if the specified robot has reached its own unique goal
        nearest_goal_idx = np.argmin([np.linalg.norm(goal - (self.robot_positions[robot_index][0], self.robot_positions[robot_index][1])) for goal in self.available_goals])
        nearest_goal = self.available_goals[nearest_goal_idx]
                        
        if self.goals[robot_index] == nearest_goal_idx:
            if np.linalg.norm(self.robot_positions[robot_index] - nearest_goal) < 0.1:
                reward += 200
                self.available_goals = np.delete(self.available_goals, nearest_goal_idx, axis=0)
        # Otherwise, give a small penalty or reward based on whether the robot is moving toward its goal
        else:
            dx = nearest_goal[0] - self.robot_positions[robot_index][0]
            dy = nearest_goal[1] - self.robot_positions[robot_index][1]
            distance_to_goal = np.sqrt(dx**2 + dy**2)
            angle_to_goal = np.arctan2(dy, dx) - self.robot_orientations[robot_index][2]
            if angle_to_goal < - np.pi:
                angle_to_goal += 2*np.pi
            elif angle_to_goal > np.pi:
                angle_to_goal -= 2*np.pi
            # Penalize if the robot is not moving toward its goal
            if distance_to_goal > self.last_distances[robot_index][self.goals[robot_index]]:
                # Robot is moving away from its goal, penalize
                reward -= 50
            else:
                # Robot is moving toward its goal, reward
                reward += 50
                
        # Check if the specified robot has reached the same goal as another robot
        for j in range(self.num_robots):
            if j == robot_index:
                continue
            for goal in range(self.num_goals):
                if np.linalg.norm(self.robot_positions[robot_index,:2] - self.robot_positions[j,:2]) < 0.2:
                    goal_pos = self.goals[goal]
                    robot_pos_i = self.robot_positions[robot_index,:2]
                    robot_pos_j = self.robot_positions[j,:2]
                    if np.linalg.norm(goal_pos - robot_pos_i) <= 0.1:
                        reward -= 100
                    elif np.linalg.norm(goal_pos - robot_pos_j) <= 0.1:
                        reward -= 100

        # Check if the maximum time has been reached for the episode
        if rospy.Time.now() - self.start_time > rospy.Duration(self.time_threshold):
            reward -= 100
            done = True
            
        return reward, done

    def calculate_observation(self, robot):

        robot_observation = []
        
        # calculate the relative positions of goals in the focal robot's polar coordinates
        self.o_t_e = np.roll(self.o_t_e, shift=1, axis=0)
        for i, goal in enumerate(self.num_goals):
            dx = goal[0] - self.robot_positions[robot][0]
            dy = goal[1] - self.robot_positions[robot][1]
            self.o_t_e[0, i, 0] = np.sqrt(dx**2 + dy**2)
            self.o_t_e[0, i, 1] = np.arctan2(dy, dx) - self.robot_orientations[robot][2]  # Correction Needed!!
        robot_observation.append(self.o_t_e)
        
        # calculate the relative positions of the goals in the other robots' polar coordinates
        self.o_t_o = np.roll(self.o_t_o, shift=1, axis=0)
        for j, other_robot in enumerate(self.num_robots):
            if robot == other_robot:
                continue
            for i, goal in enumerate(self.goals):
                dx = goal[0] - self.robot_positions[other_robot][0]
                dy = goal[1] - self.robot_positions[other_robot][1]
                self.o_t_o[j, 0, i, 0] = np.sqrt(dx**2 + dy**2)
                self.o_t_o[j, 0, i, 1] = np.arctan2(dy, dx) - self.robot_orientations[other_robot][2]   # Correction
                # Normalize the heading to be between -pi and pi
                self.o_t_o[j, 0, i, 1] = math.atan2(math.sin(self.o_t_o[j, 0, i, 1]), math.cos(self.o_t_o[j, 0, i, 1]))
        robot_observation.append(self.o_t_o)
        
        # calculate the 360 degree laser scanner data of the focal robot
        self.laser_scanner = np.roll(self.laser_scanner, shift=1, axis=0)
        self.laser_scanner[0] = self.robot_laser_data[robot]
        self.laser_scanner[0] *= self.max_laser_range
        self.laser_scanner[0] += np.array(self.robot_positions[robot][0], self.robot_positions[robot][1])

        # Perform obstacle detection using thresholding
        obstacle_mask = np.logical_or(self.robot_laser_data[robot] > self.max_laser_range, self.robot_laser_data[robot]  < self.min_laser_range)
        obstacle_mask = np.logical_or(obstacle_mask, self.robot_laser_data[robot]  < self.laser_threshold)
        self.laser_scanner[0] = np.minimum(self.laser_scanner[0], obstacle_mask)
        self.laser_scanner[0] = np.linalg.norm(self.laser_scanner[0], axis=0)
        robot_observation.append(self.laser_scanner)
        
        # calculate the time spent since the focal robot started moving and the previous action
        elapsed_time = rospy.Time.now() - self.start_time
        previous_action = np.array([self.last_linear_velocities, self.last_angular_velocities])
        robot_observation.append(np.concatenate((elapsed_time, previous_action)))
        
        return robot_observation