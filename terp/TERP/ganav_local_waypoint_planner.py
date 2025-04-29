#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
ros_path = '/opt/ros/kinetic/lib/python3.7/dist-packages'
import sys
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python3.7/dist-packages')
import numpy as np
import time
import os
import math
import threading

from math import cos, sin, radians, sqrt
from itertools import combinations

# ROS 메시지 관련
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose2D, Twist, PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu
from realsense2_camera.msg import IMUInfo

# OpenCV, CvBridge
from cv_bridge import CvBridge, CvBridgeError

# GANav 관련
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import mmcv

# PyTorch 관련
import torch as T
import torch.nn as nn

# 기타 유틸
from shutil import copyfile
import matplotlib.pyplot as plt
from skimage.graph import MCP
from scipy.stats import mode

# DDPG 관련
from ddpg_torch import Agent
from environment import Env
from utils import plot_learning_curve
from sklearn.preprocessing import normalize

# 전역 변수
vel_cmd = Twist()


class TerrainSeg():
    def __init__(self):
        # 노드 초기화 (단, 여기서는 spin을 호출하지 않음)
        rospy.init_node('TerrainSeg', anonymous=True)

        self.bridge = CvBridge()
        self.x_d = 0.0
        self.theta_d = 0.0

        # segmentation config
        self.palette = "rugd_group"
        self.config = "/root/catkin_ws/src/project/GANav-offroad/trained_models/rugd_group6/ganav_rugd_6.py"
        self.checkpoint = "/root/catkin_ws/src/project/GANav-offroad/trained_models/rugd_group6/ganav_rugd.pth"
        self.device = "cuda:0"

        self.pal = get_palette(self.palette)
        self.model = init_segmentor(self.config, self.checkpoint, device=self.device)

        # Subscriber 설정
        self.depth_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)
        self.vel_sub = rospy.Subscriber('/jackal/cmd_vel_DWA', Twist, self.getVel)

        # Publisher 설정
        self.pub_goal = rospy.Publisher('/target/position', Twist, queue_size=10)
        self.goal_state_pub = rospy.Publisher('/env_state_pub', Twist, queue_size=10)
        
        self.grids = np.zeros((4,))

        rospy.loginfo("TerrainSeg Node Initialized!")

    def run_ddpg_loop(self):
        """
        DDPG + TERP(GANav) 연동 로직을 여기에 작성하여,
        spin()과 병렬로 동작하도록 한다.
        """

        agent = Agent(alpha=0.0001, beta=0.001, tau=0.01, batch_size=8, n_actions=2)
        n_games = 10

        rospy.loginfo("CUDA Available: %s", str(T.cuda.is_available()))

        filename = 'Husky_ddpg_test2' + str(agent.alpha) + '_beta_' + \
                    str(agent.beta) + '_' + str(n_games) + '_games'
        figure_file = 'plots/' + filename + '.png'

        env = Env(False)
        agent.load_models()
        score_history = []

        rospy.loginfo("DDPG + Env 초기화 완료")

        goal_in = [3, 0]
        without_attention = False
        observation = env.reset(goal_in)

        score_history = []
        t_stamp=0
        action =[0,0]
        i=0
        arrive =False
        waypoint_arrived =False
        grid_arrived =False
        score = 0
        Tot_elevation_diff =0
        Tot_goal_heading_cost =0
        Total_dist_travelled = 0
        Tot_pose_cost=0

        action_linear = []
        action_angular = []

        grid_no =1
        current_x0,current_y0,current_z0, relative_theta_deg0,cost_pose0 = env.TGC_calculator()

        rate = rospy.Rate(10)  # 주기 설정(10Hz 등)
        rospy.loginfo("DDPG main loop 시작")

        while not rospy.is_shutdown() and not arrive:
            # DDPG 액션 선택
            action_ddpg, cbam_out, cbam_in = agent.choose_action(observation)

            # 현재 velocity는 콜백에서 받아온 vel_cmd로 설정
            action[0] = vel_cmd.linear.x
            action[1] = vel_cmd.angular.z
            action_linear.append(action[0])
            action_angular.append(action[1])

            if t_stamp == 0 or grid_arrived:
                time.sleep(0.5)

                grid_waypoint_coords_wrt_odom = env.waypoint_planner(cbam_out, observation[0], without_attention, self.grids)

                rospy.loginfo("================================ WAY POINT PLANNER ================================")
                rospy.loginfo(grid_waypoint_coords_wrt_odom)
                
                grid_no_end = np.shape(grid_waypoint_coords_wrt_odom)[0]
                rospy.loginfo("grid length: %d", grid_no_end)
                rospy.loginfo("-------Start navigating to a new waypoint-----")
                grid_no = 1
                rospy.loginfo("grid goal in to r theta cal: %s", str(grid_waypoint_coords_wrt_odom[grid_no]))

                grid_waypoints = env.angle_dist_calculator(
                    grid_waypoint_coords_wrt_odom[grid_no][0],
                    grid_waypoint_coords_wrt_odom[grid_no][1]
                )
                rospy.loginfo("Grid goal x: %f", grid_waypoint_coords_wrt_odom[grid_no][0])
                rospy.loginfo("Grid goal y: %f", grid_waypoint_coords_wrt_odom[grid_no][1])

                goal_topic = Twist()
                goal_topic.linear.x = grid_waypoints[0]
                goal_topic.linear.y = grid_waypoints[1]

                rospy.loginfo("Grid goal r, theta: %s", str(grid_waypoints))
                current_goal = [
                    grid_waypoint_coords_wrt_odom[grid_no][0],
                    grid_waypoint_coords_wrt_odom[grid_no][1]
                ]
                waypoint_goal = [
                    grid_waypoint_coords_wrt_odom[grid_no_end-1][0],
                    grid_waypoint_coords_wrt_odom[grid_no_end-1][1]
                ]
                rospy.loginfo("-------- New local grid goal is assigned -----------------")
                rospy.loginfo("New goal: %s", str((goal_topic.linear.x, goal_topic.linear.y)))
                self.pub_goal.publish(goal_topic)

                action[0] = vel_cmd.linear.x
                action[1] = vel_cmd.angular.z
                i +=1
                grid_no +=1

            # Env step
            observation_, reward, done, arrive, waypoint_arrived, grid_arrived, costmap, mask = \
                env.step(action, current_goal, waypoint_goal, grid_arrived, waypoint_arrived)

            current_x, current_y, current_z, relative_theta_deg, cost_pose = env.TGC_calculator()

            elevation_diff = abs(current_z - current_z0)
            distance_diff = math.hypot(current_x - current_x0, current_y - current_y0)

            Tot_elevation_diff += elevation_diff
            Tot_goal_heading_cost += relative_theta_deg
            Total_dist_travelled += distance_diff
            Tot_pose_cost += cost_pose

            score += reward
            observation = observation_

            current_z0 = current_z
            current_x0 = current_x
            current_y0 = current_y

            t_stamp += 1
            rate.sleep()

        # 최종 결과 출력
        Tot_TGC = Tot_elevation_diff + Tot_goal_heading_cost
        rospy.loginfo("===== DDPG 종료 =====")
        rospy.loginfo("Total distance travelled: %f", Total_dist_travelled)
        rospy.loginfo("Tot_goal_heading cost: %f", Tot_goal_heading_cost)
        rospy.loginfo("Tot_pose cost: %f", Tot_pose_cost)
        rospy.loginfo("Tot_elevation cost: %f", Tot_elevation_diff)
        rospy.loginfo("Total TGC cost: %f", Tot_TGC)

    def getVel(self, data):
        global vel_cmd
        vel_cmd.linear.x = data.linear.x
        vel_cmd.linear.y = 0
        vel_cmd.linear.z = 0
        vel_cmd.angular.x = 0
        vel_cmd.angular.y = 0
        vel_cmd.angular.z = data.angular.z

    def reset_func(self):
        vel_cmd2 = Twist()
        vel_cmd2.linear.x = 100
        self.goal_state_pub.publish(vel_cmd2)  # publish a new goal to DWA
        time.sleep(5)

    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        dim = (688, 550)
        resized_img = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)

        result = inference_segmentor(self.model, resized_img)  # result: (1, 550, 688)
        result = np.array(result)

        # 여기서는 단순히 4개 구역의 mode를 출력해 예시로 사용
        # 필요하다면 원하는 처리를 수행하면 됨
        self.grids[0] = mode(result[0, 86:344, 225:362].flatten())[0]
        self.grids[1] = mode(result[0, 344:, 225:362].flatten())[0]
        self.grids[2] = mode(result[0, 43:344, 362:].flatten())[0]
        self.grids[3] = mode(result[0, 344:, 362:].flatten())[0]
        
        mapping = {
            0: 0.0,
            1: 0.0,
            2: 0.2,
            3: 0.4,
            4: 0.8,
            5: 1,
        }
        
        self.grids = np.vectorize(mapping.get)(self.grids)

        # rospy.loginfo(f"Segment result grids: {grid1, grid2, grid3, grid4}")
        
        

def main():
    # TerrainSeg 객체 생성
    node = TerrainSeg()

    # DDPG 학습/플래닝 루프는 별도 스레드에서 실행
    ddpg_thread = threading.Thread(target=node.run_ddpg_loop)
    ddpg_thread.start()

    # 메인 스레드는 spin을 돌아 Subscriber 콜백 처리
    rospy.spin()

    # spin 종료 시(노드가 종료될 때) DDPG 스레드도 join
    ddpg_thread.join()


if __name__ == '__main__':
    main()
