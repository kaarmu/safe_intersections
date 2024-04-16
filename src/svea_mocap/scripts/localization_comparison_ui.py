#! /usr/bin/env python3

import rospy

import math
from svea_msgs.msg import VehicleState as VehicleStateMsg
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf.transformations import quaternion_matrix, euler_from_matrix, euler_matrix
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Ellipse

__license__ = "MIT"
__maintainer__ = "Michele Pestarino, Federico Sacco"
__email__ = "micpes@kth.se, fsacco@ug.kth.se"
__status__ = "Development"

__all__ = [
    'LocalizationComparisonUI',
]

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class LocalizationComparisonUI:
    """Interface handling the reception of state information from the
    localization stack and the mocap system.

    This object instatiates two subscribers (i.e. one for the state topic given by the 
    localization algorithm, one for the pose topic given by the ground truth, in this case the
    mocap system).

    Args:
        vehicle_name: Name of vehicle being controlled; The name will be
            effectively be added as a namespace to the topics used by the
            corresponding localization node i.e `namespace/vehicle_name/state`.
    """
    def __init__(self, vehicle_name: str = ''):
        """
        Init method for class LocalizationComparisonUI

        :param vehicle_name: vehicle name, defaults to ''
        :type vehicle_name: str, optional
        """

        # Offset angle between the mocap frame (of the real world) and the map frame
        self.OFFSET_ANGLE = -math.pi/2
        self.T_MATRIX_4 = euler_matrix(0, 0, self.OFFSET_ANGLE)

        # Covariance threshold for updating the covariance ellipse in the plot
        self.COVARIANCE_THRESHOLD = 5
        
        # Create rotation matrix given the offset angle and linear misalignment between mocap and map frames
        LINEAR_PART_MOCAP2MAP =  np.transpose(np.array([0.06, -0.06, 0]))
        self.T_MATRIX_4[0:3,3] = LINEAR_PART_MOCAP2MAP
        
        # Instatiate figure for trajectory plotting
        self.fig_traj, self.ax_traj = plt.subplots(num='Trajectory')
        # First line for the localization visualization
        self.line1_traj, = self.ax_traj.plot([], [], color = "r", alpha=0.5)
        # Second line for the mocap visualization
        self.line2_traj, = self.ax_traj.plot([], [], color = "g", alpha=0.5)
        # Ellipse for visualization localization covariance
        self.covariance_ellipse = Ellipse(xy = (0, 0), width=0, height=0, alpha=0.25, edgecolor='r', fc='r')
        # Annotations for rmse both for x and y positions, and yaw
        self.rmse_text_x = self.ax_traj.annotate(f'RMSE(x): 0.0000', xy = (0, -0.11), bbox=dict(boxstyle="round", fc="w"), xycoords='axes fraction')
        self.rmse_text_y = self.ax_traj.annotate(f'RMSE(y): 0.0000', xy = (0.3, -0.11), bbox=dict(boxstyle="round", fc="w"), xycoords='axes fraction')
        self.rmse_text_yaw = self.ax_traj.annotate(f'RMSE(yaw): 0.0000', xy = (0.6, -0.11), bbox=dict(boxstyle="round", fc="w"), xycoords='axes fraction')

        # Instatiate figure for rmse_x plotting
        self.fig_rmse_x, self.ax_rmse_x = plt.subplots(num='RMSE(x)')
        # Line for rmse x plotting
        self.line_rmse_x, = self.ax_rmse_x.plot([], [], color = "r")
        # Instatiate figure for rmse_y plotting
        self.fig_rmse_y, self.ax_rmse_y = plt.subplots(num='RMSE(y)')
        # Line for rmse y plotting
        self.line_rmse_y, = self.ax_rmse_y.plot([], [], color = "g")
        # Instatiate figure for rmse_x plotting
        self.fig_rmse_yaw, self.ax_rmse_yaw = plt.subplots(num='RMSE(yaw)')
        # Line for rmse yaw plotting
        self.line_rmse_yaw, = self.ax_rmse_yaw.plot([], [], color = "b")
        # Instatiate figure for velocity plotting
        self.fig_vel, self.ax_vel = plt.subplots(num='Velocity')
        # Line for svea velocity plotting
        self.line_svea_vel, = self.ax_vel.plot([], [], color = "r")
        # Line for mocap velocity plotting
        self.line_mocap_vel, = self.ax_vel.plot([], [], color = "b")

        # Get vehicle name from parameters
        self.vehicle_name = vehicle_name
        # Set svea on board localization topic's name
        self._svea_state_topic = load_param('~localization_topic', '/state')
        # Set mocap topic's name
        self._mocap_state_topic = load_param('~ground_truth_topic', f'/qualisys/{vehicle_name}/pose')
        self._mocap_vel_topic = load_param('~velocity_topic', f'/qualisys/{vehicle_name}/velocity')
        # current states
        self.svea_state = None
        self.mocap_state = None
        self.curr_vel = 0.0

        # list of measurements
        self.svea_measurements = []
        self.mocap_measurements = []
        self.svea_vel_measurements = []
        self.mocap_twist_measurements = []
        self.mocap_vel_measurements = []

        # list of measurements for RMSE
        self.rmse_x = []
        self.rmse_y = []
        self.rmse_yaw = []
        # Time counter
        self._time_tic = 0
    
    def init_and_start_listeners(self):
        """
        Inits the node and start the two listeners
        """
        rospy.loginfo("Starting Measurements Node for " + self.vehicle_name)
        self.node_name = 'measurement_node'
        # Instatiates subscribers
        self._start_listen()
        rospy.loginfo("{} Measurements Interface successfully initialized"
                      .format(self.vehicle_name))

    def _start_listen(self):
        """
        Instatiates the two listeners
        """
        # First subcriber for svea localization algorithm's pose topic
        rospy.Subscriber(self._svea_state_topic,
                         VehicleStateMsg,
                         self._svea_read_state_msg,
                         queue_size=1)
        # Second subscriber for mocap localization pose
        rospy.Subscriber(self._mocap_state_topic,
                         PoseStamped,
                         self._mocap_read_pose_msg,
                         queue_size=1)
        # Subscriber to velocity topic of mocpa
        rospy.Subscriber(self._mocap_vel_topic,
                         TwistStamped,
                         self._mocap_read_twist_msg,
                         tcp_nodelay=True,
                         queue_size=1)
        

    def _svea_read_state_msg(self, msg):
        """
        Callback method for the svea's localization algorithnm 

        :param msg: message from the localization algorithm's topic
        :type msg: VehicleStateMsg
        """
        # Append new svea state message to corresponding list
        self.svea_measurements.append(msg)
        self.curr_vel = msg.v


    def _mocap_read_pose_msg(self, msg):
        """
        Callback method for the ground truth localization algorithnm

        :param msg: message ground truth localization algorithnm
        :type msg: PoseStamped
        """
        # Append new mocap pose message to corresponding list 
        # (operation conditioned to enable set by _svea_read_state_msg,
        # slower topic dictate the pace for saving data)
        if len(self.svea_measurements) > len(self.mocap_measurements):
            self.mocap_measurements.append(msg)

    def _mocap_read_twist_msg(self, msg):
        """
        Callback method for he mocap velocity subscriber

        :param msg: message ground truth 
        :type msg: TwistStamped
        """
        # Append new mocap pose message to corresponding list 
        # (operation conditioned to enable set by _svea_read_state_msg,
        # slower topic dictate the pace for saving data)
        if len(self.svea_measurements) > len(self.mocap_twist_measurements):
            self.mocap_twist_measurements.append(np.array([msg.twist.linear.x, msg.twist.linear.y]))

    def _compute_vehicle_velocity(self, quaternion):
        """
        Method used to compute the vehicle's velocity given the mocap's twist
        
        :param quaternion: quaternion used to extract and correct yaw angle 
        :type quaternion: Quaternion

        :return: v[0] vehicle velocity
        :rtype: float
        """
        # Get svea's rotation matrix from pose quaternion wrt mocap frame
        vehicle_R_mocap = quaternion_matrix([quaternion.x, 
                                                  quaternion.y, 
                                                  quaternion.z, 
                                                  quaternion.w])
        # Get vehicle yaw wrt mocap frame
        (_, _, vehicle_yaw_mocap) = euler_from_matrix(vehicle_R_mocap)
        # Compute vehicle velocity by reprojecting the twist of the vehicle wrt mocap frame, onto the vehicle frame
        # itself, then extract the x component (which is the vehicle velocity)
        self.vehicle_R_mocap = [[math.cos(-vehicle_yaw_mocap), -math.sin(-vehicle_yaw_mocap)],
                                [math.sin(-vehicle_yaw_mocap), math.cos(-vehicle_yaw_mocap)]]

        v = np.matmul(self.vehicle_R_mocap, np.transpose(np.array(self.mocap_twist_measurements[len(self.mocap_twist_measurements) - 1])))
        return v[0]
            
    def plot_init_traj(self):
        """
        Inits plot for trajectory

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis labels
        self.ax_traj.set_xlabel('[m]')
        self.ax_traj.set_xlabel('[m]')
        # Set axis' limits to the plot
        self.ax_traj.set_xlim(-5, 5)
        self.ax_traj.set_ylim(-5, 5)
        # Set legend for the two lines
        self.ax_traj.legend(['Localization', 'Mocap'], loc='upper right')
        # Add covariance ellipse
        self.ax_traj.add_patch(self.covariance_ellipse)
        # Returns graphic widgets
        return [self.line1_traj, self.line2_traj, self.rmse_text_x, self.rmse_text_y, self.rmse_text_yaw]

    def plot_init_rmse_x(self):
        """
        Inits plot for rmse of x coordinate

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis labels
        self.ax_rmse_x.set_ylabel('RMSE(x)')
        self.ax_rmse_x.set_xlabel('time (msg received)')
        # Set legend for the two lines
        self.ax_rmse_x.legend(['RMSE(x)'], loc='upper right')
        # Returns graphic widgets
        return self.line_rmse_x
    
    def plot_init_rmse_y(self):
        """
        Inits plot for rmse of y coordinate

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis labels
        self.ax_rmse_y.set_ylabel('RMSE(y)')
        self.ax_rmse_y.set_xlabel('time (msg received)')
        # Set legend for the two lines
        self.ax_rmse_y.legend(['RMSE(y)'], loc='upper right')
        # Returns graphic widgets
        return self.line_rmse_y
    
    def plot_init_rmse_yaw(self):
        """
        Inits plot for rmse of x coordinate

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis labels
        self.ax_rmse_yaw.set_ylabel('RMSE(yaw)')
        self.ax_rmse_yaw.set_xlabel('time (msg received)')
        # Set legend for the two lines
        self.ax_rmse_yaw.legend(['RMSE(yaw)'], loc='upper right')
        # Returns graphic widgets
        return self.line_rmse_yaw
    
    def plot_init_vel(self):
        """
        Inits plot for velocity 

        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # Set axis labels
        self.ax_vel.set_ylabel('Vel')
        self.ax_vel.set_xlabel('time (msg received)')
        # Set legend for the two lines
        self.ax_vel.legend(['SVEA State Velocity', 'Mocap Velocity'], loc='upper right')
        # Returns graphic widgets
        return [self.line_svea_vel, self.line_mocap_vel]

    def _correct_mocap_coordinates(self, x, y, quaternion):
        """
        Method used to correct the mocap pose (if some misalignment between its frame and the map frame is present)
        
        :param x: x coordinate to be corrected 
        :type x: float
        :param y: y coordinate to be corrected 
        :type y: float
        :param quaternion: quaternion used to extract and correct yaw angle 
        :type quaternion: Quaternion

        :return: rotate_point[0] corrected x coordinate
        :rtype: float
        :return: rotate_point[1] corrected y coordinate
        :rtype: float
        :return: mocap_yaw corrected yaw angle
        :rtype: float
        """
        # Get svea's rotation matrix from pose quaternion
        svea_T_mocap = quaternion_matrix([quaternion.x, 
                                                  quaternion.y, 
                                                  quaternion.z, 
                                                  quaternion.w])
        # Add translational part to transofmration matrix
        svea_T_mocap[0:3,3] = np.transpose(np.array([x,y,0]))

        # Apply 4 dimension square rotation matrix (rotate svea's yaw)
        svea_T_map = np.matmul(self.T_MATRIX_4, svea_T_mocap)

        # Get correct yaw (from manipulated rotation matrix)
        (_, _, mocap_yaw) = euler_from_matrix(svea_T_map)
        return svea_T_map[0,3], svea_T_map[1,3], mocap_yaw
        

    def update_plot_traj(self, frame):
        """
        Method called by the FuncAnimation for updating the plot

        :param frame: frame of the animation
        :type frame: _type_
        :return: lines to be drawn on the figure
        :rtype: list of matplotlib.lines.Line2D
        """
        # If svea's localization measurments were received
        if self.svea_measurements:
            # Instatiate lists for xs and ys to be plotted on the figure and yaws
            svea_xs = []
            svea_ys = []
            svea_yaws = []
            # Iterate over every svea localization measurements (possibly too computationally demanding over long timespans)
            for svea_pose in self.svea_measurements:
                # Append coordinates
                svea_xs.append(svea_pose.x)
                svea_ys.append(svea_pose.y)
                #print("svea (x,y): " + str(svea_pose.x) + ", " + str(svea_pose.y))
                svea_yaws.append(svea_pose.yaw)
            
            # Update covariance Ellipse (center position, width and height based off of the variance)
            self.covariance_ellipse.set_center(xy = (self.svea_measurements[len(self.svea_measurements) - 1].x, self.svea_measurements[len(self.svea_measurements) - 1].y))

            if math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[0]) < self.COVARIANCE_THRESHOLD:
                self.covariance_ellipse.width = math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[0])

            if math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[5]) < self.COVARIANCE_THRESHOLD:
                self.covariance_ellipse.height = math.sqrt(self.svea_measurements[len(self.svea_measurements) - 1].covariance[5])
            
            # Set data for line1  
            self.line1_traj.set_data(svea_xs, svea_ys)
            
            # If mocap measurements were received
            if self.mocap_measurements:
                # Instatiate lists for xs and ys to be plotted on the figure
                mocap_xs = []
                mocap_ys = []
                mocap_yaws = []
                # Iterate over every mocap measurements (possibly too computationally demanding over long timespans)
                for mocap_pose in self.mocap_measurements:
                    # Get x and y positions
                    x = mocap_pose.pose.position.x 
                    y = mocap_pose.pose.position.y
                    # Correct mocap pose
                    corrected_x, corrected_y, corrected_yaw = self._correct_mocap_coordinates(x, y, mocap_pose.pose.orientation)
                    # Append rotated coordinates
                    mocap_xs.append(corrected_x)
                    mocap_ys.append(corrected_y)
                    #print("mocap x,y : " + str(corrected_x) + "," + str(corrected_y))
                    # Append corrected yaw
                    mocap_yaws.append(corrected_yaw)
                # Set data for line2
                self.line2_traj.set_data(mocap_xs, mocap_ys)

                if len(svea_xs) == len(mocap_xs) and len(svea_ys) == len(mocap_ys) and len(svea_yaws) == len(mocap_yaws) and len(self.svea_vel_measurements) == len(self.mocap_vel_measurements):
                    # Compute rmse for x coordinate
                    RMSE_x = np.round(math.sqrt(np.square(np.subtract(mocap_xs, svea_xs)).mean()), 4)
                    # Set text for RMSE_x
                    self.rmse_text_x.set_text(f'RMSE(x): {RMSE_x}')
                    # Compute rmse for y coordinate
                    RMSE_y = np.round(math.sqrt(np.square(np.subtract(mocap_ys, svea_ys)).mean()), 4)
                    # Set text for RMSE_y
                    self.rmse_text_y.set_text(f'RMSE(y): {RMSE_y}')
                    # Compute rmse for yaw angle
                    RMSE_yaw = np.round(math.sqrt(np.square(np.subtract(mocap_yaws, svea_yaws)).mean()), 4)
                    # Set text for RMSE_yaw
                    self.rmse_text_yaw.set_text(f'RMSE(yaw): {RMSE_yaw}')
                    # Append values in the lists
                    self.rmse_x.append(RMSE_x)
                    self.rmse_y.append(RMSE_y)
                    self.rmse_yaw.append(RMSE_yaw)
                    self.svea_vel_measurements.append(self.curr_vel)
                    if len(self.mocap_measurements) >= 1 and self.mocap_twist_measurements:
                        self.mocap_vel_measurements.append(self._compute_vehicle_velocity(self.mocap_measurements[len(self.mocap_measurements) - 1].pose.orientation))
                    self._time_tic += 1
        
        # Return graphic widgets
        return [self.line1_traj, self.line2_traj, self.rmse_text_x, self.rmse_text_y, self.rmse_text_x]

    def update_plot_rmse_x(self, frame):
        self.line_rmse_x.set_data(np.linspace(0, self._time_tic, num=(self._time_tic)), self.rmse_x) 
        self.fig_rmse_x.gca().relim()
        self.fig_rmse_x.gca().autoscale_view() 
        return self.line_rmse_x
    
    def update_plot_rmse_y(self, frame):
        self.line_rmse_y.set_data(np.linspace(0, self._time_tic, num=(self._time_tic)), self.rmse_y) 
        self.fig_rmse_y.gca().relim()
        self.fig_rmse_y.gca().autoscale_view() 
        return self.line_rmse_y
    
    def update_plot_rmse_yaw(self, frame):
        self.line_rmse_yaw.set_data(np.linspace(0, self._time_tic, num=(self._time_tic)), self.rmse_yaw) 
        self.fig_rmse_yaw.gca().relim()
        self.fig_rmse_yaw.gca().autoscale_view() 
        return self.line_rmse_yaw
    
    def update_plot_vel(self, frame):
        self.line_svea_vel.set_data(np.linspace(0, self._time_tic, num=(self._time_tic)), self.svea_vel_measurements)
        self.line_mocap_vel.set_data(np.linspace(0, self._time_tic, num=(self._time_tic)), self.mocap_vel_measurements)
        self.fig_vel.gca().relim()
        self.fig_vel.gca().autoscale_view() 
        return [self.line_svea_vel, self.line_mocap_vel]
        
   
if __name__ == '__main__':
    ## Start node ##
    rospy.init_node('localization_comparison_ui')
    # Instanciate object of class LocalizationComparisonUI
    measurement_node = LocalizationComparisonUI(vehicle_name='svea7')
    # Init node and start listeners
    measurement_node.init_and_start_listeners()
    # Create animation for the plot
    ani_traj = FuncAnimation(measurement_node.fig_traj, measurement_node.update_plot_traj, init_func=measurement_node.plot_init_traj)
    # Create animation for velocity measurements over time
    ani_vel = FuncAnimation(measurement_node.fig_vel, measurement_node.update_plot_vel, init_func=measurement_node.plot_init_vel)
    # Create animation for RMSE measurements over time
    ani_rmse_x = FuncAnimation(measurement_node.fig_rmse_x, measurement_node.update_plot_rmse_x, init_func=measurement_node.plot_init_rmse_x)
    ani_rmse_y = FuncAnimation(measurement_node.fig_rmse_y, measurement_node.update_plot_rmse_y, init_func=measurement_node.plot_init_rmse_y)
    ani_rmse_yaw = FuncAnimation(measurement_node.fig_rmse_yaw, measurement_node.update_plot_rmse_yaw, init_func=measurement_node.plot_init_rmse_yaw)
    # Show the figure
    plt.show(block=True)
    # Spin node 
    rospy.spin()
    
    