#! /usr/bin/env python3

import rospy

from svea_mocap.mocap import MotionCaptureInterface
from svea.svea_managers.svea_archetypes import SVEAManager
from svea.data import BasicDataHandler
from svea.interfaces.rc import RCInterface
from svea.interfaces.localization import LocalizationInterface

def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class RCForwarder(object):
    def __init__(self, vehicle_name=''):
        self.steering = 0
        self.velocity = 0
        self.transmission = 0

    def update_rc(self, steering, velocity, transmission):
        self.steering = steering
        self.velocity = velocity
        self.transmission = transmission

    def compute_control(self, state):
        return self.steering, self.velocity, self.transmission

class localization_comparison:

    def __init__(self):

        ## Initialize node

        rospy.init_node('localization_comparison')

        ## Parameters

        self.MOCAP_NAME = load_param('~mocap_name')

        ## Set initial values for node

        # start the SVEA manager
        self.svea = SVEAManager(MotionCaptureInterface,
                                RCForwarder,
                                data_handler=BasicDataHandler)
        self.svea.localizer.update_name(self.MOCAP_NAME)

        self.rc = RCInterface().start()
        self.svea.start(wait=True)

        self.evaluated_localizer = LocalizationInterface().start()

    def run(self):
        while self.keep_alive():
            self.spin()

    def keep_alive(self):
        return not (rospy.is_shutdown())

    def spin(self):

        # limit the rate of main loop by waiting for state
        mocap_state = self.svea.wait_for_state()
        localization_state = self.evaluated_localizer.state

        # read latest rc inputs
        rc_steering = self.rc.steering
        rc_velocity = self.rc.velocity
        rc_transmission = self.rc.gear

        self.svea.controller.update_rc(rc_steering, rc_velocity, rc_transmission)
        steering, velocity, transmission = self.svea.compute_control()
        self.svea.send_control(steering, velocity, transmission)

        rospy.loginfo_throttle(1.0, "Qualisys-given State:\n{0}".format(mocap_state))
        rospy.loginfo_throttle(1.0, "Localization-given State:\n{0}".format(localization_state))


if __name__ == '__main__':

    ## Start node ##

    localization_comparison().run()

