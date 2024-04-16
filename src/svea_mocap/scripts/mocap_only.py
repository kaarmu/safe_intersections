#! /usr/bin/env python3

import rospy

from svea_mocap.mocap import MotionCaptureInterface
from svea.svea_managers.svea_archetypes import SVEAManager
from svea.data import BasicDataHandler


def load_param(name, value=None):
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class DummyController(object):
    def __init__(self, vehicle_name=''):
        self.steering = 0
        self.velocity = 0

    def compute_control(self, state):
        return self.steering, self.velocity

class mocap_only:

    def __init__(self):

        ## Initialize node

        rospy.init_node('mocap_only')

        ## Parameters

        self.MOCAP_NAME = load_param('~mocap_name')

        ## Set initial values for node

        # start the SVEA manager
        self.svea = SVEAManager(MotionCaptureInterface,
                                DummyController,
                                data_handler=BasicDataHandler)
        self.svea.localizer.update_name(self.MOCAP_NAME)

        self.svea.start(wait=True)

    def run(self):
        while self.keep_alive():
            self.spin()

    def keep_alive(self):
        return not (rospy.is_shutdown())

    def spin(self):

        # limit the rate of main loop by waiting for state
        state = self.svea.wait_for_state()

        rospy.loginfo_throttle(1.0, "Qualisys-given State:\n{0}".format(state))


if __name__ == '__main__':

    ## Start node ##

    mocap_only().run()

