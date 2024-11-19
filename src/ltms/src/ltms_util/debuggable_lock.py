from contextlib import contextmanager

import rospy

def debuggable_lock(name, lock):
    @contextmanager
    def ctx(caller):
        rospy.logdebug(f'{caller}: {name} acquiring')
        with lock:
            rospy.logdebug(f'{caller}: {name} acquired')
            yield
        rospy.logdebug(f'{caller}: {name} released')
    return ctx
