from contextlib import contextmanager
import rospy

def debuggable_lock(name, lock):
    depth = 0
    @contextmanager
    def ctx(caller):
        nonlocal depth
        acquired = False
        try:
            rospy.logdebug('  '*depth + '> %s: %s acquiring', caller, name)
            with lock:
                depth += 1
                acquired = True
                rospy.logdebug('  '*depth + '* %s: %s acquired', caller, name)
                yield
        finally:
            if acquired:
                rospy.logdebug('  '*depth + '! %s: %s released', caller, name)
                depth -= 1
    return ctx
