import json
import rospy
from nats_ros_connector.srv import ReqRep

def _new_serviceproxy(name, *args, **kwds):
    rospy.wait_for_service(name)
    rospy.ServiceProxy(name, *args, **kwds)

class NATSManager:

    def __init__(self):

        self._subscribers = {}
        self._publishers = {}
        self._services = {}
        self._serviceproxies = {}

        rospy.wait_for_service('/nats')
        self._nats_client_caller = rospy.ServiceProxy('/nats', ReqRep)

        rospy.log_debug('NATSManager initialized')

    ## Subscribers ##

    def new_subscriber(self, name, *args, **kwds):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'new_subscriber',
            'kwds': {'name': name},
        }))
        self._subscribers[name] = rospy.Subscriber(name, *args, **kwds)
        rospy.logdebug(f'Subscriber({name=}) up!')
        return self._subscribers[name]
    
    def del_subscriber(self, name):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'del_subscriber',
            'kwds': {'name': name},
        }))
        self._subscribers.pop(name).unregister()
        rospy.logdebug(f'Subscriber({name=}) down!')
        return

    ## Publishers ##

    def new_publisher(self, name, *args, **kwds):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'new_publisher',
            'kwds': {'name': name},
        }))
        self._publishers[name] = rospy.Publisher(name, *args, **kwds)
        rospy.logdebug(f'Publisher({name=}) up!')
        return self._publishers[name]
    
    def del_publisher(self, name):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'del_publisher',
            'kwds': {'name': name},
        }))
        self._publishers.pop(name).unregister()
        rospy.logdebug(f'Publisher({name=}) down!')
        return
    
    ## Services ##

    def new_service(self, name, *args, **kwds):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'new_service',
            'kwds': {'name': name},
        }))
        self._services[name] = rospy.Service(name, *args, **kwds)
        rospy.logdebug(f'Service({name=}) up!')
        return self._services[name]
    
    def del_service(self, name):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'del_service',
            'kwds': {'name': name},
        }))
        self._services.pop(name).shutdown()
        rospy.logdebug(f'Service({name=}) down!')
        return

    ## Service Proxies ##

    def new_serviceproxy(self, name, type, *args, **kwds):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'new_serviceproxy',
            'kwds': {'name': name, 'type': type._type},
        }))
        rospy.wait_for_service(name)
        self._serviceproxies[name] = rospy.ServiceProxy(name, type, *args, **kwds)
        rospy.logdebug(f'ServiceProxy({name=}) up!')
        return self._serviceproxies[name]
    
    def del_serviceproxy(self, name):
        _ = self._nats_client_caller(data=json.dumps({
            'func': 'del_serviceproxy',
            'kwds': {'name': name},
        }))
        self._serviceproxies.pop(name).close()
        rospy.logdebug(f'ServiceProxy({name=}) down!')
        return


