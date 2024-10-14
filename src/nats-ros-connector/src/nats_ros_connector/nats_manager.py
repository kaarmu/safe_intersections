import json
import rospy
from nats_ros_connector.srv import ReqRep

class NATSManager:

    def __init__(self):

        self._subscribers = {}
        self._publishers = {}
        self._services = {}
        self._serviceproxies = {}

        rospy.wait_for_service('/nats')
        self._nats_client_caller = rospy.ServiceProxy('/nats', ReqRep)

        rospy.logdebug('NATSManager initialized')

    def _caller(self, reqd):
        req = ReqRep._request_class(json.dumps(reqd)) 
        rep = self._nats_client_caller(req)
        repd = json.loads(rep.data)
        return repd

    ## Subscribers ##

    def new_subscriber(self, name, *args, **kwds):
        err = self._caller({
            'func': 'new_subscriber',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._subscribers[name] = rospy.Subscriber(name, *args, **kwds)
        rospy.logdebug(f'Subscriber({name=}) up!')
        return self._subscribers[name]
    
    def del_subscriber(self, name):
        err = self._caller({
            'func': 'del_subscriber',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._subscribers.pop(name).unregister()
        rospy.logdebug(f'Subscriber({name=}) down!')
        return

    ## Publishers ##

    def new_publisher(self, name, *args, **kwds):
        err = self._caller({
            'func': 'new_publisher',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._publishers[name] = rospy.Publisher(name, *args, **kwds)
        rospy.logdebug(f'Publisher({name=}) up!')
        return self._publishers[name]
    
    def del_publisher(self, name):
        err = self._caller({
            'func': 'del_publisher',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._publishers.pop(name).unregister()
        rospy.logdebug(f'Publisher({name=}) down!')
        return
    
    ## Services ##

    def new_service(self, name, *args, **kwds):
        self._nats_client_caller(data=json.dumps({
            'func': 'new_service',
            'args': [name],
        }))
        self._services[name] = rospy.Service(name, *args, **kwds)
        rospy.logdebug(f'Service({name=}) up!')
        return self._services[name]
    
    def del_service(self, name):
        err = self._caller({
            'func': 'del_service',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._services.pop(name).shutdown()
        rospy.logdebug(f'Service({name=}) down!')
        return

    ## Service Proxies ##

    def new_serviceproxy(self, name, type, *args, **kwds):
        err = self._caller({
            'func': 'new_serviceproxy',
            'args': [name, type._type],
        }).get('err', None)
        assert err is None, err
        rospy.wait_for_service(name)
        self._serviceproxies[name] = rospy.ServiceProxy(name, type, *args, **kwds)
        rospy.logdebug(f'ServiceProxy({name=}) up!')
        return self._serviceproxies[name]
    
    def del_serviceproxy(self, name):
        err = self._caller({
            'func': 'del_serviceproxy',
            'args': [name],
        }).get('err', None)
        assert err is None, err
        self._serviceproxies.pop(name).close()
        rospy.logdebug(f'ServiceProxy({name=}) down!')
        return
