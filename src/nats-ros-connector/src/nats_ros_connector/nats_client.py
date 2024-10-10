import asyncio
import nats
import rospy
import json
from dataclasses import dataclass
from nats_ros_connector.srv import ReqRep
from nats_ros_connector.nats_publisher import NATSPublisher
from nats_ros_connector.nats_subscriber import NATSSubscriber
from nats_ros_connector.nats_service_proxy import NATSServiceProxy
from nats_ros_connector.nats_service import NATSService

class service:
    def __init__(self, srv_name, srv_type, srv_cb=None, *, method=False, **kwds):
        self._name = srv_name
        self._type = srv_type
        self._kwds = kwds

        self._callback = srv_cb
        self._is_method = method

        self._Req = self._type._request_class
        self._Rep = self._type._response_class

    def __call__(self, f):
        assert self._callback is None, 'Can only set callback once'
        self._callback = f
        return self

    def _new_req(self, args, kwds):
        if kwds:            return self._Req(*args, **kwds)
        if len(args) != 1:  return self._Req(*args, **kwds)
        return (args[0] if isinstance(args[0], self._Req) else 
                self._Req(*args, **kwds))

    def register(self):
        assert self._callback is not None, 'Missing callback'
        if self._is_method:
            def handler(self_, *args_, **kwds_):
                req = self._new_req(args_, kwds_)
                rep = self._Rep()
                self._callback(self_, req, rep)
                return rep
        else:
            def handler(self_, *args_, **kwds_):
                req = self._new_req(args_, kwds_)
                rep = self._Rep()
                self._callback(self_, req, rep)
                return rep
        self._service = rospy.Service(self._name, self._type, handler, self._kwds)

    def unregister(self):
        self._service.shutdown()
        self._service = None

@dataclass
class NATSConnParams:
    name=None,
    pedantic=False,
    verbose=False,
    allow_reconnect=True,
    connect_timeout=2,
    reconnect_time_wait=2,
    max_reconnect_attempts=60,
    ping_interval=120,
    max_outstanding_pings=2,
    dont_randomize=False,
    flusher_queue_size=1024,
    no_echo=False,
    tls=None,
    tls_hostname=None,
    user=None,
    password=None,
    token=None,
    drain_timeout=30,
    signature_cb=None,
    user_jwt_cb=None,
    user_credentials=None,
    nkeys_seed=None,
    srv_req_timeout=None

class NATSClient:
    def __init__(
        self,
        nats_host,
        publishers,
        subscribers,
        services,
        service_proxies,
        event_loop,
        **kwds,
    ):
        self.host = nats_host
        self.publishers = publishers
        self.subscribers = subscribers
        self.services = services
        self.service_proxies = service_proxies
        self.event_loop = event_loop

        self._subscribers = {}
        self._publishers = {}
        self._services = {}
        self._service_proxies = {}

        # NATS Connection parameters
        self.params = NATSConnParams(**kwds)

    async def run(self):
        self.nc = await nats.connect(
            # See https://nats-io.github.io/nats.py/modules.html#asyncio-client
            self.host,
            error_cb=self._error_cb,
            reconnected_cb=self._reconnected_cb,
            disconnected_cb=self._disconnected_cb,
            closed_cb=self._closed_cb,
            name=self.params.name,
            pedantic=self.params.pedantic,
            verbose=self.params.verbose,
            allow_reconnect=self.params.allow_reconnect,
            connect_timeout=self.params.connect_timeout,
            reconnect_time_wait=self.params.reconnect_time_wait,
            max_reconnect_attempts=self.params.max_reconnect_attempts,
            ping_interval=self.params.ping_interval,
            max_outstanding_pings=self.params.max_outstanding_pings,
            dont_randomize=self.params.dont_randomize,
            flusher_queue_size=self.params.flusher_queue_size,
            no_echo=self.params.no_echo,
            tls=self.params.tls,
            tls_hostname=self.params.tls_hostname,
            user=self.params.user,
            password=self.params.password,
            token=self.params.token,
            drain_timeout=self.params.drain_timeout,
            signature_cb=self.params.signature_cb,
            user_jwt_cb=self.params.user_jwt_cb,
            user_credentials=self.params.user_credentials,
            nkeys_seed=self.params.nkeys_seed
        )
        # Register Subscribers
        for topic_name in self.subscribers:
            self.new_subscriber(topic_name)            

        # Register Publishers
        for topic_name in self.publishers:
            self.new_publisher(topic_name)

        # Register Services
        for service_name in self.services:
            self.new_service(service_name)

        # Register Service Proxies
        for service_dict in self.service_proxies:
            self.new_serviceproxy(service_dict['name'],
                                  service_dict['type'])

        self._client_srv.register()

        rospy.spin() # spin ROS in background

    async def close(self):
        rospy.loginfo("NATS Connector: CLOSING CONNECTION")
        await self.nc.close()

    async def _disconnected_cb(self):
        rospy.loginfo("NATS Connector: GOT DISCONNECTED")

    async def _reconnected_cb(self):
        rospy.loginfo(f"NATS Connector: GOT RECONNECTED TO {self.nc.connected_url.netloc}")

    async def _error_cb(self, e):
        rospy.loginfo(f"NATS Connector Error: {e}")

    async def _closed_cb(self):
        rospy.loginfo("NATS Connector: CLOSED CONNECTION")

    @service('/nats', ReqRep, method=True)
    def _client_srv(self, req, rep):
        reqd = json.loads(req.data)
        func = getattr(self, reqd['func'])
        func(*reqd.get('args', []), **reqd.get('kwds', {}))

    ## Subscribers ##

    def new_subscriber(self, name):
        obj = NATSSubscriber(self.nc, name)
        self._subscribers[name] = obj
        self.event_loop.create_task(obj.start())

    def del_subscriber(self, name):
        obj: NATSSubscriber = self._subscribers.pop(name)
        self.event_loop.create_task(obj.stop())

    ## Publishers ##

    def new_publisher(self, name):
        obj = NATSPublisher(self.nc, name, self.event_loop)
        self._publishers[name] = obj
        self.event_loop.create_task(obj.start())

    def del_publisher(self, name):
        obj: NATSPublisher = self._publishers.pop(name)
        self.event_loop.create_task(obj.stop())

    ## Services ##

    def new_service(self, name):
        obj = NATSService(self.nc, name)
        self._services[name] = obj
        self.event_loop.create_task(obj.start())

    def del_service(self, name):
        obj: NATSService = self._services.pop(name)
        self.event_loop.create_task(obj.stop())

    ## Service Proxy ##

    def new_serviceproxy(self, name, type):
        obj = NATSServiceProxy(self.nc, name, type, self.event_loop, self.params.srv_req_timeout)
        self._service_proxies[name] = obj
        self.event_loop.create_task(obj.start())
    
    def del_serviceproxy(self, name):
        obj: NATSServiceProxy = self._service_proxies.pop(name)
        self.event_loop.create_task(obj.stop())
