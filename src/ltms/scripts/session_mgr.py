from threading import RLock
from contextlib import contextmanager

import rospy
from ltms_util.debuggable_lock import *

class SessionMgr:

    def __init__(self):

        self._order = ()
        self._sessions = {}
        self._reserved = set()

        self.active_sid = None
        self.reserving_sid = None

        self._lock = debuggable_lock('sessions_lock', RLock())

        rospy.Timer(rospy.Duration(10),
                    lambda event: self.clean())

    def _add(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self._lock('add'):
            self._sessions[key] = kwds if dict1 is None else dict1

    def add(self, sid, *, _addorder=False, _dbgname='', **kwds):
        if _dbgname: _dbgname += '#'
        _dbgname += 'add'

        with self._lock(_dbgname):
            self._add(sid,
                      sid=sid,
                      reserved=False,
                      lock=debuggable_lock(f'{sid}_lock', RLock()),
                      **kwds)
            if _addorder:
                self._order += (sid,)

    def _iterate_unlocked(self, *, skip=..., only=..., order=None, _dbgname):
        if order is None:
            items = list(self._sessions.items())
        else:
            items = [(sid, self._sessions[sid]) 
                     for sid in self._order]

        for sid, sess in items:
            if only is not Ellipsis and sid not in only: continue
            if skip is not Ellipsis and sid in skip: continue
            with sess['lock'](_dbgname):
                yield (sid, sess)

    def iterate(self, *, lock_all=False, _dbgname='', **kwds):
        if _dbgname: _dbgname += '#'
        _dbgname += 'iterate'
        
        if lock_all:
            with self._lock(_dbgname):
                yield from self._iterate_unlocked(_dbgname=_dbgname, **kwds)
        else:
            yield from self._iterate_unlocked(_dbgname=_dbgname, **kwds)

    def select(self, sid, *, lock_all=False, strict=False, _dbgname=''):
        if _dbgname: _dbgname += '#'
        _dbgname += 'select'

        if lock_all:
            with self._lock(_dbgname):
                if strict and sid not in self._sessions:
                    raise Exception(f'Session {sid} not found!')
                if sid not in self._sessions: return
                sess = self._sessions[sid]

                with sess['lock'](_dbgname):
                    yield sess
        else:
            with self._lock(_dbgname + '.1'):
                if strict and sid not in self._sessions:
                    raise Exception(f'Session {sid} not found!')
                if sid not in self._sessions: return
                sess = self._sessions[sid]

            with sess['lock'](_dbgname + '.2'):
                yield sess

    def select_active(self, *, _dbgname='', **kwds):
        if _dbgname: _dbgname += '#'
        _dbgname += 'select_active'

        yield from self.select(self.active_sid, _dbgname=_dbgname, **kwds)

    def select_to_reserve(self, sid, **kwds):
        for sess in self.select(sid, **kwds):
            self.reserving_sid = sid
            yield sess
            if sess['reserved']:
                self._reserved.add(sid)
            self.reserving_sid = None

    def read_prop(self, sid, name):
        for sess in self.select(sid):
            return sess[name]

    def clean(self):
        iter_opts = dict(order=False,
                         lock_all=True,
                         skip=(self.active_sid,
                               *self._order),
                         _dbgname='cleaner',)
        
        cleaned = []
        for sid in self.iterate(**iter_opts):
            del self._sessions[sid]
            self._reserved -= {sid}
            cleaned.append(sid)
        
        if cleaned:
            rospy.logdebug('\n'.join([
                'Cleaned away:',
                *(f'  - {sid}' for sid in cleaned),
            ]))

    def is_known(self, sid):
        return sid in self._sessions

    def next(self):
        with self._lock('next session'):
            if not self._order:
                rospy.logfatal('\n'.join([
                    'Trying to switch session, but nothing in queue/order',
                    f'  Active session ID: {self.active_sid}',
                ]))
                rospy.signal_shutdown('Switching session failed')
                return None
            self.active_sid, *self._order = self._order
            if self.active_sid not in self._reserved:
                rospy.logfatal('\n'.join([
                    'Switching to unreserved session:',
                    f'  Name: {self.active_sid}',
                ]))
            else:
                rospy.loginfo('\n'.join([
                    f'Switching to session {self.active_sid}',
                ]))
            return self.active_sid

