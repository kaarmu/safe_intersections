
import numpy as np


def dist(a, b):
    """
    Computes euclidean distance between two points

    :param a: vector
    :type a: _type_
    :param b: vector
    :type b: _type_
    :return: euclidean distance between two points
    :rtype: float
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b)

class TrajectoryPlanner(object):

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def create_path(self):
        """
        Function used to create a trajectory from current posistion, to target one

        :return: trajectory
        :rtype: list of floats
        """
        traj_x, traj_y = [], []
        for i in range(len(self.xs)-1):
            traj_x += np.linspace(self.xs[i], self.xs[i+1], 100).tolist()
            traj_y += np.linspace(self.ys[i], self.ys[i+1], 100).tolist()

        traj = []
        for x, y in zip(traj_x, traj_y):
            traj.append((x, y))

        return traj


class AStarNode(object):
    """
    Class defining a node of the graph

    :param object: _description_
    :type object: _type_
    """
    def __init__(self, world, ind, p, h):
        """
        Init method for node class

        :param world: _description_
        :type world: _type_
        :param ind: _description_
        :type ind: _type_
        :param p: parent node
        :type p: AStarNode
        :param h: h (heuristics) cost 
        :type h: float
        """
        self.world = world
        self.state = np.asarray(ind)
        self.parent = p
        # Heurisitcs cost (distance to target as heuristics)
        self.h_cost = h
        # G cost (distance from parent)
        self.g_cost = self.calc_g()

    def __repr__(self):
        fmt = '<AStarNode: {}, h={}, g={}>'
        return fmt.format(self.state, self.h_cost, self.g_cost)

    def __hash__(self):
            return hash(tuple(self.state))

    def __eq__(self, other):
        if isinstance(other, AStarNode):
            return np.all(self.state == other.state)
        return NotImplemented

    def calc_g(self):
        """
        Function used to compute the g_cost of the node (i.e. the distance to its parent)

        :return: g_cost 
        :rtype: float
        """
        if self.parent is None:
            # If node has no parent, then g_cost equal to 0
            return 0
        elif not self.is_free:
            # If node is occupied, then g_cost equal to ing
            return float('inf')
        else:
            # Otherwise: compute distance to parent
            cost_to_parent = dist(self.state, self.parent.state)
            return self.parent.g_cost + cost_to_parent

    def family_tree(self):
        """
        Function used to return each parent of node

        :yield: parent node
        :rtype: AStarNode
        """
        # If node has a parent
        if self.parent is not None:
            # For every parent of the parent node
            for n in self.parent.family_tree():
                # Return the current parent of the parent node (of the node) 
                yield n
        yield self

    @property
    def is_free(self):
        """
        Function used to check if node is free or not

        :return: if node is free or not
        :rtype: _type_
        """
        return self.world.is_free_ind(self.state)

    @property
    def pos(self):
        """
        Function used to get the position of a node

        :return: position of the node
        :rtype: (float, float)
        """
        return self.world.ind_to_pos(self.state)

    @property
    def f_cost(self):
        """
        Function used to retrieve the f_cost of a node (which is g_cost + h_cost)

        :return: _description_
        :rtype: _type_
        """
        return self.g_cost + self.h_cost


class AStarWorld(object):
    """
    Class describing a world

    :param object: _description_
    :type object: _type_
    :return: _description_
    :rtype: _type_
    """
    # Delta between each world's position
    DELTA = None
    # Limits of the world (list of tuples indicating the ranges of coordinate, e.g. [[0, 50], [-10, 20]], x coordinates
    # are in range [0, 50], y coordinates are in range [-10, 20])
    LIMIT = None

    # Obstacles margin (if any, i.e. how much to 'naturally' inflate the obstacles)
    OBSTACLE_MARGIN = 0
    # Obstacle array (as (x, y, radius))
    OBS = []

    _res = None
    # World as an occupancy grid
    _occupancy_grid = None

    def __init__(self, delta=None, limit=None, obstacles=None, obs_margin=0):
        """
        Init method for AStarWorld class

        :param delta: how big each cell is, defaults to None
        :type delta: float, optional
        :param limit: limits of the world, defaults to None
        :type limit: list of lists of integers, optional
        :param obstacles: obstacles of the world, defaults to None
        :type obstacles: list, optional
        :param obs_margin: obstacles' margins , defaults to 0
        :type obs_margin: list, optional
        """
        # How big is each cell
        self.DELTA = np.asarray(delta or self.DELTA)
        # 4 cells that represent the 4 corners of the world (list of lists indicating the extreme points of the map)
        self.LIMIT = np.asarray(limit or self.LIMIT)
        # Obstacle list (if obstacles exists then self.OBS = obstacles, otherwise self.OBS = self.OBS, which is None at init)
        if obstacles is not None:
            self.OBS = obstacles
        # Obstacles margins
        self.OBSTACLE_MARGIN = obs_margin or self.OBSTACLE_MARGIN
        # If no world resolution is given (so at init)
        if self._res is None:
            # Compute resolutions (number of rows/columns in an occupancy grid) for each world dimensions given world's limits
            self._res = (self.LIMIT[:, 1] - self.LIMIT[:, 0]) / self.DELTA
            # Resolution as a tuple (one element for each world's dimension)
            self._res = tuple(self._res.astype(int))
        # If no occupancy grid is given (so at init)
        if self._occupancy_grid is None:
            # Initialized occupancy grid as a matrix of zeros
            self._occupancy_grid = np.zeros(self._res).astype(bool)
            # Create obstacles array
            obs = np.array([[x, y, rad + self.OBSTACLE_MARGIN] for x, y, rad in self.OBS])
            # Insert obstacles in occupancy grid
            #!! Choose the right shape
            self._select_circle_occupants(obs)
            #self._select_rectangle_occupants(obs)

    def _select_circle_occupants(self, obs):
        """
        Method to insert obstacles in the occupancy grid as circles

        :param obs: obstacles as (x, y, radius)
        :type obs: list
        """
        # Set occupancy grid to 0 (everything free)
        occ_grid = np.zeros(self._res)
        # Get grid of discrete points of the map (x coordinate of each point will be in xx and y coordinate of each
        # point will be in yy)
        xx, yy = np.meshgrid(np.linspace(self.LIMIT[0,0], self.LIMIT[0,1], self._res[0]),
                                np.linspace(self.LIMIT[1,0], self.LIMIT[1,1], self._res[1]))
        # For every obstacle (get its coordinates)
        for xc, yc, r in obs:
            # Compute distance from the obstacle center to closest discrete world point
            dist = np.hypot(xx-xc, yy-yc)
            # Get coordinates which are inside the circle
            inside = dist < r
            # Set as obstacles each cell that falls inside the given circle
            occ_grid += inside.astype(int).T
        # Set occupied occupancy grid cells as obstacles
        self._occupancy_grid = occ_grid.astype(bool)


    def _select_rectangle_occupants(self, obs):
        """
        Method to insert obstacles in the occupancy grid as squares

        :param obs: obstacles as (x, y, radius)
        :type obs: list
        """
        # For every obstacle, get its coordinates and its radius 
        for xc, yc, r in obs:
            # Starting from x-r and going to x+r (each step is as big as the delta for the first coordinate)
            for x in np.arange(xc-r, xc+r, self.DELTA[0]):
                # Starting from y-r and going to y+r (each step is as big as the delta for the second coordinate)
                for y in np.arange(yc-r, yc+r, self.DELTA[1]):
                    # Get obstacle index given its position
                    i, j = self.pos_to_ind((x, y))
                    # Saturate i and j, withing their bounds (res[0] for x coordinate, res[1] for y coordinate)
                    i = max(0, min(i, self._res[0]-1))
                    j = max(0, min(j, self._res[1]-1))
                    # Set (i, j) cell of the occupancy grid as occupied
                    self._occupancy_grid[i,j] = True

    def __contains__(self, pos):
        """
        Function that given a position check if it is inside the world

        :param pos: position
        :type pos: list
        :return: true if position is inside the world
        :rtype: boolean
        """
        # # Set return value to true at the beginning
        # ret = 1
        # # Iterate over everything coordinate of the position, while sinchronously iterating over limits of the world
        # # (zip interates over both iterables in a synchronous way)
        # for x, (mn, mx) in zip(pos, self.LIMIT):
        #     # Check if current coordinate is inside the related world limits and bitwise and the result of this
        #     # operation with the current value of the return value
        #     ret &= int(mn <= x <= mx)
        # return bool(ret)
        ind = self.pos_to_ind(pos)
        return self.is_free_ind(ind)

    def __and__(self, other):
        """
        Function to execute the 'logic and' between two worlds

        :param other: other world
        :type other: AStarWorld
        :return: resulting world
        :rtype: World
        """
        # Assert if deltas of two different worlds are equal or not
        assert np.all(self.DELTA == other.DELTA), 'Different discretization'

        # Get minimum limit for every world dimension (between the two worlds)
        mns = np.minimum(self.LIMIT[:, 0], other.LIMIT[:, 0])
        # Get maximum limit for every world dimension (between the two worlds)
        mxs = np.maximum(self.LIMIT[:, 1], other.LIMIT[:, 1])

        # Create world given by the 'logic and' between this current world and the parameter one
        class World(AStarWorld):
            # Delta is the same for both worlds
            DELTA = self.DELTA
            # Transpose computed limits
            LIMIT = np.array([mns, mxs]).T
            # Resulting grid is given by the sovrapposition of the two occupancy grids (so logic or of them)
            _occupancy_grid = self._occupancy_grid | other._occupancy_grid
        # Return resulting world
        return World()

    def adjacent(self, ind):
        """
        Function to get adjacent discrete points, given a certain index

        :param ind: index of location
        :type ind: list
        :return: list of adjacent positions
        :rtype: list of lists
        """
        # Index as array
        v = np.asarray(ind)
        # Get +1 indexes and -1 indexes (e.g. ind=[10,0], +1_ind=[11,1], -1_ind=[9,-1])
        dv = np.array([v-1, v, v+1])            # get +1/-1 index
        adj = np.array(np.meshgrid(*dv.T))      # get combinations
        # Return adjacent positions (itself included)
        return adj.T.reshape(-1, v.shape[0])    # reshape into correct shape

    def is_free_ind(self, ind):
        """
        Function used to check if a cell (i, j coordinates) is free and inside the world's boundaries

        :param ind: index of the cell
        :type ind: (i, j) integers
        :return: true if cell is free, false if otherwise and if is outside the world's boundaries
        :rtype: boolean
        """
        # Retrieve i and j indexes from index passed as tuple
        i, j = ind[:2]
        # Return false if index is not inside the world boundaries
        if not 0 <= i < self._res[0]:
            return False
        if not 0 <= j < self._res[1]:
            return False
        # Return true if cell is free, false otherwise
        return not self._occupancy_grid[i,j]

    def pos_to_ind(self, pos):
        """
        Function used to get a cell's index given a geometrical position (round it to the closest cell)

        :param pos: position
        :type pos: (x, y)
        :return: cell corresponding to gievn position
        :rtype: (x, y)
        """
        return np.round((np.asarray(pos) - self.LIMIT[:, 0]) / self.DELTA).astype(int)

    def ind_to_pos(self, ind):
        """
        Function used to get position given a cell indexes

        :param ind: cell's indexes
        :type ind: (x, y)
        :return: position
        :rtype: (x, y)
        """
        return np.asarray(ind) * self.DELTA + self.LIMIT[:, 0]


class AStarPlanner(object):

    _world = None

    def __init__(self, *args, **kwargs):
        # Initialize world
        self._world = AStarWorld(*args, **kwargs)
        
    @property
    def world(self): return self._world

    def iter_new_children(self, node, goal):
        """
        Fucntion to get children of given node

        :param node: node
        :type node: AStarNode
        :yield: list of child nodes
        :rtype: list
        """
        for ind in self.world.adjacent(node.state):
            yield AStarNode(self.world, ind, node, dist(ind, goal))

    def _run(self, init_pos, goal_pos):
        """
        Function to run A* algorithm

        :raises Exception: no path was found
        :return: goal node if reached 
        :rtype: AStarNode
        """

        # Init cell (given position)
        init = self.world.pos_to_ind(init_pos)
        
        # Goal cell (given position)
        goal = self.world.pos_to_ind(goal_pos)


        # Initialize init node as an AStarNode
        init_node = AStarNode(self.world, init, None, dist(init, goal))

        # Initialize contour and visited set of cells
        contour = set()
        visited = set()

        # Add to the explored  set the init node
        contour.add(init_node)

        # Until there are cells to explore
        while contour:
            # Get the best cell (the one with the minimum f_cost)
            best = min(contour, key=lambda n: n.f_cost)
            # If f_cost of best cell is inf, then no path was found, stop for loop (and raise exception)
            if best.f_cost == float('inf'):
                break
            
            # Remove from explored set the best cell
            contour.remove(best)
            # Add best cell to the visited one
            visited.add(best)

            # If beste cell is in range to be considered as the goal, we got to the goal
            if dist(best.state, goal) <= 1:
                return best

            # For every child of the best cell
            for child in self.iter_new_children(best, goal):
                # If it is already visited, then skip it
                if child in visited:
                    continue
                # If it was already explored
                if child in contour:
                    # Get all other explored cells that are adjacent to the child, and remove current child
                    other = contour.intersection({child}).pop()
                    # If the child g_cost is less or equal than current child's neighboor g_cost, then remove it from explored
                    # cells and add the child to them
                    if child.g_cost <= other.g_cost:
                        # sets are funny pt.2
                        contour.remove(other)
                        contour.add(child)
                else:
                    # Otherwise (if child was not in visited nor in contour), add it to the explored cells
                    contour.add(child)
        raise Exception('Could not find a path')

    def plan(self, init_pos, goal_pos):
        """
        Function to retrieve the path from the init node to the goal 
        """
        try:
            final_node = self._run(init_pos, goal_pos)
            path = list(map(lambda n: n.pos, final_node.family_tree()))
        except Exception:
            path = []
        finally:
            return path


