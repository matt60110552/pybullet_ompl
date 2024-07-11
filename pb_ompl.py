try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import pybullet as p
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pb_ompl_utils
import time
from itertools import product
import copy
import numpy as np
from utils.utils import *

# INTERPOLATE_NUM = 500
INTERPOLATE_NUM = 30
DEFAULT_PLANNING_TIME = 5.0
# DEFAULT_PLANNING_TIME = 2.0

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id, joint_id=None, init_joint=None) -> None:
        # Public attributes
        self.id = id

        if joint_id is None:
            # prune fixed joints
            all_joint_num = p.getNumJoints(id)
            all_joint_idx = list(range(all_joint_num))
            joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
            self.num_dim = len(joint_idx)
            self.joint_idx = joint_idx
            print(self.joint_idx)
        else:
            self.num_dim = len(joint_id)
            self.joint_idx = joint_id
            print(self.joint_idx)
        self.joint_bounds = []
        self.reset(init_state=init_joint)

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self, init_state=[-0.15, -1.55, 1.8, -0.1, 1.8, 0.0, 0.0, 0.0, 0.0]):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        # state = [0] * self.num_dim
        # state = [-0.08, -1.25, 1.9, -0.1, 1.571, 0.0]
        self._set_joint_positions(self.joint_idx, init_state)
        self.state = init_state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, obstacles = [], joint_bounds=None) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        self.goal_state = None
        self.goal_mat = None
        self.sim_target_object_id = None
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        if joint_bounds is None:
            joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        # self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid_new))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("BITstar") # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if pb_ompl_utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if pb_ompl_utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True
    
    def is_state_valid_new(self, state):
        # check the state that near the goal state have appropriate end-effector orientation or not 
        
        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if pb_ompl_utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if pb_ompl_utils.pairwise_collision(body1, body2):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        
        pos, orn = p.getLinkState(self.robot_id, 7)[4:6]
        if self.sim_target_object_id is not None:
            # Check if the sim_object is collide with the obstacle
            target_pos, target_orn = p.multiplyTransforms(
                pos, orn,
                self.relative_pos, self.relative_orn
            )
            p.resetBasePositionAndOrientation(self.sim_target_object_id, target_pos, target_orn)
            for obstacle in self.obstacles:
                if np.linalg.norm(pos - self.goal_mat[:3, 3]) > 0.07 and len(p.getClosestPoints(bodyA=self.sim_target_object_id,
                                                                                                bodyB=obstacle, distance=0.)) != 0:  # getContactPoints
                    return False
        # Check the config's gripper is pointing to the target or not
        if self.goal_state is not None:
            self.robot_id
            cur_mat = np.eye(4)
            cur_mat[:3, :3] = quat2mat(tf_quat(orn))
            cur_mat[:3, 3] = pos
            if pb_ompl_utils.approaching(state, self.goal_state, cur_mat, self.goal_mat):
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions = True, allow_collision_links = []):
        self.check_link_pairs = pb_ompl_utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in pb_ompl_utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time = DEFAULT_PLANNING_TIME, interpolate_num=INTERPOLATE_NUM):
        '''
        plan a path to gaol from the given robot start state
        '''
        print("start_planning")
        print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(allowed_time)
        res = False
        sol_path_list = []
        sol_elbow_pos_list = []
        sol_gripper_pos_list = []
        sol_gripper_orn_list = []
        if solved:
            print("Found solution: interpolating into {} segments".format(interpolate_num))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()
            sol_path_geometric.interpolate(interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            # print(len(sol_path_list))
            # print(sol_path_list)
            for sol_path in sol_path_list:
                self.is_state_valid_new(sol_path)
                sol_elbow_pos_list.append(p.getLinkState(self.robot_id, 5)[4])
                sol_gripper_pos_list.append(p.getLinkState(self.robot_id, 7)[4]) # 7 is the num for gripper
                sol_gripper_orn_list.append(p.getLinkState(self.robot_id, 7)[5])
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        return res, sol_path_list, sol_elbow_pos_list, sol_gripper_pos_list, sol_gripper_orn_list

    def plan(self, goal, goal_mat = None,
             allowed_time = DEFAULT_PLANNING_TIME,
             interpolate_num=INTERPOLATE_NUM,
             sim_target_object_id=None,
             relative_pos=None,
             relative_orn=None):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        self.goal_state = goal
        self.goal_mat = goal_mat
        self.sim_target_object_id = sim_target_object_id
        self.relative_pos = relative_pos
        self.relative_orn = relative_orn
        return self.plan_start_goal(start, goal, allowed_time=allowed_time, interpolate_num=interpolate_num)

    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i],force=5 * 240.)
            else:
                self.robot.set_state(q)
            p.stepSimulation()
            time.sleep(0.01)



    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]