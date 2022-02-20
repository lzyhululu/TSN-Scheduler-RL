"""gridworld interface"""

import ctypes
import os
import importlib

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class GridWorld(Environment):
    # constant

    def __init__(self, config, **kwargs):
        """
        Parameters
        ----------
        config: str or Config Object
            if config is a string, then it is a name of stored config,
            config are stored in envs/configs
            kwargs are the arguments to the config
            if config is a Config Object, then parameters are stored in that object
        """
        Environment.__init__(self)

        # if is str, load stored configuration
        if isinstance(config, str):
            # config are stored in ./configs
            try:
                demo_game = importlib.import_module('envs.configs.' + config)
                config = getattr(demo_game, 'get_config')(**kwargs)
            except AttributeError:
                raise Exception('unknown config:"' + config + '"')
        # save flow id and info  remained to do
        self.flow_info = {}
        self.ordered_cycle = config.ordered_cycle
        # create new game
        game = ctypes.c_void_p()
        _LIB.env_new_game(ctypes.byref(game), b"GridWorld")
        self.game = game

        # set global configuration
        config_value_type = {
            'nodes_num': int, 'global_cycle': int,
            'embedding_size': int,
            'render_dir': str,
        }

        for key in config.config_dict:
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(self.game, key.encode("ascii"),
                                     ctypes.byref(ctypes.c_int(config.config_dict[key])))
            elif value_type is bool:
                _LIB.env_config_game(self.game, key.encode("ascii"),
                                     ctypes.byref(ctypes.c_bool(config.config_dict[key])))
            elif value_type is float:
                _LIB.env_config_game(self.game, key.encode("ascii"),
                                     ctypes.byref(ctypes.c_float(config.config_dict[key])))
            elif value_type is str:
                _LIB.env_config_game(self.game, key.encode("ascii"),
                                     ctypes.c_char_p(config.config_dict[key]))

        # register agent types
        for name in config.agent_type_dict:
            type_args = config.agent_type_dict[name]
            length = len(type_args)
            keys = (ctypes.c_char_p * length)(*[key.encode("ascii") for key in type_args.keys()])
            values = (ctypes.c_float * length)(*type_args.values())

            _LIB.gridworld_register_agent_type(self.game, name.encode("ascii"), length, keys, values)

        # serialize event expression, send to C++ engine
        self._serialize_event_exp(config)

        # init group handles
        self.group_handles = []
        for item in config.groups:
            handle = ctypes.c_int32()
            _LIB.gridworld_new_group(self.game, item.encode("ascii"), ctypes.byref(handle))
            self.group_handles.append(handle)

        # init observation buffer (for acceleration)
        self._init_obs_buf()

        # init view space, feature space, action space
        self.view_space = {}
        self.feature_space = {}
        self.action_space = {}
        buf = np.empty((3,), dtype=np.int32)
        for handle in self.group_handles:
            _LIB.env_get_info(self.game, handle, b"view_space",
                              buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            # -1 for total map
            self.view_space[handle.value] = (buf[0],)
            _LIB.env_get_info(self.game, handle, b"feature_space",
                              buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.feature_space[handle.value] = (buf[0],)
            _LIB.env_get_info(self.game, handle, b"action_space",
                              buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.action_space[handle.value] = (buf[0],)

    def reset(self):
        """reset environment"""
        _LIB.env_reset(self.game)

    # ====== AGENT ======
    def new_group(self, name):
        """register a new group into environment"""
        handle = ctypes.c_int32()
        _LIB.gridworld_new_group(self.game, ctypes.c_char_p(name.encode("ascii")), ctypes.byref(handle))
        return handle

    def add_agents(self, handle, delay, pkt_length, n, routes, offsets):
        """add agents to environment"""
        # copy again, to make these arrays continuous in memory
        routes, offsets = np.array(routes, dtype=np.int32), np.array(offsets, dtype=np.int32)
        _LIB.gridworld_add_agents(self.game, handle, delay, pkt_length, n,
                                  as_int32_c_array(routes), as_int32_c_array(offsets))

    # ====== RUN ======
    def _get_obs_buf(self, group, key, shape, dtype):
        """
        get buffer to receive observation from c++ engine
        first one is view_buffer
        second is feature_buffer
        """
        obs_buf = self.obs_bufs[key]
        if group in obs_buf:
            ret = obs_buf[group]
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
        else:
            ret = obs_buf[group] = np.empty(shape=shape, dtype=dtype)

        return ret

    def _init_obs_buf(self):
        """init observation buffer"""
        self.obs_bufs = []
        self.obs_bufs.append({})
        self.obs_bufs.append({})

    def get_observation(self, handle):
        """ get observation of a whole group

        Parameters
        ----------
        handle : group handle

        Returns
        -------
        obs : tuple (views, features)
            views is a numpy array, whose shape is nodes_num * global_cycle * n_channel(1)
            features is a numpy array, whose shape is agents_num *  feature_size
            feature_size = embedding_size(id size) + offsets(nodes_num) + reward(1)
            for agent i, (views, features[i]) is its observation at this step
        """
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        n = self.get_num(handle)
        view_buf = self._get_obs_buf(no, 0, (1,) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, 1, (n,) + feature_space, np.float32)

        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)
        _LIB.env_get_observation(self.game, handle, bufs)
        a = id(view_buf)
        return view_buf, feature_buf

    def set_action(self, handle, actions):
        """ set actions for whole group

        Parameters
        ----------
        handle: group handle
        actions: numpy array
            the dtype of actions must be float
        """
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.float32
        _LIB.env_set_action(self.game, handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def step(self):
        """simulation one step after set actions

        Returns
        -------
        done: bool
            whether the game is done
        """
        done = ctypes.c_int32()
        _LIB.env_step(self.game, ctypes.byref(done))
        return bool(done)

    def get_reward(self, handle):
        """ get reward for a whole group

        Returns
        -------
        rewards: numpy array (float32)
            reward for all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.float32)
        _LIB.env_get_reward(self.game, handle,
                            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    # ====== INFO ======
    def get_handles(self):
        """ get all group handles in the environment """
        return self.group_handles

    def get_num(self, handle):
        """ get the number of agents in a group"""
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b'num', ctypes.byref(num))
        return num.value

    def get_action_space(self, handle):
        """get action space

        Returns
        -------
        action_space : tuple
        """
        return self.action_space[handle.value]

    def get_view_space(self, handle):
        """get view space

        Returns
        -------
        view_space : tuple
        """
        return self.view_space[handle.value]

    def get_feature_space(self, handle):
        """ get feature space

        Returns
        -------
        feature_space : tuple
        """
        return self.feature_space[handle.value]

    def get_agent_id(self, handle):
        """ get agent id

        Returns
        -------
        ids : numpy array (int32)
            id of all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n,), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"id",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def get_pos(self, handle):
        """ get position of agents in a group

        Returns
        -------
        pos: numpy array (int)
            the shape of pos is (n, 2)
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"pos",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def set_seed(self, seed):
        """ set random seed of the engine"""
        _LIB.env_config_game(self.game, b"seed", ctypes.byref(ctypes.c_int(seed)))

    # ====== RENDER ======
    def set_render_dir(self, name):
        """ set directory to save render file"""
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name.encode("ascii"))

    def render(self):
        """ render a step """
        _LIB.env_render(self.game)

    def _get_groups_info(self):
        """ private method, for interactive application"""
        n = len(self.group_handles)
        buf = np.empty((n, 5), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"groups_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    def _get_render_info(self, x_range, y_range):
        """ private method, for interactive application"""
        n = 0
        for handle in self.group_handles:
            n += self.get_num(handle)

        buf = np.empty((n+1, 4), dtype=np.int32)
        buf[0] = x_range[0], y_range[0], x_range[1], y_range[1]
        _LIB.env_get_info(self.game, -1, b"render_window_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

        # the first line is for the number of agents in the window range
        info_line = buf[0]
        agent_ct, attack_event_ct = info_line[0], info_line[1]
        buf = buf[1:1 + info_line[0]]

        agent_info = {}
        for item in buf:
            agent_info[item[0]] = [item[1], item[2], item[3]]

        buf = np.empty((attack_event_ct, 3), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"attack_event",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        attack_event = buf

        return agent_info, attack_event

    def __del__(self):
        _LIB.env_delete_game(self.game)

    # ====== PRIVATE ======
    def _serialize_event_exp(self, config):
        """serialize event expression and sent them to game engine"""
        game = self.game

        # collect agent symbol
        symbol2int = {}
        config.symbol_ct = 0

        for rule in config.reward_rules:
            receiver = rule[1]
            for symbol in receiver:
                if symbol not in symbol2int:
                    symbol2int[symbol] = config.symbol_ct
                    config.symbol_ct += 1

        # send to C++ engine
        for sym in symbol2int:
            no = symbol2int[sym]
            _LIB.gridworld_define_agent_symbol(game, no, sym.group, sym.index)

        for rule in config.reward_rules:
            # rule = [on, receiver, value, terminal]
            op = rule[0]

            receiver = np.zeros_like(rule[1], dtype=np.int32)
            for i, item in enumerate(rule[1]):
                receiver[i] = symbol2int[item]
            if len(rule[2]) == 1 and rule[2][0] == 'auto':
                value = np.zeros(receiver, dtype=np.float32)
            else:
                value = np.array(rule[2], dtype=np.float32)
            n_receiver = len(receiver)
            _LIB.gridworld_add_reward_rule(game, op, as_int32_c_array(receiver),
                                           as_float_c_array(value), n_receiver, rule[3])


'''
the following classes are for reward description
'''


class AgentSymbol:
    """symbol to represent some agents"""
    def __init__(self, group, index):
        """ define a agent symbol, it can be the object or subject of EventNode

        group: group handle
            it is the return value of cfg.add_group()
        index: int or str
            int: a deterministic integer id
            str: can be 'all', represents all agents in a group
        """
        self.group = group if group is not None else -1
        if index == 'all':
            self.index = -1
        else:
            assert isinstance(self.index, int), "index must be a deterministic int"
            self.index = index

    def __str__(self):
        return 'agent(%d,%d)' % (self.group, self.index)


class Config:
    """configuration class of gridworld game

    Constraints:

    Basic constraint ( by default ):
        [1] All messages are sent at least once in the cluster cycle.
        [2] The start and end time of the message must be within the cycle.

    Reward-define constraints:
        OP_COLLIDE: During the cluster cycle, tany two business flows will never overlap.
        OP_E2E_DELAY: Constrain the maximum end-to-end delay of the message

    """
    OP_COLLIDE = 0
    OP_E2E_DELAY = 1

    # can extend more operation below

    def __init__(self):
        self.config_dict = {}
        self.agent_type_dict = {}
        self.groups = []
        self.reward_rules = []
        self.ordered_cycle = []

    def set(self, args):
        """ set parameters of global configuration

        Parameters
        ----------
        args : dict
            key value pair of the configuration
        """
        for key in args:
            self.config_dict[key] = args[key]

    def register_agent_type(self, name, attr):
        """ register an agent type

        Parameters
        ----------
        name : str
            name of the type (should be unique)
        attr: dict
            key value pair of the agent type
            see notes below to know the available attributes

        Notes
        -----
        height: int, height of agent body
        cycle:  int, time_slots of agent cycle

        step_reward: float, reward get in every step
        """
        if name in self.agent_type_dict:
            raise Exception("type name %s already exists" % name)
        self.agent_type_dict[name] = attr
        return name

    def add_group(self, agent_type):
        """ add a group to the configuration

        Returns
        -------
        group_handle : int
            a handle for the new added group
        """
        no = len(self.groups)
        self.groups.append(agent_type)
        return no

    def add_reward_rule(self, str_op, receiver, value, terminal=False):
        """ add a reward rule

        Some note:
        1. if the receiver is not a deterministic agent,
           it must be one of the agents involved in the triggering event

        Parameters
        ----------
        str_op: the rule index
        receiver:  (list of) AgentSymbol
            receiver of this reward rule
        value: (list of) float
            value to assign
        terminal: bool
            whether this game will terminate, all the rules should be set to true

        """
        if not (isinstance(receiver, tuple) or isinstance(receiver, list)):
            assert not (isinstance(value, tuple) or isinstance(value, list))
            receiver = [receiver]
            value = [value]
        if len(receiver) != len(value):
            raise Exception("the length of receiver and value should be equal")
        self.reward_rules.append([self.op_change(str_op), receiver, value, terminal])

    def op_change(self, str_op):
        if str_op == 'no_collide':
            int_op = self.OP_COLLIDE
        elif str_op == 'e2e_delay':
            int_op = self.OP_E2E_DELAY
        else:
            raise Exception("invalid predicate of event " + str_op)
        return int_op


if __name__ == '__main__':
    env = GridWorld("first_demo", global_cycle=4096, nodes_num=6)
