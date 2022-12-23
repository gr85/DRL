from enum import Enum
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import warnings
import numpy as np
from collections import deque, namedtuple
import json
# import gym
import gymnasium as gym
import torch


class experienceReplayBuffer():        
    def __init__(self, envs, memory_size=50000, burn_in=10000):
        '''
        Experience Replay Buffer
        ========================
    
        Used to store some experiencies to be used in the update of the networks to assure the i.i.d. (independently and identically distributed) samples
        
        :param envs: Vectorized environment
        :param memory_size: maximum capacity of the buffer. When it is reached the oldest samples are deleted and replaced for new ones
        :param burn_in: number of samples to consider the buffer is initialized with the required amount of samples        
        :return: returns nothing
        '''
        self.burn_in = burn_in
        self.envs = envs
        
        self.mem_size = memory_size
        self.mem_cntr = 0
        state_shape = 0
        if (isinstance(envs.single_observation_space, gym.spaces.Dict)):
            for key, _obs_shape in envs.single_observation_space.items():
                state_shape = state_shape + _obs_shape.shape[-1]
        else:
            state_shape = envs.single_observation_space.shape[-1]
        self.state_memory = np.zeros((self.mem_size, state_shape), dtype=float)
        self.new_state_memory = np.zeros((self.mem_size, state_shape), dtype=float)
            
        
        self.action_memory = np.zeros((self.mem_size, self.envs.single_action_space.shape[-1]), dtype=float)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=float)
        self.terminal_memory = np.zeros((self.mem_size, 1), dtype=float)
                 
    def get_obs_shape(self, observation_space: gym.spaces.Space,) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        """
        Get the shape of the observation (useful for the buffers).
        :param observation_space:
        :return:
        """
        if isinstance(observation_space, gym.spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, gym.spaces.Discrete):
            # Observation is an int
            return (1,)
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            # Number of discrete features
            return (int(len(observation_space.nvec)),)
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            # Number of binary features
            return (int(observation_space.n),)
        elif isinstance(observation_space, gym.spaces.Dict):
            return {key: self.get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    def sample_batch(self, batch_size=32):
        '''
        Gets a random batch samples from the buffer
        
        :param batch_size: number of random samples to take from the buffer
        :return: a random batch from the buffer
        '''
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones        

    def append(self, state, action, reward, next_state, done):
        '''
        Adds a sample to the buffer
        
        :param state: actual state to add
        :param action: current action to add
        :param reward: reward to add based on the action took
        :param done: if the next state is terminal or not
        :param next_state: next state to add after the action is applied to the environment
        :return: returns nothing
        '''
        
        if (isinstance(self.envs.observation_space, gym.spaces.Dict)):
            for s,des,a,r,act,nso,nsd,nsa,d in zip(state["observation"], state["desired_goal"], state["achieved_goal"], reward, action,
                        next_state["observation"], next_state["desired_goal"], next_state["achieved_goal"], done):
                index = self.mem_cntr % self.mem_size
                # state = dict(sorted(state.items()))
                # state = list(chain(*state.values()))
                self.state_memory[index] = np.concatenate([s, des], axis=1) #state             
                # next_state = dict(sorted(next_state.items()))
                # next_state = list(chain(*next_state.values()))
                self.new_state_memory[index] = np.concatenate([nso, nsd], axis=1) #next_state   
                self.mem_cntr += 1
        else:
            for s,r,act,ns,d in zip(state, reward, action, next_state, done):
                index = self.mem_cntr % self.mem_size
                self.state_memory[index] = s
                self.new_state_memory[index] = ns       
                self.action_memory[index] = act
                self.reward_memory[index] = r        
                self.terminal_memory[index] = d

                self.mem_cntr += 1        
           
    def burn_in_capacity(self):
        '''
        Checks if the buffer is filled with enough samples for the burn in process
        
        :return: percentage of the memory based on the burn in requirement (if less than 1 the burn in is not finished)
        '''
    
        return self.mem_cntr / self.burn_in
    
    def save_checkpoint(self, folder = 'tmp_extended/checkpoints'):        
        with open(folder + '/replay_buffer_state.json', 'w') as wf:            
            json.dump({"state": self.state_memory.tolist()}, wf)
        with open(folder + '/replay_buffer_next_state.json', 'w') as wf:
            json.dump({"next_state": self.new_state_memory.tolist()}, wf)
        with open(folder + '/replay_buffer_action.json', 'w') as wf:
            json.dump({"action": self.action_memory.tolist()}, wf)
        with open(folder + '/replay_buffer_reward.json', 'w') as wf:
            json.dump({"reward": self.reward_memory.tolist()}, wf)
        with open(folder + '/replay_buffer_done.json', 'w') as wf:
            json.dump({"done": self.terminal_memory.tolist()}, wf)
        with open(folder + '/replay_buffer_mem_cntr.json', 'w') as wf:
            json.dump({"mem_cntr": self.mem_cntr}, wf)
    
    def load_checkpoint(self, folder='tmp_extended'):
        with open(folder + '/checkpoints/replay_buffer_state.json') as rf:
            dict_res = json.load(rf)
            self.state_memory = np.array(dict_res['state'])
        with open(folder + '/checkpoints/replay_buffer_next_state.json') as rf:
            dict_res = json.load(rf)
            self.new_state_memory = np.array(dict_res['next_state'])
        with open(folder + '/checkpoints/replay_buffer_action.json') as rf:
            dict_res = json.load(rf)
            self.action_memory = np.array(dict_res['action'])
        with open(folder + '/checkpoints/replay_buffer_reward.json') as rf:
            dict_res = json.load(rf)
            self.reward_memory = np.array(dict_res['reward'])
        with open(folder + '/checkpoints/replay_buffer_done.json') as rf:
            dict_res = json.load(rf)
            self.terminal_memory = np.array(dict_res['done'])
        with open(folder + '/checkpoints/replay_buffer_mem_cntr.json') as rf:
            dict_res = json.load(rf)
            self.mem_cntr = int(dict_res['mem_cntr'])        



class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """

    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    "future": GoalSelectionStrategy.FUTURE,
    "final": GoalSelectionStrategy.FINAL,
    "episode": GoalSelectionStrategy.EPISODE,
}

class HERBuffer():
    """
    ----- Code taken from: 
            https://github.com/qgallouedec/stable-baselines3/blob/684364beddc53d206db38770db222aad1c599282/stable_baselines3/her/her_replay_buffer.py
        and adapted to work with gymnasium package
    ----- 
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495
    .. warning::
      For backward compatibility, we implement offline sampling. The offline
      sampling mode only works for `n_envs == 1`.
    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param online_sampling: If False, virtual transitions are saved in the replay buffer.
        Only works for `n_envs == 1`.
    """

    def __init__(self, env, memory_size: int, goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future", n_sampled_goal: int = 4,
                 env_id = 'PandaPickAndPlace-v3') -> None:
        self.env = env
        self.buffer_size = memory_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.full = False
        self.pos = 0
        self.episode = 0
        self.single_env = gym.make(env_id)

        self.obs_shape = self.get_space_shape(env.single_observation_space)
        action_shape = self.get_space_shape(env.single_action_space)

        self.states = np.zeros((self.buffer_size,) + self.obs_shape["observation"], dtype=env.single_observation_space.dtype)
        self.actions = np.zeros((self.buffer_size,) + action_shape, dtype=env.single_observation_space.dtype)
        # Use 3 dims for easier calculations without having to think about broadcasting
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.desired_goals = np.zeros((self.buffer_size,) + self.obs_shape["desired_goal"], dtype=env.single_observation_space.dtype)
        self.next_achieved_goals = np.zeros((self.buffer_size,) + self.obs_shape["achieved_goal"], dtype=env.single_observation_space.dtype)

        # Keep track of where in the data structure episodes end
        self.episode_end_indices = np.zeros((self.buffer_size), dtype=np.uint32)
        # Keep track of which transitions belong to which episodes.
        self.index_episode_map = np.zeros((self.buffer_size), dtype=np.uint32)

        if isinstance(goal_selection_strategy, str):
            self.goal_section_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_section_strategy = goal_selection_strategy

        self.her_ratio = 1 - (1.0 / (n_sampled_goal + 1))
        
    def get_space_shape(self, observation_space: gym.spaces.Space,) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        """
        Get the shape of the observation (useful for the buffers).
        :param observation_space:
        :return:
        """
        if isinstance(observation_space, gym.spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, gym.spaces.Discrete):
            # Observation is an int
            return (1,)
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            # Number of discrete features
            return (int(len(observation_space.nvec)),)
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            # Number of binary features
            return (int(observation_space.n),)
        elif isinstance(observation_space, gym.spaces.Dict):
            return {key: self.get_space_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
        
    def append(self, state: Dict[str, np.ndarray], action: Union[np.ndarray, int], reward: float, next_state: Dict[str, np.ndarray], done:Dict[str, np.ndarray]) -> None:
        for s,des,a,r,act,nso,nsd,nsa,d in zip(state["observation"], state["desired_goal"], state["achieved_goal"], reward, action,
                         next_state["observation"], next_state["desired_goal"], next_state["achieved_goal"], done):
            self.states[self.pos] = np.array(s) # state["observation"]
            self.desired_goals[self.pos] = np.array(des) # state["desired_goal"]
            self.rewards[self.pos] = np.array(r) # reward
            self.actions[self.pos] = np.array(act) # action
            self.states[(self.pos + 1) % self.buffer_size] = np.array(nso) # next_state["observation"]
            self.next_achieved_goals[self.pos] = np.array(nsa) # next_state["achieved_goal"]
            self.dones[self.pos] = np.array(d) # done
            self.index_episode_map[self.pos] = self.episode
            self.pos += 1

            if d: # done:
                self.episode_end_indices[self.episode] = self.pos
                self.episode += 1

                if self.episode == self.buffer_size:
                    self.episode = 0

            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
    
    def check_full(self) -> bool:
        return self.pos == self.buffer_size
            
    def sample_batch(self, batch_size: int):

        # Sample transitions from the last complete episode recorded
        end_idx = self.episode_end_indices[self.episode - 1]
        if self.check_full(): #self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + end_idx) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, end_idx, size=batch_size)

        return self._sample_trajectories(batch_size, batch_inds)
    
    def _sample_trajectories(self, batch_size: int, batch_inds: np.ndarray):
        """
        Get the trajectories based on batch indices calculated
        :param batch_size: number of elements to sample, included to reduce processing instead of using `len(batch_inds)`
        :param batch_inds: the indices of the elements to sample
        """
        her_batch_size = int(batch_size * self.her_ratio)

        # Separate HER and replay batch indices
        her_inds = batch_inds[:her_batch_size]
        replay_inds = batch_inds[her_batch_size:]

        her_goals = self._sample_goals(her_inds)
        # the new state depends on the previous state and action
        # s_{t+1} = f(s_t, a_t)
        # so the next_achieved_goal depends also on the previous state and action
        # because we are in a GoalEnv:
        # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
        # therefore we have to use "next_achieved_goal" and not "achieved_goal"
        
        her_rewards = self.single_env.compute_reward(self.next_achieved_goals[her_inds], her_goals, {}).reshape(-1,1)

        desired_goals = np.concatenate([her_goals, self.desired_goals[replay_inds]])
        rewards = np.concatenate([her_rewards, self.rewards[replay_inds]])
        states = np.concatenate([self.states[batch_inds], desired_goals], axis=1)
        states_ = np.concatenate([self.states[(batch_inds + 1) % self.buffer_size], desired_goals], axis=1)
        actions = self.actions[batch_inds]
        dones = self.dones[batch_inds]

        return states, actions, rewards, states_, dones
    
    def _sample_goals(self, her_inds: np.ndarray) -> np.ndarray:
        """
        Sample new episode goals to calculate rewards from
        :param her_inds: the batch indices designated for new goal sampling
        :return: the new episode goals
        """
        her_episodes = self.index_episode_map[her_inds]
        episode_end_indices = self.episode_end_indices[her_episodes]

        # Goal is the last state in the episode
        if self.goal_section_strategy == GoalSelectionStrategy.FINAL:
            goal_indices = episode_end_indices - 1

        # Goal is a random state in the same episode observed after current transition
        elif self.goal_section_strategy == GoalSelectionStrategy.FUTURE:
            # FAILURE MODE: if episode overlaps from end to beginning of buffer
            # then the randint method will likely fail due to low > high.
            # This will only happen at the overlapping episodes so quick
            # fix is to simply revert to final goal strategy in this case.
            check = False
            for i,_ in enumerate(episode_end_indices):
                if episode_end_indices[i] <= her_inds[i]:
                    # print('CUIDADUUUUUUUUUUUUUU')
                    check = True
            if check: #any(episode_end_indices < her_inds):
                goal_indices = episode_end_indices - 1
                for i,_ in enumerate(episode_end_indices):
                    if episode_end_indices[i] == 0:
                        goal_indices[i] = self.buffer_size-1
            else:
                goal_indices = np.random.randint(her_inds, episode_end_indices)

        else:
            raise ValueError(
                f"Strategy {self.goal_section_strategy} for samping goals not supported."
            )
        '''ONLY FOR DEBUG!!!'''
        debug = False
        for i,a in enumerate(goal_indices):
            if goal_indices[i] > self.buffer_size:
                print(f'HER INDX: {her_inds[i]} ; EPS INDX: {episode_end_indices[i]} ; GOAL INDX: {goal_indices[i]}')
                debug = True
        if debug: #any(goal_indices > self.buffer_size):
            print(f"Index Overflow: {[a for a in goal_indices if a > self.buffer_size]}")
            print(f"HER Indices: {her_inds}")
            print(f"Index Episode Map: {self.index_episode_map}")
            print(f"Episode End Indices: {episode_end_indices}")

        return self.next_achieved_goals[goal_indices]
        
    def save_checkpoint(self, folder = 'tmp_extended/checkpoints'):        
        with open(folder + '/replay_buffer_state.json', 'w') as wf:            
            json.dump({"state": self.states.tolist()}, wf)
        with open(folder + '/replay_buffer_desired_goals.json', 'w') as wf:
            json.dump({"desired_goals": self.desired_goals.tolist()}, wf)
        with open(folder + '/replay_buffer_next_achieved_goals.json', 'w') as wf:
            json.dump({"next_achieved_goals": self.next_achieved_goals.tolist()}, wf)
        with open(folder + '/replay_buffer_action.json', 'w') as wf:
            json.dump({"action": self.actions.tolist()}, wf)
        with open(folder + '/replay_buffer_reward.json', 'w') as wf:
            json.dump({"reward": self.rewards.tolist()}, wf)
        with open(folder + '/replay_buffer_done.json', 'w') as wf:
            json.dump({"done": self.dones.tolist()}, wf)
        with open(folder + '/replay_buffer_episode_end_indices.json', 'w') as wf:
            json.dump({"episode_end_indices": self.episode_end_indices.tolist()}, wf)            
        with open(folder + '/replay_buffer_index_episode_map.json', 'w') as wf:
            json.dump({"index_episode_map": self.index_episode_map.tolist()}, wf)            
        with open(folder + '/replay_buffer_mem_cntr.json', 'w') as wf:
            json.dump({"mem_cntr": self.pos, "episode": self.episode}, wf)
    
    def load_checkpoint(self, folder='tmp_extended'):
        with open(folder + '/checkpoints/replay_buffer_state.json') as rf:
            dict_res = json.load(rf)
            self.states = np.array(dict_res['state'])
        with open(folder + '/checkpoints/replay_buffer_desired_goals.json') as rf:
            dict_res = json.load(rf)
            self.desired_goals = np.array(dict_res['desired_goals'])
        with open(folder + '/checkpoints/replay_buffer_next_achieved_goals.json') as rf:
            dict_res = json.load(rf)
            self.next_achieved_goals = np.array(dict_res['next_achieved_goals'])
        with open(folder + '/checkpoints/replay_buffer_action.json') as rf:
            dict_res = json.load(rf)
            self.actions = np.array(dict_res['action'])
        with open(folder + '/checkpoints/replay_buffer_reward.json') as rf:
            dict_res = json.load(rf)
            self.rewards = np.array(dict_res['reward'])
        with open(folder + '/checkpoints/replay_buffer_done.json') as rf:
            dict_res = json.load(rf)
            self.dones = np.array(dict_res['done'])
        with open(folder + '/checkpoints/replay_buffer_episode_end_indices.json') as rf:
            dict_res = json.load(rf)
            self.episode_end_indices = np.array(dict_res['episode_end_indices'])            
        with open(folder + '/checkpoints/replay_buffer_index_episode_map.json') as rf:
            dict_res = json.load(rf)
            self.index_episode_map = np.array(dict_res['index_episode_map'])            
        with open(folder + '/checkpoints/replay_buffer_mem_cntr.json') as rf:
            dict_res = json.load(rf)
            self.pos = int(dict_res['mem_cntr'])
            self.episode = int(dict_res['episode'])        









def get_time_limit(env, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.
    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError as e:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            ) from e
    return current_max_episode_length

TensorDict = Dict[Union[str, int], torch.Tensor]
class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor

class HerReplayBuffer():
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495
    .. warning::
      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the replay buffer constructor.
    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.
    :param env: The training environment
    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param device: PyTorch device
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        env,
        buffer_size: int,
        device: Union[torch.device, str] = "auto",
        replay_buffer = None,
        max_episode_length = None,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
        env_id = None
    ):
        self.single_env = gym.make(env_id)
        # convert goal_selection_strategy into GoalSelectionStrategy if string
        if isinstance(goal_selection_strategy, str):
            self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        else:
            self.goal_selection_strategy = goal_selection_strategy

        # check if goal_selection_strategy is valid
        assert isinstance(
            self.goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"

        self.n_sampled_goal = n_sampled_goal
        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = buffer_size if online_sampling else self.max_episode_length

        self.env = env
        self.buffer_size = her_buffer_size

        if online_sampling:
            replay_buffer = None
        self.replay_buffer = replay_buffer
        self.online_sampling = online_sampling

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = self.get_obs_shape(self.env.observation_space.spaces["observation"])
        self.goal_shape = self.get_obs_shape(self.env.observation_space.spaces["achieved_goal"])
        
        self.action_dim = self.get_obs_shape(self.env.single_action_space)
        
        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.env.num_envs,) + self.obs_shape,
            "achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "desired_goal": (self.env.num_envs,) + self.goal_shape,
            "action": (self.env.num_envs,) + self.action_dim, #(self.action_dim[0],), #
            "reward": (1,), #(self.env.num_envs,), #
            "next_obs": (self.env.num_envs,) + self.obs_shape,
            "next_achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
            "done": (1,), #(self.env.num_envs,), #
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)
        
    def get_obs_shape(self, observation_space: gym.spaces.Space,) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        """
        Get the shape of the observation (useful for the buffers).
        :param observation_space:
        :return:
        """
        if isinstance(observation_space, gym.spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, gym.spaces.Discrete):
            # Observation is an int
            return (1,)
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            # Number of discrete features
            return (int(len(observation_space.nvec)),)
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            # Number of binary features
            return (int(observation_space.n),)
        elif isinstance(observation_space, gym.spaces.Dict):
            return {key: self.get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
        elif isinstance(self.env.action_space, gym.spaces.Tuple):
            return int(observation_space[0].shape)
   

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.
        Excludes self.env, as in general Env's may not be pickleable.
        Note: when using offline sampling, this will also save the offline replay buffer.
        """
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["env"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.
        User must call ``set_env()`` after unpickling before using.
        :param state:
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None

    def set_env(self, env) -> None:
        """
        Sets the environment.
        :param env:
        """
        if self.env is not None:
            raise ValueError("Trying to set env of already initialized environment.")

        self.env = env

    def _get_samples(self, batch_inds: np.ndarray, env = None):
        """
        Abstract method from base class.
        """
        raise NotImplementedError()

    def sample(self, batch_size: int, env = None):
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.
        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        if self.replay_buffer is not None:
            return self.replay_buffer.sample(batch_size, env)
        return self._sample_transitions(batch_size, maybe_vec_env=env, online_sampling=True)  # pytype: disable=bad-return-type

    def _sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.
        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # `maybe_vec_env=None` as we should store unnormalized transitions,
        # they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )  # pytype: disable=bad-return-type

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.
        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices], self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self._buffer["next_achieved_goal"][her_episode_indices, transitions_indices]

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env,
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ):
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

        ep_lengths = self.episode_lengths[episode_indices]

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            print(f'next_achieved_goal: {transitions["next_achieved_goal"][her_indices, 0].shape}')
            print(f'desired_goal: {transitions["desired_goal"][her_indices, 0].shape}')
            print(f'Rewards Before: {transitions["reward"][her_indices, 0].shape}')
            a= []
            a = self.single_env.compute_reward(transitions["next_achieved_goal"][her_indices, 0],transitions["desired_goal"][her_indices, 0],{})
            print(f'Rewards Calculated: {a.shape}')
            transitions["reward"][her_indices, 0] = self.single_env.compute_reward(transitions["next_achieved_goal"][her_indices, 0],
                                                                                   transitions["desired_goal"][her_indices, 0], 
                                                                                   {})
            print(f'Rewards After: {transitions["reward"][her_indices, 0]}')
            # transitions["reward"][her_indices, 0] = self.env.env_method(
            #     "compute_reward",
            #     # the new state depends on the previous state and action
            #     # s_{t+1} = f(s_t, a_t)
            #     # so the next_achieved_goal depends also on the previous state and action
            #     # because we are in a GoalEnv:
            #     # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            #     # therefore we have to use "next_achieved_goal" and not "achieved_goal"
            #     transitions["next_achieved_goal"][her_indices, 0],
            #     # here we use the new desired goal
            #     transitions["desired_goal"][her_indices, 0],
            #     transitions["info"][her_indices, 0],
            # )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            return DictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            )
        else:
            return observations, next_observations, transitions["action"], transitions["reward"]

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if any(done)==True or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            if not self.online_sampling:
                # sample virtual transitions and store them in replay buffer
                self._sample_her_transitions()
                # clear storage for current episode
                self.reset()

            self.episode_steps = 0

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals to create virtual transitions for the last episode.
        observations, next_observations, actions, rewards = self._sample_offline(n_sampled_goal=self.n_sampled_goal)

        # Store virtual transitions in the replay buffer, if available
        if len(observations) > 0:
            for i in range(len(observations["observation"])):
                self.replay_buffer.add(
                    {key: obs[i] for key, obs in observations.items()},
                    {key: next_obs[i] for key, next_obs in next_observations.items()},
                    actions[i],
                    rewards[i],
                    # We consider the transition as non-terminal
                    done=[False],
                    infos=[{}],
                )

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        current_idx = self.current_idx

        # truncate interrupted episode
        if current_idx > 0:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # get current episode and transition index
            pos = self.pos
            # set episode length for current episode
            self.episode_lengths[pos] = current_idx
            # set done = True for current episode
            # current_idx was already incremented
            self._buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
            # reset current transition index
            self.current_idx = 0
            # increment episode counter
            self.pos = (self.pos + 1) % self.max_episode_stored
            # update "full" indicator
            self.full = self.full or self.pos == 0
