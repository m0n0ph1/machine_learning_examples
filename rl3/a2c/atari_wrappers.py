# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
from collections import deque

import cv2  # opencv-python
import gym
import numpy as np
from gym import spaces

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[ 0 ] == 'NOOP'
    
    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[ 1 ] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype='uint8')
        self._skip = skip
    
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[ 0 ] = obs
            if i == self._skip - 1:
                self._obs_buffer[ 1 ] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

# class WarpFrame(gym.ObservationWrapper):
#     def __init__(self, env):
#         """Warp frames to 84x84 as done in the Nature paper and later work."""
#         gym.ObservationWrapper.__init__(self, env)
#         self.width = 84
#         self.height = 84
#         self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

#     def _observation(self, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
#         return frame[:, :, None]
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 3), dtype=np.uint8)
    
    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([ ], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[ 0 ], shp[ 1 ], shp[ 2 ] * k))
    
    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
    
    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out

def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    if frame_stack:
        env = FrameStack(env, 4)
    
    return env

class Monitor(gym.Wrapper):
    def __init__(self, env, rank=0):
        gym.Wrapper.__init__(self, env=env)
        self.rank = rank
        self.rewards = [ ]
        self.total_reward = [ ]
        self.summaries_dict = {'reward': 0, 'episode_length': 0, 'total_reward': 0, 'total_episode_length': 0}
        env = self.env
        while True:
            if hasattr(env, 'was_real_done'):
                self.episodic_env = env
            if not hasattr(env, 'env'):
                break
            env = env.env
    
    def reset(self):
        self.summaries_dict[ 'reward' ] = -1
        self.summaries_dict[ 'episode_length' ] = -1
        self.summaries_dict[ 'total_reward' ] = -1
        self.summaries_dict[ 'total_episode_length' ] = -1
        self.rewards = [ ]
        env = self.env
        if self.episodic_env.was_real_done:
            self.summaries_dict[ 'total_reward' ] = -1
            self.summaries_dict[ 'total_episode_length' ] = -1
            self.total_reward = [ ]
        return self.env.reset()
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.total_reward.append(reward)
        if done:
            # print("Done! R = %s, N = %s" % (sum(self.rewards), len(self.rewards)))
            self.summaries_dict[ 'reward' ] = sum(self.rewards)
            self.summaries_dict[ 'episode_length' ] = len(self.rewards)
            
            if self.episodic_env.was_real_done:
                self.summaries_dict[ 'total_reward' ] = sum(self.total_reward)
                self.summaries_dict[ 'total_episode_length' ] = len(self.total_reward)
        info = self.summaries_dict.copy()  # otherwise it will be overwritten
        # if done:
        #     print("info:", info)
        return observation, reward, done, info
