# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import argparse
import logging
import os

import gym

from a2c import learn
from atari_wrappers import make_atari, Monitor, wrap_deepmind
from neural_network import CNN
from subproc_vec_env import SubprocVecEnv

os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'  # Mute missing instructions errors

MODEL_PATH = 'models'
SEED = 0

def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(80e6))
    parser.add_argument('--nenv', help='No. of environments', type=int, default=16)
    return parser.parse_args()

def train(env_id, num_timesteps, num_cpu):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(SEED + rank)
            gym.logger.setLevel(logging.WARN)
            env = wrap_deepmind(env)
            
            # wrap the env one more time for getting total reward
            env = Monitor(env, rank)
            return env
        
        return _thunk
    
    env = SubprocVecEnv([ make_env(i) for i in range(num_cpu) ])
    learn(CNN, env, SEED, total_timesteps=int(num_timesteps * 1.1))
    env.close()
    pass

def main():
    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.env, args.steps, num_cpu=args.nenv)

if __name__ == "__main__":
    main()
