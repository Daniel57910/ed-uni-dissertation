import gym
import argparse
import numpy as np
import torch
import pdb

USER_INDEX = 1
SESSION_INDEX = 2
TASK_INDEX = 3

N_EVENT_INDEX = -1

USER_IN_SESSION_INDEX = 0
SESSION_COUNT_INDEX = 1
TASK_IN_SESSION_INDEX = 2
REWARD_ALLOCATED_INDEX = 3

SESSION_FINISHED_INDEX = -1

CUM_PLATFORM_TIME_INDEX = 4
METADATA_INDEX = 13

import logging
from scipy.stats import norm 
from stable_baselines3.common.logger import TensorBoardOutputFormat


class CitizenScienceEnv(gym.Env):
    
    logger = logging.getLogger(__name__) 
    metadata = {'render.modes': ['human']}
    
    def __init__(self, user_sessions, experience_dataset, n_sequences, n_features) -> None:
        """
        trajectories: dictionary of user_id to their respective trajectories.
        n_sequences: number of sequences used for preprocessing.
        n_features: number of features used for preprocessing.
        """
        super(CitizenScienceEnv, self).__init__()
        self.user_sessions = user_sessions
        self.experience_dataset = experience_dataset

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_sequences + 1, n_features), dtype=np.float32)
        self.n_sequences = n_sequences
        self.n_features = n_features
        self.current_session = None
        self.current_episode = 0
        
    def _extract_features(self, feature_array):
        
        metadata, features = feature_array[:, :METADATA_INDEX], feature_array[:, METADATA_INDEX:]
        
        features = features.reshape((features.shape[0], self.n_sequences + 1, self.n_features))
        features = np.flip(features, axis=1).squeeze(0)
        return metadata.squeeze(0), features

    def _state(self, user, session, task_count):
        
        """
        get index of current state
        """ 
        current_state = self.experience_dataset[
            (self.experience_dataset[:, USER_INDEX] == user) &
            (self.experience_dataset[:, SESSION_INDEX] == session) &
            (self.experience_dataset[:, TASK_INDEX] == task_count)
        ]

        metadata, features = self._extract_features(current_state)
        cum_platform_time = metadata[CUM_PLATFORM_TIME_INDEX]
        return features, cum_platform_time

    
    def _seed_user_session(self):
        """
        find all users sessions that have not been completed
        select random user session from list
        """
        current_session = self.user_sessions[self.user_sessions['ended'] == 0].sample(1)
        current_session['task_index'] = 1
        self.current_session = current_session
        
    def step(self, action):
        
        self._take_action(action)
            
        state, rewards, done, meta = self._calculate_next_state() 
        if not done:
            self._update_session_metadata(self.current_session)
        
        return state, rewards, done, meta

    def _update_session_metadata(self, current_session):
        self.user_sessions.loc[current_session.index] = current_session 
        
    def _calculate_next_state(self):
        
        next_state = self.current_session['task_index'] + 1
        extending = self._extending()
        if not extending:
            self.logger.debug(f'User: {self.current_session} has completed their session')
            self._user_session_terminate()
            if self.user_sessions['ended'].all():
                self.logger.debug('All users have completed their sessions')
                return None, self.user_sessions['total_reward'].sum().astype(float), True, {}
            
            self._seed_user_session()
            user, session, count = self.current_session[['user_id', 'session_id', 'task_index']].values[0]
            return (
                self._state(user, session, count)[0], 
                self.user_sessions['total_reward'].sum().astype(float),
                False,
                {}
            )
        self.logger.debug(f'User: {self.current_session} has moving to next state: {next_state}')
        self.current_session['task_index'] = next_state
        user, session, count = self.current_session[['user_id', 'session_id', 'task_index']].values[0]
        state, cum_platform_time = self._state(user, session, count)
        self.current_session['total_reward'] = cum_platform_time
        return (
            state,
            self.user_sessions['total_reward'].sum().astype(float),
            False,
            {}
        )
    
    
    def _extending(self):
        current_session = self.current_session.to_dict('records')[0]
        if current_session['task_index'] == current_session['counts']:
            return False
    
        if current_session['task_index'] <= current_session['sim_counts']:
            return True

        continue_session = self._probability_extending(current_session)
        return all([continue_session >= 0.3, continue_session <= 0.85])
    
    
    def _probability_extending(self, current_session):
        if current_session['incentive_index'] == 0:
            return 0
        else:
            scale = min(5, current_session['counts'] // 4)
            continue_session = norm(
                loc=current_session['incentive_index'],
                scale=scale
            ).cdf(current_session['task_index']) 
            
        return continue_session
        

     
    def _user_session_terminate(self):
        self.current_session['ended'] = 1
        self._update_session_metadata(self.current_session)
    
    def _take_action(self, action):
        
        current_session = self.current_session.to_dict('records')[0]
        
        if current_session['incentive_index'] > 0 or action == 0:
            self.logger.debug(f'Incentive already allocation for session or no-op: {action}, {current_session}')
            return
        
    
        self.logger.debug('Taking action and allocating incentive')
        self.current_session['incentive_index'] = self.current_session['task_index']
        self.current_session['reward_allocated'] = action
        
        self.logger.debug('Taking action and allocating incentive: updating user session')
        self.logger.debug(f'User session: {self.current_session}')

    def reset(self):
        self.user_sessions = self.user_sessions.sample(frac=1)
        self.user_sessions['incentive_index'] = 0
        self.user_sessions['task_index'] = 0
        self.user_sessions['ended'] = 0
        self.user_sessions['total_reward'] = 0
        self.user_sessions['total_reward'] = self.user_sessions['total_reward'].astype(float)
        
        self._seed_user_session()
        self._update_session_metadata(self.current_session)
        user, session, count = self.current_session[['user_id', 'session_id', 'task_index']].values[0]
        self.current_episode += 1
        return self._state(user, session, count)[0]
        
    
    def render(self, mode='human'):
        print('rendering')
        
    def dists(self):
        incentive_index = self.user_sessions['incentive_index'].values
        distance_end = (self.user_sessions['counts'] - self.user_sessions['incentive_index']).values
        distance_reward = (self.user_sessions['total_reward'] - self.user_sessions['incentive_index']).values
        return np.array([incentive_index, distance_end, distance_reward])  