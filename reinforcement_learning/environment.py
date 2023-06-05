# %load environment
import gym
import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm 
import gym
from datetime import datetime
from copy import deepcopy
from rl_constant import RL_STAT_COLS
class CitizenScienceEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset, out_features, n_sequences, evaluation=False):
        """
        trajectories: dictionary of user_id to their respective trajectories.
        n_sequences: number of sequences used for preprocessing.
        n_features: number of features used for preprocessing.
        """
        super(CitizenScienceEnv, self).__init__()
        self.dataset = dataset
        self.unique_sessions = self.dataset[['user_id', 'session_30_count_raw']].drop_duplicates()
        self.n_sequences = n_sequences
        self.current_session = None
        self.current_session_index = 0
        self.reward = 0
        self.n_sequences = n_sequences
        self.out_features = out_features
        
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(len(out_features), n_sequences + 1), dtype=np.float32)
        self.evalution = evaluation
        self.episode_bins = []
        self.exp_runs = 0

    def reset(self):
        user_to_run, session_to_run = self.unique_sessions.sample().values[0]
        self.current_session = self._get_events(user_to_run, session_to_run)
        self.metadata = self._metadata()
        self.current_session_index = 0
        self.reward = 0
        return self._state()
    
    def _row_to_dict(self, metadata):
        """
        Convert a row of metadata to a dictionary.
        """
        return metadata.to_dict()

    def step(self, action):
        
        self._take_action(action)
        next_state, done, meta = self._calculate_next_state()
        
        if done:
            current_session_index = self.current_session_index if \
                self.current_session_index != self.current_session.shape[0] else self.current_session.shape[0] - 1
            
            self.exp_runs += 1
        
            self.metadata['ended_event'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            self.metadata['ended_time'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
            self.metadata['exp_runs'] = self.exp_runs
            self.episode_bins.append(self._row_to_dict(self.metadata))
            
            return next_state, float(self.reward), done, {}
        else:
            self.reward = self.current_session.iloc[self.current_session_index]['reward'] 
            self.current_session_index += 1        
            return next_state, float(self.reward), done, meta
    
    def _metadata(self):
        session_metadata = self.current_session.iloc[0][RL_STAT_COLS]
        session_metadata['ended'] = 0
        session_metadata['incentive_index'] = 0
        return session_metadata
    
    
    def flush_episode_bins(self):
        episode_bins = self.episode_bins.copy()
        self.episode_bins = []
        return episode_bins
    
    def _calculate_next_state(self):
        
        if (self.current_session_index == self.current_session.shape[0]):
            return None, True, {}

        if self._continuing_in_session():
            return self._state(), False, {}
    
        return None, True, {}
         
    def _continuing_in_session(self):
        time_cutoff = self.current_session.iloc[self.current_session_index]['time_cutoff']
        current_session_minutes = self.current_session.iloc[self.current_session_index]['cum_session_time_raw']
        if current_session_minutes < time_cutoff:
            return True
        
        extending_session = self._probability_extending_session(current_session_minutes)
        
        return all([extending_session >= .3, extending_session <= .7])
        
    
    def _probability_extending_session(self, current_session_count):
        if self.metadata['incentive_index'] == 0:
            return 0
        
        scale = max(5, int(self.metadata['session_minutes'] / 5))
        continue_session = norm(
            loc=self.metadata['incentive_time'],
            scale=scale
        ).cdf(current_session_count)
        
        return continue_session
        

    def _get_events(self, user_id, session):
        subset = self.dataset[
            (self.dataset['user_id'] == user_id) &
            (self.dataset['session_30_count_raw'] == session).copy()
        ]

        subset = subset.sort_values(by=['date_time'])
        return subset
    
    def _take_action(self, action):
        if action == 0 or self.metadata['incentive_index'] > 0:
            return
        
        current_session_index = self.current_session_index if \
            self.current_session_index != self.current_session.shape[0] else self.current_session.shape[0] - 1
        
        self.metadata['incentive_index'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
        self.metadata['incentive_time'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
        
    def _state(self):

        if self.current_session_index > self.n_sequences:
            events = self.current_session.iloc[self.current_session_index - (self.n_sequences + 1):self.current_session_index][self.out_features].values
            
        else:
            delta = min((self.n_sequences + 1)- self.current_session_index, self.n_sequences)
            zero_cat = np.zeros((delta, len(self.out_features)))
            events = self.current_session.iloc[:max(self.current_session_index, 1)][self.out_features].values
            events = np.concatenate((zero_cat, events), axis=0)
            

        return events.astype(np.float32).T