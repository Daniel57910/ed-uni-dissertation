import gymnasium
import numpy as np
from scipy.stats import norm 
from constant import METADATA, OUT_FEATURE_COLUMNS

import gym

class CitizenScienceEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset, session_ranges, n_sequences):
        """
        trajectories: dictionary of user_id to their respective trajectories.
        n_sequences: number of sequences used for preprocessing.
        n_features: number of features used for preprocessing.
        """
        super(CitizenScienceEnv, self).__init__()
        self.dataset = dataset
        self.session_ranges = session_ranges
        self.n_sequences = n_sequences
        self.current_session = None
        self.current_session_index = 0
        self.reward = 0
        self.metadata_container = []

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_sequences + 1, 18), dtype=np.float32)
        self.n_sequences = n_sequences

    def reset(self):
        session_id = np.random.choice(self.session_ranges)
        self.current_session = self._get_events(session_id)
        self.metadata = self._metadata()
        self.current_session_index = 1
        self.reward = 0
        return self._state()

    def step(self, action):
        self._take_action(action)
        next_state, done, meta = self._calculate_next_state()
        if done:
            self.metadata['ended'] = self.current_session_index
            self.metadata['reward'] = self.reward
            self.metadata_container.append(self.metadata.values)
            return next_state, self.reward, done, meta
        self.reward += (self.current_session.iloc[self.current_session_index]['reward'] / 60)
        self.current_session_index += 1        
        return next_state, self.reward, done, meta
    
    def _metadata(self):
        session_metadata = self.current_session.iloc[0][['user_id', 'session_30_raw', 'session_size', 'sim_size', 'session_minutes']]
        session_metadata['ended'] = 0
        session_metadata['incentive_index'] = 0
        return session_metadata
    
    
    def _calculate_next_state(self):
        
        if self.current_session_index == self.current_session.shape[0]:
            return None, True, {}
        
        if self._continuing_in_session():
            return self._state(), False, {}
      
        return None, True, {}
  
    def _continuing_in_session(self):
        sim_counts = self.metadata['sim_size']
        if self.current_session_index < sim_counts:
            return True
        
        extending_session = self._probability_extending_session()
        
        return all([extending_session >= .3, extending_session <= .8])
        
    
    def _probability_extending_session(self):
        if self.metadata['incentive_index'] == 0:
            return 0
        
        scale = max(5, int(self.metadata['session_size'] / 4))
        continue_session = norm(
            loc=self.metadata['incentive_index'],
            scale=scale
        ).cdf(self.current_session_index)
        
        return continue_session
        

    def _get_events(self, session_id):
        subset = self.dataset[self.dataset['session_30_raw'] == session_id]
        subset_user = subset['user_id'].sample(1).values[0]
        subset = subset[subset['user_id'] == subset_user]
        return subset.sort_values('cum_session_event_raw').reset_index(drop=True)
    
    def _take_action(self, action):
        if action == 0 or self.metadata['incentive_index'] > 0:
            return
        
        self.metadata['incentive_index'] = self.current_session_index
        
    def _state(self):

        if self.current_session_index > self.n_sequences:
            events = self.current_session.iloc[self.current_session_index - (self.n_sequences + 1):self.current_session_index][OUT_FEATURE_COLUMNS].values
            
        else:
            delta = (self.n_sequences + 1)- self.current_session_index
            zero_cat = np.zeros((delta, len(OUT_FEATURE_COLUMNS)))
            events = self.current_session.iloc[:max(self.current_session_index, 1)][OUT_FEATURE_COLUMNS].values
            events = np.concatenate((zero_cat, events), axis=0)
            

        return events.astype(np.float32)
  
    
    def dists(self):
        return self.metadata_container