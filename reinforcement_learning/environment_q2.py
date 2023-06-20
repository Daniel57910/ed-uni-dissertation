# %load environment
# %load environment
import gym
import numpy as np
from scipy.stats import norm
MAX_EVAL_SIZE = 75
from rl_constant import RL_STAT_COLS

class CitizenScienceEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dataset, out_features, n_sequences, params=None):
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
        
        max_session_size = self.dataset['session_size'].max()
        
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=-1, high=max_session_size + 1, shape=(len(out_features) + 3, n_sequences + 1), dtype=np.float32)
        self.episode_bins = []
        self.exp_runs = 0
        self.params = params
        self.social_components = np.array([10, 20, 30, 40, 50, 70])

    def reset(self):
        random_session = np.random.randint(0, self.unique_sessions.shape[0])
        
        user_to_run, session_to_run = self.unique_sessions.iloc[random_session][['user_id', 'session_30_count_raw']]
        self.current_session = self._get_events(user_to_run, session_to_run)
        self.metadata = self._metadata()
        self.current_session_index = 1
        self.reward = 0
        return self._state()
    
    def _row_to_dict(self, metadata):
        """
        Convert a row of metadata to a dictionary.
        """
        return metadata.to_dict()
    
    def _reward_exp(self, cum_session_event_raw):
        """
        Reward shaping as
            0 if cum_session_event_raw < size_cutoff
            (cum_session_event_raw - size_cutoff) * (cum_session_event_raw / size_cutoff) otherwise
        """
        
        if all([self.metadata['inc_small'] == 0, self.metadata['inc_medium'] == 0, self.metadata['inc_large'] == 0]):
            distance_shaper = cum_session_event_raw / 4
        
        else:
            distance_shaper = np.linalg.norm([
                self.metadata['inc_small'], 
                self.metadata['inc_medium'] - self.metadata['inc_small'], 
                self.metadata['inc_large'] - self.metadata['inc_medium'] ])
        if cum_session_event_raw <= self.metadata['size_cutoff']:
            return (cum_session_event_raw / self.metadata['size_cutoff']) * distance_shaper
        
        return (cum_session_event_raw - self.metadata['size_cutoff']) * (cum_session_event_raw / self.metadata['size_cutoff']) * distance_shaper

    def step(self, action):
        
        self._take_action(action)
        self._assign_social_components()
        next_state, done, meta = self._calculate_next_state()
        
        
        if done:
            current_session_index = self.current_session_index if \
                self.current_session_index != self.current_session.shape[0] else self.current_session.shape[0] - 1
            
            self.exp_runs += 1
            self.metadata['ended_event'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            self.metadata['ended_time'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
            self.metadata['exp_runs'] = self.exp_runs
            self.episode_bins.append(self._row_to_dict(self.metadata))
           
            cum_session_event_raw = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            reward_exp = self._reward_exp(cum_session_event_raw)
            
            return next_state, reward_exp , done, {}
        else:
            self.reward = self.current_session.iloc[self.current_session_index]['cum_session_event_raw']
            cum_session_event_raw = self.current_session.iloc[self.current_session_index]['cum_session_event_raw']
            
            reward_exp = self._reward_exp(cum_session_event_raw)
    
            self.current_session_index += 1        
            
            return next_state, reward_exp, done, meta
    
    def _assign_social_components(self):
        current_event = self.current_session_index
        social_likelihood = np.searchsorted(self.social_components, current_event)
        assign_social = norm(
            loc=social_likelihood,
            scale=social_likelihood // 6,
        ).cdf(current_event) + np.random.normal(0, 0.04)
        
        if all([assign_social >= .4, assign_social <= 7]) and self.metadata[f'soc_{social_likelihood}'] == 0:
            self.metadata[f'soc_{social_likelihood}'] = current_event

 
    def _metadata(self):
        session_metadata = self.current_session.iloc[0][RL_STAT_COLS].copy()
        session_metadata['ended'] = 0
        for meta_col in ['small', 'medium', 'large'] + [f'soc_{i}' for i in self.social_components]:
            session_metadata[f'inc_{meta_col}'] = 0
            session_metadata[f'time_{meta_col}'] = 0

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
        event_cutoff = self.current_session.iloc[self.current_session_index]['size_cutoff']
        current_session_event = self.current_session.iloc[self.current_session_index]['cum_session_event_raw']
        if current_session_event <= event_cutoff or current_session_event  >= MAX_EVAL_SIZE:
            return True
        
        param_mid = 0.1 if not self.params else self.params['mid']
        param_large = 0.2 if not self.params else self.params['large']
        param_window = 0.75 if not self.params else self.params['window']
    
        extending_low = self._probability_extending(current_session_event, self.metadata['inc_small']) - \
            (0.05 + np.random.normal(-0.02, 0.1, 100).mean())

            
        extending_medium = self._probability_extending(current_session_event, self.metadata['inc_medium']) - \
            (param_mid + np.random.normal(-0.02, 0.1, 100).mean()) 
            
        extending_large = self._probability_extending(current_session_event, self.metadata['inc_large']) - \
            (param_large + np.random.normal(-0.02, 0.1, 100).mean())
        
        max_social_component = max([self.metadata[f'soc_{i}'] for i in self.social_components])
        
        extending_social = self._probability_extending(current_session_event, max_social_component) - \
            (param_large + np.random.normal(-0.02, 0.1, 100).mean())
            
        return any([
            extending_low > 0.4 and extending_low <= param_window,
            extending_medium > 0.4 and extending_medium <= param_window,
            extending_large > 0.4 and extending_large <= param_window,
            extending_social > 0.4 and extending_social <= param_window
        ])
        
           
    
    def _probability_extending(self, current_session_event, incentive_event):
        if incentive_event == 0:
            return 0
         
        continue_session = norm(
            loc=max(incentive_event, 1),
            scale=max(incentive_event *.75, 1)
        ).cdf(max(current_session_event, 1)) 
        
        return continue_session
        

    def _get_events(self, user_id, session):
        subset = self.dataset[
            (self.dataset['user_id'] == user_id) &
            (self.dataset['session_30_count_raw'] == session).copy()
        ]

        subset = subset.sort_values(by=['date_time'])
        return subset
    
    def _take_action(self, action):
        
        if action == 0:
            return 1
        
        current_session_index = self.current_session_index if \
            self.current_session_index != self.current_session.shape[0] else self.current_session.shape[0] - 1
    
        if action == 1:
            if self.metadata['inc_small'] > 0:
                return 1

            self.metadata['inc_small'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            self.metadata['time_small'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
            return 1
    
        elif action == 2:
            if self.metadata['inc_medium'] > 0:
                return 1
            self.metadata['inc_medium'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            self.metadata['time_medium'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
            return 1
        
        else:
            if self.metadata['inc_large'] > 0:
                return 1
            self.metadata['inc_large'] = self.current_session.iloc[current_session_index]['cum_session_event_raw']
            self.metadata['time_large'] = self.current_session.iloc[current_session_index]['cum_session_time_raw']
            return 1

    def _state(self):

        if self.current_session_index > self.n_sequences:
            events = self.current_session.iloc[self.current_session_index - (self.n_sequences + 1):self.current_session_index][self.out_features]
            events['inc_small'] = self.metadata['inc_small']
            events['inc_medium'] = self.metadata['inc_medium']
            events['inc_large'] = self.metadata['inc_large']
            
            events = events.values
            
            
        else:
            
            delta = min((self.n_sequences + 1)- self.current_session_index, self.n_sequences)
            zero_cat = np.zeros((delta, len(self.out_features) + 3))
            events = self.current_session.iloc[:max(self.current_session_index, 1)][self.out_features]
            
            events['inc_small'] = self.metadata['inc_small']
            events['inc_medium'] = self.metadata['inc_medium']
            events['inc_large'] = self.metadata['inc_large']
           
            
            
            events = np.concatenate((zero_cat, events), axis=0)
        
        return events.astype(np.float32).T