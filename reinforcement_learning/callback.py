# %load callback
# %load callback
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime

class DistributionCallback(BaseCallback):
    
    @classmethod
    def tensorboard_setup(cls, log_dir, log_freq):
        cls._log_dir = log_dir
        cls._log_freq = log_freq

    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('flush_episode_bins')
            values_to_log = [item for sublist in dist_list for item in sublist if len(sublist) > 0]

            values_df = pd.DataFrame(
                values_to_log
            )
            
            
            session_size, size_cutoff, session_minutes, time_cutoff, ended_event, ended_time, inc_time, inc_index = (
                values_df['session_size'].mean(),
                values_df['size_cutoff'].mean(),
                values_df['session_minutes'].mean(),
                values_df['time_cutoff'].mean(),
                values_df['ended_event'].mean(),
                values_df['ended_time'].mean(),
                values_df['incentive_time'].mean(),
                values_df['incentive_index'].mean()
            )
            
            size_stats = {
                'session_size': session_size,
                'size_cutoff': size_cutoff,
                'ended_size': ended_event,
                'inc_index': inc_index
            }
            
            
            time_stats = {
                'session_minutes': session_minutes,
                'sim_minutes': time_cutoff,
                'ended_time': ended_time,
                'inc_time': inc_time   
            }
            
            for key, value in size_stats.items():
                self.logger.record(f'size/{key}', value)
            
            for key, value in time_stats.items():
                self.logger.record(f'sess_time/{key}', value)
                
            values_df.to_csv(f'{self._log_dir}/{self.n_calls // self._log_freq}.csv')
            
        return True