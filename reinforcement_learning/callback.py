import numpy as np
import pandas as pd
from rl_constant import RL_STAT_COLS
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime
import os
class DistributionCallback(BaseCallback):
    
    @classmethod
    def tensorboard_setup(cls, log_dir, log_freq):
        cls._log_dir = log_dir
        cls._log_freq = log_freq

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))
    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('flush_episode_bins')
            values_to_log = [item for sublist in dist_list for item in sublist if len(sublist) > 0]

            values_df = pd.DataFrame(
                values_to_log
            )
            
            
            session_size, sim_size, session_minutes, sim_minutes, ended, reward, inc_time, inc_index = (
                values_df['session_size'].mean(),
                values_df['sim_size'].mean(),
                values_df['session_minutes'].mean(),
                values_df['sim_minutes'].mean(),
                values_df['ended'].mean(),
                values_df['reward'].mean(),
                values_df['incentive_time'].mean(),
                values_df['incentive_index'].mean()
            )
            
            size_stats = {
                'session_size': session_size,
                'sim_size': sim_size,
                'ended': ended,
                'inc_index': inc_index
            }
            
            
            time_stats = {
                'session_minutes': session_minutes,
                'sim_minutes': sim_minutes,
                'reward': reward,
                'inc_time': inc_time   
            }
            self.tb_formatter.writer.add_scalars('size_stats', size_stats, self.n_calls // self._log_freq)
            self.tb_formatter.writer.add_scalars('time_stats', time_stats, self.n_calls // self._log_freq)
            
            self.tb_formatter.writer.flush()
            
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            

            current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            values_df.to_parquet(f'{self._log_dir}/dist_{self.n_calls // self._log_freq}_{current_time}.parquet')
            
        return True