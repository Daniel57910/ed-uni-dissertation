import numpy as np
import pandas as pd
from rl_constant import RL_STAT_COLS
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class DistributionCallback(BaseCallback):
    
    @classmethod
    def tensorboard_setup(cls, log_dir, log_freq):
        cls._log_dir = log_dir
        cls._log_freq = 10

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))
    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('dists')
            values_to_log = np.concatenate([d for d in dist_list if d.shape[0] > 0])

            values_df = pd.DataFrame(
                values_to_log, 
                columns=RL_STAT_COLS + ['ended', 'incentive_index', 'n_episodes']
            )
            
            dist_session_time = (values_df['session_minutes'] - values_df['reward']).mean()
            dist_session_end = (values_df['session_size'] - values_df['ended']).mean()
            
            dist_sim_time = (values_df['reward'] - values_df['sim_size']).mean()
            dist_sim_end = (values_df['ended'] - values_df['sim_size']).mean()


            dist_inc_session = (values_df['session_size'] - values_df['incentive_index']).mean()
            dist_inc_end = (values_df['ended'] - values_df['incentive_index']).mean()
            dist_inc_sim_index = (values_df['sim_size'] - values_df['incentive_index']).mean()

            n_call = self.n_calls // self._log_freq
            
            self.tb_formatter.writer.add_scalar('distance/session/max_reward::decrease', dist_session_time, n_call)
            self.tb_formatter.writer.add_scalar('distance/session/max_ended::decrease', dist_session_end, n_call)
           
            self.tb_formatter.writer.add_scalar('distance/session/sim_reward::increase', dist_sim_time, n_call)
            self.tb_formatter.writer.add_scalar('distance/session/sim_ended::increase', dist_sim_end, n_call)
            
            
            self.tb_formatter.writer.add_scalar('distance/incentive/max_incentive::decrease', dist_inc_session, n_call)
            self.tb_formatter.writer.add_scalar('distance/incentive/ended_incentive', dist_inc_end, n_call)
            self.tb_formatter.writer.add_scalar('distance/incentive/sim_inc_placement::decrease', dist_inc_sim_index, n_call)
            self.tb_formatter.writer.add_scalar('distance/ended_sim_size::increase', dist_inc_sim_index, n_call)
            
            self.tb_formatter.writer.flush()
            
            values_df.to_parquet(f'{self._log_dir}/dist_{n_call}.parquet')
            
        return True