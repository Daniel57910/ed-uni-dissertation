from rl_constant import OUT_FEATURE_COLUMNS, METADATA_STAT_COLUMNS
from stable_baselines3.common.callbacks import  BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np
import pandas as pd

class DistributionCallback(BaseCallback):
    
    metadata_stat = METADATA_STAT_COLUMNS + ['time_in_session']
    @classmethod
    def tensorboard_setup(cls, log_dir, log_freq):
        cls._log_dir = log_dir
        cls._log_freq = log_freq

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))
    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('dists')
            values_to_log = np.concatenate([d for d in dist_list if d.shape[0] > 0])

            values_df = pd.DataFrame(
                values_to_log, 
                columns=METADATA_STAT_COLUMNS
            )
            
            dist_session_time = (values_df['session_minutes'] - values_df['time_in_session']).mean()
            dist_session_end = (values_df['session_size'] - values_df['ended']).mean()
            dist_inc_session = (values_df['session_size'] - values_df['incentive_index']).mean()
            dist_session_end = (values_df['ended'] - values_df['incentive_index']).mean()
            dist_inc_sim_size = (values_df['ended'] - values_df['sim_size']).mean()
            dist_inc_sim_index = (values_df['incentive_index'] - values_df['sim_size']).mean()

            n_call = self.n_calls // self._log_freq
            
            self.tb_formatter.writer.add_scalar('event/sess_time_sub_sime_time::decrease', dist_session_time, n_call)
            self.tb_formatter.writer.add_scalar('event/sess_index_sub_sim_index::decrease', dist_session_end, n_call)
            self.tb_formatter.writer.add_scalar('event/sim_incentive_index_sub_index_no_reward::increase', dist_inc_sim_size, n_call)
            
            self.tb_formatter.writer.add_scalar('event/sess_index_sub_incentive_index', dist_inc_session, n_call)
            self.tb_formatter.writer.add_scalar('event/sim_index_sub_incentive_index', dist_inc_sim_index, n_call)
            self.tb_formatter.writer.flush()
            
            values_df.to_parquet(f'{self._log_dir}/dist_{n_call}.parquet')
            
        return True