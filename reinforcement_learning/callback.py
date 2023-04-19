from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.logger import Figure
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
class DistributionCallback(BaseCallback):

    def _on_training_start(self) -> None:
        self._log_freq = 100
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))
    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('dists')
            values_to_log = np.concatenate([d for d in dist_list if d.shape[0] > 0])

            session_size, sim_size, session_minutes, ended, incentive_index, reward = (
                values_to_log[:, 0],
                values_to_log[:, 1],
                values_to_log[:, 2],
                values_to_log[:, 3],
                values_to_log[:, 4],
                values_to_log[:, 5]
            )
            
            dist_session_time = (session_minutes - reward).mean()
            dist_session_end = (session_size - ended).mean()
            dist_incentive_session = (session_size - incentive_index).mean()
            dist_incentive_end = (ended - incentive_index).mean()
            n_call = self.n_calls / 100
            
            self.tb_formatter.writer.add_scalar('time/session_time', dist_session_time, n_call)
            self.tb_formatter.writer.add_scalar('event/session_end', dist_session_end, n_call)
            self.tb_formatter.writer.add_scalar('event/incentive_session', dist_incentive_session, n_call)
            self.tb_formatter.writer.add_scalar('event/incentive_end', dist_incentive_end, n_call)
            
            self.tb_formatter.writer.flush()
        return True