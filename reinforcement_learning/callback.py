from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.logger import Figure
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class DistributionCallback(BaseCallback):

    def _on_training_start(self) -> None:
        self._log_freq = 10
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))

    
    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            dist_list = self.training_env.env_method('dists')
            dists = np.concatenate(dist_list, axis=1) 
            try:
                self.tb_formatter.writer.add_histogram('incentive_index', dists[:, 0], int(self.n_calls / self._log_freq))
                self.tb_formatter.writer.add_histogram('distance_session_end', dists[:, 1], int(self.n_calls / self._log_freq))
                self.tb_formatter.writer.add_histogram('distance_incentive_allocated', dists[:, 2], int(self.n_calls / self._log_freq))
                self.tb_formatter.writer.flush()
    
            except Exception as e:
                print(e)


