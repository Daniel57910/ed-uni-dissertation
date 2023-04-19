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
            current_values = [
                d[self.n_calls- 100:self.n_calls] for d in dist_list
            ]
            
            logging_valus = np.array(current_values)
            print(logging_valus)

                
