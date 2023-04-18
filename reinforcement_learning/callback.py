from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.logger import Figure
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
class DistributionCallback(BaseCallback):

    def _on_training_start(self) -> None:
        self._dist_log_freq = 1000
        self._reward_log_freq = 100
        output_formats = self.logger.output_formats
        self.tb_formatter = next(f for f in output_formats if isinstance(f, TensorBoardOutputFormat))
    
    def _on_step(self) -> bool:
        if self.n_calls % self._dist_log_freq == 0:
            dist_list = self.training_env.env_method('dists')
            episode_list = self.training_env.get_attr('current_episode')
            try:
                for episode, dist in zip(episode_list, dist_list):
                    self.tb_formatter.writer.add_histogram('incentive_index', dist[:, 0], episode)
                    self.tb_formatter.writer.add_histogram('distance_session_end', dist[:, 1], episode)
                    self.tb_formatter.writer.add_histogram('distance_incentive_allocated', dist[:, 2], episode)
                self.tb_formatter.writer.flush()
            except Exception as e:
                raise Exception('Unable to log distributions: {}'.format(e))
        
        if self.n_calls % self._reward_log_freq == 0:
            episode_list = self.training_env.get_attr('current_episode')
            rewards = self.training_env.get_attr('user_sessions')
            reward_dict = {}
            for episode, reward in zip(episode_list, rewards):
                reward_dict[f'episode_{episode}'] = reward['total_reward'].sum()
            
            self.tb_formatter.writer.add_scalars(
                'cum_reward', reward_dict, (self.n_calls // np.max(episode_list)) // self._reward_log_freq
            )
                
