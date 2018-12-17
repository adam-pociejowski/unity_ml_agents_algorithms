from trainers.agent_trainer import AgentTrainer
import tensorflow as tf
import os
import numpy as np
import abc


class KerasAgentTrainer(AgentTrainer):
    def __init__(self, brain, brain_name, input_num, output_num, agents_num, model_name, memory_size=5000, batch_size=32, restore_model=False,
                 min_memory_size=1000, discount_rate=0.99):
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name=model_name, memory_size=memory_size, batch_size=batch_size,
                         restore_model=restore_model, discount_rate=discount_rate)
        self.min_memory_size = min_memory_size
        __metaclass__ = abc.ABCMeta

    def init(self):
        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)
        print('[' + type(self).__name__ + ' - ' + self.brain_name + ']:  init')
        self._init_model()
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.sess = tf.Session()
        self._init_episode()
        if self.restore_model:
            self._restore_model()

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards, new_observations)
        if self.episode_step_counter > self.min_memory_size:
            self._train()

    def post_episode_actions(self, rewards, episode):
        self._delete_objects()
        self._init_episode()

    def _get_memory_samples(self):
        index_range = min(self.episode_step_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)
        return self.observation_memory[sample, :], self.new_observation_memory[sample, :], self.reward_memory[sample], self.action_memory[sample]

    def _init_episode(self):
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = np.zeros((self.memory_size, self.input_num))
        self.new_observation_memory = np.zeros((self.memory_size, self.input_num))
        self.episode_step_counter = 0

    def _store_memory(self, observations, actions, rewards, new_observations):
        for i in range(self.agents_num):
            index = self.episode_step_counter % self.memory_size
            self.action_memory[index] = actions[i]
            self.reward_memory[index] = rewards[i]
            self.observation_memory[index, :] = observations[i]
            self.new_observation_memory[index, :] = new_observations[i]
            self.episode_step_counter += 1

    def _delete_objects(self):
        del self.action_memory
        del self.reward_memory
        del self.observation_memory
        del self.new_observation_memory

    @abc.abstractmethod
    def _train(self):
        return

    @abc.abstractmethod
    def _restore_model(self):
        return
