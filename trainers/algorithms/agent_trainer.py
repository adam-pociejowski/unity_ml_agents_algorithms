import tensorflow as tf
import numpy as np
import abc


class AgentTrainer:
    def __init__(self, brain, brain_name, input_num, output_num, agents_num, model_name, learning_rate=0.0002, epsilon_max=1.0,
                 epsilon_min=0.01, decay_rate=0.0001, discount_rate=0.9, memory_size=5000, batch_size=32, use_tf=True):
        __metaclass__ = abc.ABCMeta
        print('[' + type(self).__name__ + ' - ' + brain_name + ']:  init')
        self.brain = brain
        self.brain_name = brain_name
        self.input_num = input_num
        self.output_num = output_num
        self.agents_num = agents_num
        self.model_name = model_name
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.epsilon = epsilon_max

        self._init_model()
        self.summary_writer = tf.summary.FileWriter('summary/' + model_name)
        if use_tf:
            self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self._init_episode()

    @abc.abstractmethod
    def _init_model(self):
        return

    @abc.abstractmethod
    def get_actions(self, observation):
        return

    def post_step_actions(self, observations, actions, rewards, new_observations):
        return

    def post_episode_actions(self, rewards, episode):
        self._save_model(None)
        self._init_episode()

    def _init_episode(self):
        pass

    def _reshape_observations(self, obs):
        reshaped = []
        for index in range(len(obs)):
            reshaped.append(np.asarray(obs[index]).reshape(self.input_num))
        return reshaped

    def _save_model(self, data):
        self.saver.save(self.sess, 'models/' + self.model_name + '/' + self.model_name + '.ckpt')
        print('Model '+self.model_name+' Saved')
