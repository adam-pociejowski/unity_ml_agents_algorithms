import tensorflow as tf
import numpy as np
import abc


class AgentTrainer:
    def __init__(self, brain, brain_name, input_num, output_num, agents_num, model_name, learning_rate=0.0002, epsilon_max=1.0, epsilon_min=0.01,
                 decay_rate=0.0001, discount_rate=0.99, memory_size=5000, batch_size=32, param_decrease_interval=100, epsilon_decay_rate=1.0001,
                 use_tf=True, restore_model=False):
        __metaclass__ = abc.ABCMeta
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
        self.epsilon_decay_rate = epsilon_decay_rate
        self.param_decrease_interval = param_decrease_interval
        self.use_tf = use_tf
        self.restore_model = restore_model
        self.models_dir = 'models/' + self.model_name
        self.summary_dir = 'summary/' + self.model_name
        self.save_summary = True
        self.summary_writer = None
        self.saver = None
        self.sess = None
        self.current_loss = 0
        self.step = 0

    def init(self):
        print('[' + type(self).__name__ + ' - ' + self.brain_name + ']:  init')
        self._init_model()
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
        self.sess = tf.Session()
        self._init_tensorflow()
        self._init_episode()

    def _init_tensorflow(self):
        if self.use_tf:
            self.saver = tf.train.Saver()
        if self.restore_model:
            self._restore_model()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    @abc.abstractmethod
    def _init_model(self):
        return

    @abc.abstractmethod
    def get_actions(self, observation):
        return

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self.step += 1
        if self.step % self.param_decrease_interval == 0:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay_rate**(-1*self.step))

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
        self.saver.save(self.sess, self.models_dir + '/' + self.model_name + '.ckpt')

    def _restore_model(self):
        model_path = self.models_dir + '/' + self.model_name + '.ckpt'
        self.saver.restore(self.sess, model_path)
        print(f'Model restored: {model_path}')
