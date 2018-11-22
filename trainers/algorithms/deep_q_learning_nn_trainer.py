import numpy as np
import tensorflow as tf


class DeepQLearningNNTrainer:
    def __init__(self, brain_name, input_num, output_num, agents_num, learning_rate=0.0002, epsilon_max=1.0,
                 epsilon_min=0.01, decay_rate=0.0001, discount_rate=0.9, memory_size=1000, batch_size=32, layer_1_nodes=10,
                 layer_2_nodes=10):
        self.tag = '[DeepQLearningNNTrainer - ' + brain_name + ']: '
        print(self.tag + ' started')
        self.brain_name = brain_name
        self.input_num = input_num
        self.output_num = output_num
        self.agents_num = agents_num
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

        self._init_episode()
        self.learn_step_counter = 0
        self.epsilon = epsilon_max
        self.sess = {}

        W_init = tf.contrib.layers.xavier_initializer(seed=1)
        b_init = tf.contrib.layers.xavier_initializer(seed=1)
        self._build_eval_network(layer_1_nodes, layer_2_nodes, W_init, b_init)
        self.summary_writer = tf.summary.FileWriter("summary/deep_q_learning")
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _init_episode(self):
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = np.zeros((self.input_num, self.memory_size))
        self.new_observation_memory = np.zeros((self.input_num, self.memory_size))
        self.episode_step_counter = 0

    def _reshape_observations(self, obs):
        reshaped = []
        for index in range(len(obs)):
            reshaped.append(np.asarray(obs[index]).reshape(self.input_num))
        return reshaped

    def get_actions(self, observation):
        if np.random.random() > max(self.epsilon, self.epsilon_min):
            actions_q_value = self.sess.run(self.q_eval_outputs, feed_dict={self.X: np.asarray(observation).transpose()})
            actions = np.argmax(actions_q_value, axis=0)
        else:
            actions = np.random.randint(self.output_num, size=self.agents_num)
        return actions

    def _store_memory(self, observations, actions, rewards, new_observations):
        for i in range(len(actions)):
            index = self.episode_step_counter % self.memory_size
            self.observation_memory[:, index] = observations[i]
            self.action_memory[index] = actions[i]
            self.reward_memory[index] = rewards[i]
            self.new_observation_memory[:, index] = new_observations[i]
            self.episode_step_counter += 1

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards, new_observations)
        if self.episode_step_counter > 2000:
            self._train()

    def _train(self):
        index_range = min(self.episode_step_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)

        observations_sample = self.observation_memory[:, sample]
        new_observations_sample = self.new_observation_memory[:, sample]
        rewards_sample = self.reward_memory[sample]
        actions_sample = self.action_memory[sample]

        q_next_outputs = self.sess.run(self.q_eval_outputs, feed_dict={self.X: new_observations_sample})
        q_eval_outputs = self.sess.run(self.q_eval_outputs, feed_dict={self.X: observations_sample})

        q_target_outputs = q_eval_outputs.copy()
        batch_indexes = np.arange(self.batch_size, dtype=np.int32)
        actions_indexes = actions_sample.astype(int)

        q_target_outputs[actions_indexes, batch_indexes] = \
            rewards_sample + self.discount_rate * np.max(q_next_outputs, axis=0)
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.X: observations_sample,
                                                                       self.Y: q_target_outputs})
        self.epsilon -= self.decay_rate
        self.learn_step_counter += 1

    def post_episode_actions(self, rewards, episode):
        print("Random actions: {}".format(max(self.epsilon, self.epsilon_min)))
        self.save_model()
        self._init_episode()

    def save_model(self):
        self.saver.save(self.sess, "models/deep_q_learning/deep_q_learning.ckpt")
        print("Model Saved")

    def load_model(self):
        pass

    def _build_eval_network(self, layer_1_nodes, layer_2_nodes, W_init, b_init):
        self.X = tf.placeholder(tf.float32, [self.input_num, None], name='q_learning_s')
        self.Y = tf.placeholder(tf.float32, [self.output_num, None], name='q_learning_Q_target')

        with tf.variable_scope('q_learning_eval_net'):
            c_names = ['q_learning_eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [layer_1_nodes, self.input_num], initializer=W_init, collections=c_names)
                b1 = tf.get_variable('b1', [layer_1_nodes, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [layer_2_nodes, layer_1_nodes], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [layer_2_nodes, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.output_num, layer_2_nodes], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.output_num, 1], initializer=b_init, collections=c_names)
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X) + b1
                A1 = tf.nn.relu(Z1)
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu(Z2)
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_eval_outputs = Z3

        with tf.variable_scope('q_learning_loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval_outputs, self.Y))
        with tf.variable_scope('q_learning_train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
