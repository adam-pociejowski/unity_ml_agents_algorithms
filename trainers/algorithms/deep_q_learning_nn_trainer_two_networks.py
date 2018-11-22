import numpy as np
import tensorflow as tf
from trainers.algorithms.agent_trainer import AgentTrainer


class DeepQLearningNNTrainerTwoNetworks(AgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, agents_num, memory_size=1000, batch_size=32,
                 layer_1_nodes=10, layer_2_nodes=10):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name='deep_q_learning_two_nets',
                         memory_size=memory_size, batch_size=batch_size)
        self.replace_target_iter = 100
        self.learn_step_counter = 0
        self.memory_counter = 0

    def _init_episode(self):
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = np.zeros((self.input_num, self.memory_size))
        self.new_observation_memory = np.zeros((self.input_num, self.memory_size))

    def get_actions(self, observation):
        if np.random.random() > max(self.epsilon, self.epsilon_min):
            actions_q_value = self.sess.run(self.q_eval_outputs, feed_dict={self.X: np.asarray(observation).transpose()})
            actions = np.argmax(actions_q_value, axis=0)
        else:
            actions = np.random.randint(self.output_num, size=self.agents_num)
        return actions

    def _store_memory(self, observations, actions, rewards, new_observations):
        for i in range(len(actions)):
            index = self.memory_counter % self.memory_size
            self.observation_memory[:, index] = observations[i]
            self.action_memory[index] = actions[i]
            self.reward_memory[index] = rewards[i]
            self.new_observation_memory[:, index] = new_observations[i]
            self.memory_counter += 1

    def _replace_target_net_parameters(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards, new_observations)
        if self.memory_counter > 1000:
            self._train()

    def _train(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_net_parameters()

        index_range = min(self.memory_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)

        observations_sample = self.observation_memory[:, sample]
        new_observations_sample = self.new_observation_memory[:, sample]
        rewards_sample = self.reward_memory[sample]
        actions_sample = self.action_memory[sample]

        q_next_outputs, q_eval_outputs = self.sess.run([self.q_next_outputs, self.q_eval_outputs], feed_dict={
            self.X_: new_observations_sample,
            self.X: observations_sample
        })

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
        super()._save_model(None)
        self.learning_rate *= .75

    def _init_model(self):
        W_init = tf.contrib.layers.xavier_initializer(seed=1)
        b_init = tf.contrib.layers.xavier_initializer(seed=1)
        self._build_eval_network(self.layer_1_nodes, self.layer_2_nodes, W_init, b_init)
        self._build_target_network(self.layer_1_nodes, self.layer_2_nodes, W_init, b_init)

    def _build_eval_network(self, layer_1_nodes, layer_2_nodes, W_init, b_init):
        self.X = tf.placeholder(tf.float32, [self.input_num, None], name='s')
        self.Y = tf.placeholder(tf.float32, [self.output_num, None], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
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

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval_outputs, self.Y))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_target_network(self, layer_1_nodes, layer_2_nodes, W_init, b_init):
        self.X_ = tf.placeholder(tf.float32, [self.input_num, None], name="s_")

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [layer_1_nodes, self.input_num], initializer=W_init, collections=c_names)
                b1 = tf.get_variable('b1', [layer_1_nodes, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [layer_2_nodes, layer_1_nodes], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [layer_2_nodes, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.output_num, layer_2_nodes], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.output_num, 1], initializer=b_init, collections=c_names)
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X_) + b1
                A1 = tf.nn.relu(Z1)
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu(Z2)
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_next_outputs = Z3
