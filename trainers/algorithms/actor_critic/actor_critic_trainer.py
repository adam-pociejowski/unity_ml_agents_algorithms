import numpy as np

from trainers.agent_trainer import AgentTrainer
from trainers.algorithms.model.layers import *


class ActorCriticTrainer(AgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, agents_num, memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128,
                 model_name='actor_critic', restore_model=False):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name=model_name, memory_size=memory_size, batch_size=batch_size,
                         restore_model=restore_model)
        self.learn_step_counter = 0

    def get_actions(self, observation):
        actions = []
        for i in range(len(observation)):
            probabilities = self.sess.run(self.actor.outputs_softmax,
                                          feed_dict={self.actor.X: np.asarray(observation[i]).reshape(self.input_num, 1)})
            action = np.random.choice(range(len(probabilities.ravel())), p=probabilities.ravel())
            actions.append(action)

        return actions

    def post_step_actions(self, observations, actions, rewards, new_observations):
        super().post_step_actions(observations, actions, rewards, new_observations)
        self._store_memory(observations, actions, rewards, new_observations)
        if self.episode_step_counter > 2000:
            self._train()

    def post_episode_actions(self, rewards, episode):
        del self.action_memory
        del self.reward_memory
        del self.observation_memory
        del self.new_observation_memory
        self._save_model(None)
        self._init_episode()

    def _train(self):
        index_range = min(self.episode_step_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)
        observations_sample = self.observation_memory[:, sample]
        new_observations_sample = self.new_observation_memory[:, sample]
        rewards_sample = self.reward_memory[sample]
        actions_sample = self.action_memory[sample]
        td_error = self.critic.learn(observations_sample, rewards_sample, new_observations_sample)
        action_sample_vectors = []
        for i in range(len(actions_sample)):
            action_vector = np.zeros(self.output_num)
            action_vector[int(actions_sample[i])] = 1
            action_sample_vectors.append(action_vector)

        action_sample_vectors = np.asarray(action_sample_vectors)
        self.actor.learn(observations_sample, action_sample_vectors.transpose(), td_error)

    def _store_memory(self, observations, actions, rewards, new_observations):
        for i in range(len(actions)):
            index = self.episode_step_counter % self.memory_size
            self.observation_memory[:, index] = observations[i]
            self.action_memory[index] = actions[i]
            self.reward_memory[index] = rewards[i]
            self.new_observation_memory[:, index] = new_observations[i]
            self.episode_step_counter += 1

    def init(self):
        super().init()
        self.actor.init(self.sess)
        self.critic.init(self.sess)

    def _init_episode(self):
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = np.zeros((self.input_num, self.memory_size))
        self.new_observation_memory = np.zeros((self.input_num, self.memory_size))
        self.episode_step_counter = 0

    def _init_model(self):
        self.actor = Actor(self.input_num, self.output_num, self.layer_1_nodes, self.layer_2_nodes, self.learning_rate, self.model_name)
        self.critic = Critic(self.input_num, self.output_num, self.layer_1_nodes, self.layer_2_nodes, self.learning_rate, self.discount_rate, self.model_name)


class Actor:

    def __init__(self, input_num, output_num, layer_1_nodes, layer_2_nodes, learning_rate, model_name):
        self.input_num = input_num
        self.model_name = model_name
        self.output_num = output_num
        self.learning_rate = learning_rate
        self.sess = None
        self._build_network(layer_1_nodes, layer_2_nodes)

    def init(self, sess):
        self.sess = sess

    def _build_network(self, layer_1_nodes, layer_2_nodes):
        with tf.name_scope('inputs' + self.model_name):
            self.X = tf.placeholder(tf.float32, shape=(self.input_num, None), name="X" + self.model_name)
            self.Y = tf.placeholder(tf.float32, shape=(self.output_num, None), name="Y" + self.model_name)
            self.V = tf.placeholder(tf.float32, None, name="actions_value" + self.model_name)

        with tf.name_scope('parameters' + self.model_name):
            W1 = tf.get_variable("W1", [layer_1_nodes, self.input_num],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [layer_1_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [layer_2_nodes, layer_1_nodes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [layer_2_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.output_num, layer_2_nodes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.output_num, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

        with tf.name_scope('layer_1' + self.model_name):
            Z1 = tf.add(tf.matmul(W1, self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2' + self.model_name):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3' + self.model_name):
            Z3 = tf.add(tf.matmul(W3, A2), b3)

        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        with tf.name_scope('loss' + self.model_name):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            self.loss = tf.reduce_mean(neg_log_prob * self.V)
        with tf.name_scope('train' + self.model_name):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, observations, actions, values):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: observations,
                                                                       self.Y: actions,
                                                                       self.V: values})
        return loss

    def choose_action(self, observations):
        probabilities = self.sess.run(self.outputs_softmax, {self.X: np.asarray(observations).reshape(self.input_num, 1)})
        return np.random.choice(np.arange(probabilities.shape[1]), p=probabilities.ravel())


class Critic:

    def __init__(self, input_num, output_num, layer_1_nodes, layer_2_nodes, learning_rate, discount_rate, model_name):
        self.input_num = input_num
        self.model_name = model_name
        self.output_num = output_num
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.sess = None
        self._build_network(layer_1_nodes, layer_2_nodes)

    def init(self, sess):
        self.sess = sess

    def _build_network(self, layer_1_nodes, layer_2_nodes):
        self._X = tf.placeholder(tf.float32, [self.input_num, None], name="critic_X" + self.model_name)
        self._Y = tf.placeholder(tf.float32, [1, None], name='critic_Y' + self.model_name)
        self._reward = tf.placeholder(tf.float32, [None, ], name='critic_reward' + self.model_name)

        with tf.variable_scope('critic_net' + self.model_name):
            with tf.variable_scope('parameters' + self.model_name):
                W1 = tf.get_variable('W1', [layer_1_nodes, self.input_num], initializer=tf.contrib.layers.xavier_initializer(seed=1))
                b1 = tf.get_variable('b1', [layer_1_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
                W2 = tf.get_variable('W2', [layer_2_nodes, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=1))
                b2 = tf.get_variable('b2', [layer_2_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
                W3 = tf.get_variable('W3', [1, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=1))
                b3 = tf.get_variable('b3', [1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            with tf.variable_scope('layer_1' + self.model_name):
                Z1 = tf.matmul(W1, self._X) + b1
                A1 = tf.nn.relu(Z1)
            with tf.variable_scope('layer_2' + self.model_name):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu(Z2)
            with tf.variable_scope('layer_3' + self.model_name):
                self.V = tf.matmul(W3, A2) + b3

        with tf.variable_scope('critic_loss' + self.model_name):
            self.td_error = self._reward + self.discount_rate * self._Y - self.V
            self.critic_loss = tf.square(self.td_error)
        with tf.variable_scope('critic_train' + self.model_name):
            self.critic_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def learn(self, observations, rewards, new_observations):
        v = self.sess.run(self.V, {self._X: new_observations})
        td_error, _ = self.sess.run([self.td_error, self.critic_train_op], feed_dict={self._X: observations,
                                                                                      self._Y: v,
                                                                                      self._reward: rewards})
        return td_error
