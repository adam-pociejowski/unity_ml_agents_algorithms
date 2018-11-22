import numpy as np
import tensorflow as tf
from trainers.algorithms.agent_trainer import AgentTrainer


class PolicyGradientTrainer(AgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, layer_1_nodes, layer_2_nodes, agents_num,
                 learning_rate=0.001, discount_rate=0.99, batch_size=32, train_interval=1000):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name='policy_gradient',
                         discount_rate=discount_rate, batch_size=batch_size, learning_rate=learning_rate)
        self.train_interval = train_interval

    def _init_episode(self):
        self.step_counter = 0
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        for i in range(self.agents_num):
            self.episode_observations.append([])
            self.episode_actions.append([])
            self.episode_rewards.append([])

    def get_actions(self, observation):
        actions = []
        for i in range(len(observation)):
            probabilities = self.sess.run(self.outputs_softmax,
                                          feed_dict={self.X: np.asarray(observation[i]).reshape(self.input_num, 1)})
            action = np.random.choice(range(len(probabilities.ravel())), p=probabilities.ravel())
            actions.append(action)
        return actions

    def _store_memory(self, observation, actions, reward):
        for i in range(self.agents_num):
            action_vector = np.zeros(self.output_num)
            action_vector[int(actions[i])] = 1
            self.episode_actions[i].append(action_vector)
            self.episode_observations[i].append(observation[i])
            self.episode_rewards[i].append(reward[i])
            if reward[i] > 0.5:
                print('reward: {}', reward[i])
                self._train(i)

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards)
        self.step_counter += 1
        # if self.step_counter % self.train_interval == 0:
        #     self._train()

    def _train(self, agent_index):
        discounted_episode_rewards_norm = self.discount_and_norm_rewards(self.episode_rewards[agent_index])
        print('[train] index: {}, len: {}'.format(agent_index, len(self.episode_rewards[agent_index])))
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(self.episode_observations[agent_index]).T,
            self.Y: np.vstack(np.array(self.episode_actions[agent_index])).T,
            self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })
        self.episode_rewards[agent_index] = []
        self.episode_actions[agent_index] = []
        self.episode_observations[agent_index] = []

    def discount_and_norm_rewards(self, agent_episode_rewards):
        discounted_episode_rewards = np.zeros_like(agent_episode_rewards)
        cumulative_reward = 0
        for i in reversed(range(len(agent_episode_rewards))):
            cumulative_reward = cumulative_reward * self.discount_rate + agent_episode_rewards[i]
            discounted_episode_rewards[i] = cumulative_reward

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        std_discounted_episode_rewards = np.std(discounted_episode_rewards)
        if std_discounted_episode_rewards != 0.0:
            discounted_episode_rewards /= std_discounted_episode_rewards
        return discounted_episode_rewards

    def post_episode_actions(self, rewards, episode):
        self._save_model(None)
        self.learning_rate *= .8

    def _init_model(self):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.input_num, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.output_num, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [self.layer_1_nodes, self.input_num],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [self.layer_1_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [self.layer_2_nodes, self.layer_1_nodes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [self.layer_2_nodes, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.output_num, self.layer_2_nodes],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.output_num, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1, self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)

        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)