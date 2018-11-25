import numpy as np
import tensorflow as tf
from trainers.agent_trainer import AgentTrainer
from trainers.algorithms.model.agent_experience import AgentExperience


class PolicyGradientsAgentBatchTrainer(AgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, layer_1_nodes, layer_2_nodes, agents_num,
                 learning_rate=0.0001, discount_rate=0.9, batch_size=32, train_interval=1000, epochs=1):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name='policy_gradients',
                         discount_rate=discount_rate, batch_size=batch_size, learning_rate=learning_rate,
                         epsilon_max=0.0, epsilon_min=0.0)
        self.train_interval = train_interval

    def _init_episode(self):
        self.episode_agents_experience = []
        for i in range(self.agents_num):
            self.episode_agents_experience.append(AgentExperience([], [], []))

        self.train_interval_counter = 0
        self.selected_actions = np.zeros(self.output_num, dtype=int)

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
            self.selected_actions[int(actions[i])] += 1
            self.episode_agents_experience[i].observations.append(observation[i])
            self.episode_agents_experience[i].actions.append(action_vector)
            self.episode_agents_experience[i].rewards.append(reward[i])

    def post_step_actions(self, observations, actions, rewards, new_observations):
        super().post_step_actions(observations, actions, rewards, new_observations)
        self._store_memory(observations, actions, rewards)

    def _train(self):
        for i in range(self.agents_num):
            agent_experience = self.episode_agents_experience[i]
            discounted_rewards = self.discount_and_norm_rewards(agent_experience)
            _, self.current_loss = self.sess.run([self.train_op, self.loss],
                                                 feed_dict={self.X: np.vstack(agent_experience.observations).T,
                                                            self.Y: np.vstack(np.array(agent_experience.actions)).T,
                                                            self.discounted_episode_rewards_norm: discounted_rewards
                                                            })

    def discount_and_norm_rewards(self, agent_experience):
        discounted_episode_rewards = np.zeros_like(agent_experience.rewards)
        cumulative_reward = 0
        for i in reversed(range(len(agent_experience.rewards))):
            cumulative_reward = cumulative_reward * self.discount_rate + agent_experience.rewards[i]
            discounted_episode_rewards[i] = cumulative_reward

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        std_discounted_episode_rewards = np.std(discounted_episode_rewards)
        if std_discounted_episode_rewards != 0.0:
            discounted_episode_rewards /= std_discounted_episode_rewards

        return discounted_episode_rewards

    def post_episode_actions(self, rewards, episode):
        self._train()
        self._init_episode()

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
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            self.loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
