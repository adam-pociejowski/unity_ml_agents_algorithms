import numpy as np
import tensorflow as tf


class QLearningNNTrainerOld:
    def __init__(self, brain_name, input_num, output_num, agents_num, learning_rate=0.001, random_action_chance=0.1,
                 discount_rate=0.99):
        self.tag = '[QLearningNNTrainer - ' + brain_name + ']: '
        print(self.tag + ' started')
        self.brain_name = brain_name
        self.input_num = input_num
        self.output_num = output_num
        self.agents_num = agents_num
        self.random_action_chance = random_action_chance
        self.current_random_action_chance = random_action_chance
        self.discount_rate = discount_rate

        self.agentsTargetQ = []
        self.actions = []
        self.observations = []
        self.sess = {}
        self.init_graph(learning_rate)

    def init_graph(self, learning_rate):
        self.features = tf.placeholder(tf.float32, [None, self.input_num])
        self.Qnext = tf.placeholder(tf.float32, [None, self.output_num])

        units_num = 87
        self.w1 = tf.Variable(tf.random_normal([self.input_num, units_num]))
        self.b1 = tf.Variable(tf.random_normal([units_num]))
        z1 = tf.add(tf.matmul(self.features, self.w1), self.b1)
        h1 = tf.nn.relu(z1)

        self.w2 = tf.Variable(tf.random_normal([units_num, units_num]))
        self.b2 = tf.Variable(tf.random_normal([units_num]))
        z2 = tf.add(tf.matmul(h1, self.w2), self.b2)
        h2 = tf.nn.relu(z2)

        self.w_out = tf.Variable(tf.random_normal([units_num, self.output_num]))
        self.b_out = tf.Variable(tf.random_normal([self.output_num]))
        self.Q = tf.add(tf.matmul(h2, self.w_out), self.b_out)

        cost = tf.reduce_sum(tf.square(self.Qnext - self.Q))
        self.predict = tf.argmax(self.Q, 1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    def _reshape_observations(self, obs):
        return np.asarray(obs).reshape(1, self.input_num)

    def set_session(self, sess):
        self.sess = sess

    def get_actions(self, observation):
        self.actions = []
        self.agentsTargetQ = []
        self.observations = []
        for index in range(self.agents_num):
            agent_observation_reshaped = self._reshape_observations(observation[index])
            action, Q = self.sess.run([self.predict, self.Q], feed_dict={self.features: agent_observation_reshaped})
            action = action[0]
            if np.random.rand(1) < self.current_random_action_chance:
                action = np.random.randint(self.output_num)

            self.actions.append(action)
            self.agentsTargetQ.append(Q)
            self.observations.append(agent_observation_reshaped)
        return self.actions

    def post_step_actions(self, new_observation, rewards):
        for index in range(self.agents_num):
            if np.abs(rewards[index]) > 0.5:
                agent_observation_reshaped = self._reshape_observations(new_observation[index])
                new_Q = self.sess.run(self.Q, feed_dict={self.features: agent_observation_reshaped})

                target_Q = self.agentsTargetQ[index]
                target_Q[0, self.actions[index]] = rewards[index] + self.discount_rate * np.max(new_Q)
                self.sess.run(self.optimizer, feed_dict={self.features: self.observations[index],
                                                         self.Qnext: target_Q})

    def post_episode_actions(self, rewards, episode):
        self.current_random_action_chance = (1.5**(-1*episode))*self.random_action_chance
        print('Episode: {}, random_action_chance: {}'.format(episode, self.current_random_action_chance))
