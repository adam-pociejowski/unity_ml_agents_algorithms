import numpy as np
import tensorflow as tf
import pandas as pd


class ExperienceBatch:
    def __init__(self, observations, targetQ):
        self.observations = observations
        self.targetQ = targetQ


class QLearningNNTrainerWithExperience:
    def __init__(self, brain_name, input_num, output_num, agents_num, learning_rate=0.001, random_action_chance=0.1,
                 discount_rate=0.9, experience_batch_size=20):
        self.tag = '[QLearningNNTrainer - ' + brain_name + ']: '
        print(self.tag + ' started')
        self.brain_name = brain_name
        self.input_num = input_num
        self.output_num = output_num
        self.agents_num = agents_num
        self.random_action_chance = random_action_chance
        self.current_random_action_chance = random_action_chance
        self.discount_rate = discount_rate
        self.experience_batch_size = experience_batch_size
        self.experience_batch = self._init_experience_batch()

        self.agentsTargetQ = []
        self.actions = []
        self.observations = []
        self.sess = {}
        self.init_graph(learning_rate)

    def _init_experience_batch(self):
        experience_batch = []
        for _ in range(self.agents_num):
            experience_batch.append(ExperienceBatch([], []))
        return experience_batch

    def init_graph(self, learning_rate):
        self.features = tf.placeholder(tf.float32, [None, self.input_num])
        self.Qnext = tf.placeholder(tf.float32, [None, self.output_num])

        units_num = 87
        self.w1 = tf.Variable(tf.random_normal([self.input_num, units_num]))
        self.b1 = tf.Variable(tf.random_normal([units_num]))
        z1 = tf.add(tf.matmul(self.features, self.w1), self.b1)
        h1 = tf.nn.relu(z1)

        # self.w2 = tf.Variable(tf.random_normal([units_num, units_num]))
        # self.b2 = tf.Variable(tf.random_normal([units_num]))
        # z2 = tf.add(tf.matmul(h1, self.w2), self.b2)
        # h2 = tf.nn.relu(z2)

        self.w_out = tf.Variable(tf.random_normal([units_num, self.output_num]))
        self.b_out = tf.Variable(tf.random_normal([self.output_num]))
        self.Q = tf.add(tf.matmul(h1, self.w_out), self.b_out)

        cost = tf.reduce_sum(tf.square(self.Qnext - self.Q))
        self.predict = tf.argmax(self.Q, 1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    def _reshape_observations(self, obs):
        reshaped = []
        for index in range(len(obs)):
            reshaped.append(np.asarray(obs[index]).reshape(self.input_num))
        return reshaped

    def _append_random_actions(self, actions):
        for index in range(len(actions)):
            if np.random.rand(1) < self.current_random_action_chance:
                actions[index] = np.random.randint(self.output_num)
        return actions

    def _update_experience_batch(self, observations, agentsTargetQ):
        for agent in range(self.agents_num):
            self.experience_batch[agent].observations.append(observations[agent])
            self.experience_batch[agent].targetQ.append(agentsTargetQ[agent])
            if len(self.experience_batch[agent].observations) > self.experience_batch_size:
                self.experience_batch[agent].observations.pop(0)
                self.experience_batch[agent].targetQ.pop(0)

    def _optimize(self):
        for agent in range(self.agents_num):
            self.sess.run(self.optimizer, feed_dict={self.features: self.experience_batch[agent].observations,
                                                     self.Qnext: self.experience_batch[agent].targetQ})

    def set_session(self, sess):
        self.sess = sess

    def get_actions(self, observation):
        self.observations = self._reshape_observations(observation)
        self.actions, self.agentsTargetQ = self.sess.run([self.predict, self.Q],
                                                         feed_dict={self.features: self.observations})
        return self._append_random_actions(self.actions)

    def post_step_actions(self, observations, actions, rewards, new_observations):
        agent_observation_reshaped = self._reshape_observations(new_observations)
        new_Q = self.sess.run(self.Q, feed_dict={self.features: agent_observation_reshaped})
        for index in range(self.agents_num):
            self.agentsTargetQ[index][self.actions[index]] = rewards[index] + self.discount_rate * np.max(new_Q[index])

        self._update_experience_batch(self.observations, self.agentsTargetQ)
        self._optimize()

    def post_episode_actions(self, rewards, episode):
        self.current_random_action_chance = (1.5**(-1*episode))*self.random_action_chance
        print('Episode: {}, random_action_chance: {}'.format(episode, self.current_random_action_chance))
        # self.save_model()

    def save_model(self):
        w1, b1, w2, b2, w_out, b_out = self.sess.run([self.w1, self.b1, self.w2, self.b2, self.w_out, self.b_out])
        model_to_save = np.asarray(w1).reshape(self.input_num, self.input_num)
        model_to_save = np.append(model_to_save, np.asarray(b1).reshape(1, self.input_num), axis=0)
        model_to_save = np.append(model_to_save, np.asarray(w2).reshape(self.input_num, self.input_num), axis=0)
        model_to_save = np.append(model_to_save, np.asarray(b2).reshape(1, self.input_num), axis=0)
        w_out_extended = np.append(np.asarray(w_out), np.zeros([self.input_num, self.input_num - self.output_num]), axis=1)
        model_to_save = np.append(model_to_save, np.asarray(w_out_extended).reshape(self.input_num, self.input_num), axis=0)
        b_out_extended = np.append(np.asarray(b_out), np.zeros([1, self.input_num - self.output_num]))
        model_to_save = np.append(model_to_save, np.asarray(b_out_extended).reshape(1, self.input_num), axis=0)
        df = pd.DataFrame(model_to_save)
        df.to_csv('models/' + self.brain_name + '.csv', index=False)

    def load_model(self):
        df = pd.read_csv('models/' + self.brain_name + '.csv')
        d = df.as_matrix()

