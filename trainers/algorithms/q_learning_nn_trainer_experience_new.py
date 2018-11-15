import numpy as np
import tensorflow as tf
import pandas as pd


class Memory:
    def __init__(self, observation, action, reward, new_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.new_observation = new_observation


class QLearningNNTrainerWithExperienceNew:
    def __init__(self, brain_name, input_num, output_num, agents_num, learning_rate=0.001, random_action_chance=0.1,
                 discount_rate=0.99, memory_size=20, batch_size=32):
        self.tag = '[QLearningNNTrainer - ' + brain_name + ']: '
        print(self.tag + ' started')
        self.brain_name = brain_name
        self.input_num = input_num
        self.output_num = output_num
        self.agents_num = agents_num
        self.random_action_chance = random_action_chance
        self.current_random_action_chance = random_action_chance
        self.discount_rate = discount_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.action_memory = np.zeros(self.memory_size)
        self.reward_memory = np.zeros(self.memory_size)
        self.observation_memory = np.zeros((self.memory_size, self.input_num))
        self.new_observation_memory = np.zeros((self.memory_size, self.input_num))
        self.memory_counter = 0

        self.agentsTargetQ = []
        self.actions = []
        self.observations = []
        self.sess = {}

        # Graph init
        self.X = tf.placeholder(tf.float32, [None, self.input_num])
        self.Y = tf.placeholder(tf.float32, [None, self.output_num])

        units_num = 10
        self.w1 = tf.Variable(tf.random_normal([self.input_num, units_num]))
        self.b1 = tf.Variable(tf.random_normal([units_num]))
        z1 = tf.add(tf.matmul(self.X, self.w1), self.b1)
        h1 = tf.nn.relu(z1)

        self.w2 = tf.Variable(tf.random_normal([units_num, units_num]))
        self.b2 = tf.Variable(tf.random_normal([units_num]))
        z2 = tf.add(tf.matmul(h1, self.w2), self.b2)
        h2 = tf.nn.relu(z2)

        self.w_out = tf.Variable(tf.random_normal([units_num, self.output_num]))
        self.b_out = tf.Variable(tf.random_normal([self.output_num]))
        self.Q = tf.add(tf.matmul(h2, self.w_out), self.b_out)

        self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.Q))
        self.predict = tf.argmax(self.Q, 1)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

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
        pass

    def _optimize(self):
        for agent in range(self.agents_num):
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: self.experience_batch[agent].observations,
                                                                            self.Y: self.experience_batch[agent].targetQ})

    def set_session(self, sess):
        self.sess = sess

    def get_actions(self, observation):
        self.observations = self._reshape_observations(observation)
        self.actions, self.agentsTargetQ = self.sess.run([self.predict, self.Q],
                                                         feed_dict={self.X: self.observations})
        return self._append_random_actions(self.actions)

    def store_memory(self, observations, rewards, actions, new_observations):
        for i in range(len(actions)):
            index = self.memory_counter % self.memory_size
            self.observation_memory[index, :] = observations[i]
            self.action_memory[index] = actions[i]
            self.reward_memory[index] = rewards[i]
            self.new_observation_memory[index, :] = new_observations[i]
            print('add: {}'.format(self.memory_counter))
            self.memory_counter += 1

    def post_step_actions(self, observations, rewards):
        index_range = min(self.memory_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)
        observations_sample = self.observation_memory[sample]
        new_observations_sample = self.new_observation_memory[sample]
        rewards_sample = self.reward_memory[sample]
        actions_sample = self.actions[sample]

    def post_episode_actions(self, rewards, episode):
        self.current_random_action_chance = (1.5**(-1*episode))*self.random_action_chance

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

