from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from trainers.keras.keras_agent_trainer import *


class DeepQLearningTrainer(KerasAgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, agents_num, memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128,
                 model_name='deep_q_learning', restore_model=False):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name=model_name, memory_size=memory_size, batch_size=batch_size,
                         restore_model=restore_model)
        self.learn_step_counter = 0

    def get_actions(self, observation):
        self.epsilon = 0
        if np.random.random() > self.epsilon:
            prediction = self.eval_network.predict(observation, batch_size=self.agents_num)
            actions = np.argmax(prediction, axis=1)
        else:
            actions = np.random.randint(self.output_num, size=self.agents_num)
        return actions

    def post_episode_actions(self, rewards, episode):
        self.eval_network.save_weights(self.models_dir + "/eval_network.h5")
        super().post_episode_actions(rewards, episode)

    def _train(self):
        observations_sample, new_observations_sample, rewards_sample, actions_sample = self._get_memory_samples()
        q_value = self.eval_network.predict(observations_sample, batch_size=self.batch_size)
        q_target = self.eval_network.predict(new_observations_sample, batch_size=self.batch_size)
        batch_indexes = np.arange(self.batch_size, dtype=np.int32)
        actions_indexes = actions_sample.astype(int)

        q_value[batch_indexes, actions_indexes] = rewards_sample + self.discount_rate * np.max(q_target, axis=1)
        self.eval_network.fit(observations_sample, q_value, epochs=1, verbose=0)
        self.learn_step_counter += 1

    def _init_model(self):
        self.eval_network = self._build_eval_network()

    def _restore_model(self):
        self._init_model()
        self.eval_network.load_weights(self.models_dir + "/eval_network.h5")

    def _build_eval_network(self):
        network = Sequential()
        network.add(Dense(self.layer_1_nodes, input_dim=self.input_num, activation='relu', kernel_initializer='he_uniform'))
        network.add(Dense(self.layer_2_nodes, input_dim=self.layer_1_nodes, activation='relu', kernel_initializer='he_uniform'))
        network.add(Dense(self.output_num, activation='linear', kernel_initializer='he_uniform'))
        network.summary()
        network.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return network
