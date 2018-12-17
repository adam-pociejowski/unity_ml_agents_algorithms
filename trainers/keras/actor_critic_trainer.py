from trainers.keras.keras_agent_trainer import *
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class ActorCriticTrainer(KerasAgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, agents_num, memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128,
                 model_name='actor_critic_keras', restore_model=False, discount_rate=0.99):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name=model_name, memory_size=memory_size, batch_size=batch_size,
                         restore_model=restore_model, discount_rate=discount_rate)
        self.save_summary = False
        self.learn_step_counter = 0

    def get_actions(self, observation):
        actions = []
        policy = self.actor.predict(observation, batch_size=self.agents_num)
        for i in range(self.agents_num):
            actions.append(np.random.choice(self.output_num, 1, p=policy[i])[0])
        return actions

    def post_step_actions(self, observations, actions, rewards, new_observations):
        self._store_memory(observations, actions, rewards, new_observations)
        if self.episode_step_counter > 1000:
            self._train()

    def post_episode_actions(self, rewards, episode):
        self.actor.save_weights("models/"+self.model_name+"/actor.h5")
        self.critic.save_weights("models/"+self.model_name+"/critic.h5")
        super().post_episode_actions(rewards, episode)

    def _train(self):
        index_range = min(self.episode_step_counter, self.memory_size)
        sample = np.random.choice(index_range, size=self.batch_size)
        observations_sample = self.observation_memory[sample, :]
        new_observations_sample = self.new_observation_memory[sample, :]
        rewards_sample = self.reward_memory[sample]
        actions_sample = self.action_memory[sample]

        value = self.critic.predict(observations_sample)[:, 0]
        next_value = self.critic.predict(new_observations_sample)[:, 0]
        advantages = np.zeros((self.batch_size, self.output_num))
        target = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            advantages[i][int(actions_sample[i])] = rewards_sample[i] + self.discount_rate * next_value[i] - value[i]

        target[:, 0] = rewards_sample + self.discount_rate * next_value
        self.actor.fit(observations_sample, advantages, epochs=1, verbose=0)
        self.critic.fit(observations_sample, target, epochs=1, verbose=0)

    def _init_model(self):
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _restore_model(self):
        self._init_model()
        self.actor.load_weights("models/" + self.model_name + "/actor.h5")
        self.critic.load_weights("models/" + self.model_name + "/critic.h5")

    def _build_actor(self):
        actor = Sequential()
        actor.add(Dense(self.layer_1_nodes, input_dim=self.input_num, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.layer_2_nodes, input_dim=self.layer_1_nodes, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.output_num, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return actor

    def _build_critic(self):
        critic = Sequential()
        critic.add(Dense(self.layer_1_nodes, input_dim=self.input_num, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.layer_2_nodes, input_dim=self.layer_1_nodes, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return critic
