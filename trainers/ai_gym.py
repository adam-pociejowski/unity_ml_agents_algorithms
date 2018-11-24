import gym
from trainers.algorithms.policy_gradients_trainer import PolicyGradientsTrainer
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

render = False
episodes = 5000
rewards = []
total_steps_counter = 0

if __name__ == '__main__':
    trainer = PolicyGradientsTrainer(None, 'PPOBrain', input_num=4, output_num=2, layer_1_nodes=10, layer_2_nodes=2,
                                     agents_num=1, learning_rate=0.01, discount_rate=0.95)
    trainer.init()
    for episode in range(episodes):
        observation = env.reset()
        episode_reward = 0

        while True:
            if render:
                env.render()

            actions = trainer.get_actions([observation])
            new_observation, reward, done, info = env.step(actions[0])
            episode_reward += reward
            trainer.post_step_actions([observation], actions, [reward], [new_observation])
            if done:
                rewards.append(episode_reward)
                max_reward_so_far = np.amax(rewards)
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", round(episode_reward, 2))
                print("Epsilon: ", round(trainer.epsilon, 2))
                print("Max reward so far: ", max_reward_so_far)
                trainer.post_episode_actions([episode_reward], episode)
                break
            observation = new_observation
            total_steps_counter += 1
