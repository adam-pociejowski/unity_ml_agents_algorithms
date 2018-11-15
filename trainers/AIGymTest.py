import gym
from trainers.algorithms.q_learning_nn_trainer_experience_new import QLearningNNTrainerWithExperienceNew
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RENDER_ENV = False
EPISODES = 500
rewards = []
RENDER_REWARD_MIN = 800
total_steps_counter = 0


if __name__ == '__main__':
    trainer = QLearningNNTrainerWithExperienceNew('PPOBrain', input_num=4, output_num=2, agents_num=1,
                                                  memory_size=2000, batch_size=32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer.set_session(sess)

        for episode in range(400):
            observation = env.reset()
            episode_reward = 0

            while True:
                if RENDER_ENV:
                    env.render()

                actions = trainer.get_actions([observation])
                action = actions[0]
                new_observation, reward, done, info = env.step(action)

                x, x_dot, theta, theta_dot = new_observation
                r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
                step_reward = r1 + r2

                trainer.store_memory([observation], [action], [step_reward], [new_observation])
                trainer.post_step_actions([observation], [action])
                episode_reward += step_reward

                if done:
                    rewards.append(episode_reward)
                    max_reward_so_far = np.amax(rewards)
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", round(episode_reward, 2))
                    print("Epsilon: ", round(trainer.current_random_action_chance, 2))
                    print("Max reward so far: ", max_reward_so_far)
                    trainer.post_episode_actions([episode_reward], episode)

                    if max_reward_so_far > RENDER_REWARD_MIN:
                        RENDER_ENV = True
                    break
                observation = new_observation
                total_steps_counter += 1
