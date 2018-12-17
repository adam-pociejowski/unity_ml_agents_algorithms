import numpy as np
from gym_unity.envs import UnityEnv
from trainers.algorithms.deep_q_learning.deep_q_learning_trainer import DeepQLearningTrainer

number_of_agents = 40
number_of_episodes = 20
episode_max_length = 500


def run_episode(_env, _trainer):
    observation = _env.reset()
    total_rewards = [0.0] * number_of_agents
    for t in range(episode_max_length):
        actions = _trainer.get_actions(observation)
        new_observation, reward, done, info = _env.step(actions)
        _trainer.post_step_actions(observation, actions, reward, new_observation)
        total_rewards = [sum(x) for x in zip(total_rewards, reward)]
        observation = new_observation

    return total_rewards


if __name__ == '__main__':
    print('main')
    env_name = "../../env/multi-6-agent/Tanks"
    env = UnityEnv(env_name, worker_id=0, use_visual=False, multiagent=True)
    trainer = DeepQLearningTrainer(
                        None, 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                        layer_1_nodes=128, layer_2_nodes=128)

    for episode in range(number_of_episodes):
        rewards = run_episode(env, trainer)
        trainer.post_episode_actions(rewards, episode)
        best = np.amax(rewards)
        avg = np.average(rewards)
        print("Episode: {}, Avg: {}, Best: {}".format(episode + 1, avg, best))

    env.close()
