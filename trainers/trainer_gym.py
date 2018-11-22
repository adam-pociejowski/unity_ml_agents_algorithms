from trainers.algorithms.genetic_algorithm_trainer import GeneticAlgorithmTrainer
from gym_unity.envs import UnityEnv
import numpy as np

number_of_agents = 40
number_of_episodes = 20
episode_max_length = 500


def run_episode(env, trainer):
    observation = env.reset()
    total_rewards = [0.0] * number_of_agents
    # Running episode for every agent
    for t in range(episode_max_length):
        actions = trainer.get_actions(observation)
        observation, reward, done, info = env.step(actions)
        total_rewards = [sum(x) for x in zip(total_rewards, reward)]

    return total_rewards


if __name__ == '__main__':
    print('main')
    env_name = "../../env/multi-6-agent/Tanks"
    env = UnityEnv(env_name, worker_id=0, use_visual=False, multiagent=True)
    trainer = GeneticAlgorithmTrainer('PPOBrain', number_of_observations=42, number_of_actions=6,
                                      number_of_chromosomes=40, elite_chromosomes=8, hidden_layer_nodes=128)
    for episode in range(number_of_episodes):
        rewards = run_episode(env, trainer)
        trainer.update_model(rewards)
        best = np.amax(rewards)
        avg = np.average(rewards)
        print("Episode: {}, Avg: {}, Best: {}".format(episode + 1, avg, best))

    env.close()
