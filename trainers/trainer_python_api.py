from trainers.algorithms.genetic_algorithm_trainer import GeneticAlgorithmTrainer
from trainers.algorithms.q_learning_nn_trainer import QLearningNNTrainer
from trainers.algorithms.q_learning_nn_trainer_experience import QLearningNNTrainerWithExperience
from trainers.algorithms.deep_q_learning_nn_trainer_two_networks import DeepQLearningNNTrainerTwoNetworks
from trainers.algorithms.deep_q_learning_nn_trainer import DeepQLearningNNTrainer
from trainers.algorithms.policy_gradient_trainer import PolicyGradientTrainer
from trainers.algorithms.policy_gradient_trainer2 import PolicyGradientTrainer2
from mlagents.envs import UnityEnvironment
from trainers.trainer_python_api_utils import *

number_of_episodes = 40
episode_max_length = 5000
train_mode = True


if __name__ == '__main__':
    # env_name = "../../env/two-4v4/Tanks"
    env_name = "../../env/single-64/Tanks"
    env = UnityEnvironment(worker_id=0, file_name=env_name, no_graphics=False)
    print('Brains: {}'.format(env.brains))
    # model = Model('PPOBrain', env.brains['PPOBrain'],
    #               GeneticAlgorithmTrainer('Genetic', number_of_observations=87, number_of_actions=6,
    #                                       number_of_chromosomes=64, number_of_elite_chromosomes=6,
    #                                       hidden_layer_nodes=128), 64)
    model = Model('PPOBrain', env.brains['PPOBrain'],
                  DeepQLearningNNTrainer('QLearning1', input_num=87, output_num=6, agents_num=64,
                                         memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128), 64)
    # model = Model('GABrain', env.brains['GABrain'],
    #                DeepQLearningNNTrainerTwoNetworks('QLearning2', input_num=87, output_num=6, agents_num=4,
    #                                                  memory_size=2000, batch_size=64, layer_1_nodes=128,
    #                                                  layer_2_nodes=128), 4)
    # model = Model('PPOBrain', env.brains['PPOBrain'],
    #               PolicyGradientTrainer('PolicyGradient', input_num=87, output_num=6, layer_1_nodes=128,
    #                                     layer_2_nodes=128, agents_num=8), 8)
    # model = Model('PPOBrain', env.brains['PPOBrain'],
    #               PolicyGradientTrainer('PolicyGradient', input_num=87, output_num=6, layer_1_nodes=128,
    #                                     layer_2_nodes=128, agents_num=64), 64)
    models = [model]

    for episode in range(number_of_episodes):
        rewards = run_episode(models, env, episode, train_mode, episode_max_length)
        post_episode_actions(models, rewards, episode)

    env.close()
