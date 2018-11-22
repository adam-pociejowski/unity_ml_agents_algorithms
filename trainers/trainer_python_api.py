from trainers.algorithms.genetic_algorithm_trainer import *
from trainers.algorithms.deep_q_learning_nn_trainer_two_networks import *
from trainers.algorithms.policy_gradient_trainer import *
from trainers.algorithms.deep_q_learning_nn_trainer import *
from mlagents.envs import UnityEnvironment
from trainers.trainer_python_api_utils import *

number_of_episodes = 40
episode_max_length = 5000
train_mode = True


if __name__ == '__main__':
    # env_name = "../../env/two-4v4/Tanks"
    env_name = "../../env/single-64/Tanks"
    env = UnityEnvironment(worker_id=0, file_name=env_name, no_graphics=True)
    print('Brains: {}'.format(env.brains))
    genetic_trainer = GeneticAlgorithmTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6,
                                              agents_num=64, elite_chromosomes=6, hidden_layer_nodes=128)

    q_learn_trainer = DeepQLearningNNTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6,
                            agents_num=64, memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128)

    q_learn_trainer_2_nets = DeepQLearningNNTrainerTwoNetworks(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6,
                            agents_num=64, memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128)

    policy_gradient = PolicyGradientTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, layer_1_nodes=128,
                            layer_2_nodes=128, agents_num=8)
    trainers = [q_learn_trainer]

    for episode in range(number_of_episodes):
        run_episode(trainers, env, episode, train_mode, episode_max_length)

    env.close()
