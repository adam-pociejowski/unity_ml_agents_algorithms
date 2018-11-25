from trainers.algorithms.deep_q_learning_improved_trainer import *
from trainers.algorithms.genetic_algorithm_trainer import *
from trainers.algorithms.policy_gradients_agent_batch_trainer import *
from trainers.algorithms.policy_gradients_trainer import *
from trainers.algorithms.deep_q_learning_trainer import *
from trainers.trainer_python_api_utils import *
from mlagents.envs import UnityEnvironment

NUMBER_OF_EPISODES = 1000
EPISODE_LENGTH = 1001
LOG_INTERVAL = 1000
train_mode = True


if __name__ == '__main__':
    # env_name = "../../env/two-4v4/Tanks"
    env_name = "../../env/single-64/Tanks"
    env = UnityEnvironment(worker_id=0, file_name=env_name, no_graphics=True)
    print('Brains: {}'.format(env.brains))
    genetic_trainer = GeneticAlgorithmTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64,
                            elite_chromosomes=6, hidden_layer_nodes=128)

    q_learn_trainer = DeepQLearningTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64,
                            memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128)

    q_learn_improved_trainer = DeepQLearningImprovedTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64,
                            memory_size=5000, batch_size=32, layer_1_nodes=128, layer_2_nodes=128)

    policy_gradients = PolicyGradientsTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64,
                            layer_1_nodes=128, layer_2_nodes=128, discount_rate=0.95, learning_rate=0.001)

    policy_gradients_agent_batch = PolicyGradientsAgentBatchTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64,
                            layer_1_nodes=128, layer_2_nodes=128, discount_rate=0.95, learning_rate=0.001)

    trainers = [policy_gradients_agent_batch]
    for i in range(len(trainers)):
        trainers[i].init()

    for episode in range(NUMBER_OF_EPISODES):
        run_episode(trainers, env, episode, train_mode, EPISODE_LENGTH)

    env.close()
