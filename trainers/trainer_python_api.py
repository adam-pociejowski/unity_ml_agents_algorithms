from mlagents.envs import UnityEnvironment
from trainers.algorithms.deep_q_learning.deep_q_learning_improved_trainer import *
from trainers.algorithms.deep_q_learning.deep_q_learning_trainer import *
from trainers.algorithms.genetic.genetic_algorithm_trainer import *
from trainers.algorithms.genetic.genetic_algorithm_trainer_old import *
from trainers.keras.actor_critic_trainer import *
from trainers.keras.deep_q_learning_improved_trainer import *
from trainers.keras.deep_q_learning_trainer import *
from trainers.trainer_python_api_utils import *

if __name__ == '__main__':
    env_name = "../../env/single-64-training/Tanks"
    env = UnityEnvironment(worker_id=0, file_name=env_name, no_graphics=True)
    print('Brains: {}'.format(env.brains))
    genetic_trainer = GeneticAlgorithmOldTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, elite_chromosomes=6,
                            hidden_layer_nodes=128)

    genetic_trainer_new = GeneticAlgorithmTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=64, layer_2_nodes=64,
                            elite_chromosomes=8)

    q_learn_trainer = DeepQLearningTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128, model_name='deep_q_learning_6')

    q_learn_improved_trainer = DeepQLearningImprovedTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128)

    ########################### KERAS
    actor_critic_keras = ActorCriticTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                            discount_rate=0.99)

    q_learn_trainer_keras = KerasDeepQLearningTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128)

    q_learn_trainer_improved_keras = KerasDeepQLearningImprovedTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128)

    trainers = [q_learn_improved_trainer]
    for i in range(len(trainers)):
        trainers[i].init()

    for episode in range(40):
        run_episode(trainers, env, episode, train_mode=True, episode_max_length=5001, log_interval=1000, verbose=True)
    env.close()
