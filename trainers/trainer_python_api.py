# from trainers.algorithms.deep_q_learning.deep_q_learning_improved_trainer import *
from trainers.keras.deep_q_learning_improved_trainer import *
from trainers.algorithms.genetic.genetic_algorithm_trainer_old import *
from trainers.algorithms.genetic.genetic_algorithm_trainer import *
from trainers.algorithms.policy_gradients.policy_gradients_agent_batch_trainer import *
from trainers.algorithms.policy_gradients.policy_gradients_trainer import *
from trainers.algorithms.deep_q_learning.deep_q_learning_trainer import *
from trainers.algorithms.actor_critic.actor_critic_trainer import *
from trainers.algorithms.actor_critic.actor_critic_keras_trainer import *
from trainers.trainer_python_api_utils import *
from mlagents.envs import UnityEnvironment


if __name__ == '__main__':
    env_name = "../../env/single-64/Tanks"
    env = UnityEnvironment(worker_id=9999, file_name=env_name, no_graphics=True)
    print('Brains: {}'.format(env.brains))
    genetic_trainer = GeneticAlgorithmOldTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, elite_chromosomes=6,
                            hidden_layer_nodes=128)

    genetic_trainer_new = GeneticAlgorithmTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=64, layer_2_nodes=64,
                            elite_chromosomes=8)

    q_learn_trainer = DeepQLearningTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128)

    q_learn_improved_trainer = DeepQLearningImprovedTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                            layer_1_nodes=128, layer_2_nodes=128)

    policy_gradients = PolicyGradientsTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                            discount_rate=0.95, learning_rate=0.001)

    policy_gradients_agent_batch = PolicyGradientsAgentBatchTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                            discount_rate=0.99, learning_rate=0.01)

    actor_critic = ActorCriticTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128)

    actor_critic_keras = ActorCriticKerasTrainer(
                            env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                            discount_rate=0.99)

    trainers = [q_learn_improved_trainer]
    for i in range(len(trainers)):
        trainers[i].init()

    for episode in range(1000):
        run_episode(trainers, env, episode, train_mode=True, episode_max_length=5001, log_interval=1000, verbose=True)
    env.close()
