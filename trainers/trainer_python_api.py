from trainers.algorithms.genetic_algorithm_trainer import GeneticAlgorithmTrainer
from trainers.algorithms.q_learning_nn_trainer import QLearningNNTrainer
from trainers.algorithms.q_learning_old import QLearningNNTrainerOld
from trainers.algorithms.q_learning_nn_trainer_experience import QLearningNNTrainerWithExperience
from mlagents.envs import UnityEnvironment
import tensorflow as tf
from operator import add
import numpy as np

number_of_episodes = 40
episode_max_length = 5001
train_mode = True

class Model:
    def __init__(self, brain_name, brain, trainer, agents_num):
        self.brain_name = brain_name
        self.brain = brain
        self.trainer = trainer
        self.agents_num = agents_num
        self.actions_size = brain.vector_action_space_size


def get_actions(models, env_info):
    actions = {}
    for m in models:
        observations = env_info[m.brain_name].vector_observations
        action = m.trainer.get_actions(observations)
        actions[m.brain_name] = action
    return actions


def init_rewards_dict(models):
    rewards = {}
    for m in models:
        rewards[m.brain_name] = [0.0]*m.agents_num
    return rewards


def update_rewards(rewards, models, env_info):
    for m in models:
        model_rewards = env_info[m.brain_name].rewards
        observations = env_info[m.brain_name].vector_observations
        m.trainer.post_step_actions(observations, model_rewards)
        rewards[m.brain_name] = list(map(add, rewards[m.brain_name],  model_rewards))
    return rewards


def run_episode(models, env):
    env_info = env.reset(train_mode=train_mode)
    rewards = init_rewards_dict(models)
    rewards_log_copy = rewards.copy()
    for step in range(episode_max_length):
        actions = get_actions(models, env_info)
        env_info = env.step(actions)
        rewards = update_rewards(rewards, models, env_info)
        if step % 1000 == 0 and step > 0:
            rewards_log_copy = log_stats(models, rewards_log_copy, rewards, episode, step)
    return rewards


def post_episode_actions(models, rewards, episode):
    for m in models:
        model_rewards = rewards[m.brain_name]
        total_reward = sum(model_rewards)
        m.trainer.post_episode_actions(model_rewards, episode)
        print("[EPISODE END]: {:8}, avg: {:8.4f}, max: {:8.4f}\n".format(m.brain_name, total_reward / m.agents_num, np.amax(model_rewards)))


def log_stats(models, rewards_log_copy, rewards, episode, step):
    print("##########  EPISODE: {},  STEP: {}  ############".format(episode + 1, step))
    for m in models:
        rewards_change = [a_i - b_i for a_i, b_i in zip(rewards[m.brain_name], rewards_log_copy[m.brain_name])]
        best = np.amax(rewards_change)
        avg = np.average(rewards_change)
        print("[BRAIN]:       {:8}, avg: {:8.4f}, best: {:8.4f}".format(m.brain_name, avg, best))
    return rewards.copy()


def set_sessions(models, sess):
    for m in models:
        m.trainer.set_session(sess)


if __name__ == '__main__':
    env_name = "../../env/two-32v32/Tanks"
    env = UnityEnvironment(file_name=env_name, no_graphics=False)
    print('Brains: {}'.format(env.brains))
    # ppo_model = Model('PPOBrain',  env.brains['PPOBrain'],
    #                   GeneticAlgorithmTrainer('PPOBrain', number_of_observations=42, number_of_actions=6,
    #                                           number_of_chromosomes=35, number_of_elite_chromosomes=6,
    #                                           hidden_layer_nodes=128), 35)
    ga_model = Model('GABrain', env.brains['GABrain'],
                     GeneticAlgorithmTrainer('GABrain', number_of_observations=87, number_of_actions=6,
                                             number_of_chromosomes=32, number_of_elite_chromosomes=6,
                                             hidden_layer_nodes=128), 32)
    q_model = Model('PPOBrain', env.brains['PPOBrain'],
                    QLearningNNTrainerWithExperience('PPOBrain', input_num=87, output_num=6, agents_num=32), 32)
    # q_model2 = Model('GABrain', env.brains['GABrain'],
    #                  QLearningNNTrainerWithExperience('QLearningNNTrainerWithExperience', input_num=87, output_num=6, agents_num=16), 16)

    models = [ga_model, q_model]

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        set_sessions(models, sess)
        for episode in range(number_of_episodes):
            rewards = run_episode(models, env)
            post_episode_actions(models, rewards, episode)

    env.close()
