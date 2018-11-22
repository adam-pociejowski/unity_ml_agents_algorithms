import numpy as np
from operator import add
import tensorflow as tf


class Model:
    def __init__(self, brain_name, brain, trainer, agents_num):
        self.brain_name = brain_name
        self.brain = brain
        self.trainer = trainer
        self.agents_num = agents_num
        self.actions_size = brain.vector_action_space_size


def update_rewards(rewards, models, new_env_info, actions, env_info):
    for m in models:
        brain_rewards = new_env_info[m.brain_name].rewards
        brain_new_observations = new_env_info[m.brain_name].vector_observations
        brain_observations = env_info[m.brain_name].vector_observations
        brain_actions = actions[m.brain_name]
        m.trainer.post_step_actions(brain_observations, brain_actions, brain_rewards, brain_new_observations)
        rewards[m.brain_name] = list(map(add, rewards[m.brain_name],  brain_rewards))
    return rewards


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


def run_episode(models, env, episode, train_mode, episode_max_length):
    env_info = env.reset(train_mode=train_mode)
    rewards = init_rewards_dict(models)
    rewards_log_copy = rewards.copy()
    for step in range(episode_max_length):
        actions = get_actions(models, env_info)
        new_env_info = env.step(actions)
        rewards = update_rewards(rewards, models, new_env_info, actions, env_info)
        if step % 1000 == 0 and step > 0:
            rewards_log_copy = log_stats(models, rewards_log_copy, rewards, episode, step)
        env_info = new_env_info
    return rewards


def post_episode_actions(models, rewards, episode):
    print("\n################  EPISODE END: {} ################".format(episode + 1))
    for m in models:
        model_rewards = rewards[m.brain_name]
        total_reward = sum(model_rewards)
        m.trainer.post_episode_actions(model_rewards, episode)
        print("[BRAIN]:       {:8}, avg: {:8.4f}, max: {:8.4f}".format(m.brain_name, total_reward / m.agents_num, np.amax(model_rewards)))
    print("\n")


def log_stats(models, rewards_log_copy, rewards, episode, step):
    print("##########  EPISODE: {},  STEP: {}  ############".format(episode + 1, step))
    for m in models:
        rewards_change = [a_i - b_i for a_i, b_i in zip(rewards[m.brain_name], rewards_log_copy[m.brain_name])]
        best = np.amax(rewards_change)
        avg = np.average(rewards_change)
        print("[BRAIN]:       {:8}, avg: {:8.4f}, best: {:8.4f}".format(m.brain_name, avg, best))
        summary, _, _, _ = sess.run([merged, mean_reward, max_reward, learning_rate],
                                    feed_dict={mean_reward_placeholder: avg,
                                               max_reward_placeholder: best,
                                               learning_rate_placeholder: m.trainer.learning_rate})
        m.trainer.summary_writer.add_summary(summary, (episode*5000 + step))
    return rewards.copy()


sess = tf.Session()
init = tf.global_variables_initializer()

with tf.name_scope('Summaries'):
    mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='mean_reward')
    mean_reward = tf.summary.scalar('mean_reward', mean_reward_placeholder)

    max_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='max_reward')
    max_reward = tf.summary.scalar('max_reward', max_reward_placeholder)

    learning_rate_placeholder = tf.placeholder(tf.float32, shape=None, name='learning_rate')
    learning_rate = tf.summary.scalar('learning_rate', learning_rate_placeholder)

    # epsilon_placeholder = tf.placeholder(tf.float32, shape=None, name='epsilon')
    # epsilon = tf.summary.scalar('epsilon', epsilon_placeholder)
    merged = tf.summary.merge_all()
