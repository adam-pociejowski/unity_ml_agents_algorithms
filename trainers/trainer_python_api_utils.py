import numpy as np
from operator import add
import tensorflow as tf

episode_length = 0


def update_rewards(rewards, trainers, new_env_info, actions, env_info):
    for t in trainers:
        brain_rewards = new_env_info[t.brain_name].rewards
        brain_new_observations = new_env_info[t.brain_name].vector_observations
        brain_observations = env_info[t.brain_name].vector_observations
        brain_actions = actions[t.brain_name]
        t.post_step_actions(brain_observations, brain_actions, brain_rewards, brain_new_observations)
        rewards[t.brain_name] = list(map(add, rewards[t.brain_name], brain_rewards))
    return rewards


def get_actions(trainers, env_info):
    actions = {}
    for t in trainers:
        observations = env_info[t.brain_name].vector_observations
        action = t.get_actions(observations)
        actions[t.brain_name] = action
    return actions


def init_rewards_dict(trainers):
    rewards = {}
    for t in trainers:
        rewards[t.brain_name] = [0.0]*t.agents_num
    return rewards


def run_episode(trainers, env, episode, train_mode, episode_max_length=5000, log_interval=1000, verbose=True):
    env_info = env.reset(train_mode=train_mode)
    rewards = init_rewards_dict(trainers)
    rewards_log_copy = rewards.copy()
    for step in range(episode_max_length):
        actions = get_actions(trainers, env_info)
        new_env_info = env.step(actions)
        rewards = update_rewards(rewards, trainers, new_env_info, actions, env_info)
        if step % log_interval == 0 and (step > 0 or episode == 0):
            rewards_log_copy = log_stats(trainers, rewards_log_copy, rewards, episode, step, episode_max_length, verbose)
        env_info = new_env_info

    post_episode_actions(trainers, rewards, episode, verbose)
    del rewards
    del rewards_log_copy


def run_episode_from_restored_model(trainers, env, episode, train_mode, episode_max_length=5000, log_interval=1000, verbose=True, init_step=0):
    env_info = env.reset(train_mode=train_mode)
    rewards = init_rewards_dict(trainers)
    rewards_log_copy = rewards.copy()
    for step in range(init_step + 1, episode_max_length):
        actions = get_actions(trainers, env_info)
        new_env_info = env.step(actions)
        rewards = update_rewards(rewards, trainers, new_env_info, actions, env_info)
        if step % log_interval == 0 and (step > 0 or episode == 0):
            rewards_log_copy = log_stats(trainers, rewards_log_copy, rewards, episode, step, episode_max_length, verbose)
        env_info = new_env_info

    post_episode_actions(trainers, rewards, episode, verbose)
    del rewards
    del rewards_log_copy


def post_episode_actions(trainers, rewards, episode, verbose):
    if verbose:
        print("\n################  EPISODE END: {} ################".format(episode + 1))

    for t in trainers:
        model_rewards = rewards[t.brain_name]
        t.post_episode_actions(model_rewards, episode)
        best = np.amax(model_rewards)
        avg = np.average(model_rewards)
        if verbose:
            print("[BRAIN]:       {:8}, avg: {:8.4f}, max: {:8.4f}".format(t.brain_name, avg, best))

    if verbose:
        print("\n")


def log_stats(trainers, rewards_log_copy, rewards, episode, step, episode_max_length, verbose):
    if verbose:
        print("##########  EPISODE: {},  STEP: {}  ############".format(episode + 1, step))

    for t in trainers:
        rewards_change = [a_i - b_i for a_i, b_i in zip(rewards[t.brain_name], rewards_log_copy[t.brain_name])]
        best = np.amax(rewards_change)
        avg = np.average(rewards_change)
        if verbose:
            print("[BRAIN]:       {:8}, avg: {:8.4f}, best: {:8.4f}".format(t.brain_name, avg, best))

        save_summary(t, avg, best, episode*(episode_max_length - 1) + step)
    return rewards.copy()


def save_summary(trainer, avg_reward, best_reward, step):
    summary, _, _, _, _, _ = trainer.sess.run([merged, mean_reward, max_reward, learning_rate, random_action_chance, loss],
                                              feed_dict={mean_reward_placeholder: avg_reward,
                                                         max_reward_placeholder: best_reward,
                                                         loss_placeholder: trainer.current_loss,
                                                         learning_rate_placeholder: trainer.learning_rate,
                                                         random_action_chance_placeholder: trainer.epsilon})
    trainer.summary_writer.add_summary(summary, step)


with tf.name_scope('Summaries'):
    mean_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='mean_reward')
    mean_reward = tf.summary.scalar('mean_reward', mean_reward_placeholder)

    max_reward_placeholder = tf.placeholder(tf.float32, shape=None, name='max_reward')
    max_reward = tf.summary.scalar('max_reward', max_reward_placeholder)

    learning_rate_placeholder = tf.placeholder(tf.float32, shape=None, name='learning_rate')
    learning_rate = tf.summary.scalar('learning_rate', learning_rate_placeholder)

    random_action_chance_placeholder = tf.placeholder(tf.float32, shape=None, name='random_action_chance')
    random_action_chance = tf.summary.scalar('random_action_chance', random_action_chance_placeholder)

    loss_placeholder = tf.placeholder(tf.float32, shape=None, name='loss')
    loss = tf.summary.scalar('loss', loss_placeholder)
    merged = tf.summary.merge_all()
