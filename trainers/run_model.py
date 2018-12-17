import sys
import logging
import time
sys.path.append("C:\\\\Users\\Valverde\\Dropbox\\projects\\python\\unity_models")
from tqdm import tqdm
from docopt import docopt
from mlagents.envs import UnityEnvironment
from trainers.algorithms.deep_q_learning.deep_q_learning_improved_trainer import *
from trainers.algorithms.genetic.genetic_algorithm_trainer_old import *
from trainers.algorithms.genetic.genetic_algorithm_trainer import *
from trainers.algorithms.policy_gradients.policy_gradients_agent_batch_trainer import *
from trainers.algorithms.policy_gradients.policy_gradients_trainer import *
from trainers.algorithms.deep_q_learning.deep_q_learning_trainer import *
from trainers.algorithms.actor_critic.actor_critic_trainer import *
from trainers.algorithms.actor_critic.actor_critic_keras_trainer import *
from trainers.trainer_python_api_utils import *

logger = logging.getLogger("mlagents.envs")
logger.setLevel(logging.ERROR)

model_dict = {'ga_old': 'genetic_algorithm_old',
              'ga': 'genetic_algorithm',
              'dql': 'deep_q_learning',
              'dqli': 'deep_q_learning_improved',
              'pol': 'policy_gradients',
              'poli': 'policy_gradients_improved',
              'a2c_old': 'actor_critic_old',
              'a2c': 'actor_critic'}


def run():
    model_id = init_model_id
    env_name = "../../env/single-64/Tanks"
    while True:
        print('Model name: {}'.format(model_dict[model] + '_' + str(model_id)))
        env = UnityEnvironment(worker_id=int(worker_id), file_name=env_name, no_graphics=no_graphics)
        trainer = _choose_trainer(env, model_id)
        trainer.init()

        for episode in tqdm(range(episodes_amount)):
            if episode > 0 and episode % 10 == 0:
                env.close()
                time.sleep(5)
                env = UnityEnvironment(worker_id=int(worker_id), file_name=env_name, no_graphics=no_graphics)
                time.sleep(5)
                trainer.brain = env.brains['PPOBrain']

            run_episode([trainer], env, episode, train_mode, episode_max_length=episode_length, log_interval=log_interval, verbose=False)

        env.close()
        if not loop_forever:
            break
        else:
            del env
            del trainer
            model_id = model_id + 1
            time.sleep(5)


def _choose_trainer(env, model_id):
    model_name_with_id = model_dict[model] + '_' + str(model_id)
    if model == 'ga_old':
        return GeneticAlgorithmOldTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, elite_chromosomes=6,
                                          hidden_layer_nodes=128, model_name=model_name_with_id)
    elif model == 'ga':
        return GeneticAlgorithmTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=64,
                                       layer_2_nodes=64, elite_chromosomes=8, model_name=model_name_with_id)
    elif model == 'dql':
        return DeepQLearningTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                                    layer_1_nodes=128, layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'dqli':
        return DeepQLearningImprovedTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, memory_size=5000,
                                            batch_size=32, layer_1_nodes=128, layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'pol':
        return PolicyGradientsTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128,
                                      layer_2_nodes=128, discount_rate=0.95, learning_rate=0.001, model_name=model_name_with_id)
    elif model == 'poli':
        return PolicyGradientsAgentBatchTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128,
                                                layer_2_nodes=128, discount_rate=0.99, learning_rate=0.01, model_name=model_name_with_id)
    elif model == 'a2c_old':
        return ActorCriticTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                                  model_name=model_name_with_id)
    elif model == 'a2c':
        return ActorCriticKerasTrainer(env.brains['PPOBrain'], 'PPOBrain', input_num=87, output_num=6, agents_num=64, layer_1_nodes=128,
                                       layer_2_nodes=128, discount_rate=0.99, model_name=model_name_with_id)


if __name__ == '__main__':
    _USAGE = '''
    Usage:
      learn [options]

    Options:
      --worker=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0]
      --model=<dt>            Model name
      --model_id=<n>          ID of model
      --episodes=<n>          Number of episodes [default: 1000]
      --episode-length=<n>    Number of steps in episode [default: 5001]
      --log-interval=<n>      Number of steps in episode [default: 1000]
      --no-graphics           Whether to run the Unity simulator in no-graphics mode [default: False]
      --loop-forever          Run model in loop [default: False]
    '''

    options = docopt(_USAGE)
    print(options)
    worker_id = options['--worker']
    model = options['--model']
    init_model_id = int(options['--model_id'])
    episodes_amount = int(options['--episodes'])
    episode_length = int(options['--episode-length'])
    log_interval = int(options['--log-interval'])
    no_graphics = options['--no-graphics']
    loop_forever = options['--loop-forever']
    train_mode = True
    run()


