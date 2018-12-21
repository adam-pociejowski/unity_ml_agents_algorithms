import sys
import logging
import time
sys.path.append("C:\\\\Users\\Valverde\\Dropbox\\projects\\python\\unity_models")
from tqdm import tqdm
from docopt import docopt
from mlagents.envs import UnityEnvironment
from trainers.algorithms.genetic.genetic_algorithm_trainer_old import *
from trainers.algorithms.genetic.genetic_algorithm_trainer import *
from trainers.algorithms.deep_q_learning.deep_q_learning_improved_trainer import *
from trainers.algorithms.deep_q_learning.deep_q_learning_trainer import *
from trainers.algorithms.actor_critic.actor_critic_trainer import *
from trainers.keras.deep_q_learning_improved_trainer import *
from trainers.keras.deep_q_learning_trainer import *
from trainers.keras.actor_critic_trainer import *
from trainers.trainer_python_api_utils import *

logger = logging.getLogger("mlagents.envs")
logger.setLevel(logging.ERROR)

model_dict = {'ga_old': 'genetic_algorithm_old',
              'ga': 'genetic_algorithm',
              'dql': 'deep_q_learning',
              'dqlk': 'deep_q_learning_keras',
              'dqli': 'deep_q_learning_improved',
              'dqlik': 'deep_q_learning_improved_keras',
              'a2c_old': 'actor_critic_old',
              'a2c': 'actor_critic'}


def start_training():
    model_id = init_model_id
    env_name = "../../env/single-64-training/Tanks"
    while True:
        print('Model name: {}'.format(model_dict[model] + '_' + str(model_id)))
        env = UnityEnvironment(worker_id=int(worker_id), file_name=env_name, no_graphics=no_graphics)
        trainer = _choose_trainer(env, model_id)
        trainer.init()
        start_episode = 0
     
        for episode in tqdm(range(start_episode, episodes_amount)):
            run_episode([trainer], env, episode, train_mode, episode_max_length=episode_length, log_interval=step_interval, verbose=False)

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
    brain = env.brains['PPOBrain']
    brain_name = 'PPOBrain'
    
    if model == 'ga_old':
        return GeneticAlgorithmOldTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, elite_chromosomes=6, hidden_layer_nodes=128,
                                          model_name=model_name_with_id)
    elif model == 'ga':
        return GeneticAlgorithmTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, layer_1_nodes=64, layer_2_nodes=64,
                                       elite_chromosomes=8, model_name=model_name_with_id)
    elif model == 'dql':
        return DeepQLearningTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32, layer_1_nodes=128,
                                    layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'dqlk':
        return KerasDeepQLearningTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                                         layer_1_nodes=128, layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'dqli':
        return DeepQLearningImprovedTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                                            layer_1_nodes=128, layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'dqlik':
        return KerasDeepQLearningImprovedTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, memory_size=5000, batch_size=32,
                                                 layer_1_nodes=128, layer_2_nodes=128, model_name=model_name_with_id)
    elif model == 'a2c':
        return ActorCriticTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                                  model_name=model_name_with_id)
    elif model == 'a2ck':
        return KerasActorCriticTrainer(brain, brain_name, input_num=87, output_num=6, agents_num=64, layer_1_nodes=128, layer_2_nodes=128,
                                       model_name=model_name_with_id)


if __name__ == '__main__':
    _USAGE = '''
    Usage:
      learn [options]

    Options:
      --worker=<n>            Number to add to communication port (5005). Used for multi-environment [default: 0]
      --model=<dt>            Model name
      --model_id=<n>          ID of model
      --episodes=<n>          Number of episodes [default: 40]
      --episode-length=<n>    Number of steps in episode [default: 5001]
      --step-interval=<n>     Number of steps in single interval [default: 1000]
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
    step_interval = int(options['--step-interval'])
    no_graphics = options['--no-graphics']
    loop_forever = options['--loop-forever']
    train_mode = True
    start_training()
