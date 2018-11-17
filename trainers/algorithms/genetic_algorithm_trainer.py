import random
import numpy as np
import pandas as pd
import tensorflow as tf


class GeneticAlgorithmTrainer:
    def __init__(self, brain_name, number_of_observations, number_of_actions,  mutation_rate=0.05, max_mutation_value_change=0.2,
                 number_of_chromosomes=40, number_of_elite_chromosomes=8, hidden_layer_nodes=87):
        self.brain_name = brain_name
        self.tag = '[GeneticAlgorithmTrainer - '+brain_name+']: '
        print(self.tag+' started')
        self.number_of_observations = number_of_observations
        self.number_of_actions = number_of_actions

        # Genetic algorithm parameters
        self.mutation_rate = mutation_rate
        self.max_mutation_value_change = max_mutation_value_change
        self.number_of_chromosomes = number_of_chromosomes
        self.number_of_elite_chromosomes = number_of_elite_chromosomes

        # Neural network parameters
        self.input_layer_nodes = number_of_observations + 1
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = number_of_actions

        self.population = []
        self.prepare_init_model()
        self.summary_writer = tf.summary.FileWriter("summary/genetic")
        # self.load_model()

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def prepare_init_model(self):
        for i in range(self.number_of_chromosomes):
            hidden_layer_weights = np.random.rand(self.input_layer_nodes, self.hidden_layer_nodes) * 2 - 1
            output_layer_weights = np.random.rand(self.hidden_layer_nodes, self.output_layer_nodes) * 2 - 1
            self.population.append([hidden_layer_weights, output_layer_weights])

    def _change_to_flatten_list(self, chromosome):
        input_layer = chromosome[0]
        input_layer = input_layer.reshape(input_layer.shape[1], -1)
        hidden_layer = chromosome[1]
        return np.append(input_layer, hidden_layer.reshape(hidden_layer.shape[1], -1))

    def _mutation(self, chromosome):
        random_value = np.random.randint(0, len(chromosome))
        if random_value < self.mutation_rate:
            n = np.random.randint(0, len(chromosome))
            chromosome[n] += (np.random.rand() * self.max_mutation_value_change) - self.max_mutation_value_change / 2
        return chromosome

    def _crossover(self, best_chromosomes):
        new_population = best_chromosomes
        for index in range(self.number_of_chromosomes - self.number_of_elite_chromosomes):
            parents = random.sample(range(self.number_of_elite_chromosomes), 2)
            cut_point = random.randint(0, len(best_chromosomes[0]))
            new_chromosome = np.append(best_chromosomes[parents[0]][:cut_point],
                                       best_chromosomes[parents[1]][cut_point:])
            new_chromosome = self._mutation(new_chromosome)
            new_population.append(new_chromosome)
        return new_population

    def predict_using_neural_network(self, observation, chromosome):
        input_values = observation / max(np.max(np.linalg.norm(observation)), 1)
        input_values = np.insert(1.0, 1, input_values)
        hidden_layer_values = self._relu(np.dot(input_values, chromosome[0]))
        output_layer_values = self._relu(np.dot(hidden_layer_values, chromosome[1]))
        return np.argmax(output_layer_values)

    def post_episode_actions(self, rewards, episode):
        best_chromosomes_indexes = np.asarray(rewards).argsort()[-self.number_of_elite_chromosomes:][::-1]
        best_chromosome = self.population[best_chromosomes_indexes[0]]
        best_chromosomes_list = []
        for index in best_chromosomes_indexes:
            chromosome_flatten = self._change_to_flatten_list(self.population[index])
            best_chromosomes_list.append(chromosome_flatten)

        new_population_flatten = self._crossover(best_chromosomes_list)
        new_population = []
        for chromosome_flatten in new_population_flatten:
            input_layer_flatten = np.array(chromosome_flatten[:self.hidden_layer_nodes * self.input_layer_nodes])
            input_layer_reshaped = np.reshape(input_layer_flatten, (-1, self.population[0][0].shape[1]))
            hidden_layer_flatten = np.array(chromosome_flatten[self.hidden_layer_nodes * self.input_layer_nodes:])
            hidden_layer_reshaped = np.reshape(hidden_layer_flatten, (-1, self.population[0][1].shape[1]))
            new_population.append([input_layer_reshaped, hidden_layer_reshaped])

        self.population = new_population
        self.save_model(best_chromosome)

    def get_actions(self, observation):
        actions = []
        for index in range(len(self.population)):
            actions.append(self.predict_using_neural_network(observation[index], self.population[index]))
        return actions

    def append_summary(self, mean_reward, max_reward, step):
        pass

    def post_step_actions(self, observations, actions, rewards, new_observations):
        pass

    def set_session(self, sess):
        pass

    def save_model(self, chromosome):
        model_to_save = np.asarray(chromosome[0])
        model_to_save = np.append(model_to_save, np.asarray(chromosome[1]).reshape(self.number_of_actions, self.hidden_layer_nodes), axis=0)
        df = pd.DataFrame(model_to_save)
        df.to_csv('models/' + self.brain_name + '.csv', index=False)

    def load_model(self):
        df = pd.read_csv('models/' + self.brain_name + '.csv')
        data = df.as_matrix()
        input_layer = data[:self.input_layer_nodes]
        output_layer = np.asarray(data[self.input_layer_nodes:]).reshape(self.hidden_layer_nodes, 6)
        chromosome = [input_layer, output_layer]
        self.population[0] = chromosome
        self.population[1] = chromosome
