from trainers.agent_trainer import AgentTrainer
from trainers.algorithms.model.chromosome import Chromosome
import tensorflow as tf
import pandas as pd
import numpy as np
import random


class GeneticAlgorithmTrainer(AgentTrainer):

    def __init__(self, brain, brain_name, input_num, output_num, layer_1_nodes, layer_2_nodes,  mutation_rate=0.001, max_mutation_value_change=0.2,
                 agents_num=40, elite_chromosomes=8, model_name='genetic_algorithm_new', restore_model=False):
        self.layer_1_nodes = layer_1_nodes
        self.layer_2_nodes = layer_2_nodes
        self.population = []

        self.mutation_rate = mutation_rate
        self.max_mutation_value_change = max_mutation_value_change
        self.elite_chromosomes = elite_chromosomes
        super().__init__(brain, brain_name, input_num, output_num, agents_num, model_name=model_name, epsilon_max=0.0, epsilon_min=0.0, use_tf=False,
                         restore_model=restore_model)

    def test(self):
        print(f'population: {len(self.population)})')

        for i, c in enumerate(self.population):
            print(f'W1: {c.W1}')
            print(f'b1: {c.b1}')
            print(f'W2: {c.W2}')
            print(f'b2: {c.b2}')
            print(f'W3: {c.W3}')
            print(f'b3: {c.b3}')
            flatten_list = self._change_to_flatten_list(c)
            print(f'flatten_list: {flatten_list}')

            result = self._predict([1.4, 0.453, 0.3], c)
            print(f'predict: {result}')
            self.post_episode_actions([1], 1)

    def _init_model(self):
        for i in range(self.agents_num):
            W1 = np.random.rand(self.input_num, self.layer_1_nodes) * 2 - 1
            b1 = np.random.rand(1, self.layer_1_nodes)*2 - 1

            W2 = np.random.rand(self.layer_1_nodes, self.layer_2_nodes) * 2 - 1
            b2 = np.random.rand(1, self.layer_2_nodes) * 2 - 1

            W3 = np.random.rand(self.layer_2_nodes, self.output_num) * 2 - 1
            b3 = np.random.rand(1, self.output_num)*2 - 1
            self.population.append(Chromosome(W1, b1, W2, b2, W3, b3))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _change_to_flatten_list(chromosome):
        flatten = chromosome.W1.flatten()
        flatten = np.append(flatten, chromosome.b1)
        flatten = np.append(flatten, chromosome.W2.flatten())
        flatten = np.append(flatten, chromosome.b2)
        flatten = np.append(flatten, chromosome.W3.flatten())
        flatten = np.append(flatten, chromosome.b3)
        return flatten

    def _predict(self, observation, chromosome):
        A1 = self._relu(np.add(np.dot(observation, chromosome.W1), chromosome.b1))
        A2 = self._relu(np.add(np.dot(A1, chromosome.W2), chromosome.b2))
        Z3 = np.add(np.dot(A2, chromosome.W3), chromosome.b3)
        return np.argmax(Z3)

    def _mutation(self, flatten_chr):
        for i in range(len(flatten_chr)):
            if np.random.random() < self.mutation_rate:
                flatten_chr[i] += (np.random.rand() * self.max_mutation_value_change) - self.max_mutation_value_change / 2
        return flatten_chr

    def _crossover(self, best_flatten_chromosomes):
        new_population = best_flatten_chromosomes
        for index in range(self.agents_num - self.elite_chromosomes):
            parents = random.sample(range(self.elite_chromosomes), 2)
            cut_point = random.randint(0, len(best_flatten_chromosomes[0]))
            new_flatten_chromosome = np.append(best_flatten_chromosomes[parents[0]][:cut_point],
                                               best_flatten_chromosomes[parents[1]][cut_point:])
            new_flatten_chromosome = self._mutation(new_flatten_chromosome)
            new_population.append(new_flatten_chromosome)
        return new_population

    def post_episode_actions(self, rewards, episode):
        best_chromosomes_indexes = np.asarray(rewards).argsort()[-self.elite_chromosomes:][::-1]
        best_chromosomes_list = []
        for i in best_chromosomes_indexes:
            chromosome_flatten = self._change_to_flatten_list(self.population[i])
            best_chromosomes_list.append(chromosome_flatten)

        new_population_flatten = self._crossover(best_chromosomes_list)
        new_population = []
        for c in new_population_flatten:
            W1, b1 = self._extract_layer(c, 0, self.input_num * self.layer_1_nodes, self.layer_1_nodes)
            actual_index = self.input_num*self.layer_1_nodes + self.layer_1_nodes
            W2, b2 = self._extract_layer(c, actual_index, self.layer_1_nodes*self.layer_2_nodes, self.layer_2_nodes)
            actual_index += self.layer_1_nodes*self.layer_2_nodes + self.layer_2_nodes
            W3, b3 = self._extract_layer(c, actual_index, self.layer_2_nodes*self.output_num, self.output_num)
            new_population.append(Chromosome(W1, b1, W2, b2, W3, b3))

        del self.population
        del new_population_flatten
        del best_chromosomes_list
        self.population = new_population

    @staticmethod
    def _extract_layer(chromosome_flatten, start_index, layer_nodes, next_layer_nodes):
        W_flatten = np.array(chromosome_flatten[start_index: start_index + layer_nodes])
        W = np.reshape(W_flatten, (-1, next_layer_nodes))
        start_index += layer_nodes
        b = np.array(chromosome_flatten[start_index: start_index + next_layer_nodes])
        return W, b

    def get_actions(self, observation):
        actions = []
        for i, c in enumerate(self.population):
            actions.append(self._predict(observation[i], c))
        return actions

    def _save_model(self, chromosome):
        model_to_save = np.asarray(chromosome[0])
        model_to_save = np.append(model_to_save, np.asarray(chromosome[1]).reshape(self.output_num, self.layer_2_nodes), axis=0)
        df = pd.DataFrame(model_to_save)
        df.to_csv('models/' + self.brain_name + '.csv', index=False)
