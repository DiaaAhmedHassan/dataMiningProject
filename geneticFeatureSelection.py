import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class geneticFeatureSelection():

    def __init__(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            model,

        ) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.4)
        self.x = x
        self.y = y
        self.model = model
    
    def generate_population(self, n_children, n_features):
        # print('generating population')
        population = []
        for i in range(n_children):
            # give all the features true
            chromosome = np.ones(n_features)

            # get the first 30% of the features and give them false
            chromosome[: int(0.3*n_features)] = False

            # shuffle this false and true randomly
            np.random.shuffle(chromosome)

            # append the chromosome to our population
            population.append(chromosome)
        return population
    
    def fitness_scores(self, population):
        scores = []

        # print('calculating the fitness \n')
        for chromosome in population:
            self.model.fit(self.x_train.iloc[:, chromosome], self.y_train)
            prediction = self.model.predict(self.x_test.iloc[:, chromosome])
            scores.append(accuracy_score(self.y_test, prediction))
            # print('iteration finished')
        
        scores, population = np.array(scores), np.array(population)
        inds = np.argsort(scores)
        # print(list(population[inds, :][::-1]))
        return list(scores[inds][::-1]), list(population[inds, :][::-1])

    def selection(self, population_after_fitting, n_parents):
        population_next_generation = []
        for i in range(n_parents):
            population_next_generation.append(population_after_fitting[i])
        return population_next_generation

    def crossover(self, population_after_selection):
        population_next_generation = population_after_selection

        for i in range(0, len(population_after_selection), 2):
            child1, child2 = population_after_selection[i], population_after_selection[i+1]
            new_parents = np.concatenate((child1[:len(child1)//2], child2[len(child2)//2:]), axis=None)
            population_next_generation.append(new_parents)
        return population_next_generation

    def mutate(self, population_after_crossover, mutation_rate, n_features):
        mutation_range = int(mutation_rate * n_features)
        population_next_generation = []

        for n in range(0, len(population_after_crossover)):
            chromosome = population_after_crossover[n]

            locus = []
            for i in range(mutation_range):
                pos = np.random.randint(0, n_features-1)
                locus.append(pos)
            for j in  locus:
                chromosome[j] = not chromosome[j]
            population_next_generation.append(chromosome)
        return population_next_generation

    def create_generations(self, df, label, size, n_features, n_parents, mutation_rate, n_generations):

        best_chromosome = []
        best_score = []
        next_generation = self.generate_population(size, n_features)
        for i in range(n_generations):
            # print(f"length before fitting: {len(next_generation)}")
            
            scores, population_after_fitting = self.fitness_scores(next_generation)
            # print(f'best score in generation {i+1} : {scores[:1]}')
            # print(f"length after fitting: {len(population_after_fitting)}")
            
            population_after_selection = self.selection(population_after_fitting, n_parents)
    #         print(f"length after selection: {len(population_after_selection)}")

            population_after_crossover = self.crossover(population_after_selection)
    #         print(f"length after crossover: {len(population_after_crossover)}")
            
            next_generation = self.mutate(population_after_crossover, mutation_rate, n_features)
            best_chromosome.append(population_after_fitting[0])
            best_score.append(scores[0])
            # np.set_printoptions(threshold=np.inf)
        return best_chromosome, best_score
