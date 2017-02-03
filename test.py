import keras, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import SReLU
#from keras.regularizers import l2, activity_l2
#from keras.optimizers import SGD
import mod_mem_net as mod
import pickle
import cPickle

class SSNE_param:
    def __init__(self, is_memoried):
        self.num_input = 1
        self.num_hnodes = 10
        self.num_output = 1
        if is_memoried: self.type_id = 'memoried'
        else: self.type_id = 'normal'

        self.elite_fraction = 0.1
        self.crossover_prob = 0.2
        self.mutation_prob = 0.9
        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            #Normalize network flexibility by changing hidden nodes
            naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
            mem_weights = 3 * (
                 self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                 self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
            normalization_factor = int(mem_weights/naive_total_num_weights)

            #Set parameters for comparable flexibility with memoried net
            self.num_hnodes *= normalization_factor + 1
            self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.depth = 3
            self.interleaving_lower_bound = 10
            self.interleaving_upper_bound = 20
            self.is_memoried = 1

            #DEAP/SSNE stuff
            self.use_ssne = 1
            self.use_deap = 0
            if self.use_deap or self.use_ssne:
                self.ssne_param = SSNE_param( self.is_memoried)
            self.total_gens = 10000

parameters = Parameters() #Create the Parameters class

def unpickle(filename = 'def.pickle'):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

if __name__ == "__main__":

    agent = mod.SSNE(parameters, parameters.ssne_param)
    agent.fitness_evals[9] = 5
    pickle_object(agent, 'Agent')

    test_agent = unpickle('Agent')

    print 'k'




















