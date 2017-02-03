import numpy as np, os
#import MultiNEAT as NEAT
import mod_mem_net as mod, sys
from random import randint
import random



save_foldername = 'RSeq_Recall'
class tracker(): #Tracker
    def __init__(self, parameters, foldername = save_foldername):
        self.foldername = foldername
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        if parameters.is_memoried:
            self.file_save = 'mem_seq_recall.csv'
        else:
            self.file_save = 'norm_seq_recall.csv'


    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/' + 'rough_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/' + 'hof_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class SSNE_param:
    def __init__(self, is_memoried):
        self.num_input = 2
        self.num_hnodes = 20
        self.num_output = 1
        if is_memoried: self.type_id = 'memoried'
        else: self.type_id = 'normal'

        self.elite_fraction = 0.1
        self.crossover_prob = 0
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000000
        self.mut_distribution = 0  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s
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
            self.depth = 5
            self.interleaving_lower_bound = 10
            self.interleaving_upper_bound = 20
            self.is_memoried = 1
            self.repeat_trials = 10
            self.test_trials = 50

            #DEAP/SSNE stuff
            self.use_ssne = 1
            if self.use_ssne:
                self.ssne_param = SSNE_param( self.is_memoried)
            self.total_gens = 10000
            self.arch_type = 1  # 0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture

            #Reward scheme
            #1 Coarse and Order matters
            #2 Coarse and Order doesn't matter - evaluate remember performance
            #3 Binary and Order matters
            #4 Binary and Order doesn't matter - evaluate remember performance
            #5 Test - final performance only matters
            #6 Combine #2 and #3
            self.reward_scheme = 6

            self.tolerance = 1
            self.test_tolerance = 1
            if self.arch_type == 0: self.arch_type = 'quasi_gru'
            elif self.arch_type ==1: self.arch_type = 'quasi_ntm'
            else: sys.exit('Invalid choice of neural architecture')
parameters = Parameters() #Create the Parameters class
tracker = tracker(parameters) #Initiate tracker

class Seq_Recall:
    def __init__(self, parameters):
        self.parameters = parameters; self.ssne_param = self.parameters.ssne_param
        self.depth = self.parameters.depth
        self.interleaving_upper_bound = self.parameters.interleaving_upper_bound; self.interleaving_lower_bound = self.parameters.interleaving_lower_bound

        self.agent = mod.SSNE(self.parameters, self.ssne_param, parameters.arch_type)

    def generate_input(self):
        #Distraction/Hallway parts
        distractions_macro = []
        for i in range(self.depth):
            num_distractions = randint(self.interleaving_lower_bound, self.interleaving_upper_bound)
            distractions_macro.append(num_distractions)
            if i == 0: distractions_inp = np.zeros(self.depth) + num_distractions #ALongside the signals
            distractions_inp = np.concatenate((distractions_inp, np.arange(num_distractions)[::-1])) #From then on
        distractions_inp = np.reshape(distractions_inp, (len(distractions_inp), 1))

        #Real signal
        target = []
        signal_inp = np.zeros(distractions_inp.shape[0])
        for i in range(self.depth):
            if random.random() < 0.5: direction = -1
            else: direction = 1
            signal_inp[i] = direction
            target.append(direction)
        signal_inp = np.reshape(signal_inp, (len(signal_inp), 1))

        #Final Preparation: Join the two halves to construct the final input sequence
        final_input = np.concatenate((signal_inp, distractions_inp), axis=1)

        return final_input, target

    def get_reward(self, target, output):

        reward = 0.0
        if self.parameters.reward_scheme == 1: #Coarse and Order matters
            for i, j in zip(target, output):
                if i * j < 0: break
                reward += i * j

        elif self.parameters.reward_scheme == 2: #Coarse and Order doesn't matter - evaluate remember performance
            for i, j in zip(target, output):
                reward += i * j

        elif self.parameters.reward_scheme == 3: #Binary and Order matters
            for i, j in zip(target, output):
                if i * j < 0: break
                reward += 1.0

        elif self.parameters.reward_scheme == 4: # Binary and Order doesn't matter - evaluate remember performance
            for i, j in zip(target, output):
                if i * j > 0: reward += 1.0

        elif self.parameters.reward_scheme == 5: #Test - final performance only matters
            for i, j in zip(target, output):
                if i * j > 0: reward += 1.0
                else:
                    reward = 0
                    break

        elif self.parameters.reward_scheme == 6: #Combine #2 and #5
            for i, j in zip(target, output):
                reward += (i * j)/2.0

            for i, j in zip(target, output):
                if i * j < 0: break
                reward += 0.5

        return reward/self.depth

    def run_simulation(self, index, epoch_inputs, epoch_targets):
        reward = 0.0
        for input, target in zip(epoch_inputs, epoch_targets):
            self.agent.pop[index].reset_bank()
            net_output = []
            for inp in input: #Run network to get output
                net_out = (self.agent.pop[index].feedforward(inp)[0][0] - 0.5) * 2
                if inp[1] == 0: net_output.append(net_out)
            reward += self.get_reward(target, net_output) #get reward or fitness of the individual

        #print net_output
        reward /= self.parameters.repeat_trials #Normalize
        self.agent.fitness_evals[index] = reward #Encode reward as fitness for individual
        return reward

    def evolve(self, gen):
        best_epoch_reward = -1000000

        #Generate epoch input
        epoch_inputs = []; epoch_targets = []
        for i in range(parameters.repeat_trials):
            input, target = self.generate_input()
            epoch_inputs.append(input)
            epoch_targets.append(target)

        for i in range(self.parameters.population_size): #Test all genomes/individuals
            reward = self.run_simulation(i, epoch_inputs, epoch_targets)
            if reward > best_epoch_reward: best_epoch_reward = reward

        #HOF test net
        hof_index = self.agent.fitness_evals.index(max(self.agent.fitness_evals))
        hof_score = self.test_net(hof_index)

        #Save population and HOF
        if (gen + 1) % 1 == 0:
            mod.pickle_object(self.agent.pop, save_foldername + '/seq_recall_pop')
            mod.pickle_object(self.agent.pop[hof_index], save_foldername + '/seq_recall_hof')
            np.savetxt(save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        self.agent.epoch()
        return best_epoch_reward, hof_score

    def test_net(self, index): #Test is binary
        reward = 0.0
        test_boost = 5
        for trial in range(self.parameters.test_trials):
            self.agent.pop[index].reset_bank()
            trial_reward = 0.0
            input, target = self.generate_input()  # get input
            net_output = []
            for inp in input:  # Run network to get output
                net_out = (self.agent.pop[index].feedforward(inp)[0][0] - 0.5) *2
                if inp[1] == 0: net_output.append(net_out)

            #Reward

            for i, j in zip(target, net_output):
                #print i,j
                if i * j > 0: trial_reward = 1.0
                else:
                    trial_reward = 0.0
                    break
            reward += trial_reward

        return reward/(self.parameters.test_trials)

if __name__ == "__main__":
    print 'Running SEQUENCE RECALL with ', parameters.arch_type
    task = Seq_Recall(parameters)
    for gen in range(parameters.total_gens):
        epoch_reward, hof_score = task.evolve(gen)
        print 'Gen:', gen, ' Ep_rew:', "%0.2f" % epoch_reward, ' HOF_rew:', "%0.2f" % hof_score, ' Cml_Hof_rew:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(epoch_reward, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(hof_score, gen)  # Add average global performance to tracker














