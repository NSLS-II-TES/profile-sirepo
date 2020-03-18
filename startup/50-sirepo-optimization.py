import bluesky.plans as bp

import sirepo_bluesky.sirepo_flyer as sf

import numpy as np
import random
import matplotlib.pyplot as plt


# def omea_evaluation(params_to_change, first_scan):
def omea_evaluation(first_scan):
    # get data from databroker
    pop_intensity = []
    pop_positions = []
    t = db[-1].table('sirepo_flyer')
    between_intensities = []
    between_positions = []
    for j in range(len(t)):
        if first_scan:
            if j == 0 or j == (len(t) - 1):
                pop_intensity.append(t['sirepo_flyer_mean'][j + 1])
                # indv = {}
                # for elem, param in params_to_change[0].items():
                #     indv[elem] = {}
                #     for param_name in param.keys():
                #         indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                # pop_positions.append(indv)
                pop_positions.append(t['sirepo_flyer_parameters'][j + 1])
            else:
                between_intensities.append(t['sirepo_flyer_mean'][j + 1])
                # indv = {}
                # for elem, param in params_to_change[0].items():
                #     indv[elem] = {}
                #     for param_name in param.keys():
                #         indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                # between_positions.append(indv)
                between_positions.append(t['sirepo_flyer_parameters'][j + 1])
        else:
            if j == (len(t) - 1):
                pop_intensity.append(t['sirepo_flyer_mean'][j + 1])
                # indv = {}
                # for elem, param in params_to_change[0].items():
                #     indv[elem] = {}
                #     for param_name in param.keys():
                #         indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                # pop_positions.append(indv)
                pop_positions.append(t['sirepo_flyer_parameters'][j + 1])
            else:
                between_intensities.append(t['sirepo_flyer_mean'][j + 1])
                # indv = {}
                # for elem, param in params_to_change[0].items():
                #     indv[elem] = {}
                #     for param_name in param.keys():
                #         indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                # between_positions.append(indv)
                between_positions.append(t['sirepo_flyer_parameters'][j + 1])
    current_max_int = np.max(between_intensities)
    curr_max_indx = between_intensities.index(current_max_int)
    if current_max_int > pop_intensity[-1]:
        pop_intensity[-1] = current_max_int
        pop_positions[-1] = between_positions[curr_max_indx]
    return pop_positions, pop_intensity


param_bounds = {'Aperture': {'horizontalSize': [1, 10],
                             'verticalSize': [.1, 1]},
                'Lens': {'horizontalFocalLength': [10, 30]},
                'Obstacle': {'horizontalSize': [1, 10]}}

best_fitness = [0]


def run_fly_sim(flyers):
    # run serial
    RE(bp.fly([flyer for flyer in flyers]))


def generate_flyers(population, num_between_vals,
                    sim_id, server_name, root_dir, watch_name, run_parallel):
    flyers = []
    params_to_change = []
    for i in range(len(population) - 1):
        curr_params_to_change = []
        between_param_linspaces = []
        if i == 0:
            curr_params_to_change.append(population[i])
        for elem, param in population[i].items():
            for param_name, pos in param.items():
                between_param_linspaces.append(np.linspace(pos, population[i + 1][elem][param_name],
                                                           num_between_vals)[1:-1])

        for j in range(len(between_param_linspaces[0])):
            ctr = 0
            indv = {}
            for elem, param in population[0].items():
                indv[elem] = {}
                for param_name in param.keys():
                    indv[elem][param_name] = between_param_linspaces[ctr][j]
                    ctr += 1
            curr_params_to_change.append(indv)
        curr_params_to_change.append(population[i + 1])
        params_to_change.append(curr_params_to_change)
        # "fly" scan
        sim_flyer = sf.SirepoFlyer(sim_id=sim_id, server_name=server_name,
                                   root_dir=root_dir, params_to_change=curr_params_to_change,
                                   watch_name=watch_name, run_parallel=run_parallel)
        flyers.append(sim_flyer)
    return flyers, params_to_change


def ensure_bounds(vec, bounds):
    # Makes sure each individual stays within bounds and adjusts them if they aren't
    vec_new = {}
    # cycle through each variable in vector
    for elem, param in vec.items():
        vec_new[elem] = {}
        for param_name, pos in param.items():
            # variable exceeds the minimum boundary
            if pos < bounds[elem][param_name][0]:
                vec_new[elem][param_name] = bounds[elem][param_name][0]
            # variable exceeds the maximum boundary
            if pos > bounds[elem][param_name][1]:
                vec_new[elem][param_name] = bounds[elem][param_name][1]
            # the variable is fine
            if bounds[elem][param_name][0] <= pos <= bounds[elem][param_name][1]:
                vec_new[elem][param_name] = pos
    return vec_new


def rand_1(pop, popsize, target_indx, mut, bounds):
    # mutation strategy
    # v = x_r1 + F * (x_r2 - x_r3)
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b, c = np.random.choice(idxs, 3, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]
    x_3 = pop[c]

    v_donor = {}
    for elem, param in x_1.items():
        v_donor[elem] = {}
        for param_name in param.keys():
            v_donor[elem][param_name] = x_1[elem][param_name] + mut *\
                                        (x_2[elem][param_name] - x_3[elem][param_name])
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def best_1(pop, popsize, target_indx, mut, bounds, ind_sol):
    # mutation strategy
    # v = x_best + F * (x_r1 - x_r2)
    x_best = pop[ind_sol.index(np.max(ind_sol))]
    idxs = [idx for idx in range(popsize) if idx != target_indx]
    a, b = np.random.choice(idxs, 2, replace=False)
    x_1 = pop[a]
    x_2 = pop[b]

    v_donor = {}
    for elem, param in x_best.items():
        v_donor[elem] = {}
        for param_name in param.items():
            v_donor[elem][param_name] = x_best[elem][param_name] + mut *\
                                        (x_1[elem][param_name] - x_2[elem][param_name])
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def mutate(population, strategy, mut, bounds, ind_sol):
    mutated_indv = []
    for i in range(len(population)):
        if strategy == 'rand/1':
            v_donor = rand_1(population, len(population), i, mut, bounds)
        elif strategy == 'best/1':
            v_donor = best_1(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'current-to-best/1':
        #     v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'best/2':
        #     v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
        # elif strategy == 'rand/2':
        #     v_donor = rand_2(population, len(population), i, mut, bounds)
        mutated_indv.append(v_donor)
    return mutated_indv


def crossover(population, mutated_indv, crosspb):
    crossover_indv = []
    for i in range(len(population)):
        x_t = population[i]
        v_trial = {}
        for elem, param in x_t.items():
            v_trial[elem] = {}
            for param_name, pos in param.items():
                crossover_val = random.random()
                if crossover_val <= crosspb:
                    v_trial[elem][param_name] = mutated_indv[i][elem][param_name]
                else:
                    v_trial[elem][param_name] = x_t[elem][param_name]
        crossover_indv.append(v_trial)
    return crossover_indv


def select(population, crossover_indv, ind_sol, num_between_vals,
           sim_id, server_name, root_dir, watch_name, run_parallel):
    positions = [elm for elm in crossover_indv]
    positions.insert(0, population[0])
    flyers, changed_params = generate_flyers(positions, num_between_vals, sim_id, server_name,
                                             root_dir, watch_name, run_parallel)
    for i in range(len(flyers)):
        run_fly_sim([flyers[i]])
        if i == 0:
            # new_population, new_ind_sol = omea_evaluation(changed_params[i], first_scan=True)
            new_population, new_ind_sol = omea_evaluation(first_scan=True)
        else:
            # partial_pop_pos, partial_pop_int = omea_evaluation(changed_params[i], first_scan=False)
            partial_pop_pos, partial_pop_int = omea_evaluation(first_scan=False)
            new_population.extend(partial_pop_pos)
            new_ind_sol.extend(partial_pop_int)
    new_population = new_population[1:]
    new_ind_sol = new_ind_sol[1:]
    for i in range(len(new_ind_sol)):
        if new_ind_sol[i] > ind_sol[i]:
            population[i] = new_population[i]
            ind_sol[i] = new_ind_sol[i]
    population.reverse()
    ind_sol.reverse()
    return population, ind_sol


def optimize(bounds, num_between_vals, sim_id, server_name, root_dir, watch_name,
             popsize=3, crosspb=.8, mut=.1, mut_type='rand/1', threshold=0,
             max_iter=100, run_parallel=True):
    # Initial population
    initial_population = []
    for i in range(popsize):
        indv = {}
        for elem, param in param_bounds.items():
            indv[elem] = {}
            for param_name, bound in param.items():
                indv[elem][param_name] = random.uniform(bound[0], bound[1])
        initial_population.append(indv)
    first_optic = list(param_bounds.keys())[0]
    first_param_name = list(param_bounds[first_optic].keys())[0]
    initial_population = sorted(initial_population, key=lambda kv: kv[first_optic][first_param_name])

    flyers, changed_params = generate_flyers(initial_population, num_between_vals, sim_id, server_name,
                                             root_dir, watch_name, run_parallel)

    population = []
    intensities = []
    for i in range(len(flyers)):
        run_fly_sim([flyers[i]])
        if i == 0:
            # population, intensities = omea_evaluation(changed_params[i], first_scan=True)
            population, intensities = omea_evaluation(first_scan=True)
        else:
            # partial_pop_pos, partial_pop_int = omea_evaluation(changed_params[i], first_scan=False)
            partial_pop_pos, partial_pop_int = omea_evaluation(first_scan=False)
            population.extend(partial_pop_pos)
            intensities.extend(partial_pop_int)
    population.reverse()
    intensities.reverse()
    v = 0
    consec_best_ctr = 0
    old_best_fit_int = 0
    # termination conditions
    # while not (v > 0):
    while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_int >= threshold)):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(population, mut_type, mut, bounds, ind_sol=intensities)
        # crossover
        cross_trial_pop = crossover(population, mutated_trial_pop, crosspb)
        # select
        population, intensities = select(population, cross_trial_pop, intensities, num_between_vals,
                                         sim_id, server_name, root_dir, watch_name, run_parallel)
        # score keeping
        gen_best = np.max(intensities)
        best_indv = population[intensities.index(gen_best)]
        best_gen_sol.append(best_indv)
        best_fitness.append(gen_best)

        print('      > FITNESS:', gen_best)
        print('         > BEST POSITIONS:', best_indv)

        v += 1
        if np.round(gen_best, 6) == np.round(old_best_fit_int, 6):
            consec_best_ctr += 1
            print(f'Counter: {consec_best_ctr}')
        else:
            consec_best_ctr = 0
        old_best_fit_int = gen_best

        if consec_best_ctr >= 5 and old_best_fit_int >= threshold:
            print('Finished')
            break
        else:
            # randomize individual
            pos_to_check = []
            change_indx = intensities.index(np.min(intensities))
            pos_to_check.append(population[0])
            indv = {}
            for elem, param in param_bounds.items():
                indv[elem] = {}
                for param_name, bound in param.items():
                    indv[elem][param_name] = random.uniform(bound[0], bound[1])
            pos_to_check.append(indv)
            flyers, changed_params = generate_flyers(pos_to_check, num_between_vals, sim_id, server_name,
                                                     root_dir, watch_name, run_parallel)
            for i in range(len(flyers)):
                run_fly_sim([flyers[i]])
                if i == 0:
                    # rand_position, rand_intensity = omea_evaluation(changed_params[i], first_scan=True)
                    rand_position, rand_intensity = omea_evaluation(first_scan=True)
                else:
                    # partial_pop_pos, partial_pop_int = omea_evaluation(changed_params[i], first_scan=False)
                    partial_pop_pos, partial_pop_int = omea_evaluation(first_scan=False)
                    rand_position.extend(partial_pop_pos)
                    rand_intensity.extend(partial_pop_int)
            rand_position = rand_position[1:]
            rand_intensity = rand_intensity[1:]
            if rand_intensity[0] > intensities[change_indx]:
                population[change_indx] = rand_position[0]
                intensities[change_indx] = rand_intensity[0]
    x_best = best_gen_sol[-1]
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')
    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)
