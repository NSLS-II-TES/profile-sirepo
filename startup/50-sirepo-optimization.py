import bluesky.plans as bp

import sirepo_bluesky.sirepo_flyer as sf

import numpy as np
import random
import matplotlib.pyplot as plt


def omea_evaluation(param_bounds, popsize, num_interm_vals, num_scans_at_once,
                    flyer_name, intensity_name, uids):
    """Look at data from flyers and pick best individuals

        Parameters
        ----------
        param_bounds : Dict of dict of list
                       In the form of {optical element:
                                       {parameter name: [lower bound, upper bound],
                                       ...
                                       }
                                      }
        population : list of dicts of dicts
                     Population of individuals. Needed for comparison later.
        num_interm_vals : int
                          Number of positions to look at in between two individuals
        num_scans_at_once : int
                            Number of parallel scans to run at a time
    """
    # TODO: parameterize flyer name
    pop_positions = []
    pop_intensities = []
    # get number of records to look at from databroker
    num_records = calc_num_records(popsize, num_interm_vals, num_scans_at_once)
    # get data from databroker
    fly_data = []
    # for i in range(-int(num_records), 0):
    for uid in uids:
        fly_data.append(db[uid].table(flyer_name))
    interm_pos = []
    interm_int = []
    for i in fly_data:
        print(i)
    # Create all sets of indices for population values first
    pop_indxs = [[0, 1]]
    while len(pop_indxs) < popsize:
        i_index = pop_indxs[-1][0]
        j_index = pop_indxs[-1][1]
        pre_mod_val = j_index + num_interm_vals + 1
        mod_res = pre_mod_val % num_scans_at_once
        int_div_res = pre_mod_val // num_scans_at_once
        if mod_res == 0:
            i_index = i_index + (int_div_res - 1)
            j_index = pre_mod_val
        else:
            i_index = i_index + int_div_res
            j_index = mod_res
        pop_indxs.append([i_index, j_index])
    curr_pop_index = 0
    for i in range(len(fly_data)):
        curr_interm_pos = []
        curr_interm_int = []
        for j in range(1, len(fly_data[i]) + 1):
            if (i == pop_indxs[curr_pop_index][0] and
                    j == pop_indxs[curr_pop_index][1]):
                pop_intensities.append(fly_data[i][f'{flyer_name}_{intensity_name}'][j])
                indv = {}
                for elem, param in param_bounds.items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = fly_data[i][f'{flyer_name}_{elem}_{param_name}'][j]
                pop_positions.append(indv)
                curr_pop_index += 1
            else:
                curr_interm_int.append(fly_data[i][f'{flyer_name}_{intensity_name}'][j])
                indv = {}
                for elem, param in param_bounds.items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = fly_data[i][f'{flyer_name}_{elem}_{param_name}'][j]
                curr_interm_pos.append(indv)
        interm_pos.append(curr_interm_pos)
        interm_int.append(curr_interm_int)
    # picking best positions
    interm_max_idx = []
    print('OMEA: LEN OF INTERM_INT', len(interm_int))
    for i in range(len(interm_int)):
        curr_max_int = np.max(interm_int[i])
        interm_max_idx.append(interm_int[i].index(curr_max_int))
    print('OMEA: LEN OF interm_max_idx', len(interm_max_idx))
    print('OMEA: LEN OF pop_intensities', len(pop_intensities))
    for i in range(len(interm_max_idx)):
        if interm_int[i][interm_max_idx[i]] > pop_intensities[i + 1]:
            pop_intensities[i + 1] = interm_int[i][interm_max_idx[i]]
            pop_positions[i + 1] = interm_pos[i][interm_max_idx[i]]
    return pop_positions, pop_intensities


# TODO: move this somewhere; make into a function?
param_bounds = {'Aperture': {'horizontalSize': [1, 10],
                             'verticalSize': [.1, 1]},
                'Lens': {'horizontalFocalLength': [10, 30]},
                'Obstacle': {'horizontalSize': [1, 10]}}

best_fitness = [0]


def run_fly_sim(population, num_interm_vals, num_scans_at_once,
                sim_id, server_name, root_dir, watch_name, run_parallel):
    uid_list = []
    flyers = generate_flyers(population, num_interm_vals,
                             sim_id, server_name, root_dir, watch_name, run_parallel)
    # make list of flyers into list of list of flyers
    # pass 1 sublist of flyers at a time
    flyers = [flyers[i:i+num_scans_at_once] for i in range(0, len(flyers), num_scans_at_once)]
    for i in range(len(flyers)):
        # uids = (yield from bp.fly(flyers[i]))
        yield from bp.fly(flyers[i])
        # uid_list.append(uids)
    for i in range(-len(flyers), 0):
        uid_list.append(i)
    return uid_list


def generate_flyers(population, num_between_vals,
                    sim_id, server_name, root_dir, watch_name, run_parallel):
    flyers = []
    params_to_change = []
    for i in range(len(population) - 1):
        between_param_linspaces = []
        if i == 0:
            params_to_change.append(population[i])
        for elem, param in population[i].items():
            for param_name, pos in param.items():
                between_param_linspaces.append(np.linspace(pos, population[i + 1][elem][param_name],
                                                           (num_between_vals + 2))[1:-1])

        for j in range(len(between_param_linspaces[0])):
            ctr = 0
            indv = {}
            for elem, param in population[0].items():
                indv[elem] = {}
                for param_name in param.keys():
                    indv[elem][param_name] = between_param_linspaces[ctr][j]
                    ctr += 1
            params_to_change.append(indv)
        params_to_change.append(population[i + 1])
    for param in params_to_change:
        sim_flyer = sf.SirepoFlyer(sim_id=sim_id, server_name=server_name,
                                   root_dir=root_dir, params_to_change=[param],
                                   watch_name=watch_name, run_parallel=run_parallel)
        flyers.append(sim_flyer)
    return flyers


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


def create_selection_params(population, cross_indv):
    positions = [elm for elm in cross_indv]
    positions.insert(0, population[0])
    return positions


def create_rand_selection_params(population, intensities, param_bounds):
    positions = []
    change_indx = intensities.index(np.min(intensities))
    positions.append(population[0])
    indv = {}
    for elem, param in param_bounds.items():
        indv[elem] = {}
        for param_name, bound in param.items():
            indv[elem][param_name] = random.uniform(bound[0], bound[1])
    positions.append(indv)
    return positions, change_indx


def select(param_bounds, population, intensities, num_interm_vals, num_scans_at_once,
           flyer_name, intensity_name, uids):
    # OMEA
    new_population, new_intensities = omea_evaluation(param_bounds, len(population) + 1,
                                                      num_interm_vals, num_scans_at_once,
                                                      flyer_name, intensity_name, uids=uids)
    # # cut first value; it's already been evaluated in population
    del new_population[0]
    del new_intensities[0]
    # analysis
    print('LEN OF NEW_INTENSITIES', len(new_intensities))
    for i in range(len(new_intensities)):
        print(intensities[i], '\n', new_intensities[i])
        if new_intensities[i] > intensities[i]:
            population[i] = new_population[i]
            intensities[i] = new_intensities[i]
    population.reverse()
    intensities.reverse()
    return population, intensities

# def select(population, crossover_indv, ind_sol, num_between_vals,
#            sim_id, server_name, root_dir, watch_name, run_parallel):
#     # TODO: FIX THIS
#     positions = [elm for elm in crossover_indv]
#     positions.insert(0, population[0])
#     flyers, changed_params = generate_flyers(positions, num_between_vals, sim_id, server_name,
#                                              root_dir, watch_name, run_parallel)
#     for i in range(len(flyers)):
#         run_fly_sim([flyers[i]])
#         if i == 0:
#             new_population, new_ind_sol = omea_evaluation(param_bounds, first_scan=True)
#             # new_population, new_ind_sol = omea_evaluation(first_scan=True)
#         else:
#             partial_pop_pos, partial_pop_int = omea_evaluation(param_bounds, first_scan=False)
#             # partial_pop_pos, partial_pop_int = omea_evaluation(first_scan=False)
#             new_population.extend(partial_pop_pos)
#             new_ind_sol.extend(partial_pop_int)
#     new_population = new_population[1:]
#     new_ind_sol = new_ind_sol[1:]
#     for i in range(len(new_ind_sol)):
#         if new_ind_sol[i] > ind_sol[i]:
#             population[i] = new_population[i]
#             ind_sol[i] = new_ind_sol[i]
#     population.reverse()
#     ind_sol.reverse()
#     return population, ind_sol


def calc_num_records(popsize, num_interm_vals, num_scans_at_once):
    """Calculate number of records to look at"""
    num_flyers = popsize + (popsize - 1) * num_interm_vals
    num_records = np.ceil(num_flyers / num_scans_at_once)
    return num_records


def optimize(fly_plan, bounds, num_interm_vals, sim_id, server_name, root_dir, watch_name,
             popsize=3, crosspb=.8, mut=.1, mut_type='rand/1', threshold=0,
             max_iter=100, run_parallel=True, num_scans_at_once=5,
             flyer_name='sirepo_flyer', intensity_name='mean'):
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
    print('INITIAL POPULATION')
    for i in initial_population:
        print(i)

    # TODO: test this!
    uid_list = (yield from fly_plan(initial_population, num_interm_vals, num_scans_at_once,
                                        sim_id, server_name, root_dir, watch_name, run_parallel))

    # OMEA evaluation
    population, intensities = omea_evaluation(param_bounds=bounds, popsize=len(initial_population),
                                              num_interm_vals=num_interm_vals, num_scans_at_once=num_scans_at_once,
                                              flyer_name=flyer_name, intensity_name=intensity_name, uids=uid_list)
    population.reverse()
    intensities.reverse()
    v = 0
    consec_best_ctr = 0
    old_best_fit_int = 0
    # termination conditions
    while not (v > 0):
        print('*********************', len(population), len(intensities))
        # while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_int >= threshold)):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(population, mut_type, mut, bounds, ind_sol=intensities)
        print('MUTATED_TRIAL_POP')
        for i in mutated_trial_pop:
            print(i)
        # crossover
        cross_trial_pop = crossover(population, mutated_trial_pop, crosspb)
        print('CROSS_TRIAL_POP')
        for i in cross_trial_pop:
            print(i)
        # select
        positions = create_selection_params(population, cross_trial_pop)
        uid_list = (yield from fly_plan(positions, num_interm_vals, num_scans_at_once,
                                           sim_id, server_name, root_dir, watch_name, run_parallel))
        population, intensities = select(param_bounds, population, intensities,
                                         num_interm_vals, num_scans_at_once,
                                         flyer_name, intensity_name, uids=uid_list)
        print('select')
        for i in range(len(population)):
            print(population[i], intensities[i])
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
            positions, change_indx = create_rand_selection_params(population, intensities,
                                                                  param_bounds)
            print('CHANGED: ', population[change_indx], intensities[change_indx])
            uid_list = (yield from fly_plan(positions, num_interm_vals, num_scans_at_once,
                                               sim_id, server_name, root_dir, watch_name, run_parallel))
            rand_pop, rand_int = select(param_bounds, [population[change_indx]], [intensities[change_indx]],
                                        num_interm_vals, num_scans_at_once, flyer_name, intensity_name,
                                        uids=uid_list)
            print('00000', len(rand_pop), len(rand_int))
            population[change_indx] = rand_pop[0]
            intensities[change_indx] = rand_int[0]
            print('NEW POP AND INT')
            for i in range(len(population)):
                print(population[i], intensities[i])
    #         # randomize individual
    #         # TODO: FIX THIS
    #         pos_to_check = []
    #         change_indx = intensities.index(np.min(intensities))
    #         pos_to_check.append(population[0])
    #         indv = {}
    #         for elem, param in param_bounds.items():
    #             indv[elem] = {}
    #             for param_name, bound in param.items():
    #                 indv[elem][param_name] = random.uniform(bound[0], bound[1])
    #         pos_to_check.append(indv)
    #         flyers, changed_params = generate_flyers(pos_to_check, num_between_vals, sim_id, server_name,
    #                                                  root_dir, watch_name, run_parallel)
    #         for i in range(len(flyers)):
    #             run_fly_sim([flyers[i]])
    #             if i == 0:
    #                 rand_position, rand_intensity = omea_evaluation(param_bounds, first_scan=True)
    #                 # rand_position, rand_intensity = omea_evaluation(first_scan=True)
    #             else:
    #                 partial_pop_pos, partial_pop_int = omea_evaluation(param_bounds, first_scan=False)
    #                 # partial_pop_pos, partial_pop_int = omea_evaluation(first_scan=False)
    #                 rand_position.extend(partial_pop_pos)
    #                 rand_intensity.extend(partial_pop_int)
    #         rand_position = rand_position[1:]
    #         rand_intensity = rand_intensity[1:]
    #         if rand_intensity[0] > intensities[change_indx]:
    #             population[change_indx] = rand_position[0]
    #             intensities[change_indx] = rand_intensity[0]
    x_best = best_gen_sol[-1]
    print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
    print('It took', v, 'generations')
    plot_index = np.arange(len(best_fitness))
    plt.figure()
    plt.plot(plot_index, best_fitness)
