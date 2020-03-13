import bluesky.plans as bp

import sirepo_bluesky.sirepo_flyer as sf

import numpy as np
import random
from multiprocessing import Process


def omea_evaluation(params_to_change, first_scan):  # , num_scans=1):
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
                indv = {}
                for elem, param in params_to_change[0].items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                pop_positions.append(indv)
            else:
                between_intensities.append(t['sirepo_flyer_mean'][j + 1])
                indv = {}
                for elem, param in params_to_change[0].items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                between_positions.append(indv)
        else:
            if j == (len(t) - 1):
                pop_intensity.append(t['sirepo_flyer_mean'][j + 1])
                indv = {}
                for elem, param in params_to_change[0].items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                pop_positions.append(indv)
            else:
                between_intensities.append(t['sirepo_flyer_mean'][j + 1])
                indv = {}
                for elem, param in params_to_change[0].items():
                    indv[elem] = {}
                    for param_name in param.keys():
                        indv[elem][param_name] = t[f'sirepo_flyer_{elem}_{param_name}'][j + 1]
                between_positions.append(indv)
    current_max_int = np.max(between_intensities)
    curr_max_indx = between_intensities.index(current_max_int)
    if current_max_int > pop_intensity[-1]:
        print('updating population/intensities')
        pop_intensity[-1] = current_max_int
        pop_positions[-1] = between_positions[curr_max_indx]
    return pop_positions, pop_intensity


param_bounds = {'Aperture': {'horizontalSize': [1, 10],
                             'verticalSize': [.1, 1]},
                'Lens': {'horizontalFocalLength': [10, 30]},
                'Obstacle': {'horizontalSize': [1, 10]}}

best_fitness = [0]


def run(flyer):
    # print(f'flying with {flyer}')
    RE(bp.fly([flyer]))


def run_fly_sim(flyers, run_scans_parallel):
    if run_scans_parallel:
        procs = []
        for i in range(len(flyers)):
            p = Process(target=run, args=(flyers[i],))
            p.start()
            procs.append(p)
        # wait for procs to finish
        for p in procs:
            p.join()
    else:
        # run serial
        RE(bp.fly([flyer for flyer in flyers]))


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

    x_diff = {}
    for motor_name, pos in x_2.items():
        x_diff[motor_name] = x_2[motor_name] - x_3[motor_name]
    v_donor = {}
    for motor_name, pos in x_1.items():
        v_donor[motor_name] = x_1[motor_name] + mut * x_diff[motor_name]
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

    x_diff = {}
    for motor_name, pos in x_1.items():
        x_diff[motor_name] = x_1[motor_name] - x_2[motor_name]
    v_donor = {}
    for motor_name, pos in x_best.items():
        v_donor[motor_name] = x_best[motor_name] + mut * x_diff[motor_name]
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


# def crossover(population, mutated_indv, crosspb):
#     crossover_indv = []
#     for i in range(len(population)):
#         x_t = population[i]
#         v_trial = {}
#         for motor_name, pos in x_t.items():
#             crossover_val = random.random()
#             if crossover_val <= crosspb:
#                 v_trial[motor_name] = mutated_indv[i][motor_name]
#             else:
#                 v_trial[motor_name] = x_t[motor_name]
#         crossover_indv.append(v_trial)
#     return crossover_indv


# def select(population, ind_sol, motors, crossover_indv, max_velocity):
#     positions = [elm for elm in crossover_indv]
#     positions.insert(0, population[0])
#     velocities_list, times_list = generate_flyer_params(population, max_velocity)
#     for param, vel, time_ in zip(population, velocities_list, times_list):
#         hf = HardwareFlyer(params_to_change=param,
#                            velocities=vel,
#                            time_to_travel=time_,
#                            detector=xs, motors=motors)
#         yield from bp.fly([hf])
#
#         hf_flyers.append(hf)
#
#     positions, intensities = omea_evaluation(len(population))
#     positions = positions[1:]
#     intensities = intensities[1:]
#     for i in range(len(intensities)):
#         if intensities[i] > ind_sol[i]:
#             population[i] = positions[i]
#             ind_sol[i] = intensities[i]
#     return population, ind_sol


def optimize(bounds=param_bounds, popsize=3, crosspb=.8, mut=.1, mut_type='rand/1', threshold=0,
             max_iter=100, run_scans_parallel=True):
    # Initial population
    initial_population = []
    flyers = []
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

    for i in initial_population:
        print(i)

    # create linspaces between initial_population
    params_to_change = []
    for i in range(popsize-1):
        curr_params_to_change = []
        between_param_linspaces = []
        if i == 0:
            curr_params_to_change.append(initial_population[i])
        for elem, param in initial_population[i].items():
            for param_name, pos in param.items():
                between_param_linspaces.append(np.linspace(pos, initial_population[i + 1][elem][param_name], 4)[1:-1])

        for j in range(len(between_param_linspaces[0])):
            ctr = 0
            indv = {}
            for elem, param in initial_population[0].items():
                indv[elem] = {}
                for param_name in param.keys():
                    indv[elem][param_name] = between_param_linspaces[ctr][j]
                    ctr += 1
            curr_params_to_change.append(indv)
        curr_params_to_change.append(initial_population[i + 1])
        params_to_change.append(curr_params_to_change)

        # "fly" scan
        sim_flyer = sf.SirepoFlyer(sim_id='87XJ4oEb', server_name='http://10.10.10.10:8000',
                                   root_dir=root_dir, params_to_change=curr_params_to_change,
                                   watch_name='W60', run_parallel=True)
        flyers.append(sim_flyer)

    population = []
    intensities = []
    for i in range(len(flyers)):
        run_fly_sim([flyers[i]], run_scans_parallel=False)
        if i == 0:
            population, intensities = omea_evaluation(params_to_change[i], True)
        else:
            partial_pop_pos, partial_pop_int = omea_evaluation(params_to_change[i], False)
            population.extend(partial_pop_pos)
            intensities.extend(partial_pop_int)
    print('\nFINAL NEW', population, intensities)
    v = 0
    consec_best_ctr = 0
    old_best_fit_int = 0
    # termination conditions
    while not (v > 0):
        print(f'GENERATION {v + 1}')
        best_gen_sol = []
        # mutate
        mutated_trial_pop = mutate(population, mut_type, mut, bounds, ind_sol=intensities)
        # crossover

        # select

        # score keeping

        # randomize individual

#     #
#     # # Termination conditions
#     # v = 0  # generation number
#     # consec_best_ctr = 0  # counting successive generations with no change to best value
#     # old_best_fit_val = 0
#     # # while not v > 0:
#     # while not ((v > max_iter) or (consec_best_ctr >= 5 and old_best_fit_val >= threshold)):
#     #     print(f'GENERATION {v + 1}')
#     #     best_gen_sol = []
#     #     # mutate
#     #     mutated_trial_pop = mutate(pop_positions, mut_type, mut, bounds, ind_sol=pop_intensity)
#     #     # crossover
#     #     cross_trial_pop = crossover(pop_positions, mutated_trial_pop, crosspb)
#     #
#     #     # select, how can this be it's own function?
#     #     select_positions = [elm for elm in cross_trial_pop]
#     #     indv = {}
#     #     for motor_name, motor_obj in motors.items():
#     #         indv[motor_name] = motor_obj.user_readback.get()
#     #     select_positions.insert(0, indv)
#     #     for params in select_positions:
#     #         sim_flyer = sf.SirepoFlyer(sim_id='87XJ4oEb', server_name='http://10.10.10.10:8000',
#     #                                    root_dir=root_dir, params_to_change=[params],
#     #                                    watch_name='W60', run_parallel=False)
#     #         flyers.append(sim_flyer)
#     #     positions, intensities = omea_evaluation(len(select_positions))
#     #     positions = positions[1:]
#     #     intensities = intensities[1:]
#     #     for i in range(len(intensities)):
#     #         if intensities[i] > pop_intensity[i]:
#     #             pop_positions[i] = positions[i]
#     #             pop_intensity[i] = intensities[i]
#     #     # pop_positions, pop_intensity = select(positions, intensities, motors,
#     #     #                                       cross_trial_pop, max_velocity)
#     #
#     #     # get best solution
#     #     gen_best = np.max(pop_intensity)
#     #     best_indv = pop_positions[pop_intensity.index(gen_best)]
#     #     best_gen_sol.append(best_indv)
#     #     best_fitness.append(gen_best)
#     #
#     #     print('      > FITNESS:', gen_best)
#     #     print('         > BEST POSITIONS:', best_indv)
#     #
#     #     v += 1
#     #     if np.round(gen_best, 6) == np.round(old_best_fit_val, 6):
#     #         consec_best_ctr += 1
#     #         print('Counter:', consec_best_ctr)
#     #     else:
#     #         consec_best_ctr = 0
#     #     old_best_fit_val = gen_best
#     #
#     #     if consec_best_ctr >= 5 and old_best_fit_val >= threshold:
#     #         print('Finished')
#     #         break
#     #     else:
#     #         # randomize worst individual and repeat from mutate
#     #         pos_to_check = []
#     #         change_indx = pop_intensity.index(np.min(pop_intensity))
#     #         changed_indv = pop_positions[change_indx]
#     #         indv = {}
#     #         for motor_name, motor_obj in motors.items():
#     #             indv[motor_name] = motor_obj.user_readback.get()
#     #         pos_to_check.append(indv)
#     #         indv = {}
#     #         for motor_name, pos in changed_indv.items():
#     #             indv[motor_name] = random.uniform(bounds[motor_name]['low'],
#     #                                               bounds[motor_name]['high'])
#     #         pos_to_check.append(indv)
#     #
#     #         for params in pos_to_check:
#     #             sim_flyer = sf.SirepoFlyer(sim_id='87XJ4oEb', server_name='http://10.10.10.10:8000',
#     #                                        root_dir=root_dir, params_to_change=[params],
#     #                                        watch_name='W60', run_parallel=False)
#     #             flyers.append(sim_flyer)
#     #         rand_position, rand_intensity = omea_evaluation(len(pos_to_check))
#     #         rand_position = rand_position[1:]
#     #         rand_intensity = rand_intensity[1:]
#     #         if rand_intensity[0] > pop_intensity[change_indx]:
#     #             pop_positions[change_indx] = rand_position[0]
#     #             pop_intensity[change_indx] = rand_intensity[0]
#     #
#     # # best solution overall should be last one
#     # x_best = best_gen_sol[-1]
#     # print('\nThe best individual is', x_best, 'with a fitness of', gen_best)
#     # print('It took', v, 'generations')
#     #
#     # plot_index = np.arange(len(best_fitness))
#     # plt.figure()
#     # plt.plot(plot_index, best_fitness)
#
#
# # def ensure_bounds(vec, bounds):
# #     # Makes sure each individual stays within bounds and adjusts them if they aren't
# #     vec_new = {}
# #     # cycle through each variable in vector
# #     for motor_name, pos in vec.items():
# #         # variable exceeds the minimum boundary
# #         if pos < bounds[motor_name]['low']:
# #             vec_new[motor_name] = bounds[motor_name]['low']
# #         # variable exceeds the maximum boundary
# #         if pos > bounds[motor_name]['high']:
# #             vec_new[motor_name] = bounds[motor_name]['high']
# #         # the variable is fine
# #         if bounds[motor_name]['low'] <= pos <= bounds[motor_name]['high']:
# #             vec_new[motor_name] = pos
# #     return vec_new
# #
# #
# # def rand_1(pop, popsize, target_indx, mut, bounds):
# #     # mutation strategy
# #     # v = x_r1 + F * (x_r2 - x_r3)
# #     idxs = [idx for idx in range(popsize) if idx != target_indx]
# #     a, b, c = np.random.choice(idxs, 3, replace=False)
# #     x_1 = pop[a]
# #     x_2 = pop[b]
# #     x_3 = pop[c]
# #
# #     x_diff = {}
# #     for motor_name, pos in x_2.items():
# #         x_diff[motor_name] = x_2[motor_name] - x_3[motor_name]
# #     v_donor = {}
# #     for motor_name, pos in x_1.items():
# #         v_donor[motor_name] = x_1[motor_name] + mut * x_diff[motor_name]
# #     v_donor = ensure_bounds(v_donor, bounds)
# #     return v_donor
# #
# #
# # def mutate(population, strategy, mut, bounds, ind_sol):
# #     mutated_indv = []
# #     for i in range(len(population)):
# #         if strategy == 'rand/1':
# #             v_donor = rand_1(population, len(population), i, mut, bounds)
# #         # elif strategy == 'best/1':
# #         #     v_donor = best_1(population, len(population), i, mut, bounds, ind_sol)
# #         # elif strategy == 'current-to-best/1':
# #         #     v_donor = current_to_best_1(population, len(population), i, mut, bounds, ind_sol)
# #         # elif strategy == 'best/2':
# #         #     v_donor = best_2(population, len(population), i, mut, bounds, ind_sol)
# #         # elif strategy == 'rand/2':
# #         #     v_donor = rand_2(population, len(population), i, mut, bounds)
# #         mutated_indv.append(v_donor)
# #     return mutated_indv
# #
# #
# # def crossover(population, mutated_indv, crosspb):
# #     crossover_indv = []
# #     for i in range(len(population)):
# #         x_t = population[i]
# #         v_trial = {}
# #         for motor_name, pos in x_t.items():
# #             crossover_val = random.random()
# #             if crossover_val <= crosspb:
# #                 v_trial[motor_name] = mutated_indv[i][motor_name]
# #             else:
# #                 v_trial[motor_name] = x_t[motor_name]
# #         crossover_indv.append(v_trial)
# #     return crossover_indv
#
#
# # def select(population, ind_sol, motors, crossover_indv, max_velocity):
# #     positions = [elm for elm in crossover_indv]
# #     positions.insert(0, population[0])
# #     velocities_list, times_list = generate_flyer_params(population, max_velocity)
# #     for params in params_to_change:
# #         sim_flyer = sf.SirepoFlyer(sim_id='87XJ4oEb', server_name='http://10.10.10.10:8000',
# #                                    root_dir=root_dir, params_to_change=[params],
# #                                    watch_name='W60', run_parallel=False)
# #
# #         flyers.append(sim_flyer)
# #
# #     positions, intensities = omea_evaluation(len(population))
# #     positions = positions[1:]
# #     intensities = intensities[1:]
# #     for i in range(len(intensities)):
# #         if intensities[i] > ind_sol[i]:
# #             population[i] = positions[i]
# #             ind_sol[i] = intensities[i]
# #     return population, ind_sol
#


if __name__ == '__main__':
    optimize()
