import bluesky.plans as bp

import sirepo_bluesky.sirepo_flyer as sf


def run_fly_sim(population, num_interm_vals, num_scans_at_once,
                sim_id, server_name, root_dir, watch_name, run_parallel):
    uid_list = []
    flyers = generate_sim_flyers(population=population, num_between_vals=num_interm_vals,
                             sim_id=sim_id, server_name=server_name, root_dir=root_dir,
                             watch_name=watch_name, run_parallel=run_parallel)
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


def generate_sim_flyers(population, num_between_vals, sim_id, server_name,
                        root_dir, watch_name, run_parallel):
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