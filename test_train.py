"""
A test with four types of cycles
"""
import tsn_scheduler
from parameters import *
import time
import logging as log
import numpy as np
from models import buffer
from utility.archi2graph import Graph


def generate_map(env, graph: Graph):
    """
    mainly used for add flow agents into the environment

    in c++ engine, used as
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    flow shaped as:
    source_node, end_node, cycle, delay, pkt_len
    """
    handles = env.get_handles()
    flow_dic = graph.flow_dic
    routes_dic = graph.routes_dic
    for flow in flow_dic:
        routes = routes_dic[str(flow[0])][str(flow[1])].split(',')
        routes = [int(i) for i in routes]
        offsets = np.random.randint(flow[2], size=len(routes))
        handle = handles[env.ordered_cycle.index(flow[2])]
        env.add_agents(handle, flow[3], flow[4], len(routes), routes, offsets)


def play_a_round(env, graph: Graph, handles, models, print_every, train=True, render=False, eps=None):
    env.reset()
    generate_map(env, graph)
    # random initialize
    # buffer.sample_observation(env, handles, n_obs=1, step=1)

    step_ct = 0
    done = False

    n = len(handles)
    obs = [[] for _ in range(n)]
    next_obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %.2f number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            # let rl-models infer action in parallel (non-blocking)
            action = models[i].infer_action(obs[i]).numpy()
            # add gauss noises, delete OUnoises after consideration
            np.clip(action + np.random.normal(0, eps*args.sigma, size=action.shape), 0.0, 0.15, out=action)
            acts[i] = action

        for i in range(n):
            # acts[i] = rl-models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(handles[i], acts[i], models[i].ignore_offsets)

        # simulate one step
        done = env.step(models[0].ignore_offsets)
        if done:
            log.info("done")
            print("done")

        for i in range(n):
            next_obs[i] = env.get_observation(handles[i])

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            if train:
                # store samples in replay buffer (non-blocking)
                models[i].sample_step(ids[i], obs[i], acts[i], next_obs[i], rewards)
            s = sum(rewards) if not models[i].ignore_offsets else rewards[0]
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # check return message of previous called non-blocking function sample_step()
        # if args.train:
        #     for model in rl-models:
        #         model.check_done()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                  (step_ct, nums, np.around(step_reward[0], 2), np.around(total_reward[0], 2)))
        step_ct += 1
        if step_ct > 100:
            break

    # train
    # total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train rl-models in parallel
        train_parallel(models, n)

        # for i in range(n):
        #     total_loss[i], value[i] = rl-models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    def round_list(list_): return [round(x, 2) for x in list_]
    # return round_list(total_loss), nums, round_list(total_reward), round_list(value)
    return nums, round_list(total_reward)


def train_parallel(models, agent_size):
    target_actions = [None for _ in range(agent_size)]
    features_batch = [None for _ in range(agent_size)]
    target_actions_batch = [None for _ in range(agent_size)]
    rewards_batch = [None for _ in range(agent_size)]
    next_features_batch = [None for _ in range(agent_size)]
    last_actions_batch = [None for _ in range(agent_size)]
    # Get sampling range
    record_range = min(models[0].sample_buffer.capacity, models[0].sample_buffer.counter())

    # Randomly sample indices
    batch_indices = np.random.choice(record_range, models[0].batch_size)
    # get actions from target policy network muon'
    for i in range(agent_size):
        target_actions[i], features_batch[i], target_actions_batch[i], rewards_batch[i], next_features_batch[i], \
            last_actions_batch[i] = models[i].train_first(models[i].sample_buffer, batch_indices)
    # calculate the gradient of the online Q network with total actions and states
    # and update online Q network and online policy network(actor network)
    for i in range(agent_size):
        models[i].train_second(models[i].sample_buffer, batch_indices, target_actions, features_batch,
                               target_actions_batch, rewards_batch, next_features_batch, last_actions_batch)
    # Updating target networks for each agent
    for i in range(agent_size):
        # soft update, tau: update rate
        models[i].update_target(0.01)


def main():
    # set logger
    buffer.init_logger(args.name)

    # init the game
    graph = Graph()
    env = tsn_scheduler.envs.GridWorld('first_demo', global_cycle=args.global_cycle*4, nodes_num=len(graph.nodes))
    # four groups of agents
    handles = env.get_handles()

    # reset environment and add agents
    env.reset()
    generate_map(env, graph)

    # sample eval observation set and initialize observation space
    eval_obs = [None, None, None, None]
    # if args.eval:
    #     print("sample eval set...")
    #     eval_obs = buffer.sample_observation(env, handles, n_obs=50, step=100)

    # DDPG
    from models import DDPolicyGradient
    RLModel = DDPolicyGradient

    # load rl-models
    names = [args.name + "-g0", args.name + "-g1", args.name + "-g2", args.name + "-g3"]
    models = []

    # initializing parameters
    # xx_lr: Learning rate for actor-critic rl-models
    # reward_decay: Discount factor for future rewards
    # target_update: Used to update target networks
    nums = [env.get_num(handle) for handle in handles]
    for i in range(len(names)):
        model_args = {'eval_obs': eval_obs[i], 'memory_size': 8 * 625, 'critic_lr': 1e-4, 'ignore_offsets': True,
                      'actor_lr': 5e-5, 'reward_decay': 0.95, 'target_update': 0.005, 'nums_all_agent': nums}
        # rl-models.append(ProcessingModel(env, handles[i], names[i], 20000+i, 1000, RLModel, **model_args))
        models.append(RLModel(env, handles[i], names[i], **model_args))

    # load if
    savedir = './DataSaved/Vehicle_NetWork/save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print state info
    print("view_size", env.get_view_space(handles[0]))
    print("feature_size", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = buffer.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05])
        # loss, num, reward, value
        num, reward = play_a_round(env, graph, handles, models,
                                   train=args.train, print_every=50,
                                   render=args.render or (k+1) % args.render_every == 0,
                                   eps=eps)  # for e-greedy
        # log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        log.info("round %d\t num: %s\t reward: %s" % (k, num, reward))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save rl-models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    # for model in rl-models:
    #     model.quit()


if __name__ == "__main__":
    main()
