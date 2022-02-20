""" demo of scheduler """
import tsn_scheduler


def get_config(global_cycle, nodes_num):
    gw = tsn_scheduler.envs.gridworld
    cfg = gw.Config()

    cfg.set({"global_cycle": global_cycle, "nodes_num": nodes_num})
    cfg.set({"embedding_size": 10})
    cfg.set({"render_dir": b"___debug___"})
    cfg.ordered_cycle = [64, 128, 256, 512]

    cycle64 = cfg.register_agent_type(
        "cycle64",
        {'height': 1, 'cycle': 256, 'step_reward': -0.005})

    cycle128 = cfg.register_agent_type(
        "cycle128",
        {'height': 1, 'cycle': 512, 'step_reward': -0.005})

    cycle256 = cfg.register_agent_type(
        "cycle256",
        {'height': 1, 'cycle': 1024, 'step_reward': -0.005})

    cycle512 = cfg.register_agent_type(
        "cycle512",
        {'height': 1, 'cycle': 2048, 'step_reward': -0.005})

    # g: group  number: index of this group
    g0 = cfg.add_group(cycle64)
    g1 = cfg.add_group(cycle128)
    g2 = cfg.add_group(cycle256)
    g3 = cfg.add_group(cycle512)

    a = gw.AgentSymbol(g0, index='all')
    b = gw.AgentSymbol(g1, index='all')
    c = gw.AgentSymbol(g2, index='all')
    d = gw.AgentSymbol(g3, index='all')

    # reward shaping to meet the constraints
    cfg.add_reward_rule("no_collide", receiver=[a, b, c, d], value=[0.1, 0.1, 0.1, 0.1])
    cfg.add_reward_rule("e2e_delay", receiver=[a, b, c, d], value=[0.1, 0.1, 0.1, 0.1])

    return cfg


if __name__ == '__main__':
    get_config(1, 1)
