/**
 * \file runtime_api.cc
 * \brief Runtime library interface
 */

#include "Environment.h"
#include "gridworld/GridWorld.h"
#include "utility/utility.h"
#include "runtime_api.h"
#include <iostream>
using namespace std;
/**
 *  General Environment
 */
int env_new_game(EnvHandle *game, const char *name) {
    using ::magent::utility::strequ;

    if (strequ(name, "GridWorld")) {
        *game = new ::magent::gridworld::GridWorld();
    } else {
        cout << "invalid name of game" << endl;
    }
    return 0;
}

int env_delete_game(EnvHandle game) {
    LOG(TRACE) << "env delete game.  ";
    delete game;
    return 0;
}

int env_config_game(EnvHandle game, const char *name, void *p_value) {
    LOG(TRACE) << "env config game.  ";
    game->set_config(name, p_value);
    return 0;
}

int env_show_config(EnvHandle game) {
    game->show_config();
    return 0;
}

// run step
int env_reset(EnvHandle game) {
    LOG(TRACE) << "env reset.  ";
    game->reset();
    return 0;
}

int env_get_observation(EnvHandle game, GroupHandle group, float **buffer) {
    LOG(TRACE) << "env get observation.  ";
    game->get_observation(group, buffer);
    return 0;
}

int env_set_action(EnvHandle game, GroupHandle group, const float *actions, bool ignore_offsets) {
    LOG(TRACE) << "env set action.  ";
    game->set_action(group, actions, ignore_offsets);
    return 0;
}

int env_step(EnvHandle game, int *done, bool ignore_offsets) {
    LOG(TRACE) << "env step.  ";
    game->step(done, ignore_offsets);
    return 0;
}

int env_get_reward(EnvHandle game, GroupHandle group, float *buffer) {
    LOG(TRACE) << "env get reward.  ";
    game->get_reward(group, buffer);
    return 0;
}

// info getter
int env_get_info(EnvHandle game, GroupHandle group, const char *name, void *buffer) {
    LOG(TRACE) << "env get info " << name << ".  ";
    game->get_info(group, name, buffer);
    return 0;
}

// render
int env_render(EnvHandle game) {
    LOG(TRACE) << "env render.  ";
    game->render();
    return 0;
}

int env_render_next_file(EnvHandle game) {
    LOG(TRACE) << "env render next file.  ";
    // temporally only needed in DiscreteSnake
    //((::magent::discrete_snake::DiscreteSnake *)game)->render_next_file();
    return 0;
}

/**
 *  GridWorld special
 */
// agent
int gridworld_register_agent_type(EnvHandle game, const char *name, int n,
                                  const char **keys, float *values) {
    LOG(TRACE) << "gridworld register agent type.  ";
    ((::magent::gridworld::GridWorld *)game)->register_agent_type(name, n, keys, values);
    return 0;
}

int gridworld_new_group(EnvHandle game, const char *agent_type_Name, GroupHandle *group) {
    LOG(TRACE) << "gridworld new group.  ";
    ((::magent::gridworld::GridWorld *)game)->new_group(agent_type_Name, group);
    return 0;
}

int gridworld_add_agents(EnvHandle game, GroupHandle group, int worst_delay, int pkt_length,
                        int n, int *routes, int *offsets) {
    LOG(TRACE) << "gridworld add agents.  ";
    ((::magent::gridworld::GridWorld *)game)->add_agent(group, worst_delay, pkt_length, n, routes, offsets);
    return 0;
}

// reward description
int gridworld_define_agent_symbol(EnvHandle game, int no, int group, int index) {
    LOG(TRACE) << "gridworld define agent symbol";
    ((::magent::gridworld::GridWorld *)game)->define_agent_symbol(no, group, index);
    return 0;
}

int gridworld_add_reward_rule(EnvHandle game, int op, int *receiver, float *value, int n_receiver, bool is_terminal) {
    LOG(TRACE) << "gridworld add reward rule";
    ((::magent::gridworld::GridWorld *)game)->add_reward_rule(op, receiver, value, n_receiver, is_terminal);
    return 0;
}
