/**
 * \file runtime_api.h
 * \brief Runtime library interface
 */

#ifndef MAGENT_RUNTIME_API_H
#define MAGENT_RUNTIME_API_H

#include "Environment.h"

extern "C" {

using ::magent::environment::EnvHandle;
using ::magent::environment::GroupHandle;

/**
 *  General Environment
 */
// game
int env_new_game(EnvHandle *game, const char *name);
int env_delete_game(EnvHandle game);
int env_config_game(EnvHandle game, const char *name, void *p_value);
int env_show_config(EnvHandle game);

// run step
int env_reset(EnvHandle game);
int env_get_observation(EnvHandle game, GroupHandle group, float **buffer);
int env_set_action(EnvHandle game, GroupHandle group, const float *actions, bool ignore_offsets=false);
int env_step(EnvHandle game, int *done, bool ignore_offsets=false);
int env_get_reward(EnvHandle game, GroupHandle group, float *buffer);

// info getter
int env_get_info(EnvHandle game, GroupHandle group, const char *name, void *buffer);

// render
int env_render(EnvHandle game);
int env_render_next_file(EnvHandle game);

/**
 *  GridWorld special
 */
// agent
int gridworld_register_agent_type(EnvHandle game, const char *name, int n, const char **keys, float *values);
int gridworld_new_group(EnvHandle game, const char *agent_type_name, GroupHandle *group);
int gridworld_add_agents(EnvHandle game, GroupHandle group, int worst_delay, int pkt_length,
                        int n, int *routes, int *offsets);


// reward description
int gridworld_define_agent_symbol(EnvHandle game, int no, int group, int index);
int gridworld_add_reward_rule(EnvHandle game, int op, int *receiver, float *value, int n_receiver, bool is_terminal);
}

#endif // MAGENT_RUNTIME_API_H
