#if 1
/**
 * \file test.cc
 * \brief unit test for some function
 */
#include <iostream>
#include "../runtime_api.h"

using namespace std;

int main() {
    // env_new_game
    EnvHandle *game = new EnvHandle;
    env_new_game(game, "GridWorld");
    // set render direction
    char* dir = "___debug___";
    env_config_game(*game, "render_dir", dir);
    // env_config_game
    int global_cycle = 4096;
    env_config_game(*game, "global_cycle", &global_cycle);
    int nodes_num = 6;
    env_config_game(*game, "nodes_num", &nodes_num);
    int embedding_size = 10;
    env_config_game(*game, "embedding_size", &embedding_size);
    env_show_config(*game);

    // register_agent_type
    const char* keys[3] = {
        "height",
        "cycle",
        "step_reward"
    };
    float values[3] = {1, 64, -0.001};
    gridworld_register_agent_type(*game, "cycle64", 3, keys, values);
    values[1] = 128;
    gridworld_register_agent_type(*game, "cycle128", 3, keys, values);
    int group1 = 0;
    gridworld_new_group(*game, "cycle64", &group1);
    int group2 = 1;
    gridworld_new_group(*game, "cycle128", &group2);

    // add reward rule
    gridworld_define_agent_symbol(*game, 0, group1, 0);
    int receiver[1] = {0};
    float value[1] = {0.1};
    gridworld_add_reward_rule(*game, 0, receiver, value, 1, false);
    gridworld_add_reward_rule(*game, 1, receiver, value, 1, false);

    // initialize env
    env_reset(*game);

    // add agents
    int buf[3] = {};
    int routes[3] = {0, 1, 2};
    int offsets[3] = {62, 0, 0};
    int worst_delay = 13; // represented by timeslot
    int pkt_length = 4;
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group1, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group2, worst_delay, pkt_length, 3, routes, offsets);
    offsets[0] = rand() % 64; offsets[1] = rand() % 64; offsets[2] = rand() % 64;
    gridworld_add_agents(*game, group2, worst_delay, pkt_length, 3, routes, offsets);

    // get env info
    env_get_info(*game, group1, "num", buf);

    // return observation
    float* linear_buffers[2];
    float view_space[6*4096] = {0};
    float feature_space[10+6+1] = {0};
    linear_buffers[0] = view_space;
    linear_buffers[1] = feature_space;

    // get agent id
    env_get_info(*game, group1, "id", buf);

    // set action
    float actions[30] = {0.92123, 0.32121, 0.13746, 0, 0, 0, 0.92123, 0.32121, 0.13746, 0, 0, 0, 0.13746, 0, 0, 0, 0.92123, 0.32121, 0.13746, 0, 0, 0, 0.92123, 0.32121, 0.13746, 0, 0, 0, 0.92123, 0.32121};
    env_set_action(*game, group1, actions, true);
    env_set_action(*game, group2, actions+18);


    // env step
    int done = false;
    env_step(*game, &done, true);

    // env render
    env_get_observation(*game, group1, linear_buffers);
    env_render(*game);

    // env get reward
    float reward_buffer[5] = {0, 0, 0, 0, 0};
    env_get_reward(*game, group1, reward_buffer);

    env_delete_game(*game);
    cout << "finish" << endl;
}
#endif
