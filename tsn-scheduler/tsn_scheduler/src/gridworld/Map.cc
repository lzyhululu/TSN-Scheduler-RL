/**
 * \file Map.cc
 * \brief The map for the game engine
 */

#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "Map.h"
#include "GridWorld.h"

namespace magent {
namespace gridworld {

#define MAP_INNER_Y_ADD w

void Map::reset(int global_cycle, int node_num) {
    this->w = global_cycle;
    this->h = node_num;
    this->s_node = node_num;
    this->g_cycle = global_cycle;

    // time slots with conplement information
    if (map_slots != nullptr)
        delete [] map_slots;
    map_slots = new int[w * h];

    // 1 is the initial value, 0 represents this time slot is occupied, <0 represents occupied by several flows
    for(int i = 0; i < w * h; i++)
        map_slots[i] = 1;
}
    // add total at one-time
void Map::add_agent(Agent *agent) {
    int length = agent->get_length(), height = agent->get_height();
    int cycle = agent->get_type().cycle;
    // use quote to save the memory
    Position &pos = agent->get_pos();
    int offsets_n = pos.n;
    int *routes = pos.routes;
    int *offsets = pos.offsets;

    // fill in map
    fill_area(length, height, cycle, offsets_n, routes, offsets);
}
    // remove total flow at one-time
void Map::remove_agent(Agent *agent) {
    int length = agent->get_length(), height = agent->get_height();
    int cycle = agent->get_type().cycle;
    // use quote to save the memory
    Position &pos = agent->get_pos();
    int offsets_n = pos.n;
    int *routes = pos.routes;
    int *offsets = pos.offsets;

    // clear map
    clear_area(length, height, cycle, offsets_n, routes, offsets);
}

// do move for agent, new_offsets means the new position in the flow cycle
Reward Map::do_move(Agent *agent, const int *action_int) {
    int length = agent->get_length(), height = agent->get_height();
    int cycle = agent->get_type().cycle;
    // use quote to save the memory
    Position &pos = agent->get_pos();
    int offsets_n = pos.n;

    int *routes = pos.routes;
    int *offsets = pos.offsets;
    // old position
    clear_area(length, height, cycle, offsets_n, routes, offsets);
    // update to new position
    agent->update_pos(action_int);
    fill_area(length, height, cycle, offsets_n, routes, offsets);
    return 0.0;
}

// calculate global reward for OP_COLLIDE rule
void Map::calc_global(float value){
    float reward = 0;
    # pragma omp parallel for
    for (int i = 0; i < h * w; i++) {
        if(map_slots[i] < 0)
            reward -= map_slots[i] * value;
    }
    static_cast<Reward>(reward);
    // here replace the last reward with this new reward directly
    add_reward(reward);
}

/**
 * Utility to operate map
 */

// fill several rectangle
inline void Map::fill_area(int length, int height, int cycle, int offsets_n, int *routes, int *offsets) {
    // the node index in the route
    int node_idx;
    // global cycle / flow cycle * painting height
    int basic_N = MAP_INNER_Y_ADD / cycle * height;
    // offsets[] defines the offset in each node, the rest of it based on first value 
    # pragma omp parallel for
    for (int n = 0; n < offsets_n; n++) {
        node_idx = routes[n];
        for (int i = 0; i < length; i++) {
            PositionInteger pos_int = pos2int((offsets[n] + i) % cycle, node_idx * height);
            for (int j = 0; j < basic_N; j++) {
                map_slots[pos_int] -= 1;
                pos_int += cycle;
            }
        }
    }
}

// clear a rectangle
inline void Map::clear_area(int length, int height, int cycle, int offsets_n, int *routes, int *offsets) {
    // the node index in the route
    int node_idx;
    // global cycle / flow cycle * painting height
    int basic_N = MAP_INNER_Y_ADD / cycle * height;

    # pragma omp parallel for
    for (int n = 0; n < offsets_n; n++) {
        node_idx = routes[n];
        for (int i = 0; i < length; i++) {
            PositionInteger pos_int = pos2int((offsets[n] + i) % cycle, node_idx * height);
            for (int j = 0; j < basic_N; j++) {
                map_slots[pos_int] += 1;
                pos_int += cycle;
            }
        }
    }
}

/**
 * Render for debug, print the map to terminal screen
 */
void Map::render() {
    for (int y = 0; y < h; y++) {
        // printf("%2d ", y);
        for (int x = 0; x < w; x++) {
            int s = map_slots[pos2int(x, y)];
        }
    }
}

} // namespace magent
} // namespace gridworld
