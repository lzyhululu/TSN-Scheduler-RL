/**
 * \file Map.h
 * \brief The map for the game engine
 */

#ifndef MAGNET_GRIDWORLD_MAP_H
#define MAGNET_GRIDWORLD_MAP_H

#include <vector>
#include <random>
#include "grid_def.h"
#include "../Environment.h"
#include "Range.h"

namespace magent {
namespace gridworld {


class Map {
public:
    Map(): map_slots(nullptr), w(-1), h(-1), g_cycle(-1), s_node(-1), interval(3){
        init_reward();
    }

    ~Map() {
        delete [] map_slots;
    }

    void reset(int global_cycle, int node_num);

    void add_agent(Agent *agent);
    void remove_agent(Agent *agent);

    void init_reward() { next_reward = 0; }
    Reward get_reward()         { return next_reward; }
    void add_reward(Reward add) { next_reward = add; }
    void calc_global(float value);
    int *get_slots(){ return map_slots; }

    Reward do_move(Agent *agent, const int delta[2]);

    void render();

private:
    int *map_slots;  // status of each time_slots
    int w, h;
    int s_node; // sum of nodes in archi
    int g_cycle; // global cycle
    const int interval; // the interval painting time slots within the nodes
    Reward next_reward; // global reward

    /**
     * Utility
     */
    PositionInteger pos2int(int x, int y) const {
        //return (PositionInteger)x * h + y;
        return (PositionInteger)y * w + x;
    }

    inline void clear_area(int length, int height, int cycle, int offsets_n, int *routes, int *offsets);
    inline void fill_area(int length, int height, int cycle, int offsets_n, int *routes, int *offsets);
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_MAP_H
