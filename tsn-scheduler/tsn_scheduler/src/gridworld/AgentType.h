/**
 * \file AgentType.h
 * \brief implementation of AgentType (mainly initialization)
 */

#ifndef MAGENT_GRIDWORLD_AGENTTYPE_H
#define MAGENT_GRIDWORLD_AGENTTYPE_H

#include <vector>

#include "grid_def.h"
#include "Range.h"

namespace magent {
namespace gridworld {

class AgentType {
public:
    AgentType(int n, std::string name, const char **keys, float *values);

    // user defined setting
    int height;
    int cycle;
    Reward step_reward;

    /***** system calculated setting *****/
    std::string name;
    int n_channel; // obstacle
    Range *move_range;
};


} // namespace magent
} // namespace gridworld

#endif //MAGENT_GRIDWORLD_AGENTTYPE_H
