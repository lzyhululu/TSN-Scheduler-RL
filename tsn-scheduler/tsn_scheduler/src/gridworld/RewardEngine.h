/**
 * \file reward_description.h
 * \brief Data structure for reward description
 */

#ifndef MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H
#define MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H

#include <vector>
#include <set>
#include <map>
#include "grid_def.h"

namespace magent {
namespace gridworld {

class AgentSymbol {
public:
    int group;
    int index;           // -1 for all
    void *entity;

    bool is_all() {
        return index == -1;
    }

    bool bind_with_check(void *entity);
};

class RewardRule {
public:
    RuleOp op;
    // remain for further research
    std::vector<AgentSymbol*> input_symbols;
    std::vector<AgentSymbol*> infer_obj;

    std::vector<AgentSymbol*> receivers;
    std::vector<float> values;
    bool is_terminal;

    std::vector<int> raw_parameter; // serialized parameter from python end

    bool trigger;
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_REWARD_DESCRIPTION_H
