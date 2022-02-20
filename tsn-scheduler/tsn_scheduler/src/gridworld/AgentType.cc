/**
 * \file AgentType.cc
 * \brief implementation of AgentType (mainly initialization)
 */

#include "AgentType.h"

namespace magent {
namespace gridworld {


#define AGENT_TYPE_SET_INT(name) \
    if (strequ(keys[i], #name)) {\
        name = (int)(values[i] + 0.5);\
        is_set = true;\
    }

#define AGENT_TYPE_SET_FLOAT(name) \
    if (strequ(keys[i], #name)) {\
        name = values[i];\
        is_set = true;\
    }

#define AGENT_TYPE_SET_BOOL(name)\
    if (strequ(keys[i], #name)) {\
        name = bool(int(values[i] + 0.5));\
        is_set = true;\
    }

AgentType::AgentType(int n, std::string name, const char **keys, float *values) {
    this->name = name;

    // default value
    height = 1;
    cycle = 0;
    step_reward = 0.0;

    // init member vars from str (reflection)
    bool is_set;
    for (int i = 0; i < n; i++) {
        is_set = false;
        AGENT_TYPE_SET_INT(height);
        AGENT_TYPE_SET_INT(cycle);
        AGENT_TYPE_SET_FLOAT(step_reward);
        if (!is_set) {
            LOG(FATAL) << "invalid agent config in AgentType::AgentType : " << keys[i];
        }
    }

    // NOTE:
    
    move_range   = new Range(cycle);
}

} // namespace magent
} // namespace gridworld
