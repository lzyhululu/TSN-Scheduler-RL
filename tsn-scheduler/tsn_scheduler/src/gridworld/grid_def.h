/**
 * \file grid_def.h
 * \brief some global definition for gridworld
 */

#ifndef MAGNET_GRIDWORLD_GRIDDEF_H
#define MAGNET_GRIDWORLD_GRIDDEF_H

#include "../Environment.h"
#include "../utility/utility.h"

namespace magent {
namespace gridworld {

typedef enum {
    OP_COLLIDE,
    OP_E2E_DELAY,
    OP_NULL,
} RuleOp;


struct Position {
    int n;
    int *routes;
    int *offsets;
};
typedef long long PositionInteger;

typedef float Reward;
typedef float*  Action;

// some forward declaration
class Agent;
class AgentType;
class Group;

struct MoveAction;

// reward description
class AgentSymbol;
class RewardRule;

using ::magent::environment::Environment;
using ::magent::environment::GroupHandle;
using ::magent::utility::strequ;
using ::magent::utility::NDPointer;

} // namespace gridworld
} // namespace magent


#endif //MAGNET_GRIDWORLD_GRIDDEF_H
