/**
 * \file reward_description.cc
 * \brief implementation of reward description
 */

#include "assert.h"

#include "RewardEngine.h"
#include "GridWorld.h"

namespace magent {
namespace gridworld {

bool AgentSymbol::bind_with_check(void *entity) {
    // bind agent symbol to entity with correctness check
    Agent *agent = (Agent *)entity;
    if (group != agent->get_group())
        return false;
    if (index != -1 && index != agent->get_index())
        return false;
    this->entity = agent;
    return true;
}

/**
 * some interface for python bind
 */
void GridWorld::define_agent_symbol(int no, int group, int index) {
//    LOG(TRACE) << "define agent symbol %d (group=%d index=%d)\n", no, group, index);
    if (no >= agent_symbols.size()) {
        agent_symbols.resize((unsigned)no + 1);
    }
    agent_symbols[no].group = group;
    agent_symbols[no].index = index;
}

void GridWorld::add_reward_rule(int op, int *receivers, float *values, int n_receiver, bool is_terminal) {

    RewardRule rule;
    rule.op = (RuleOp)op;
    for (int i = 0; i < n_receiver; i++) {
        rule.raw_parameter.push_back(receivers[i]);
        rule.values.push_back(values[i]);
    }
    rule.is_terminal = is_terminal;
    reward_rules.push_back(rule);
}

void GridWorld::init_reward_description() {
    // from serial data to pointer
    // assign the constraints(reward rules) to specify flows
    for (int i = 0; i < reward_rules.size(); i++) {
        RewardRule &rule = reward_rules[i];
        for (int j = 0; j < rule.raw_parameter.size(); j++) {
            rule.receivers.push_back(&agent_symbols[rule.raw_parameter[j]]);
        }
        switch (rule.op) {
            case OP_COLLIDE:
                break;
            case OP_E2E_DELAY:
                break;
            default:
                LOG(FATAL) << "invalid rule op in GridWorld::init_reward_description";
        }
    }

    // for every reward rule, add new flows for specify rule
    for (int i = 0; i < reward_rules.size(); i++) {
        // TODO: case select, dynamic scheduling
    }

    // print rules for debug
    /*for (int i = 0; i < reward_rules.size(); i++) {
        printf("op: %d\n", (int)reward_rules[i].op);
        printf("input symbols: ");
        for (int j = 0; j < reward_rules[i].input_symbols.size(); j++) {
            printf("(%d,%d) ", reward_rules[i].input_symbols[j]->group,
                               reward_rules[i].input_symbols[j]->index);
            if (reward_rules[i].infer_obj[j] != nullptr) {
                printf("-> (%d,%d)  ", reward_rules[i].infer_obj[j]->group,
                                     reward_rules[i].infer_obj[j]->index);
            }
        }
        printf("\n");
    }*/
}

void GridWorld::calc_rule(RewardRule &rule) {
    rule.trigger = true;
    switch (rule.op) {
        case OP_COLLIDE:
            map.calc_global(rule.values[0]);
            break;
        case OP_E2E_DELAY:
            // for every agent in groups
            for(Group i :groups){
                i.calc_delay(rule.values[0]);
            }
            break;
        default:
            LOG(FATAL) << "invalid op of Rulw in GridWorld::calc_rule";
    }
}



} // namespace gridworld
} // namespace magent