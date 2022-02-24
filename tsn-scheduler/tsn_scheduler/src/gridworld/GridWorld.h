/**
 * \file GridWorld.h
 * \brief core game engine of the gridworld
 */

#ifndef MAGNET_GRIDWORLD_GRIDWORLD_H
#define MAGNET_GRIDWORLD_GRIDWORLD_H

#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../Environment.h"
#include "grid_def.h"
#include "Map.h"
#include "Range.h"
#include "AgentType.h"
#include "RenderGenerator.h"
#include "RewardEngine.h"

namespace magent {
namespace gridworld {

// the main engine
class GridWorld: public Environment {
public:
    GridWorld();
    ~GridWorld() override;

    // game
    void reset() override;
    void set_config(const char *key, void *p_value) override;
    void show_config() override;

    // run step
    void get_observation(GroupHandle group, float **linear_buffers) override;
    void set_action(GroupHandle group, const float *actions, bool ignore_offsets) override;
    void step(int *done, bool ignore_offsets) override;
    void get_reward(GroupHandle group, float *buffer) override;

    // info getter
    void get_info(GroupHandle group, const char *name, void *buffer) override;

    // render
    void render() override;

    // agent
    void register_agent_type(const char *name, int n, const char **keys, float *values);
    void new_group(const char *agent_name, GroupHandle *group);
    void add_agent(GroupHandle group, int worst_delay, int pkt_length, int n, int *routes, int *offsets);

    // reward description
    void define_agent_symbol(int no, int group, int index);
    void add_reward_rule(int op, int *receivers, float *values, int n_receiver, bool is_terminal);

private:
    // reward description
    void init_reward_description();
    void calc_reward();
    void calc_rule(RewardRule &rule);

    // utility
    // to make channel layout in observation symmetric to every group
    std::vector<int> make_channel_trans(
            GroupHandle group, int base, int n_channel, int n_group);
    int group2channel(GroupHandle group);
    int get_feature_size();

    // game config
    int nodes_num;
    int global_cycle;
    int embedding_size;  // default = 0

    // game states : map, agent and group
    Map map;
    std::map<std::string, AgentType> agent_types;
    std::vector<Group> groups;
    std::default_random_engine random_engine;

    // reward description
    std::vector<AgentSymbol> agent_symbols;
    std::vector<RewardRule>  reward_rules;
    bool reward_des_initialized;

    // action buffer
    // split the events to small regions and boundary for parallel
    int NUM_SEP_BUFFER;
    std::vector<MoveAction> *move_buffers, move_buffer_bound;

    // render
    RenderGenerator render_generator;
    int id_counter;
    bool first_render;

    // statistic recorder
    int *counter_x, *counter_y;
};


class Agent {
public:
    Agent(AgentType &type, int id, GroupHandle group, int worst_delay, int pkt_length, int nodes_num, int n, int *routes_py, int *offsets_py) :   
                                                        group(group),
                                                        worst_delay(worst_delay),
                                                        next_reward(0),
                                                        type(type),
                                                        nodes_num(nodes_num),
                                                        n(n), pkt_length(pkt_length),
                                                        last_op(OP_NULL), op_obj(nullptr), index(0) {
        this->id = id;
        // occupy additional memory, needed to be modified
        last_action = new float[nodes_num]();
        routes = new int[n];
        offsets = new int[n]; 
        for(int i = 0; i < n; i += 1){
            routes[i] = routes_py[i];
            offsets[i] = offsets_py[i];
            last_action[routes[i]] = static_cast<float>(offsets_py[i]) / type.cycle;
        }
        next_reward = 0;
        init_reward();
        height = type.height;
        // initialize the position
        pos = { n, routes, offsets };
    }
    ~Agent(){
        delete [] last_action;
    }

    Position &get_pos()             { return pos; }
    const Position &get_pos() const { return pos; }
    void update_pos(const int *action_int) {
        for(int i = 0; i < n; i++){
            offsets[i] = action_int[routes[i]];
        }
    }

    AgentType &get_type()             { return type; }
    const AgentType &get_type() const { return type; }

    int get_id() const            { return id; }
    int get_length() const { return pkt_length; }
    int get_height() const { return height; }
    int *get_routes() const { return routes; }
    int *get_offsets() const { return offsets; }
    int get_offsets_num() const { return n; }
    int get_worst_delay() const { return worst_delay; }


    void get_embedding(float *buf, int size) {
        // embedding are binary form of id
        if (embedding.empty()) {
            int t = id;
            for (int i = 0; i < size; i++, t >>= 1) {
                embedding.push_back((float)(t & 1));
            }
        }
        memcpy(buf, &embedding[0], sizeof(float) * size);
    }

    void init_reward() {
        last_reward = next_reward;
        last_op = OP_NULL;
        step_reward = 0;
        next_reward = step_reward;
        op_obj = nullptr;
        be_involved = false;
    }

    void update_reward() {
        last_reward = next_reward;
        // last_op = OP_NULL;
        step_reward += type.step_reward;
        next_reward = step_reward;
        // op_obj = nullptr;
        // be_involved = false;
    }

    Reward get_reward()         { return next_reward; }
    Reward get_last_reward()    { return last_reward; }
    void add_reward(Reward add) { next_reward += add; }
    // reward of delay
    bool calc_delay(float value){
        int delay = 0;
        for(int i = 1; i < n; i++){
            if(offsets[i] < offsets[i-1])
                delay += type.cycle + offsets[i] - offsets[i-1];
            else
                delay += offsets[i] - offsets[i-1];
        }
        if(delay > worst_delay){
            add_reward(Reward((worst_delay - delay) * value));
            return false;
        }
        return true;
    };

    // should be modified
    void set_involved(bool value) { be_involved = value; }
    bool get_involved() { return be_involved; }

    void set_action(Action act, bool ignore_offsets) { 
        if(!ignore_offsets){
            for(int i = 0; i < nodes_num; i++){
                last_action[i] = act[i]; 
            }
        }
        else
            last_action[0] = act[0];
    }
    Action get_action()         { return last_action; }

    GroupHandle get_group() const { return group; }
    int get_index() const { return index; }
    void set_index(int i) { index = i; }

    RuleOp get_last_op() const { return last_op; }
    void set_last_op(RuleOp op){ last_op = op; }

    void *get_op_obj() const   { return op_obj; }
    void set_op_obj(void *obj) { op_obj = obj; }


private:
    int id;
    int pkt_length, height;
    int n; // nodes included in the nodes
    int *routes; // indicator of routes
    int *offsets; // offsets in each 
    int worst_delay;
    int nodes_num;

    Position pos;

    RuleOp last_op;
    void *op_obj;

    Action last_action;
    Reward next_reward, last_reward, step_reward;
    AgentType &type;
    GroupHandle group;
    int index;

    bool be_involved;

    std::vector<float> embedding;
};


class Group {
public:
    Group(AgentType &type) : type(type), next_reward(0),
                             center_x(0), center_y(0), recursive_base(0) {
        init_reward();
    }

    void add_agent(Agent *agent) {
        agents.push_back(agent);
    }

    int get_num()       { return (int)agents.size(); }
    size_t get_size()   { return agents.size(); }

    std::vector<Agent*> &get_agents() { return agents; }
    AgentType &get_type()             { return type; }

    void clear() {
        agents.clear();
    }
    // reward
    void init_reward() { next_reward = 0; }
    Reward get_reward()         { return next_reward; }
    void add_reward(Reward add) { next_reward += add; }
    // agent reward
    bool calc_delay(float value){
        bool flag = true;
        for(Agent *i:agents){
            flag &= i->calc_delay(value);
            i->update_reward();
        }
        return flag;
    }

    // use a base to eliminate duplicates in Gridworld::calc_reward
    int get_recursive_base() {
        return recursive_base;
    }
    void set_recursive_base(int base) { recursive_base = base; }

private:
    AgentType &type;
    std::vector<Agent*> agents;

    Reward next_reward; // group reward
    float center_x, center_y;

    int recursive_base;
};

struct MoveAction {
    Agent *agent;
    float*  action;
};

} // namespace magent
} // namespace gridworld

#endif //MAGNET_GRIDWORLD_GRIDWORLD_H