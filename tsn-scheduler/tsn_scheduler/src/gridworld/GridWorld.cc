/**
 * \file GridWorld.cc
 * \brief core game engine of the gridworld
 */

#include <iostream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cassert>

#include "GridWorld.h"

namespace magent {
namespace gridworld {

GridWorld::GridWorld() {
    first_render = true;

    reward_des_initialized = false;
    embedding_size = 0;
    random_engine.seed(0);

    counter_x = counter_y = nullptr;
}

GridWorld::~GridWorld() {
    for (int i = 0; i < groups.size(); i++) {
        std::vector<Agent*> &agents = groups[i].get_agents();

        // free agents
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int j = 0; j < agent_size; j++) {
            delete agents[j];
        }

        // free ranges
        AgentType &type = groups[i].get_type();
        if (type.move_range != nullptr) {
            delete type.move_range;
            type.move_range = nullptr;
        }
    }

    if (counter_x != nullptr)
        delete [] counter_x;
    if (counter_y != nullptr)
        delete [] counter_y;

}

void GridWorld::reset() {
    id_counter = 0;

    NUM_SEP_BUFFER = 1;

    // reset map
    map.reset(global_cycle, nodes_num);

    if (counter_x != nullptr)
        delete [] counter_x;
    if (counter_y != nullptr)
        delete [] counter_y;
    counter_x = new int [global_cycle];
    counter_y = new int [nodes_num];

    render_generator.next_file();

    for (int i = 0;i < groups.size(); i++) {
        std::vector<Agent*> &agents = groups[i].get_agents();

        // free agents
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int j = 0; j < agent_size; j++) {
            delete agents[j];
        }
        groups[i].clear();
        
        // channel should be used only when the different between the agents is emphasized, default 1
        groups[i].get_type().n_channel = 1;
    }
    if (!reward_des_initialized) {
        init_reward_description();
        reward_des_initialized = true;
    }
}

void GridWorld::set_config(const char *key, void *p_value) {
    float fvalue = *(float *)p_value;
    int ivalue   = *(int *)p_value;
    bool bvalue   = *(bool *)p_value;
    const char *strvalue = (const char *)p_value;

    if (strequ(key, "nodes_num"))
        nodes_num = ivalue;
    else if (strequ(key, "global_cycle"))
        global_cycle = ivalue;
    else if (strequ(key, "embedding_size")) // embedding size in the observation.feature
        embedding_size = ivalue;
    else if (strequ(key, "render_dir"))     // the directory of saved videos
        render_generator.set_render("save_dir", strvalue);
    else if (strequ(key, "seed"))           // random seed
        random_engine.seed((unsigned long)ivalue);

    else
        LOG(FATAL) << "invalid argument in GridWorld::set_config : " << key;
}

void GridWorld::show_config(){
    using namespace std;
    cout << "global cycle: " << global_cycle << " nodes count: " << nodes_num << " embedding_size: " << embedding_size << endl;
}

void GridWorld::register_agent_type(const char *name, int n, const char **keys, float *values) {
    std::string str(name);

    if (agent_types.find(str) != agent_types.end())
        LOG(FATAL) << "duplicated name of agent type in GridWorld::register_agent_type : " << str;

    agent_types.insert(std::make_pair(str, AgentType(n, str, keys, values)));
}

void GridWorld::new_group(const char *agent_name, GroupHandle *group) {
    *group = (GroupHandle)groups.size();

    auto iter = agent_types.find(std::string(agent_name));
    if (iter == agent_types.end()) {
        LOG(FATAL) << "invalid name of agent type in new_group : " << agent_name;
    }

    groups.push_back(Group(iter->second));
}

void GridWorld::add_agent(GroupHandle group, int worst_delay, int pkt_length, int n, int *routes, int *offsets) {
    // group >= 0 for agents
    if (group > groups.size()) {
        LOG(FATAL) << "invalid group handle in GridWorld::add_agents : " << group;
    }
    Group &g = groups[group];
    int cycle = g.get_type().cycle, height = g.get_type().height;
    AgentType &agent_type = g.get_type();
    Agent *agent = new Agent(agent_type, id_counter, group, worst_delay, pkt_length, nodes_num, n, routes, offsets);
    map.add_agent(agent);
    g.add_agent(agent);
    id_counter++;
}

void GridWorld::get_observation(GroupHandle group, float **linear_buffers) {
    Group &g = groups[group];
    AgentType &type = g.get_type();

    const int n_channel   = g.get_type().n_channel;
    const int n_group = (int)groups.size();
    const int n_action = (int)nodes_num;
    const int feature_size = get_feature_size();

    std::vector<Agent*> &agents = g.get_agents();
    size_t agent_size = agents.size();

    // transform buffers
    // view buffer should load the entire map
    NDPointer<float, 2> view_buffer(linear_buffers[0], {-1, nodes_num * global_cycle * n_channel});
    NDPointer<float, 2> feature_buffer(linear_buffers[1], {-1, feature_size});

    memset(view_buffer.data, 0, sizeof(float)* nodes_num * global_cycle * n_channel);
    memset(feature_buffer.data, 0, sizeof(float) * agent_size * feature_size);
    
    // fill local view for every agents
    std::vector<float> offset_feature(6);
    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];
        // get non-spatial feature
        agent->get_embedding(feature_buffer.data + i*feature_size, embedding_size);
        // record last action
        Action last_action = agent->get_action();
        for (int j = 0; j < nodes_num; j++) {
            offset_feature[j] = static_cast<float>(last_action[j]);
        }
        // return probability of the offsets in each nodes, should be combined with routes
        // id + action + reward: 10 + nodes_num + 1
        memcpy(feature_buffer.data + i*feature_size + embedding_size, &offset_feature[0], sizeof(float) * nodes_num);
        // last reward
        feature_buffer.at(i, embedding_size + nodes_num) = agent->get_last_reward();
    }
    int *slots = map.get_slots();
    for (size_t i = 0; i < nodes_num * global_cycle; i++)
    {
        view_buffer.at(0, i) = static_cast<float>(slots[i]);
    }
}

void GridWorld::set_action(GroupHandle group, const float *actions) {
    std::vector<Agent*> &agents = groups[group].get_agents();
    const AgentType &type = groups[group].get_type();
    // action space layout : offsets in each node
    size_t agent_size = agents.size();

    for (int i = 0; i < agent_size; i++) {
        Agent *agent = agents[i];
        Action act = (Action) (actions +6*i);
        agent->set_action(act);
        // move
        move_buffer_bound.push_back(MoveAction{agent, act});
    }
}

void GridWorld::step(int *done) {
    const bool stat = false;

    LOG(TRACE) << "gridworld step begin.  ";
    size_t group_size  = groups.size();

    // do move
    auto do_move_for_a_buffer = [] (std::vector<MoveAction> &move_buf, Map &map, int nodes_num) {
        //std::random_shuffle(move_buf.begin(), move_buf.end());
        size_t move_size = move_buf.size();
        for (int j = 0; j < move_size; j++) {
            Action act = move_buf[j].action;
            Agent *agent = move_buf[j].agent;
            int dx[nodes_num];
            // act represent an array of several probability, dx change it to the actual position
            agent->get_type().move_range->num2delta(act, dx, nodes_num);
            map.do_move(agent, dx);
        }
        move_buf.clear();
    };

    LOG(TRACE) << "move boundary.  ";
    do_move_for_a_buffer(move_buffer_bound, map, nodes_num);

    LOG(TRACE) << "calc_reward.  ";
    calc_reward();

    LOG(TRACE) << "game over check.  ";
    int flag = 0;  // default game over condition: 

    // Scheduling terminates when all the rules are met
    size_t rule_size = reward_rules.size();
    for (int i = 0; i < rule_size; i++) {
        if (reward_rules[i].trigger && reward_rules[i].is_terminal)
            flag++;
    }
    if(flag == rule_size)
        *done = (int)true;
}

void GridWorld::calc_reward() {
    size_t rule_size = reward_rules.size();
    for (int i = 0; i < groups.size(); i++)
        groups[i].set_recursive_base(0);

    for (int i = 0; i < rule_size; i++) {
        reward_rules[i].trigger = false;
        calc_rule(reward_rules[i]);
    }
}

void GridWorld::get_reward(GroupHandle group, float *buffer) {
    // temporarily ignoring this
    std::vector<Agent*> &agents = groups[group].get_agents();

    size_t agent_size = agents.size();
    Reward group_reward = groups[group].get_reward();
    Reward global_reward = map.get_reward();

    #pragma omp parallel for
    for (int i = 0; i < agent_size; i++) {
        buffer[i] = agents[i]->get_last_reward() + group_reward + global_reward;
    }
}

/**
 * info getter
 */
void GridWorld::get_info(GroupHandle group, const char *name, void *void_buffer) {
    // for more information from the engine, add items here
    std::vector<Agent*> &agents = groups[group].get_agents();
    int   *int_buffer   = (int *)void_buffer;
    float *float_buffer = (float *)void_buffer;
    bool  *bool_buffer  = (bool *)void_buffer;

    if (strequ(name, "num")) {         // int
        int_buffer[0] = groups[group].get_num();
    } else if (strequ(name, "id")) {   // int
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int_buffer[i] = agents[i]->get_id();
        }
    } else if (strequ(name, "pos")) {   // int
        size_t agent_size = agents.size();
        #pragma omp parallel for
        for (int i = 0; i < agent_size; i++) {
            int n = agents[i]->get_pos().n;
            int_buffer[i * (2*n+1)] = agents[i]->get_pos().n;
            for(int j = 0; j < n; j++){
                int_buffer[i * (2*n+1) + 2 * j] = agents[i]->get_pos().routes[j];
                int_buffer[i * (2*n+1) + 2 * j + 1] = agents[i]->get_pos().offsets[j];
            }
        }
    } else if (strequ(name, "action_space")) {  // int
        int_buffer[0] = (int)nodes_num;
    } else if (strequ(name, "feature_space")) {
        int_buffer[0] = get_feature_size();
    } else if (strequ(name, "view_space")) {    // int
        int_buffer[0] = global_cycle * nodes_num;
    } else if (strequ(name, "groups_info")) {
        const int colors[][3] = {
                {192, 64, 64},
                {64, 64, 192},
                {64, 192, 64},
                {64, 64, 64},
        };
        NDPointer<int, 2> info(int_buffer, {-1, 3});

        for (int i = 0; i < groups.size(); i++) {
            info.at(i, 0) = colors[i][0];
            info.at(i, 1) = colors[i][1];
            info.at(i, 2) = colors[i][2];
        }
    } else {
        LOG(FATAL) << "unsupported info name in GridWorld::get_info : " << name;
    }
}

// private utility
int GridWorld::get_feature_size() {
    // feature space layout : [embedding, last_action (one hot), last_reward]
    int feature_space = embedding_size + nodes_num + 1;
    return feature_space;
}

/**
 * render
 */
void GridWorld::render() {
    if (render_generator.get_save_dir() == "___debug___")
        map.render();
    else {
        if (first_render) {
            first_render = false;
            render_generator.gen_config(groups, global_cycle, nodes_num);
        }
        render_generator.render_a_frame(groups, map);
    }
}

} // namespace magent
} // namespace gridworld
