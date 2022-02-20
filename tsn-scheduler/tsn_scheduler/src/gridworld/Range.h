/**
 * \file Range.h
 * \brief View range in GridWorld
 */

#ifndef MAGNET_GRIDWORLD_RANGE_H
#define MAGNET_GRIDWORLD_RANGE_H

#include <cstdio>
#include <tgmath.h>
#include <cstring>

namespace magent {
namespace gridworld {

class Range {
public:
    Range(int view_range) : width(view_range) {

    }

    Range(const Range &other) :  width(other.width) {

    }

    ~Range() {

    }

    int get_width()  const { return width; }

    void num2delta(Action act, int *dx, int nodes_num) const {
        // do not check boundary
        for(int i = 0; i < nodes_num; i++){
            // 0 ~ cycle-1
            dx[i] =int(width*act[i]) % 64;
        }
    }

    void print_self() {

    }

protected:
    int width;
};

} // namespace gridworld
} // namespace magent

#endif //MAGNET_GRIDWORLD_RANGE_H
