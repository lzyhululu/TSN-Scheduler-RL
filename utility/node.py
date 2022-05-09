from parameters import *


class Node:
    """
    mainly used to record the occupation of the time slots
    buffer is used when a flow was arrived but hasn't been sent out
    """
    def __init__(self, index, capacity):
        # Basic node information
        self.id = index
        self.buffer_capacity = capacity
        self.global_cycle = args.global_cycle * args.slot_per_millisecond

        # initialize all the time slots and buffers
        self.buffer_avaiable = [self.buffer_capacity for _ in range(args.global_cycle * args.slot_per_millisecond)]
        self.time_slot = [1 for _ in range(args.global_cycle * args.slot_per_millisecond)]

    def check_buffer(self, start, cycle):
        for pos in range(args.global_cycle):
            if pos % cycle == start:
                if self.buffer_avaiable[pos] - 1 < 0:
                    return False
        return True

    def occupy_buffer(self, start, cycle):
        for pos in range(args.global_cycle):
            if pos % cycle == start:
                self.buffer_avaiable[pos] -= 1

    def check_buffers(self, start, end, cycle):
        for pos in range(args.global_cycle):
            offset = pos % cycle
            if (start < end and start < offset <= end) or \
                    (start > end and (offset <= end or start < offset)):
                if self.buffer_avaiable[pos] - 1 < 0:
                    return False
        return True

    def occupy_buffers(self, start, end, cycle):
        for pos in range(args.global_cycle):
            offset = pos % cycle
            if (start < end and start < offset <= end) or \
                    (start > end and (offset <= end or start < offset)):
                self.buffer_avaiable[pos] -= 1

    def occupy_single_time_slot(self, time_slot: int, cycle: int):
        assert time_slot < cycle, "time_slot parameter shouldn't be larger than cycle"
        pos = time_slot
        while pos < self.global_cycle:
            self.time_slot[pos] -= 1
            pos += cycle

    def occupy_time_slot(self, time_slot, cycle, length):
        """
        time_slot: should be the location of the time slot
        """
        for i in range(length):
            self.occupy_single_time_slot(time_slot+i, cycle)

    def reset(self):
        # info initializion
        self.buffer_avaiable = [self.buffer_capacity for _ in range(args.global_cycle)]
        self.time_slot = [1 for _ in range(args.global_cycle)]

    def show(self):
        for i in range(self.global_cycle):
            print(self.time_slot[i], end=' ')


def main():
    import pandas as pd
    node = Node(0, 1)
    node.occupy_time_slot(3, 256, 2)
    ds = pd.Series(node.time_slot)
    print(ds[ds == 0])
    node.show()


if __name__ == '__main__':
    main()
