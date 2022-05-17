from utility.topo_and_streams_generator import construct_topo_and_streams
from utility.archi import ArchiGenerater
from data_handler import init_topo_and_stream_obj_set_for_frame_demo
from frame_demo.constraints_constructor_for_frame_demo import construct_constraints_for_frame_demo
from frame_demo.z3_constraints_solver import construct_solver, push_and_solve_constraints
from transformer_tsn import subsequent_mask
# torch
import torch
# torch中变量封装函数Variable
from torch.autograd import Variable
# 参数,时间等杂项
from parameters import *
import time
import random
import copy


def basic_data_generator(file_name, stream_num=60):
    # 1. 输入拓扑需求和流量需求
    data_gene = ArchiGenerater()
    sw_links = ((0, 1), (1, 2), (1, 3), (1, 5), (3, 4), (5, 4))
    data_gene.gene_all(rand_min=1, rand_max=1, tt_num=30, delay_min=2, delay_max=20,
                       pkt_min=1, pkt_max=5, node_links=sw_links)
    # 2. 根据拓扑和流量需求生成拓扑文件和流量文件
    print('{}: generating topo and stream json file...'.format(file_name))
    construct_topo_and_streams(filename=file_name,
                               generator=data_gene,
                               es_num_per_sw_set=[2],
                               speed_set=[8],
                               st_queues_set=[4],
                               stream_num=stream_num,
                               size_set=[1],
                               period_set=args.tt_flow_cycles,
                               latency_requirement_set=args.tt_flow_cycles,
                               jitter_requirement_set=[0 for _ in args.tt_flow_cycles],
                               macrotick=1,
                               show_topo_graph=False)
    # 3. 初始化链路、流量及流实例集合
    (link_obj_set,
     stream_obj_set,
     stream_instance_obj_set) = init_topo_and_stream_obj_set_for_frame_demo(file_name)
    # 4. 初始化约束公式栈
    print('constructing basic constraints...')
    constraint_formula_set = construct_constraints_for_frame_demo(link_obj_set,
                                                                  stream_obj_set,
                                                                  stream_instance_obj_set,
                                                                  sync_precision=1)

    # 5. 添加约束
    print('adding constraints...')
    solver = construct_solver(constraint_formula_set, timeout=-1)
    return stream_obj_set, solver


class Batch:
    """
    Object for holding a batch of data with mask during training.
    创建一个批处理对象
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def relative_data_gen(stream_obj_set, solver, num_batch, diff_num):
    """
    stream_obj_set, 流量的集合
    num_batch, 总共输入多少轮次的数据
    diff_num, 每轮变化的流量的条数
    """
    # 使用20条流量作为备选流量
    length = len(stream_obj_set)
    basic_choice = set(range(length - 20))
    reserve_choice = list(range(length - 20, length))
    for i in range(num_batch):
        change_choice = copy.deepcopy(basic_choice)
        for j in range(diff_num):
            change_choice.remove(random.choice(list(change_choice)))
        for j in range(diff_num):
            change_choice.add(random.choice(reserve_choice))
        for j in change_choice:
            stream_obj_set[j].unactivate = False
        # print(change_choice)
        result = push_and_solve_constraints(solver, stream_obj_set, change_choice, timeout=1 * 10 * 60 * 1000)
        if not result:
            continue
        yield Batch(stream_obj_set, change_choice)
        for j in change_choice:
            stream_obj_set[j].unactivate = True


def run_epoch(data_iter, constraint_formula_set, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# def run(model, loss, epochs=1, relative_epochs=10, diff_num=1):
#     """
#     生成epochs次主环境，每个环境衍生出relative_epochs个相似环境
#     相似环境中改变diff_num条流量
#     模型设置为最多支持输入160条
#     """
#     #
#     for epoch in range(epochs):
#         file_name = 'main_env_{}'.format(epoch)
#         # stream num 至少大于20,生成相似环境时会使用20条流量作为备选流量进行替换
#         stream_obj_set, constraint_formula_set = basic_data_generator(file_name=file_name)
#
#         for relative_epoch in range(relative_epochs):
#             model.train()
#             run_epoch(relative_data_gen(stream_obj_set, relative_epochs, diff_num), constraint_formula_set, model, loss)
#             model.eval()
#             run_epoch(relative_data_gen(stream_obj_set, 5, diff_num), constraint_formula_set, model, loss)
#
#     model.eval()
#
#     source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))
#     source_mask = Variable(torch.ones(1, 1, 10))
#
#     result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
#     print(result)


def main():
    stream_obj, solver = basic_data_generator(file_name='main_env_1', stream_num=30)
    g = relative_data_gen(stream_obj, solver, 10, diff_num=1)
    next(g)
    run_epoch(g, solver, 0, 0)
    return


if __name__ == '__main__':
    main()
