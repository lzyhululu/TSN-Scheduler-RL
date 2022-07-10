from utility.topo_and_streams_generator import construct_topo_and_streams
from utility.archi import ArchiGenerater
from data_handler import init_topo_and_stream_obj_set_for_frame_demo
from frame_demo.constraints_constructor_for_frame_demo import construct_constraints_for_frame_demo
from frame_demo.z3_constraints_solver import construct_solver, push_and_solve_constraints
from transformer_tsn import *
# torch
import torch
# torch中变量封装函数Variable
from torch.autograd import Variable
# 参数,时间等杂项
from parameters import *
import time
import random
import copy
import pickle


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
                               st_queues_set=[1],
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
    # print('constructing basic constraints...')
    constraint_formula_set = construct_constraints_for_frame_demo(link_obj_set,
                                                                  stream_obj_set,
                                                                  stream_instance_obj_set,
                                                                  sync_precision=1)

    # 5. 添加约束
    # print('adding constraints...')
    solver = construct_solver(constraint_formula_set, timeout=-1)
    return stream_obj_set, solver


class Batch:
    """
    Object for holding a batch of data with mask during training.
    创建一个批处理对象
    """
    def __init__(self, src, trg=None, ac_stream_ids=None, pad=0):
        # self.src_mask = (src != pad)
        self.src = src
        self.ac_stream_ids = ac_stream_ids
        # self.src = src.masked_fill(self.src_mask == 0, -1e9)
        self.ntokens = self.src.shape[0] * self.src.shape[1]
        if trg is not None:
            self.trg = trg

    @staticmethod
    def make_std_mask(tgt, pad=0):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def relative_data_gen(stream_obj_set, solver, batch, num_batch, diff_num):
    """
    stream_obj_set, 流量的集合
    num_batch, 总共输入多少轮次的数据
    diff_num, 每轮变化的流量的条数
    """
    # 初始化输入流量特征的列表
    stream_features = []
    for stream_obj in stream_obj_set:
        temp = list()
        # 特征主要为1 大小 1 周期 1 时延要求 7 路由
        temp.append(stream_obj.size)
        temp.append(stream_obj.period)
        temp.append(stream_obj.latency_requirement)
        assert temp[-1] != 0
        route_set = stream_obj.route_set
        for i in range(7):
            if i >= len(route_set):
                temp.append(0)
            else:
                temp.append(stream_obj.route_set[i] + 1)
        stream_features.append(temp)
    # print(stream_features)
    # 使用20条流量作为备选流量 100 20
    length = len(stream_obj_set)
    act_len = length - 20
    basic_choice = set(range(act_len))
    reserve_choice = list(range(act_len, length))
    for _ in range(num_batch):
        batch_data = []
        batch_result = []
        change_choices = []
        k = 0
        while k < batch:
            k += 1
            change_choice = copy.deepcopy(basic_choice)
            for j in range(diff_num):
                change_choice.remove(random.choice(list(change_choice)))
            while len(change_choice) < act_len:
                change_choice.add(random.choice(reserve_choice))
            for j in change_choice:
                stream_obj_set[j].unactivate = False
            # print(change_choice)
            start = time.time_ns()
            # schedulability verification with extra constraints
            result = push_and_solve_constraints(solver, stream_obj_set, change_choice, timeout=1 * 10 * 60 * 1000)
            end = time.time_ns()
            if not result:
                k -= 1
                continue
            data = [stream_features[i] for i in change_choice]
            # data.insert(0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            batch_data.append(data)
            # batch_result.append(result)
            for j in change_choice:
                stream_obj_set[j].unactivate = True
            change_choices.append(change_choice)
        batch_data = torch.FloatTensor(batch_data)
        batch_result = torch.FloatTensor(batch_result)
        source = Variable(batch_data, requires_grad=False)
        target = Variable(batch_result, requires_grad=False)
        yield Batch(source, target, change_choices)


def run_epoch(data_iter, model, loss_compute, stream_obj_set, solver):
    """Standard Training and Logging Function"""
    # start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # out = model.forward(batch.src, batch.trg,
        #                     None, batch.trg_mask)
        out = model.forward(batch.src, None)
        loss = loss_compute(out, batch.ac_stream_ids, batch.ntokens, stream_obj_set, solver)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            # elapsed = time.time() - start
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            # start = time.time()
            tokens = 0
    return total_loss / total_tokens


def run(model, loss, epochs=1000, relative_epochs=2, diff_num=3):
    """
    生成epochs次主环境，每个环境衍生出relative_epochs个相似环境
    相似环境中改变diff_num条流量
    模型设置为最多支持输入160条
    """
    for epoch in range(epochs):
        file_name = 'main_env_{}'.format(epoch)
        # stream num 至少大于20,生成相似环境时会使用20条流量作为备选流量进行替换
        stream_obj_set, basic_solver = basic_data_generator(file_name=file_name, stream_num=30)

        model.train()
        run_epoch(relative_data_gen(stream_obj_set, basic_solver, relative_epochs, 10, diff_num), model,
                  loss, stream_obj_set, basic_solver)
        model.eval()
        run_epoch(relative_data_gen(stream_obj_set, basic_solver, 1, 2, diff_num), model,
                  loss, stream_obj_set, basic_solver)

    model.eval()

    source = [[[1, 256, 256, 16, 13, 0, 0, 0, 0, 0],
              [1, 256, 256, 34, 35, 0, 0, 0, 0, 0],
              [1, 128, 128, 32, 29, 0, 0, 0, 0, 0],
              [1, 256, 256, 34, 8, 19, 0, 0, 0, 0],
              [1, 256, 256, 34, 8, 5, 25, 0, 0, 0],
              [1, 256, 256, 22, 4, 2, 15, 0, 0, 0],
              [1, 512, 512, 20, 7, 35, 0, 0, 0, 0],
              [1, 128, 128, 20, 5, 9, 29, 0, 0, 0],
              [1, 256, 256, 30, 10, 6, 3, 23, 0, 0],
              [1, 128, 128, 34, 8, 5, 25, 0, 0, 0],
              [1, 512, 512, 32, 10, 6, 2, 13, 0, 0],
              [1, 128, 128, 22, 4, 2, 13, 0, 0, 0],
              [1, 64, 64, 28, 6, 3, 21, 0, 0, 0],
              [1, 128, 128, 20, 17, 0, 0, 0, 0, 0],
              [1, 64, 64, 30, 10, 6, 2, 13, 0, 0],
              [1, 256, 256, 22, 4, 2, 15, 0, 0, 0],
              [1, 256, 256, 22, 4, 2, 15, 0, 0, 0],
              [1, 256, 256, 22, 4, 2, 15, 0, 0, 0],
              [1, 256, 256, 22, 4, 2, 15, 0, 0, 0],
              [1, 256, 256, 16, 13, 0, 0, 0, 0, 0]]]
    source = torch.FloatTensor(source)
    source = Variable(source, requires_grad=False)
    source_mask = None

    result = greedy_decode(model, source, source_mask, max_len=20, start_symbol=1)
    print(result)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, 10).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1]).contiguous().view(-1, 10, 513)
        _, next_word = torch.max(prob, dim=2)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
    return ys


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    try:
        f = open(filename, 'ab+')
        r = pickle.load(f)
        f.close()
        return r

    except EOFError:
        return ""


def main():
    model = ScheT(
        feature_nums=10,
        target_size=1,
        slot_nums=512,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=2048,
        dropout=0.0
    )
    model_optimizer = get_std_opt(model)
    loss = FrameLossCompute(model.generator, model_optimizer)
    run(model, loss)
    return


if __name__ == '__main__':
    main()
