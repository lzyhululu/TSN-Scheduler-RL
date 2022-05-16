from frame_demo.constraints_constructor_for_frame_demo import construct_constraints_for_frame_demo
from frame_demo.topo_and_streams_txt_parser_for_frame_demo import init_topo_and_stream_obj_set_for_frame_demo
from frame_demo.z3_model_parser_for_frame_demo import write_declare_set_to_txt
from utility.topo_and_streams_generator import construct_topo_and_streams
from utility.archi import ArchiGenerater
from z3_constraints_solver import add_and_solve_constraints


def main():
    # 1. 输入拓扑需求和流量需求
    data_gene = ArchiGenerater()
    sw_links = ((0, 1), (1, 2), (1, 3), (1, 5), (3, 4), (5, 4))
    data_gene.gene_all(rand_min=1, rand_max=1, tt_num=30, delay_min=2, delay_max=20,
                       pkt_min=1, pkt_max=5, node_links=sw_links)
    # 2. 根据拓扑和流量需求生成拓扑文件和流量文件
    print('phase 1: generating topo and stream json file...')
    construct_topo_and_streams(filename='frame_based',
                               generator=data_gene,
                               es_num_per_sw_set=[2],
                               speed_set=[8],
                               st_queues_set=[4],
                               stream_num=30,
                               size_set=[1],
                               period_set=[64, 128, 256, 512],
                               latency_requirement_set=[64, 128, 256, 512],
                               jitter_requirement_set=[0 for _ in [64, 128, 256, 512]],
                               macrotick=1,
                               show_topo_graph=True)
    # # 转换流量文件
    # transform_window_stream_txt_to_frame_stream_txt(window_txt='../log/stream_win_8_stream_50',
    #                                                 frame_txt='../log/stream_macrotick_1_stream_50')

    # 3. 按照window_demo的数据结构初始化链路、流量及流实例集合
    print('phase 2: initializing topo and stream object set...')
    (link_obj_set,
     stream_obj_set,
     stream_instance_obj_set) = init_topo_and_stream_obj_set_for_frame_demo('frame_based')

    # 4. 生成约束
    print('phase 3: constructing constraints...')
    constraint_formula_set = construct_constraints_for_frame_demo(link_obj_set,
                                                                  stream_obj_set,
                                                                  stream_instance_obj_set,
                                                                  sync_precision=1)

    # 5. 添加约束并求解
    print('phase 4: adding and solving constraints...')
    result_set = add_and_solve_constraints(constraint_formula_set, timeout=-1)

    # 6. 解析z3的解，并将解输出到文本文件
    print('phase 5: writing solution...')
    write_declare_set_to_txt(result_set, link_obj_set, './solution')

    return


if __name__ == '__main__':
    main()
