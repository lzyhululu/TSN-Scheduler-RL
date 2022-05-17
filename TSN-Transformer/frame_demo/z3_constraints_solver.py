import time
import re
from z3 import *


def _parse_z3_model(model):
    solution = []
    for declare in model.decls():
        name = declare.name()
        value = model[declare]
        solution.append({'name': name, 'value': value})
    return solution


# 传入的timeout的单位是ms
def construct_solver(constraint_set, timeout=-1):
    """返回一个求解器对象"""
    s = Solver()
    if timeout > 0:
        s.set(timeout=timeout)

    for constraint in constraint_set:
        s.add(constraint)
    return s


# 传入的timeout的单位是ms
def push_and_solve_constraints(solver, stream_obj_set, ac_stream_ids, timeout=-1):
    # start = 0
    end = 0
    s = solver
    s.push()
    if timeout > 0:
        s.set(timeout=timeout)

    for stream_obj in stream_obj_set:
        if not stream_obj.unactivate:
            un_active = Bool(f'A_{stream_obj.stream_id}')
            s.add(Not(un_active))
        else:
            un_active = Bool(f'A_{stream_obj.stream_id}')
            s.add(un_active)

    declare_set = []
    unknown_reason = ''
    # 开始计时
    start = time.time_ns()
    # 判断是否有可行解
    sat_or_not = s.check()
    if sat_or_not == sat:
        model = s.model()
        end = time.time_ns()
        print("end time: %f" % end)
        # 输出变量声明的集合
        declare_set = _parse_z3_model(model)
    elif sat_or_not == unsat:
        # 输出时间
        end = time.time_ns()
        # 输出一个空的declare_set
    elif sat_or_not == unknown:
        end = time.time_ns()
        # 输出一个空的declare_set
        # 输出unknown的原因
        unknown_reason = s.reason_unknown()
        pass
    s.pop()
    time_used_in_second = (end - start) / 1000000000
    print('time_used:')
    print(time_used_in_second)
    if str(sat_or_not) == 'sat':
        # frame_demo的结果变量分为两类
        # 1. offset，命名：O_stream_id^(link_id)
        # 2. prio，命名：P_stream_id^(link_id)
        for declare in declare_set:
            name = declare['name']
            if name == 'p':
                continue
            stream_id = int(name.split('_')[1].split('^')[0])
            if re.match(r'A', name) or stream_id not in ac_stream_ids:
                continue
            value = declare['value']
            value = str(value)
            if re.match(r'O', name):
                # 解析link_id
                link_id = int(name.split('(')[1].split(')')[0])
                stream_obj_set[stream_id].offsets[link_id] = int(value)
            elif re.match(r'P', name):
                # 解析link_id
                link_id = int(name.split('(')[1].split(')')[0])
                stream_obj_set[stream_id].priority[link_id] = int(value)

    # 返回是否成功
    return str(sat_or_not) == 'sat'


def _main():
    return


if __name__ == '__main__':
    _main()
