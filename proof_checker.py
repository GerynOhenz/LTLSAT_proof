# -*- coding:utf-8 -*-
import os
import sys
import subprocess
import queue
import json
from copy import deepcopy
import time
import timeout_decorator
import argparse
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually, LTLfImplies, LTLfRelease
from ltl_model_check_proof import check
import signal
from multiprocessing import Pool
from queue import Queue

TIME_LIMIT=60

def get_proof(data_s):
    vocab = [i for i in 'abcdefghij']
    vocab = set(vocab)

    ret_data=[]
    cnt=0
    for data in data_s:
        cnt+=1
        if cnt%100==0:
            print('\r',cnt,end='')
        v, root_node, proof_dic, pair_set, trace = check(data['ltl'], data['trace'], vocab) #用生成的路径和公式进行证明生成
        que = Queue()
        que.put(root_node)
        visited = set()
        visited.add(root_node)
        while not que.empty():
            cur_node = que.get()
            cur_list = proof_dic.get(cur_node, -1)
            if cur_list != -1:
                for son in cur_list:
                    if son in visited:
                        continue
                    else:
                        que.put(son)
                        visited.add(son)

        # 对比证明访问过的节点visited和生成的节点进行对比
        # node:(t,(s,e),v)
        true_node=0 #真阳
        false_node=0 # 假阳
        unfound_node=0 # 假阴
        # data['proof']=set(data['proof'])
        for i in range(len(data['proof'])):
            node=data['proof'][i]
            data['proof'][i]=(node[0],(node[1][0],node[1][1]),node[2])
            if data['proof'][i] in visited:
                true_node+=1
            else:
                false_node+=1
        data['proof']=set(data['proof'])
        for node in visited:
            if node not in data['proof']:
                unfound_node+=1
        ret_data.append((true_node,false_node,unfound_node,unfound_node==0 and false_node==0))

    return ret_data


if __name__ == "__main__":

    #并行化完成任务，每个进程完成minidx到maxidx部分

    #python3 proof_checker.py --testfile tree-proof-spot-5t20-test.json --netfile res-model149-tree-proof-spot-5t20-test.json -o tree-proof-spot-5t20-test-result.json -t 40 -s 40
    parser = argparse.ArgumentParser(description='Main script for active learning')
    parser.add_argument('--testfile', type=str, required=True , help='test file in json format')
    parser.add_argument('--netfile', type=str, required=True, help='network output file in json format')
    parser.add_argument('-o', type=str, required=True, help='output result file')
    parser.add_argument('-t', type=int, required=False, default=40, help='thread number')
    parser.add_argument('-s', type=int, required=False, default=10000, help='size')

    args = parser.parse_args()
    test_file=args.testfile
    net_file=args.netfile
    ofile=args.o

    f=open(test_file,'r')
    data_test=json.load(f)
    f.close()

    f=open(net_file,'r')
    data_net=json.load(f)
    f.close()

    for i in range(args.s):
        data_net[i]['ltl']=data_test[i]['ltl']

    pool = Pool(processes=args.t)
    job_size=args.s//args.t
    result = []
    for i in range(args.t):
        result.append(pool.apply_async(get_proof, ([data_net[i*job_size:(i+1)*job_size]])))
    pool.close()
    pool.join()

    # 对比结果写到文件里
    ret=[]
    result = [x.get() for x in result]
    for i in result:
        ret+=i

    true_node = 0  # 真阳
    false_node = 0  # 假阳
    unfound_node = 0  # 假阴
    complete_right = 0 # 完全正确的样例数
    for i in ret:
        true_node+=i[0]
        false_node+=i[1]
        unfound_node+=i[2]
        complete_right+=i[3]

    f=open(ofile,'w')
    json.dump({'all_result':ret,'true_node':true_node,'false_node':false_node,'unfound_node':unfound_node,'complete_right':complete_right},f)
    f.close()
    print('\n',{'true_node':true_node,'false_node':false_node,'unfound_node':unfound_node,'complete_right':complete_right})

