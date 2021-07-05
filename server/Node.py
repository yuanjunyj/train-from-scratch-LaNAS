# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from Classifier import Classifier
import json
import numpy as np
import math
from collections import namedtuple
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

class Node:
    obj_counter   = 0
    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.
    
    def __init__(self, parent = None,  is_good_kid = False, arch_code_len = 0, is_root = False):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        if not is_root:
            assert type( parent ) == type( self )
        self.is_root       = is_root
        self.ARCH_CODE_LEN = arch_code_len
        self.x_bar         = float("inf")
        self.n             = 0
        self.classifier    = Classifier({}, self.ARCH_CODE_LEN)
        self.parent        = parent
        self.is_good_kid   = is_good_kid
        self.uct           = 0
        
        #insert curt into the kids of parent
        if parent is not None:
            self.parent.kids.append(self)
            if self.parent.is_leaf  == True:
                self.parent.is_leaf = False
            assert len(self.parent.kids) <= 2
        self.kids          = []
        self.bag           = { }
        self.good_kid_data = {}
        self.bad_kid_data  = {}

        self.is_leaf       = True
        self.id            = Node.obj_counter
        
        #data for good and bad kids, respectively
        Node.obj_counter += 1
    
    def visit(self):
        self.n += 1
    
    def collect_sample(self, arch, acc):
        self.bag[json.dumps(arch) ] = acc
        self.n                      = len( self.bag )
    
    def print_bag(self):
        print("BAG"+"#"*10)
        for k, v in self.bag.items():
            print("arch:", k, "acc:", v)
        print("BAG"+"#"*10)
        print('\n')

    
    def put_in_bag(self, net, acc):
        assert type(net) == type([])
        assert type(acc) == type(float(0.1))
        net_k = json.dumps(net)
        self.bag[net_k] = acc
    
    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)
    
    def pad_str_to_8chars(self, ins):
        if len(ins) <= 14:
            ins += ' '*(14 - len(ins) )
            return ins
        else:
            return ins
    
    def __str__(self):
        name   = self.get_name()
        name   = self.pad_str_to_8chars(name)
        name  += ( self.pad_str_to_8chars( 'lf:' + str(self.is_leaf)) )
        
        val    = 0
        name  += ( self.pad_str_to_8chars( ' val:{0:.4f}   '.format(round(self.get_xbar(), 4) ) ) )
        name  += ( self.pad_str_to_8chars( ' uct:{0:.4f}   '.format(round(self.get_uct(), 4) ) ) )

        name  += self.pad_str_to_8chars( 'n:'+str(self.n) )
        name  += self.pad_str_to_8chars( 'sp:'+ str(len(self.bag)) )
        name  += ( self.pad_str_to_8chars( 'g_k:' + str( len(self.good_kid_data) ) ) )
        name  += ( self.pad_str_to_8chars( 'b_k:' + str( len(self.bad_kid_data ) ) ) )

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent)
        
        name += (' parent:' + parent)
        
        kids = ''
        kid  = ''
        for k in self.kids:
            kid   = self.pad_str_to_8chars( k.get_name() )
            kids += kid
        name  += (' kids:' + kids)
        
        return name
    

    def get_uct(self):
        if self.is_root and self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        Cp = 0.5
        return self.x_bar + 2*Cp*math.sqrt( 2* math.log(self.parent.n) / self.n )
    
    def get_xbar(self):
        return self.x_bar

    def get_n(self):
        return self.n
    
    def get_parent_str(self):
        return self.parent.get_name()

    def train(self):
        if self.parent == None and self.is_root == True:
        # training starts from the bag
            assert len(self.bag) > 0
            self.classifier.update_samples(self.bag )
            self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.classifier.update_samples(self.parent.good_kid_data )
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
            else:
                self.bag = self.parent.bad_kid_data
                self.classifier.update_samples(self.parent.bad_kid_data )
                self.good_kid_data, self.bad_kid_data = self.classifier.split_data()
        if len(self.bag) == 0:
           self.x_bar = float('inf')
           self.n     = 0
        else:
           self.x_bar = np.mean( np.array(list(self.bag.values())) )
           self.n     = len( self.bag.values() )

    def predict(self):
        if self.parent == None and self.is_root == True and self.is_leaf == False:
            self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.bag)
        elif self.is_leaf:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
            else:
                self.bag = self.parent.bad_kid_data
        else:
            if self.is_good_kid:
                self.bag = self.parent.good_kid_data
                self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.parent.good_kid_data)
            else:
                self.bag = self.parent.bad_kid_data
                self.good_kid_data, self.bad_kid_data = self.classifier.split_predictions(self.parent.bad_kid_data)

    def sample_arch(self):
        while True:
            if len(self.bag) == 0:
                return None
            net_str = np.random.choice( list(self.bag.keys() ) )
            del self.bag[net_str]
            if principles_validate(net_str):
                return json.loads(net_str)
    
    def clear_data(self):
        self.bag.clear()
        self.bad_kid_data.clear()
        self.good_kid_data.clear()


def principles_validate(net_str):
    net = json.loads(net_str)
    node_count = int((len(net) / 4))
    code = gen_code_from_list(net, node_num=node_count)
    genotype = translator([code, code], max_node=node_count)
    skip0 = [1, 0] + [0] * (node_count)
    skip1 = [0, 1] + [0] * (node_count)
    conv0 = [1, 0] + [0] * (node_count)
    conv1 = [0, 1] + [0] * (node_count)
    poolings = 0
    index = 0
    for i in range(2, node_count + 2):
        for j in range(2):
            op, src = genotype.normal[index]
            if 'skip' in op:
                skip0[i] |= skip0[src]
                skip1[i] |= skip1[src]
                conv0[i] = max(conv0[i], conv0[src])
                conv1[i] = max(conv1[i], conv1[src])
            if 'pool' in op:
                poolings += 1
                conv0[i] = max(conv0[i], conv0[src])
                conv1[i] = max(conv1[i], conv1[src])
            if 'conv_3x3' in op:
                conv0[i] = max(conv0[i], conv0[src] + 1)
                conv1[i] = max(conv1[i], conv1[src] + 1)
            if 'conv_5x5' in op:
                conv0[i] = max(conv0[i], conv0[src] + 2)
                conv1[i] = max(conv1[i], conv1[src] + 2)
            index += 1
    s0, s1 = 0, 0
    c0, c1 = 0, 0
    for i in genotype.normal_concat:
        s0 |= skip0[i]
        s1 |= skip1[i]
        c0 = max(c0, conv0[i])
        c1 = max(c1, conv1[i])
    return s0 == 1 and s1 == 1 and poolings <= 2 and c0 > 2 and c1 > 2

operations = ['sep_conv_3x3',
    'max_pool_3x3',
    'skip_connect',
    'sep_conv_5x5']

def gen_code_from_list(sample, node_num=6):
    node = [[-1 for col in range(4)] for row in range(node_num)]
    for i in range(node_num):
        for j in range(4):
            if j <= 1:
                node[i][j] = sample[i * 2 + j]
            else:
                node[i][j] = sample[i * 2 + j + (node_num - 1) * 2]
    return node

def translator(code, max_node=6):
    # input: code type
    # output: geno type
    n = 0
    normal = []
    normal_concat = []
    reduce_concat = []
    for i in range(max_node+2):
        normal_concat.append(i)
        reduce_concat.append(i)
    reduce = []

    for cell in range(len(code)):
        if cell == 0: # for normal cell
            for block in range(len(code[cell])):
                normal.append((operations[code[cell][block][0]], code[cell][block][2]))
                normal.append((operations[code[cell][block][1]], code[cell][block][3]))
                if code[cell][block][2] in normal_concat:
                    normal_concat.remove(code[cell][block][2])
                if code[cell][block][3] in normal_concat:
                    normal_concat.remove(code[cell][block][3])

        else: # for reduction cell
            for block in range(len(code[cell])):
                reduce.append((operations[code[cell][block][0]], code[cell][block][2]))
                reduce.append((operations[code[cell][block][1]], code[cell][block][3]))
                if code[cell][block][2] in reduce_concat:
                    reduce_concat.remove(code[cell][block][2])
                if code[cell][block][3] in reduce_concat:
                    reduce_concat.remove(code[cell][block][3])

    if 0 in reduce_concat:
        reduce_concat.remove(0)
    if 1 in reduce_concat:
        reduce_concat.remove(1)

    gen_type = Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)

    # print(normal, normal_concat, reduce, reduce_concat)
    return gen_type
