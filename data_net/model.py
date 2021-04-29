import torch
import math
from  torch.nn.parameter import Parameter
import torch.nn as nn

def reset_param(w): #从w的第一维的变换中得到均匀分布的张量参数。
    stdv=2./math.sqrt(w.size(0))
    w.data.uniform_(-stdv,stdv)

class GCN(torch.nn.Module):
    def __init__(self, args, activation):
        super().__init__()
        self.activation=activation
        self.num_layers=args.num_layers

        self.w_list=nn.ParameterList()
        for i in range(self.num_layers):
            if i==0: #第一层
                w_i=Parameter(torch.Tensor(args.feats_per_node, args_layer_1_feats))
            else:
                w_i=Parameter(torch.Tensor(args.layer_1_feats,args.layers_2_feats))
                reset_param(w_i) #只对除第一层之外的数据重参数化
            self.w_list.append(w_i)

    def forward(self, A_list,Nodes_list,nodes_mask_list):
        node_feats=Nodes_list[-1] #最后一层信息

        Ahat=A_list[-1] #归一化拉普拉斯矩阵

        last_l=self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0]))) #GCN的计算公式
        for i in range(1,self.num_layers):
            last_l=self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l #最终的GCN输出
