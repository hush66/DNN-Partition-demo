import torch
import numpy as np
from Model_Pair import *
from config import *
from collections import OrderedDict

def infer(cORs, ep, pp, input):
    '''
    DNN model inference
    :param cORs: client or server
    :param pp: partition point
    :param ep: exit point
    :return: intermediate data or final result
    '''
    netPair = 'NetExit' + str(ep) + 'Part' + str(pp)
    net = eval(netPair)[cORs]()

    # load params
    LOrR = 'L' if cORs == CLIENT else 'R'
    params_path = PARAM_PATH + netPair + LOrR
    net.load_state_dict(torch.load(params_path))
    print(net.state_dict().keys())

    net.eval()

    return net(input)





