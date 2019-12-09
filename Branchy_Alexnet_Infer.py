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
    '''
    model_dict = net.state_dict()
    params = torch.load('./alexnet_data_out/models/epoch_910_model.pt', map_location=torch.device('cpu'))
    # filter out needed keys
    #loaded_dict = {k:v for k,v in params.items() if k in model_dict}
    loaded_dict = OrderedDict()
    for k, v in params.items():
        if k[7:] in model_dict:
            loaded_dict[k[7:]] = v
    # overwrite entries in the existing state dict
    model_dict.update(loaded_dict)
    net.load_state_dict(model_dict)
    '''

    net.eval()

    return net(input)





