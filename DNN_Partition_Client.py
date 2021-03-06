import thriftpy2 as thriftpy
import numpy as np
from thriftpy2.rpc import make_client
from Branchy_Alexnet_Infer import infer
from utils import test_data
from config import *
from Optimize import Optimize
import time

FILENAME = 'intermediate.npy'

def client_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    return make_client(partition_thrift.Partition, '127.0.0.1', 6000)

def file_info(filename):
    with open(filename, 'rb') as file:
        file_content = file.read()
    return {filename: file_content}

if __name__ == '__main__':

    # get time threshold
    threshold = float(input('Please input latency threshold: '))

    # get test data
    dataiter = test_data()
    images, labels = dataiter.next()

    start = time.time()

    # get partition point and exit point
    ep, pp = Optimize(threshold)
    print('Branch is %d, and partition point is %d' %(ep, pp))

    # infer left part
    out = infer(CLIENT, ep, pp, images)

    print('Left part of model inference complete.')

    # save intermediate for RPC process
    intermediate = out.detach().numpy()
    np.save('intermediate.npy', intermediate)

    client = client_start()
    info = file_info(FILENAME)
    print('Predict answer is: ' + client.partition(info, ep, pp))
    print('True label is: %d' %labels[0])

    end = time.time()
    print('Total time: %f' %(end-start))
