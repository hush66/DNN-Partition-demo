import thriftpy2 as thriftpy
import numpy as np
import torch
from thriftpy2.rpc import make_server
from Branchy_Alexnet_Infer import infer
from config import *

def server_start():
    partition_thrift = thriftpy.load('partition.thrift', module_name='partition_thrift')
    server = make_server(partition_thrift.Partition, Dispacher(), '127.0.0.1', 6000)
    print('Thriftpy server is listening...')
    server.serve()


class Dispacher(object):
    def partition(self, file):
        for filename, content in file.items():
            with open('recv_'+filename, 'wb') as f:
                f.write(content)

        readed = np.load('intermediate.npy')
        input = torch.from_numpy(readed)

        out = infer(SERVER, 3, 1, input)
        print(out)
        max_index = str(torch.max(out,1))
        return max_index

if __name__ == '__main__':
    server_start()
