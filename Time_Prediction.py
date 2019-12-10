from config import *


branch1 = ['conv1', 'relu0', 'pool0', 'norm0', 'convB1', 'relu1', 'convB2', 'relu2', 'pool1', 'linear']
branch2 = ['conv1', 'relu0', 'norm0', 'pool0', 'conv2', 'relu1', 'pool1', 'norm1', 'convB1', 'relu2', 'pool2',
           'linear']
branch3 = ['conv1', 'relu0', 'norm0', 'pool0', 'conv2', 'relu1', 'norm1', 'pool1', 'conv3', 'relu2', 'conv4',
           'relu3', 'conv5', 'relu4', 'pool2', 'classifier']

branch1_partition_index = [2, 8]
branch2_partition_index = [3, 6, 10]
branch3_partition_index = [3, 7, 14]

# partitiion point number for every layer
partition_point_number = [2, 2, 3]

branches_info = [(branch1, branch1_partition_index), (branch2, branch2_partition_index), (branch3, branch3_partition_index)]


###############################################
# Mobile device side time prediction class
###############################################
class DeviceTime:
    def __init__(self):
        self.branch1 = {
            'conv1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.device_relu(63 * 32 * 32),
            'pool0': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            'norm0': self.device_lrn(64 * 15 * 15),
            'convB1': self.device_conv(64, (3 * 3 * 64) ** 2 * 32),
            'relu1': self.device_relu(32 * 15 * 15),
            'convB2': self.device_conv(32, (3 * 3 * 32) ** 2 * 32),
            'relu2': self.device_relu(32 * 15 * 15),
            'pool1': self.device_pool(32 * 15 * 15, 32 * 7 * 7),
            'linear': self.device_linear(1568, 10),
        }
        self.branch2 = {
            'conv1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.device_relu(64 * 32 * 32),
            'norm0': self.device_lrn(64 * 32 * 32),
            'pool0': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            'conv2': self.device_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu1': self.device_relu(192 * 13 * 13),
            'pool1': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            'norm1': self.device_lrn(192 * 6 * 6),
            'convB1': self.device_conv(192, (3 * 3 * 192) ** 2 * 32),
            'relu2': self.device_relu(32 * 6 * 6),
            'pool2': self.device_pool(32 * 6 * 6, 32 * 2 * 2),
            'linear': self.device_linear(128, 10),
        }
        self.branch3 = {
            'conv1': self.device_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.device_relu(64 * 32 * 32),
            'norm0': self.device_lrn(64 * 32 * 32),
            'pool0': self.device_pool(64 * 32 * 32, 64 * 15 * 15),
            'conv2': self.device_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu1': self.device_relu(192 * 13 * 13),
            'norm1': self.device_lrn(192 * 13 * 13),
            'pool1': self.device_pool(192 * 13 * 13, 192 * 6 * 6),
            'conv3': self.device_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu2': self.device_relu(384 * 6 * 6),
            'conv4': self.device_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu3': self.device_relu(256 * 6 * 6),
            'conv5': self.device_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu4': self.device_relu(256 * 6 * 6),
            'pool2': self.device_pool(256 * 6 * 6, 256 * 2 * 2),
            'classifier': self.device_dropout(1024) + self.device_linear(1024, 4096) +
                          self.device_relu(4096) + self.device_dropout(4096) +
                          self.device_linear(4096, 4096) + self.device_relu(4096) +
                          self.device_linear(4096, 10)
        }
        self.branches = [self.branch1, self.branch2, self.branch3]

    # time predict function
    def device_lrn(self, data_size):
        return 1.504004534429181e-08 * data_size + 0.00023856992705946142

    def device_pool(self, input_data_size, output_data_size):
        return 2.6848359497725075e-10 * input_data_size + 1.4361124920215772e-08 * output_data_size + 1.7488092089789508e-05

    def device_relu(self, input_data_size):
        return 1.0365086209870186e-09 * input_data_size + 0.0002997479361697905

    def device_dropout(self, input_data_size):
        return 3.5615447448579893e-09 * input_data_size + 0.0010654349730239024

    def device_linear(self, input_data_size, output_data_size):
        return 1.4168476912236977e-09 * input_data_size + 9.367565952531078e-06 * output_data_size + 0.00021007062556504494

    def device_conv(self, feature_map_amount, compution_each_pixel):
        # compution_each_pixel stands for (filter size / stride)^2 * (number of filters)
        return 3.7532886332610986e-06 * feature_map_amount + 7.285089421250621e-12 * compution_each_pixel + 0.00038511489900182815

    def device_model_load(self, model_size):
        return 8.081666725164214e-09 * model_size + 0.02417584685455454

    # tool
    def predict_time(self, branch_number, partition_point_number):
        '''
        :param branch_number: the index of branch
        :param partition_point_number: the index of partition point
        :return:
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point =  partition_point_index_set[partition_point_number]
        # layers in partitioned model
        layers = branch_layer[:partition_point+1]
        time_dict = self.branches[branch_number]

        time = 0
        for layer in layers:
            time += time_dict[layer]
        return time


###############################################
# Edge server side time prediction class
###############################################
class ServerTime:
    def __init__(self):
        self.branch1 = {
            'conv1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.server_relu(63 * 32 * 32),
            'pool0': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            'norm0': self.server_lrn(64* 15 * 15),
            'convB1': self.server_conv(64, (3 * 3 * 64) ** 2 * 32),
            'relu1': self.server_relu(32 * 15 * 15),
            'convB2': self.server_conv(32, (3 * 3 * 32) ** 2 * 32),
            'relu2': self.server_relu(32 * 15 * 15),
            'pool1': self.server_pool(32 * 15 *15, 32 *  7 * 7),
            'linear': self.server_linear(1568, 10),
        }
        self.branch2 = {
            'conv1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.server_relu(64 * 32 * 32),
            'norm0': self.server_lrn(64 * 32 * 32),
            'pool0': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            'conv2': self.server_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu1': self.server_relu(192 * 13 * 13),
            'pool1': self.server_pool(192 * 13 * 13, 192 * 6 * 6),
            'norm1': self.server_lrn(192 * 6 * 6),
            'convB1': self.server_conv(192, (3 * 3 * 192) ** 2 * 32),
            'relu2': self.server_relu(32 * 6 * 6),
            'pool2': self.server_pool(32 * 6 * 6, 32 * 2 * 2),
            'linear': self.server_linear(128, 10),
        }
        self.branch3 = {
            'conv1': self.server_conv(3, (5 * 5 * 3) ** 2 * 64),
            'relu0': self.server_relu(64 * 32 * 32),
            'norm0': self.server_lrn(64 * 32 * 32),
            'pool0': self.server_pool(64 * 32 * 32, 64 * 15 * 15),
            'conv2': self.server_conv(64, (5 * 5 * 64) ** 2 * 192),
            'relu1': self.server_relu(192 * 13 * 13),
            'norm1': self.server_lrn(192 * 13 * 13),
            'pool1': self.server_pool(192 * 13 * 13 , 192 * 6 * 6),
            'conv3': self.server_conv(192, (3 * 3 * 192) ** 2 * 384),
            'relu2': self.server_relu(384 * 6 * 6),
            'conv4': self.server_conv(384, (3 * 3 * 384) ** 2 * 256),
            'relu3': self.server_relu(256 * 6 * 6),
            'conv5': self.server_conv(256, (3 * 3 * 256) ** 2 * 256),
            'relu4': self.server_relu(256 * 6 * 6),
            'pool2': self.server_pool(256 * 6 * 6, 256 * 2 * 2),
            'classifier': self.server_dropout(1024) + self.server_linear(1024, 4096) +
                          self.server_relu(4096) + self.server_dropout(4096) +
                          self.server_linear(4096, 4096) + self.server_relu(4096) +
                          self.server_linear(4096, 10)
        }
        self.branches = [self.branch1, self.branch2, self.branch3]


    def server_lrn(self, data_size):
        return 2.111544033139625e-08 * data_size + 0.0285872721707483

    def server_pool(self, input_data_size, output_data_size):
        return -3.08201145e-10 * input_data_size + 1.19458883e-09 * output_data_size - 0.0010152380964514613

    def server_relu(self,input_data_size):
        return 2.332339368254984e-09 * input_data_size + 0.005070494191853819

    def server_dropout(self, input_data_size):
        return 3.962833398808942e-09 * input_data_size + 0.015458175165054516

    def server_linear(self, input_data_size, output_data_size):
        return 9.843676646891836e-12 * input_data_size + 4.0100716666407315e-07 * output_data_size + 0.015619779485748695

    def server_conv(self, feature_map_amount, compution_each_pixel):
        # compution_each_pixel stands for (filter size / stride)^2 * (number of filters)
        return 1.513486447521604e-06 * feature_map_amount + 4.4890001480985655e-12 * compution_each_pixel + 0.009816023641653768

    def server_model_load(self, model_size):
        return 7.073959838308404e-09 * model_size + 3.9334279905883087

    # tool
    def predict_time(self, branch_number, partition_point_number):
        '''
        :param branch_number: the index of branch
        :param partition_point_number: the index of partition point
        :return:
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point = partition_point_index_set[partition_point_number]
        # layers in partitioned model(right part)
        layers = branch_layer[partition_point + 1:]
        time_dict = self.branches[branch_number]

        time = 0
        for layer in layers:
            time += time_dict[layer]
        return time
    
class OutputSizeofPartitionLayer:   
    def __init__(self):
        self.branch1={
            'pool0':64 * 15 * 15* 32,
            'pool1':32 *  7 * 7 * 32,
        }
        self.branch2 = {
            'pool0': 64 * 15 * 15 * 32,
            'pool1': 192 * 6 * 6 *32,
            'pool2': 32 * 2 * 2 *32,
        }
        self.branch3 = {
            'pool0': 64 * 15 * 15 *32,
            'pool1': 192 * 6 * 6 *32,
            'pool2':256 * 2 * 2 *32,
        }
        self.branches = [self.branch1, self.branch2, self.branch3]
        
    def output_size(self, branch_number, partition_point_number):
        '''
        :return:unit(bit)
        '''
        branch_layer, partition_point_index_set = branches_info[branch_number]
        partition_point =  partition_point_index_set[partition_point_number]
        # layers in partitioned model
        layer = branch_layer[partition_point:partition_point+1][0]
        outputsize_dict = self.branches[branch_number]
        return outputsize_dict[layer]
        
if __name__=='__main__':
    ospl=OutputSizeofPartitionLayer()
    print(ospl.output_size(0,0))