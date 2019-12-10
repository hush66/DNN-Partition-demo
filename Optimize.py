from config import BRANCH_NUMBER
from Time_Prediction import ServerTime, DeviceTime, partition_point_number, model_size

def Optimize(latency_threshold):
    server_time_predictor, device_time_predictor = ServerTime(), DeviceTime()
    for exit_branch in range(BRANCH_NUMBER-1,-1,-1):
        for partition_point in range(partition_point_number[exit_branch]-1, -1, -1):
            # model load time
            left_model_size = model_size['branch'+str(exit_branch+1)+'_part'+str(partition_point+1)+'L']
            right_model_size = model_size['branch'+str(exit_branch+1)+'_part'+str(partition_point+1)+'R']
            model_load_time = device_time_predictor.device_model_load(left_model_size) + \
                              device_time_predictor.device_model_load(right_model_size)
            # TODO: B? data?
            total_time = device_time_predictor.predict_time(exit_branch, partition_point) + \
                         server_time_predictor.predict_time(exit_branch, partition_point) + model_load_time

            if total_time < latency_threshold:
                return exit_branch, partition_point


