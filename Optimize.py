from config import BRANCH_NUMBER
from Time_Prediction import ServerTime, DeviceTime, partition_point_number,OutputSizeofPartitionLayer

def Optimize(latency_threshold):
    server_time_predictor, device_time_predictor,ospl = ServerTime(), DeviceTime(),OutputSizeofPartitionLayer()
    for exit_branch in range(BRANCH_NUMBER-1,-1,-1):
        for partition_point in range(partition_point_number[exit_branch]-1, -1, -1):
            # TODO: B? data?
            total_time = device_time_predictor.predict_time(exit_branch, partition_point) +
                         server_time_predictor.predict_time(exit_branch, partition_point) +
            
            outputsize=ospl.output_size(exit_branch,partition_point)
            
            if total_time < latency_threshold:
                return exit_branch, partition_point


    