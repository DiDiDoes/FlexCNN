import numpy as np
import os
from tqdm import tqdm

#root = "/mnt/data1/chengdi/nb201/dse_no_systolic/"
root = "/nfs/data/chengdicao/data/flexcnn_nb201/dse_nosystolic_dt1/"

results = []
for i in tqdm(range(15625)):
    result_filename = os.path.join(root, str(i), "results.npy")
    try:
        result = np.load(result_filename)
        results.append(result)
    except:
        print(i)
results = np.array(results)
print(results.shape)

prev_results_filename = "/home/chengdicao/zero-shot-nas-dse/results/naswot_result_nasbench201_cifar10_False.npy"
prev_results = np.load(prev_results_filename)
print(prev_results.shape)

output_filename = "/home/chengdicao/zero-shot-nas-dse/results/dse_dt1_nosystolic_nasbench201_cifar10.npy"
aggregated_results = np.concatenate([prev_results, results], axis=1)
print(aggregated_results.shape)
print(aggregated_results[:5])
np.save(output_filename, aggregated_results)
