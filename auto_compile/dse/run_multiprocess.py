import os
import argparse
import subprocess
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="number of processes")
args = parser.parse_args()

root = "/mnt/data1/chengdi/nb201/"
cmds = []
for i in range(15625):
    cmds.append("python3 dse_p.py -m {} -i ./cifar10.json -b ./vu9p.json -o {} --parallel -dt 1".format(
        os.path.join(root, "models", f"{i}.model"), os.path.join(root, "dse_dt1", str(i))))

queue = multiprocessing.Queue(maxsize=args.n)
def _worker(pid, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        _, cmd = token
        print("Process #{}: CMD: {}".format(pid, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(pid))

for pid in range(args.n):
    p = multiprocessing.Process(target=_worker, args=(pid, queue))
    p.start()

for i_cmd, cmd in enumerate(cmds):
    queue.put((i_cmd, cmd))

# close all the workers
for _ in range(args.n):
    queue.put(None)
