import os
import time
import shlex
import subprocess
import itertools

datasets = os.listdir('/home/covpreduser/Blob/v-hangtingye/TabAD/datasets')
for i in range(len(datasets)):
    datasets[i] = datasets[i].split('.')[0]   

pseudo_models = ['pca', 'iforest', 'hbos', 'ocsvm', 'lof', 'cblof', 'cof', 'knn', 'sod', 'ecod', 'deep_svdd', 'loda', 'copod', 'gmm', 'vae']
# pseudo_models = ['pca', 'iforest', 'hbos', 'ocsvm']
# pseudo_models = ['lof', 'cblof', 'cof', 'knn']
# pseudo_models = ['sod', 'ecod', 'deep_svdd', 'loda']
# pseudo_models = ['copod', 'gmm']

# experiment_types = ['uadb', 'base_mean', 'base_std', 'base_mean_cascade', 'base_std_cascade']
experiment_types = ['uadb', 'base_mean', 'base_std', 'base_mean_cascade', 'base_std_cascade']

gpu_cnt = 4

# submitting experiments in parallel to multiple gpus
def run(cmds, cuda_id, gpu_cnt):
    _cur = 0

    def recycle_devices():
        running_jobs = 0
        for cid in cuda_id:
            if cuda_id[cid] is not None:
                proc = cuda_id[cid]
                if proc.poll() is not None:
                    cuda_id[cid] = None
                else:
                    running_jobs += 1
        return running_jobs

    def available_device_id():
        for cid in cuda_id:
            if cuda_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(int(cid) % gpu_cnt)

        print('Submit Job:')
        print(cmd)

        cmd_args = shlex.split(cmd)
        #log_file = open(f'log/{cmd_args[-1]}', 'w')

        proc = subprocess.Popen(cmd_args, env=env)

        cuda_id[cid] = proc

    while 1:
        running_jobs = recycle_devices()
        cid = available_device_id()

        if _cur >= len(cmds) and running_jobs == 0:
            break

        if cid is not None and _cur < len(cmds):
            print('CUDA {} available'.format(cid))
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(5)


def start():
    cmds = []
    options = list(itertools.product(datasets, pseudo_models, experiment_types))
    
    # generate cmds of different experiments
    for dataset, pseudo_mdoel, experiment_type in options:            
        cmd_parts = [
            # 'python main_teacher.py',
            'python main.py',
            f'--data_path {dataset}',
            f'--pseudo_model {pseudo_mdoel}',
            f'--experiment_type {experiment_type}'
        ]
        cmd = ' '.join(cmd_parts)
        cmds.append(cmd)

    cuda_id = dict([(str(i), None) for i in range(gpu_cnt)])    
    run(cmds, cuda_id, gpu_cnt)


if __name__ == '__main__':
    start()