import subprocess
import time
import signal

CHECK_INTERVAL = 60

models = [
    'nonstylized_bilateral_vgg19_in_sm_single_tune_all_confirm',
    'nonstylized_bilateral_vgg19_in_sm_single_tune_all_confirm_again'
]

NODES = [404, 406, 410, 414, 416, 424, 426, 428, 430, 432, 434, 436]

# for beta in [0.0, 0.2, 1.0, 10.0, 100.0]:
#     for zdim in [32, 128]:
#         for gamma in [0.0, 10.0, 50.0, 100.0, 200.0]:
#             model_name = 'nonstylized_vae{0}_beta{1}_gamma{2} --train --gamma {2} --zdim {0} --beta {1} --inputSize 128 --batchSize 128'.format(zdim, beta, gamma)
#             models.append(model_name)

completed_models = []

def get_queue_count(status='PENDING'):
    jobs = subprocess.run(['squeue', '-t', status], capture_output=True)
    output = [s.strip() for s in jobs.stdout.splitlines()]
    return len(output) - 1

def is_node_available():
    jobs = subprocess.run([
        'sinfo',
        '-n',
        'node[{}]'.format(','.join(map(str, NODES)))
        ], capture_output=True)
    output = [s.strip() for s in jobs.stdout.splitlines()]
    return len(output) > 2

def start_job(model):
    command = [
        'srun',
        '--gres=gpu:1',
        '--constraint=TitanX',
        '--time=48:00:00',
        'python',
        'run.py',
        '--train',
        '--bilateral',
        '--batchSize',
        '24',
        '--model',
        model
        ]
    job = subprocess.Popen(command, stdout=subprocess.PIPE)
    return job

def kill_job(pid):
    return subprocess.run(['kill', '-9', str(pid)])

running_jobs = {}

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

if __name__ == '__main__':
    killer = GracefulKiller()
    while True:
        time.sleep(CHECK_INTERVAL)
        running_models = list(running_jobs.keys())
        print('# ALL PENDING: {}'.format(get_queue_count('PENDING')))
        print('# ALL RUNNING: {}'.format(get_queue_count('RUNNING')))
        print('# MY RUNNING: {}'.format(len(running_models)))
        print('MY RUNNING: {}'.format(running_models))

        if get_queue_count('PENDING') < 1:
            print('IS TITANX AVAILABLE: {}'.format(is_node_available()))
            if is_node_available():
                skip_models = running_models + completed_models
                pending_models = [ model for model in models if model not in skip_models ]
                if len(pending_models) > 0:
                    current_model = pending_models[0]
                    started_job = start_job(current_model)
                    running_jobs[current_model] = started_job
                    print('STARTED {}'.format(current_model))
                else:
                    print('NO MODELS IN QUEUE {}'.format(pending_models))
        else:
            if len(running_models) > 1:
                current_model = running_models.pop(-1)
                current_job_id = running_jobs[current_model].pid
                kill_job(current_job_id)
                running_jobs.pop(current_model)
                print('KILLED {}'.format(current_model))

        for running_model in running_models:
            if running_jobs[running_model].poll() == 0:
                completed_models.append(running_model)
                running_jobs.pop(running_model)
                print('COMPLETED {}'.format(running_model))

        print('PID\tMODEL\tSTATUS')
        for key, value in running_jobs.items():
            print('{}\t{}\t{}'.format(value.pid, key, value.poll()))

        if killer.kill_now:
            for running_model in running_models:
                current_job_id = running_jobs[running_model].pid
                kill_job(current_job_id)
            break

    print('Graceful Termination')
