import time
import psutil
import subprocess as sp

def start_process():
    print('starting process...')
    cmd = "nohup /opt/conda/bin/python /home/oinnerednib/ai-research/ann_hyperparam_search.py >> logs/ann_hyperparam_search.log 2>&1 &"
    sp.Popen(cmd, shell=True, executable="/bin/bash")

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def find_process():
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['cmdline'] and 'ann_hyperparam_search.py' in ' '.join(proc.info['cmdline']):
            return proc
    return None

def kill_process(proc):
    try:
        proc.terminate()  # Gracefully terminate the process
        proc.wait(timeout=10)  # Wait for the process to exit
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        proc.kill()  # Forcefully kill the process if it doesn't exit in time

def main():
    sp.Popen('pkill -f "/opt/conda/bin/python /home/oinnerednib/ai-research/ann_hyperparam_search.py"', shell=True, executable='/bin/bash')

    while True:
        proc = find_process()

        # If process crashed, start it
        if not proc:
            print('Process not found, must have crashed, waiting 5s and restarting...')
            time.sleep(5)
            start_process()
            last_restart_time = time.time()  # Reset the time when starting a new process

        # Check if 30 minutes have passed since the last restart, maybe the process is hanging
        if time.time() - last_restart_time >= 30 * 60:
            print('30mins passed, restarting process...')
            if proc:
                kill_process(proc)  # Kill the existing process
            start_process()  # Restart the process
            last_restart_time = time.time()  # Update the restart time

        time.sleep(2)

if __name__ == "__main__":
    main()


# nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9
# nohup python control_hyperparam_search_processes.py > logs/control_hyperparam_search_processes.log 2>&1 &
# pkill -f "python control_hyperparam_search_processes.py"
