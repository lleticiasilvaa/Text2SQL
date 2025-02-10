import subprocess
import time
import csv

# Function to run nvidia-smi and get GPU stats
def get_gpu_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE
    )
    # Parse the result
    stats = result.stdout.decode().strip().split(', ')
    free_mem = int(stats[0])
    used_mem = int(stats[1])
    total_mem = int(stats[2])
    utilization = int(stats[3])
    return free_mem, used_mem, total_mem, utilization

# Set the duration for 10 minutes
duration = 12 * 60  # 10 minutes in seconds
end_time = time.time() + duration

# Prepare CSV file to save data
with open('gpu_usage.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Free Memory (MB)', 'Used Memory (MB)', 'Total Memory (MB)', 'GPU Utilization (%)'])

    while time.time() < end_time:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        free_mem, used_mem, total_mem, utilization = get_gpu_usage()
        writer.writerow([timestamp, free_mem, used_mem, total_mem, utilization])
        time.sleep(1)  # Record data every second

print("GPU usage data has been saved to gpu_usage.csv.")
