import itertools
import subprocess
import threading
import time
import os

def generate_files(filename):
    count = 0
    current_time = time.time()
    with open(filename) as f:
            os.makedirs(os.path.splitext(filename)[0], exist_ok=True)
            file_str = ""
            for line in f:
                file_str += line
                if count % 1000 == 0:
                    print(count)
                if count % 1000000 == 0:
                    print(count)
                    print(time.time() - current_time)
                    with open(f"{os.path.splitext(filename)[0]}/{int(count / 1000000)}.json", "w") as f_new:
                        f_new.write(file_str)
                    file_str = ""
                count += 1

def zip_longest_with_empty_string(*iterables):
    return itertools.zip_longest(*iterables, fillvalue="")


def run_command(command, out=False):
    print(f"Running command {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read and print the output in real-time
    output = ""
    if out:
        for out, err in zip_longest_with_empty_string(process.stdout, process.stderr):
            print(out, end='')
            print(err, end='')
            output += out
    else:
        for err in process.stderr:
            print(err, end='')
            output += err

    # Wait for the process to finish and capture the return code
    return_code = process.wait()
    return_code = process.poll()
    if return_code != 0:
        print(f"Error: Command '{command}' failed with return code {return_code}")
        error_output, _ = process.communicate()
        print(f"Error output: {error_output}")
    return output

def put(local_filename, cluster_filename):
    command = f"/opt/hadoop/bin/hadoop fs -put {local_filename} {cluster_filename}"
    run_command(command)

def delete(filename):
    command = f"rm -r -f {filename}"
    run_command(command)

def mkdir_cluster(filename):
    command = f"/opt/hadoop/bin/hadoop fs -mkdir thesis/{os.path.splitext(filename)[0]}"
    run_command(command)

def unzip(filename, new_file_name=None, zstd=True):
    if new_file_name is None:
        new_file_name = os.path.splitext(filename)[0]
    if os.path.exists(new_file_name):
        print(f"File {filename} exists!")
        return
    if zstd:
        command = f"/usr/bin/zstd -d {filename} --long=31 -f -o {new_file_name}"
    else:
        command = f"unzip {filename} -d {new_file_name}"
    run_command(command)


def check_for_new_files(filename):
    mkdir_cluster(filename)
    while True:
        for i in range(1000):
            current_file = f"{os.path.splitext(filename)[0]}/{i}.json"
            if os.path.exists(f"{current_file}/{i}.json"):
                put(f"{current_file}", "thesis/{current_file}")
                delete(f"{current_file}")

def list_files_on_cluster():
    command = f"/opt/hadoop/bin/hadoop fs -ls /user/s3049221/thesis/"
    output = run_command(command, True)
    return output

def check_existing_files():
    file_list_str = list_files_on_cluster()
    lines = file_list_str.split("\n")
    files = []
    for line in lines[1:]:
        if "RC" in line and '_COPYING_' not in line:
            file = line.split("thesis/")[-1]
            file = file.replace("_filtered", "")
            file += ".zst"
            files.append(str(file).strip())
    return files



def main():
    for root, dirs, files in os.walk(os.getcwd()):
        for filename in sorted(files, reverse=True):
            if ".zst" not in filename:
                continue
            existing_files = check_existing_files()
            if filename in existing_files:
                print("File exists: " + filename)
                continue
            
            print(filename)
            local_unprocessed_filename = f"{os.path.splitext(filename)[0]}.json"
            unzip(filename, local_unprocessed_filename)

            # Create and start the threads
            file_generation_thread = threading.Thread(target=generate_files, args=(local_unprocessed_filename,))
            file_checking_thread = threading.Thread(target=check_for_new_files, args=(local_unprocessed_filename,))

            file_generation_thread.start()
            file_checking_thread.start()

            # Wait for both threads to finish (You can use Ctrl+C to stop the application)
            file_generation_thread.join()
            file_checking_thread.join()

main()