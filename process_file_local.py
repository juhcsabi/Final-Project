import argparse
import os
import subprocess


def run_command(command):
    print(f"Running command {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read and print the output in real-time
    for line in process.stderr:
        print(line, end='')

    # Wait for the process to finish and capture the return code
    return_code = process.wait()
    return_code = process.poll()
    if return_code != 0:
        print(f"Error: Command '{command}' failed with return code {return_code}")
        error_output, _ = process.communicate()
        print(f"Error output: {error_output}")

def unzip(filename):
    command = f"/usr/bin/zstd -d {filename} --long=31 -f -o unprocessed.json"
    run_command(command)

def delete(filename):
    command = f"rm -r -f {filename}"
    run_command(command)

def put(local_filename, cluster_filename):
    command = f"/opt/hadoop/bin/hadoop fs -put {local_filename} {cluster_filename}"
    run_command(command)


def process(comments=True):
    command = f"/opt/spark/bin/spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --conf spark.dynamicAllocation.minExecutors=5 process_file{'_s' if not comments else ''}.py 2"
    run_command(command)


def delete_on_cluster(filename_cluster):
    command = f"/opt/hadoop/bin/hadoop fs -rm -r -f {filename_cluster}"
    run_command(command)


def get(cluster_foldername, local_foldername):
    command = f"/opt/hadoop/bin/hadoop fs -get {cluster_foldername} {local_foldername}"
    run_command(command)


def zip_processed(local_processed_foldername, local_processed_zipname):
    command = f"zip -r {local_processed_zipname} {local_processed_foldername}"
    run_command(command)


if __name__ == '__main__':
    for root, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if ".zst" not in filename:
                continue
            print(filename)
            cluster_dir = "thesis"
            local_unprocessed_filename = "unprocessed.json"
            cluster_unprocessed_filename = f"{cluster_dir}/{local_unprocessed_filename}"
            local_processed_foldername = "processed"
            cluster_processed_foldername = f"{cluster_dir}/{local_processed_foldername}"
            local_processed_zipname = f"{os.path.splitext(filename)[0]}_processed.zip"
            cluster_processed_zipname = f"{cluster_dir}/{local_processed_zipname}"
            unzip(filename)
            delete(filename)
            put(local_unprocessed_filename, cluster_unprocessed_filename)
            delete(local_unprocessed_filename)
            if "C" in filename:
                process(comments=True)
            else:
                process(comments=False)
            delete_on_cluster(cluster_unprocessed_filename)
            get(cluster_processed_foldername, local_processed_foldername)
            delete_on_cluster(cluster_processed_foldername)
            zip_processed(local_processed_foldername, local_processed_zipname)
            delete(local_processed_foldername)
            put(local_processed_zipname, cluster_processed_zipname)
            delete(local_processed_zipname)