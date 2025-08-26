import json
import os
import subprocess
import time
import zipfile
from math import ceil

import paramiko as paramiko
import zstandard
from pw import pwd, student_number

file_path = 'C:\\Users\\csabs\\Downloads\\reddit'
subfolders = ['comments', 'submissions']
cluster_path = f"thesis"



def decompress_file(zst_name, json_name):
    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
    with open(zst_name, 'rb') as ifh, open(json_name, 'wb') as ofh:
        for chunk in dctx.read_to_iter(ifh):
            ofh.write(chunk)

def compress_file(json_name, zst_name):
    ctx = zstandard.ZstdCompressor()
    with open(json_name, 'rb') as ifh, open(zst_name, 'wb') as ofh:
        for chunk in ctx.read_to_iter(ifh):
            ofh.write(chunk)

def process_file_zip_individual(file_json, zipf):
    print("Extracting relevant attributes for ", file_json)
    file_size = os.path.getsize(f'{file_json}')

    print(file_size)

    path = os.path.splitext(file_json)[0]

    # Initialize a variable to keep track of the processed bytes
    processed_bytes = 0
    with open(f'{file_json}', 'r') as file:
        if os.path.exists(f'{path}'):
            print("Folder exists")
            # return False
        os.makedirs(f'{path}', exist_ok=True)
        lines = []
        selected_keys = ["author", "score", "parent_id", "id", "link_id", "body", "controversiality", "subreddit"]
        # file_length = int(subprocess.check_output(f"wc -l {os.getcwd()}/{path}/{filename}", shell=True).split()[0])
        for i, line in enumerate(file):
            if os.path.exists(f'{path}/{ceil(i / 1000000) * 1000000}.json'):
                continue
            if i % 1000000 == 0 and i > 0:
                with open(f'{path}/{i}.json', 'w') as f_new:
                    json.dump(lines, f_new)
                lines = []
                file_path = f'{path}/{i}.json'
                file_path_zstd = f'{path}/{i}.zst'
                command = f"zstd {file_path} -o {file_path_zstd} -v"
                process = subprocess.Popen(command, shell=True)
                process.wait()
                zipf.write(file_path_zstd, os.path.relpath(file_path_zstd, os.getcwd() + f"/{path}"))
                os.remove(file_path)
                os.remove(file_path_zstd)
            original_dict = json.loads(line)
            new_dict = {key: original_dict[key] for key in selected_keys if key in original_dict}
            lines.append(json.dumps(new_dict))
            # Update the processed bytes count
            processed_bytes += len(line.encode('utf-8'))

            # Calculate the percentage of progress
            progress_percentage = (processed_bytes / file_size) * 100

            # Print the progress percentage (optional)
            if i % 1000000 == 0:
                print(f"Progress: {progress_percentage:.2f}%")
        with open(f'{path}/{i}.json', 'w') as f_new:
            json.dump(lines, f_new)
        file_path = f'{path}/{i}.json'
        file_path_zstd = f'{path}/{i}.zst'
        command = f"zstd {file_path} -o {file_path_zstd} -v"
        process = subprocess.Popen(command, shell=True)
        process.wait()
        zipf.write(file_path_zstd, os.path.relpath(file_path_zstd, os.getcwd() + f"/{path}"))
        os.remove(file_path)
        os.remove(file_path_zstd)
        return True

def process_file_zip_folder(file_json, year, month, contr_type):
    print("Extracting relevant attributes for ", file_json)
    file_size = os.path.getsize(f'{file_json}')

    print(file_size)

    json_path = os.path.join(os.getcwd(), "data", f"{year}_{month}_{contr_type}.json")

    # Initialize a variable to keep track of the processed bytes
    processed_bytes = 0
    with open(f'{file_json}', 'r', encoding="iso-8859-1") as file:
        if os.path.exists(f'{json_path}'):
            print("JSON exists")
            print(json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst")
            return json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst"
        lines = []
        if contr_type == "comments":
            selected_keys = ["author", "score", "parent_id", "id", "link_id", "body", "controversiality", "subreddit"]
        elif contr_type == "submissions":
            selected_keys = ["author", "brand_safe", "hidden", "id", "is_self", "num_crossposts", "selftext", "score",
                             "subreddit", "title"]
        # file_length = int(subprocess.check_output(f"wc -l {os.getcwd()}/{path}/{filename}", shell=True).split()[0])
        for i, line in enumerate(file):
            #print(i, line)
            if i % 1000000 == 0 and i > 0:
                with open(json_path, 'a') as f_new:
                    json.dump(lines, f_new)
                lines = []
            try:
                original_dict = json.loads(line)
            except:
                print(i, line)
            new_dict = {key: original_dict[key] for key in selected_keys if key in original_dict}
            lines.append(new_dict)
            # Update the processed bytes count
            processed_bytes += len(line.encode('utf-8'))

            # Calculate the percentage of progress
            progress_percentage = (processed_bytes / file_size) * 100

            # Print the progress percentage (optional)
            if i % 1000000 == 0:
                print(f"Progress: {progress_percentage:.2f}%")
        with open(json_path, 'a') as f_new:
            json.dump(lines, f_new)
    compress_file(json_path, json_path.replace('.json', '.zst'))
    os.remove(file_json)
    os.remove(json_path)
    return json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst"


def upload_to_local(zst_path):
    print("Uploading to local")
    command = f"pscp -P 22 -pw {pwd} {zst_path} s3049221@spark-head1.eemcs.utwente.nl:"
    powershell_command = f'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe {command}'
    process = subprocess.Popen(powershell_command, shell=True)
    return_code = process.wait()
    print(zst_path, return_code)
    if return_code != 0:
        raise ValueError
    return zst_path


def connect_to_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect("spark-head1.eemcs.utwente.nl", username=f"{student_number}", password=pwd)
    except TimeoutError:
        print("TimeoutError. Please do something")
        input("Have you done something? [input]")
        ssh.connect("spark-head1.eemcs.utwente.nl", username=f"{student_number}", password=pwd)
    return ssh


def put(ssh, zst_name):
    try:
        input_file, channel_file, error_file = ssh.exec_command(
            f"/opt/hadoop/bin/hadoop fs -put {zst_name} /user/{student_number}/thesis/{zst_name}")
        chan_lines = channel_file.readlines()
        if chan_lines:
            print(chan_lines)
        err_lines = error_file.readlines()
        if err_lines:
            print(err_lines)
            if "Exception encountered" in "".join(err_lines):
                raise TimeoutError
    except TimeoutError:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect("spark-head1.eemcs.utwente.nl", username=f"{student_number}", password=pwd)
        input_file, channel_file, error_file = ssh.exec_command(
            f"/opt/hadoop/bin/hadoop fs -put {zst_name} /user/{student_number}/thesis/{zst_name}")
        chan_lines = channel_file.readlines()
        if chan_lines:
            print(chan_lines)
        err_lines = error_file.readlines()
        if err_lines:
            print(err_lines)
    return ssh


def check_existing_files(ssh):
    command = f"/opt/hadoop/bin/hadoop fs -ls /user/{student_number}/thesis/"
    input_file, channel_file, error_file = ssh.exec_command(command)
    chan_lines = channel_file.readlines()
    if chan_lines or True:
        lines = chan_lines[1:]
        files = []
        for line in lines:
            file = line.split("thesis/")[-1][:-1]
            files.append(str(file).strip())
    print(files)
    return files


def original():
    years = [str(i) for i in range(2006, 2016)]
    ssh = connect_to_ssh()
    existing_files = check_existing_files(ssh)
    for subfolder in subfolders:
        for root, dirs, files in os.walk(os.path.join(file_path, subfolder)):
            for file in files:
                year = file.split("_")[1].split('-')[0]
                month = os.path.splitext(file.split("_")[1].split('-')[1])[0]
                print(file)
                if year in years and '.zst' in file and [year, month, subfolder] not in existing_files:
                    print(file)
                    decompress_file(os.path.join(root, file), os.path.join(root, file.replace('.zst', '.json')))
                    zst_path, zst_name = process_file_zip_folder(os.path.join(root, file.replace('.zst', '.json')), year, month, subfolder)
                    return_code = upload_to_local(zst_path)
                    if return_code == 0:
                        os.remove(zst_path)
                    ssh = put(ssh, zst_name)

def execute_in_ssh(ssh, command):
    print(command)
    input_file, channel_file, error_file = ssh.exec_command(command, get_pty=True)
    chan_lines = channel_file.readlines()
    if chan_lines:
        print(chan_lines)
    err_lines = error_file.readlines()
    if err_lines:
        print(err_lines)


def unzip_in_local(ssh, file):
    print(f"Unzipping {file} in local")
    file_json = file.replace(".zst", ".json")
    command = f"zstd -d {file} --long=31 -f -o unprocessed.json"
    execute_in_ssh(ssh, command)
    return file_json


def delete_in_local(ssh, file):
    print(f"Deleting {file} in local")
    command = f"rm -rf {file}"
    execute_in_ssh(ssh, command)
    return

def put_cluster(ssh, file):
    print(f"Putting {file} to cluster")
    command = f"/opt/hadoop/bin/hadoop fs -put {file} {cluster_path}/{file}"
    execute_in_ssh(ssh, command)
    return file

def cluster_process_c(ssh, file):
    print(f"Processing {file} on cluster")
    command = "time /opt/spark/bin/spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --conf spark.dynamicAllocation.minExecutors=5 process_file.py 2"
    execute_in_ssh(ssh, command)


def cluster_process_s(ssh, file):
    print(f"Processing {file} on cluster")
    command = "time /opt/spark/bin/spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=10 --conf spark.dynamicAllocation.minExecutors=5 process_file_s.py 2"
    execute_in_ssh(ssh, command)


def cluster_delete(ssh, file):
    print(f"Deleting {file} on cluster")
    command = f"/opt/hadoop/bin/hadoop fs -rm -r -f {file}"
    execute_in_ssh(ssh, command)
    return file

def cluster_to_local(ssh, file):
    print(f"Getting {file} from cluster")
    command = f"/opt/hadoop/bin/hadoop fs -get {file}"
    execute_in_ssh(ssh, command)
    return file

def zip_processed(ssh, file):
    print(f"Zipping {file}")
    command = f"zip -r {file} processed"
    execute_in_ssh(ssh, command)


def set_up_required_file_names(file):
    file_no_ext = os.path.splitext(file)[0]
    initial_zst_file_name = file
    initial_json_file_name = "unprocessed.json"
    cluster_json_file_name = f'{cluster_path}/unprocessed.json'
    cluster_folder_name = f'{cluster_path}/processed'
    local_processed_folder_name = "processed"
    local_processed_zst_name = file_no_ext + "_" + local_processed_folder_name + '.zip'
    cluster_processed_zst_name = f'{cluster_path}/{local_processed_zst_name}'
    file_list = []
    file_list.extend([initial_zst_file_name for i in range(2)])
    file_list.extend([initial_json_file_name for i in range(2)])
    file_list.extend([cluster_json_file_name for i in range(2)])
    file_list.extend([cluster_folder_name for i in range(2)])
    file_list.extend([local_processed_zst_name for i in range(1)])
    file_list.extend([local_processed_folder_name for i in range(1)])
    file_list.extend([local_processed_zst_name for i in range(2)])
    file_list.extend([cluster_processed_zst_name for i in range(1)])
    return file_list




def main():
    years = [str(i) for i in range(2006, 2020)]
    ssh = connect_to_ssh()
    existing_files = check_existing_files(ssh)
    for subfolder in subfolders:
        steps = [unzip_in_local, delete_in_local, put_cluster, delete_in_local, cluster_process_c if subfolder == "comments" else cluster_process_s,
                 cluster_delete, cluster_to_local, cluster_delete, zip_processed, delete_in_local, put_cluster,
                 delete_in_local]
        for root, dirs, files in os.walk(os.path.join(file_path, subfolder)):
            for file in files:
                file_names = set_up_required_file_names(file)
                if file.endswith(".zst") and file_names[-1].replace(".zip", "") not in existing_files:
                    upload_to_local(os.path.join(root, file_names[0]))
                    #for i, step in enumerate(steps):
                        #try:
                            #step(ssh, file_names[i])
                        #except:
                            #ssh = connect_to_ssh()
                            #step(ssh, file_names[i])
            # fel a localba
            # localban unzip
            # localban töröl zst
            # put clusterbe
            # localban töröl json
            # clusterben feldolgoz
            # clusterben töröl eredeti
            # clusterből localba másol
            # localban zip
            # localban feldolgozott töröl
            # clusterbe zippelt


def delete_old_files():
    ssh = connect_to_ssh()
    existing_files = check_existing_files(ssh)
    for existing_file in existing_files:
        if "_comments" in existing_file or "_submissions" in existing_file:
            cluster_delete(ssh, f"{cluster_path}/{existing_file}.zst")


def main():
    ssh = connect_to_ssh()
    existing_files = check_existing_files(ssh)
    for subfolder in subfolders:
        steps = [unzip_in_local, delete_in_local, put_cluster, delete_in_local,
                 cluster_process_c if subfolder == "comments" else cluster_process_s,
                 cluster_delete, cluster_to_local, cluster_delete, zip_processed, delete_in_local, put_cluster,
                 delete_in_local]
        for root, dirs, files in os.walk(os.path.join(file_path, subfolder)):
            for file in files:
                file_names = set_up_required_file_names(file)
                if file.endswith(".zst") and file_names[-2] not in existing_files and "2019" in file:
                    upload_to_local(os.path.join(root, file_names[0]))
                    # stdin, stdout, stderr = ssh.exec_command(f"python3 process_file_local.py {file_names[0]}")
                    # while not stdout.channel.exit_status_ready():
                    #     if stdout.channel.recv_ready():
                    #         output = stdout.channel.recv(64)
                    #         print(output.decode('utf-8'), end='')

if __name__ == '__main__':
    main()






