import json
import os
import subprocess
import time
import zipfile
from math import ceil

import paramiko as paramiko
import zstandard
from pw import pwd, student_number

file_path = os.path.join(os.getcwd(), "raw_data")
subfolders = ['submissions']


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
            processed_bytes += len(line.encode('utf-8'))

            progress_percentage = (processed_bytes / file_size) * 100

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

    processed_bytes = 0
    with open(f'{file_json}', 'r') as file:
        if os.path.exists(f'{json_path}'):
            print("JSON exists")
            print(json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst")
            return json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst"
        lines = []
        if contr_type == "comments":
            selected_keys = ["author", "score", "parent_id", "id", "link_id", "body", "controversiality", "subreddit"]
        elif contr_type == "submissions":
            selected_keys = ["author", "brand_safe", "hidden", "id", "is_self", "num_crossposts", "selftext", "score", "subreddit", "title"]
        # file_length = int(subprocess.check_output(f"wc -l {os.getcwd()}/{path}/{filename}", shell=True).split()[0])
        for i, line in enumerate(file):
            if i % 1000000 == 0 and i > 0:
                with open(json_path, 'a') as f_new:
                    json.dump(lines, f_new)
                lines = []
            original_dict = json.loads(line)
            new_dict = {key: original_dict[key] for key in selected_keys if key in original_dict}
            lines.append(new_dict)
            processed_bytes += len(line.encode('utf-8'))

            progress_percentage = (processed_bytes / file_size) * 100

            if i % 1000000 == 0:
                print(f"Progress: {progress_percentage:.2f}%")
        with open(json_path, 'a') as f_new:
            json.dump(lines, f_new)
    compress_file(json_path, json_path.replace('.json', '.zst'))
    os.remove(file_json)
    os.remove(json_path)
    return json_path.replace('.json', '.zst'), f"{year}_{month}_{contr_type}.zst"


def main():
    for year in range(2017, 2023):
        for month in range(1, 13):
            month_ = str(month) if month >= 10 else f"0{month}"
            zst_name = f"RC_{year}-{month_}.zst"
            json_name = f"RC_{year}-{month_}.json"
            file_zst = os.path.join(file_path, zst_name)
            file_json = os.path.join(file_path, json_name)
            processed_folder = os.path.splitext(file_json)[0]
            print(processed_folder)
            if not os.path.exists(file_zst) and not os.path.exists(file_json):
                print(f"Downloading {file_zst}")
                res = subprocess.run(f"aws s3 cp s3://reddit-juhcsabi/reddit/comments_zstd/{zst_name} {file_zst}", shell=True)
                if res.returncode == 1:
                    continue
            if not os.path.exists(file_json):
                print(f"Unzipping {file_zst}")
                command = f"zstd -d {file_zst} --long=31 -o {file_json}"
                process = subprocess.Popen(command, shell=True)
                return_code = process.wait()
                print(file_json, return_code)
                os.remove(file_zst)
            zst_path, zst_name = process_file_zip_folder(file_json, year, month,
                                                         "comments")
            return_code = upload_to_local(zst_path)
            if return_code == 0:
                os.remove(zst_path)
            ssh = put(ssh, zst_name)


def upload_to_local(zst_path):
    command = f"pscp -P 22 -pw {pwd} {zst_path} s3049221@spark-head1.eemcs.utwente.nl:"
    powershell_command = f'C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe {command}'
    process = subprocess.Popen(powershell_command, shell=True)
    return_code = process.wait()
    print(zst_path, return_code)
    return return_code


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
    except (TimeoutError, paramiko.SSHException):
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
            files.append(os.path.splitext(file)[0].split("_"))
    print(files)
    return files


if __name__ == '__main__':
    years = [str(i) for i in range(2006, 2012)]
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





