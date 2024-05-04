import argparse
import json
import os
import sqlite3

print("processing table info...")
parser = argparse.ArgumentParser()
parser.add_argument('--databases_file_path', type=str, required=True)
parser.add_argument('--table_info_file_path', type=str, required=True)
args = parser.parse_args()

database_names = [f for f in os.listdir(args.databases_file_path) if os.path.isdir(os.path.join(args.databases_file_path, f))]
database_data = {}
for idx, database_name in enumerate(database_names):
    database_data[database_name] = {}
    database_data[database_name]["tables"] = []
    # 获取数据库对应的文件路径
    database_file = args.databases_file_path + "/" + database_name + "/" + database_name + ".sqlite"
    # 连接数据库
    conn = sqlite3.connect(database_file)
    # 创建一个游标对象
    cursor = conn.cursor()
    # 查询 sqlite_master 表以获取数据库中所有表的定义语句
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    # 获取查询结果
    tables = cursor.fetchall()
    # 遍历每个表，查询其列信息
    for t_idx, table in enumerate(tables):
        table_name = table[0]
        if table_name == "sqlite_sequence":
            continue
        database_data[database_name]["tables"].append(table_name)
        # 执行 PRAGMA table_info 查询
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        # 获取查询结果
        columns = cursor.fetchall()
        database_data[database_name][table_name] = []
        for c_idx, column in enumerate(columns):
            column_name = column[1]
            database_data[database_name][table_name].append(column_name)
with open(args.table_info_file_path, 'w') as json_file:
    json.dump(database_data, json_file)