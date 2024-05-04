
"""
获取列对应的描述信息，并保存在一个json文件中
"""
import json
import os
import sqlite3
import pandas as pd
import argparse


def get_dev_describe(root_path, db_name, table_name):
    table_csv_path = root_path + "/" + db_name + "/database_description" + "/" + table_name + ".csv"
    instruction = {}
    if not os.path.exists(table_csv_path):
        return instruction
    df = pd.read_csv(table_csv_path, encoding='unicode_escape')
    df.dropna()
    df.fillna('undefine')
    for index, row in df.iterrows():
        key1 = [key for key in row.keys() if 'column_name' in key]
        key2 = [key for key in row.keys() if 'column_description' in key]
        # key2 = [key for key in row.keys() if 'data_format' in key]
        # key2 = [key for key in row.keys() if 'column_name' in key]
        # print(key1[0])
        cname = str(row[key1[0]])
        # cname = row['column_name'] if pd.isna('column_name') else row[key1[0]]
        if type(row[key2[0]]) == str:
            describe = row[key2[0]].strip("")
        else:
            describe = ""
        instruction[cname] = describe
    return instruction

if __name__ == "__main__":
    print("processing columns describe...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--databases_file_path', type=str, required=True)
    parser.add_argument('--columns_describe_file_path', type=str, required=True)
    args = parser.parse_args()
    database_names = [f for f in os.listdir(args.databases_file_path) if os.path.isdir(os.path.join(args.databases_file_path, f))]
    columns_describe_data = {}
    for idx, database_name in enumerate(database_names):
        columns_describe_data[database_name] = {}
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
        for t_idx, table in enumerate(tables):
            table_name = table[0]
            if table_name == "sqlite_sequence":
                continue
            columns_describe_data[database_name][table_name] = {}
            # 获取表对应的所有描述信息
            describe_info = get_dev_describe(args.databases_file_path, database_name, table_name)
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            # 获取查询结果
            columns = cursor.fetchall()
            for c_idx, column in enumerate(columns):
                column_name = column[1]
                col_describe_info = describe_info.get(column_name, "")
                if column_name.lower() == col_describe_info.lower():
                    col_describe_info = ""
                columns_describe_data[database_name][table_name][column_name] = col_describe_info.strip()

    with open(args.columns_describe_file_path, 'w') as json_file:
        json.dump(columns_describe_data, json_file, ensure_ascii=False)