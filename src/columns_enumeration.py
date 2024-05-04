"""
获取列对应的枚举值，并保存在一个json文件中
"""
import json
import os
import sqlite3
import argparse

def get_enumeration_values(database_file, table_name, column_name, data_type):
    # 连接数据库
    conn = sqlite3.connect(database_file)
    # 创建一个游标对象
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL")

    # 获取所有不同的取值
    distinct_values = cursor.fetchall()
    if len(distinct_values) == 0:
        return ""
    distinct_values = [value[0] if type(value) == tuple else value for value in distinct_values]
    if "int" in data_type:
        if 2 <= len(distinct_values) <= 3:
            value_info = "(" + distinct_values.__str__().strip("[").strip("]") + ")"
            return value_info
    else:
        sorted_distinct_values = sorted(distinct_values, key=len, reverse=True)
        if 2 <= len(distinct_values) <= 5:
            if len(sorted_distinct_values[0]) <= 15:
                value_info = "(" + distinct_values.__str__().strip("[").strip("]") + ")"
                return value_info
    return ""


if __name__ == "__main__":
    print("processing columns enumeration...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--databases_file_path', type=str, required=True)
    parser.add_argument('--columns_enumeration_values_file_path', type=str, required=True)
    args = parser.parse_args()
    # databases_file = "/public14_data/wtl/work_point/open_sql_v1/datasets/bird/dev/dev_databases"
    database_names = [f for f in os.listdir(args.databases_file_path) if os.path.isdir(os.path.join(args.databases_file_path, f))]
    # columns_enumeration_values_file = "/public14_data/wtl/work_point/open_sql_v1/datasets/bird/dev/schema/columns_enumeration_values.json"
    columns_enumeration_values_data = {}
    for idx, database_name in enumerate(database_names):
        print(f"database_name:{idx}")
        columns_enumeration_values_data[database_name] = {}
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
            print(f"\ttables:{t_idx}")
            table_name = table[0]
            if table_name == "sqlite_sequence":
                continue
            columns_enumeration_values_data[database_name][table_name] = {}
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            # 获取查询结果
            columns = cursor.fetchall()
            for c_idx, column in enumerate(columns):
                print(f"\t\tcolumns:{c_idx}")
                column_name = column[1]
                data_type = column[2].lower()
                if "char" in data_type or data_type == "text" or "int" in data_type:
                    values_info = get_enumeration_values(database_file, table_name, column_name, data_type)
                else:
                    values_info = ""
                columns_enumeration_values_data[database_name][table_name][column_name] = values_info

    with open(args.columns_enumeration_values_file_path, 'w') as json_file:
        json.dump(columns_enumeration_values_data, json_file, ensure_ascii=False)