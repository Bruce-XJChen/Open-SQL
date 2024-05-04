# dev数据集对应的数据库文件地址
databases_file_path=''
columns_describe_file_path='./data/columns_describe.json'
columns_enumeration_values_file_path='./data/columns_enumeration_values.json'
table_info_file_path='./data/table_info.json'
schema_D_file_path="./data/schema_D.json"
columns_describe_file_path="./data/columns_describe.json"
# dev数据集对应的tables.json文件存放地址
dev_tables_file_path=""
schema_A_file_path="./data/schema_A.json"


python -u ./src/columns_describe.py     --databases_file_path ${databases_file_path} --columns_describe_file_path ${columns_describe_file_path}
python -u ./src/columns_enumeration.py     --databases_file_path ${databases_file_path} --columns_enumeration_values_file_path ${columns_enumeration_values_file_path}
python -u ./src/table_info.py     --table_info_file_path ${table_info_file_path} --databases_file_path ${databases_file_path}

python -u ./src/schema_D.py     --schema_D_file_path ${schema_D_file_path} --columns_describe_file_path ${columns_describe_file_path} \
                                --databases_file_path ${databases_file_path} \
                                --dev_tables_file_path ${dev_tables_file_path}

python -u ./src/schema_A.py     --schema_A_file_path ${schema_A_file_path} --databases_file_path ${databases_file_path} \
                                --columns_describe_file_path ${columns_describe_file_path} \
                                --columns_enumeration_values_file_path ${columns_enumeration_values_file_path} \
                                --dev_tables_file_path ${dev_tables_file_path}

