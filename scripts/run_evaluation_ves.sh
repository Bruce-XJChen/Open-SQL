# dev数据集对应数据库文件存放目录
db_root_path=''
data_mode='dev'
predicted_sql_path='./output/'
# dev数据集真实sql的json文件对应的存放目录
ground_truth_path=''
# dev数据集文件
diff_json_path=""
num_cpus=16
time_out=60
mode_gt='gt'
mode_predict='gpt'

python -u ./src/evaluation_ves.py \
    --db_root_path ${db_root_path} \
    --predicted_sql_path ${predicted_sql_path} \
    --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} \
    --num_cpus ${num_cpus} \
    --time_out ${time_out} \
    --mode_gt ${mode_gt} \
    --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path}
