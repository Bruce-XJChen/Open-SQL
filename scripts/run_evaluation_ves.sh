# dev���ݼ���Ӧ���ݿ��ļ����Ŀ¼
db_root_path=''
data_mode='dev'
predicted_sql_path='./output/'
# dev���ݼ���ʵsql��json�ļ���Ӧ�Ĵ��Ŀ¼
ground_truth_path=''
# dev���ݼ��ļ�
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
