# 模型存放地址
model_name_or_path="./model/CodeLlama-7b-hf/"
# 日志文件路径
log_file="./log.txt"
# 推理结果的文件路径
result_file="./output/predict_dev.json"
# dev数据文件
dev_data_path=""

schema_D_info_file_path="./data/schema_D.json"
schema_A_info_file_path="./data/schema_A.json"
table_info_file_path="./data/table_info.json"
dev_table_info_file_path=""

CUDA_VISIBLE_DEVICES=0 python ./src/infer.py \
    --model_name_or_path ${model_name_or_path} \
    --ckpt_path ./model/adapter_model/ \
	  --log_file  ${log_file} \
	  --result_file ${result_file} \
	  --dev_data_path ${dev_data_path} \
	  --schema_D_info_file_path ${schema_D_info_file_path} \
	  --schema_A_info_file_path ${schema_A_info_file_path} \
	  --table_info_file_path ${table_info_file_path} \
	  --dev_table_info_file_path ${dev_table_info_file_path}
