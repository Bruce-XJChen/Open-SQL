

# 如何使用进行测试
### 1.安装运行的环境

```
conda create -n open_sql python=3.8
pip install -r requirements.txt
```



### 2. 进行数据预处理

修改processing_data.sh文件

设置db_root_path为dev数据集对应的数据库文件保存地址

设置dev_tables_file_path为test数据集对应的dev_tables.json文件存放地址

```sh
 sh ./scripts/process_data.sh
```



### 3. 进行推理

修改infer.sh文件

设置result_file="./output/predict_dev.json"

设置dev_data_path为test数据集对应的json文件保存地址

设置dev_table_info_file_path为test数据集对应的test_tables.json文件存放地址

```sh
 sh ./scripts/infer.sh
```



### 4.评估

###### 4.1EX 评估

修改run_evaluation.sh文件

设置data_mode 为dev

设置db_root_path为dev数据集对应的数据库文件地址

设置ground_truth_path为dev数据集对应真实sql的json文件对应的存放目录

设置diff_json_path为dev数据集对应的json文件保存地址

     sh ./scirpts/run_evaluation.sh
###### 4.2 VES 评估

修改run_evaluation_ves.sh文件

设置db_root_path为dev数据集对应的数据库文件地址

设置ground_truth_path为dev数据集对应真实sql的json文件对应的存放目录

设置diff_json_path为dev数据集对应的json文件保存地址

     sh ./scirpts/run_evaluation_ves.sh