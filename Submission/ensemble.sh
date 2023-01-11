python /opt/ml/input/Submission/ensemble.py \
--file_list \
"['/opt/ml/input/Submission/Csv_files/output (5).csv',
'/opt/ml/input/Submission/Csv_files/output (6).csv',
'/opt/ml/input/Submission/Csv_files/output (7).csv',
'/opt/ml/input/Submission/Csv_files/output (3).csv',
'/opt/ml/input/Submission/Csv_files/output (4).csv']" \
--output_file_name 'test' \
--method 'majority' \
# --Weight '[5.3, 7.3, 6.8]' 