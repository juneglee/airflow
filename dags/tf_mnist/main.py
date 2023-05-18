import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import timedelta
import os

source_dir = '__source_dir__'
data_dir = "__data_dir__"

data_filename = "mnist.npz"
data_load_filename = "data_load.py"
train_filename = "train.py"
evaluate_filename = "evaluate.py"
model_foldername = "models/mnist"
log_filename = "training-result.log"


train_data_path = os.path.join(
    data_dir,
    "mnist",
    "train",
    data_filename,
)
test_data_path = os.path.join(
    data_dir,
    "mnist",
    "test",
    data_filename,
)
model_path = os.path.join(
    data_dir,
    model_foldername,
)
log_path = os.path.join(
    data_dir,
    log_filename
)
t1_data_load_path = os.path.join(
    source_dir,
    data_load_filename,
)
t2_train_path = os.path.join(
    source_dir,
    train_filename
)
t3_evaluate_path = os.path.join(
    source_dir,
    evaluate_filename
)

t1_command = f"python {t1_data_load_path} --train_data_path {train_data_path} --test_data_path {test_data_path}"
t2_command = f"python {t2_train_path} --data_path {train_data_path} --model_path {model_path}"
t3_command = f"python {t3_evaluate_path} --data_path {test_data_path} --model_path {model_path} --log_path {log_path}"

default_args = {
    'start_date': airflow.utils.dates.days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'airflow-mnist',
    default_args=default_args,
    description='Training mnist dag',
    schedule_interval=None,
    dagrun_timeout=timedelta(minutes=20)
)
t1 = BashOperator(
    task_id='data_load',
    bash_command=t1_command,
    dag=dag,
)

t2 = BashOperator(
    task_id='train',
    bash_command=t2_command,
    dag=dag,
)
t3 = BashOperator(
    task_id='evaluate',
    bash_command=t3_command,
    dag=dag,
)
t1 >> t2 >> t3