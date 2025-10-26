apt install -y python3.12-venv
python3 -m venv libs/modern-bert
source libs/modern-bert/bin/activate
sudo apt-get install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install -r requirements-training.txt
export HF_HOME=/scratch/
python3 train_encoder.py