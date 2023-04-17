# shellcheck disable=SC2046
echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8" #Change or update python as per need
conda create --prefix ./env python=3.8 -y
echo [$(date)]: "activate env"
source activate ./env
echo [$(date)]: "intalling requirements.txt"
pip install -r requirements.txt
echo [$(date)]: "END"
