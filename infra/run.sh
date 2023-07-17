VENV=~/venv310_nohf
# if the venv doesn't exist, make it
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv at $VENV"
    python3.10 -m venv $VENV
fi

source $VENV/bin/activate

pip install -U pip
pip install -U wheel

# jax
pip install -U "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install matplotlib==3.7.1
pip install midi2audio==0.1.1
pip install mido==1.2.10
pip install numpy==1.22.4
pip install torch==2.0.1
pip install tqdm==4.65.0
pip install huggingface-hub==0.15.1

ANT_ROOT=$(dirname "$(readlink -f $0)")/..

echo $VENV > anticipation-kathli/infra/venv_path.txt

cd transformers-levanter
pip install -e .

cd $ANT_ROOT
cd anticipation-kathli

pip install -e .

umask 000

PYTHONPATH=${ANT_ROOT}:${ANT_ROOT}/scripts:${ANT_ROOT}/tests:$PYTHONPATH "$@"