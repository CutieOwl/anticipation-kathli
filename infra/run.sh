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

echo $VENV > anticipation-kathli/infra/venv_path.txt

cd anticipation-kathli

pip install -e .

umask 000

ANT_ROOT=$(dirname "$(readlink -f $0)")/..

PYTHONPATH=${ANT_ROOT}:${ANT_ROOT}/scripts:${ANT_ROOT}/tests:$PYTHONPATH "$@"