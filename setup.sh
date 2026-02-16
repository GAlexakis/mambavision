git clone https://github.com/NVlabs/MambaVision.git

pipx install uv

uv venv --python=3.10 --seed

source .venv/bin/activate

uv pip install torch==2.5 torchvision==0.20

wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post2/mamba_ssm-2.2.6.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install mamba_ssm-2.2.6.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

cp setup_to_replace.py ./MambaVision/setup.py
cd MambaVision
uv pip install . --no-build-isolation