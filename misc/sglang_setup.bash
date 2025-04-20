python3.10 -m venv .sglang-venv   # (or whatever path)
. ./sglang-venv/bin/activate
pip install -U 'pip<25'
pip install -U setuptools packaging wheel cmake ninja 'numpy<2'
pip install -U 'sglang[all]' --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
