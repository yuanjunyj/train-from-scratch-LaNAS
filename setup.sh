pip install jsonpickle;
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html;
pip install pillow!=8.3.0;
git clone https://github.com/NVIDIA/apex;
cd "apex";
python setup.py install;
cd ".."
