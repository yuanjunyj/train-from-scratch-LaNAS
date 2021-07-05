# Requirements

Cuda version: cu111

Required Libraries: Pytorch & Apex 

For Pytorch:

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
或者
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
For Apex:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
(apex这一句我没成功安装，如果报错的话可以改为下面这一句，但是似乎性能会差一点)
python setup.py install
```

## Starting the server

Here are the steps to start:
1. go to server folder, unzip search_space.zip.
2. ifconfig get your ip address
3. change the line 227 in MCTS.py
```
address = ('166.111.81.72', 13237), # replace to your ip address and your port
```
4. To start the server, ``` python MCTS.py & ```.

## Starting the clients
Once the server starts running, here is what you need to start clients.
1. go to clientX folder, open client.py
2. change line 25, line 76, line 114 to <b>the server's ip address</b>.
3. change the line 7 in launch_clients.sh to the available GPUs, each GPU corresponds to one client process.
4. run the script launch_clients.sh

By default, if there are 16 GPUs, the script will create folders from client0 to client15, and run 16 clients for the search.
