rm -rf "checkpoint"
mkdir "checkpoint"
cd "server"
unzip -o "search_space.zip"
nohup python MCTS.py >"../OUTPUT/server.txt" 2>&1 &
# nohup python MCTS.py &
sleep 5m
cd ".."

GPUS=`nvidia-smi -L | wc -l`
for (( c=0; c < $GPUS; c++ ))
do
   echo "---------------------------------"
   echo $PWD
   rm -rf "client$c"
   cp -rf "clientX" "client$c"
   cd "client$c"
   echo $PWD
   CUDA_VISIBLE_DEVICES=$c nohup python client.py >"../OUTPUT/"$c".txt" 2>&1 &
   # nohup echo "$c" &
   # touch $!".pid"
   cd ".."
   echo "$PWD"
   sleep 30s
done

python query.py >"./OUTPUT/query_result.txt"

