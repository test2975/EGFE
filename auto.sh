#! /home/d3-server1/anaconda3/envs/cynenv/bin/python3
flag=1
result=1
echo "training script is started"
while [ "$flag" -eq 1 ]
do
    sleep 1s
    PID=1599477
    PID_EXIST=$(ps u | awk '{print $2}'| grep -w $PID)
    if [ ! $PID_EXIST ]; then
        echo $(date +%F%n%T)
        echo "process is finished"
        flag=0
        sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
    fi
done
# /media/ssd/cyn-workspace/envs/bin/python
torchrun --nnodes 1 --nproc_per_node 4  main_ddp.py --batch_size 10 --lr 5e-4 --task set-weight
# /home/d3-server1/anaconda3/envs/cynenv/bin/python3 main.py --task WFF --aggregation sum