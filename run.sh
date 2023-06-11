time=$(date "+%m%d-%H%M")
tag=$1
key=$2
resultPath="./work_dir"

if [ ! $tag ]; then
  echo "Usage: ./run.sh [tag] [key]"
  exit
fi

if [ ! $key ]; then
  key="PDU7213T0doLjdaekgRVhljn5i809caCuC8VFcDp"
fi

if [ ! -w "$resultPath" ]; then
  mkdir $resultPath
  mkdir "$resultPath/logs"
fi

if [ ! -w "$resultPath/logs" ]; then
  mkdir "$resultPath/logs"
fi

if [ $3 ]; then
  python -u main.py --task ${tag} \
    --train /var/app/dataset/sketch_transformer_dataset/index_train.json \
     --test /var/app/dataset/sketch_transformer_dataset/index_test.json \
     --cache /hy-tmp/cache \
     --batch_size 40
  # End
  time=$(date "+%m%d-%H%M%S")
  echo "Run End: $time"
  # 压缩包名称
  file="$tag-$time.zip"
  # 把 result 目录做成 zip 压缩包
  zip -q -r "${file}" ${resultPath} -x "*/checkpoints/*"
  oss cp "${file}" oss://result/
  curl "https://api2.pushdeer.com/message/push?pushkey=$key&text=$1完成训练"
  echo "Shutdown!"
  shutdown
else
  # 杀掉其他进程
  pid=$(top -b -n1 | grep 'python' | head -1 | awk '{print $1}')
  if [ $pid ]; then
    kill "$pid"
    echo "Killed $pid"
  else
    echo "NOT FOUND OTHER PROCESS"
  fi

  nohup ./run.sh $tag $key 1 > ${resultPath}/logs/$tag-${time}.log 2>&1 &

  echo "Log File: ${resultPath}/logs/$tag-${time}.log"

  sleep 3

  tail -f ${resultPath}/logs/$tag-${time}.log

fi
