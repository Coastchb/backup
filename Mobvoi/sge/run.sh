#qsub -tc 2 -q g.q -t 1:10 sub.sh
#qsub -tc 2 -t 1:10 -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/sub.log sub.sh >> exp/sub.log 2>&1
. cmd.sh

export PATH=$PATH:`pwd`

$train_cmd --max-jobs-run 20 JOB=1:40 log/sub.JOB.log \
bash sub.sh || exit 1

$train_cmd --max-jobs-run 30 JOB=1:40 log/SUB.JOB.log \
bash sub.sh || exit 1

