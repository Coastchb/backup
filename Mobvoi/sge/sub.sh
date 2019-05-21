date > log/$SGE_TASK_ID.log
sleep ${SGE_TASK_ID}s
echo "sleep for ${SGE_TASK_ID}s" >> log/$SGE_TASK_ID.log
date >> log/$SGE_TASK_ID.log
