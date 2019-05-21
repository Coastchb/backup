. conf.sh

mkdir -p $target_feat_dir $target_lab_dir $batch_dir

python get_batchs.py $raw_dir $job_num $batch_dir


qsub -V -cwd -t 1:$job_num -o stdout -l ram_free=8G,q=140.q -j  yes sub.sh