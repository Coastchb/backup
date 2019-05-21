# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

host_opt="-q mobvoi-queue@@rheahosts"
host_opt="-l h=*rhea*"
host_opt=

export train_cmd="queue.pl -q mobvoi-queue --max_jobs_run 20" #@mobvoi-maryland -q mobvoi-queue@mobvoi-rhea-01 -q mobvoi-queue@mobvoi-rhea-02 -q mobvoi-queue@mobvoi-rhea-05"
export decode_cmd="queue.pl -q mobvoi-queue" #@mobvoi-rhea-01 -q mobvoi-queue@mobvoi-rhea-02 -q mobvoi-queue@mobvoi-rhea-05" # --mem 4G" # $host_opt"
export mkgraph_cmd="queue.pl -q mobvoi-queue" # --mem 8G" # $host_opt"
export cuda_cmd="queue.pl -q g.q" #@mobvoi-rhea-02 -q g.q@mobvoi-rhea-05" # --gpu 1"
export other_cmd="queue.pl -q mobvoi-queue" # $host_opt"

