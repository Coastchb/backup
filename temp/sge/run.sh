
#qsub -V -cwd -t 1:50 -o stdout -l ram_free=8G -j yes sub.sh  # run on all.q (default queue)
### -V: export all the environment variables(to sub.sh), so sub.sh can call any environment variable as current qsub.


#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q all.q@dear-B85-D3V-A sub.sh 	#run only on dear-B85-D3V-A host of all.q

qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q 102.q sub.sh
#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q 102.q,140.q sub.sh #run on two queues:102.q and 140.q
#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q sub.q sub.sh  # run on sub.q (self-defined queue)


