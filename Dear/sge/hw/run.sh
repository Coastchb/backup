
qsub -v PATH -cwd -t 1:50 -o stdout -j yes sub.sh  # run on all.q (default queue)

#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q all.q@dear-B85-D3V-A sub.sh 	#run only on dear-B85-D3V-A host of all.q

#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q 102.q,140.q sub.sh #run on two queues:102.q and 140.q
#qsub -v PATH -cwd -t 1:50 -o stdout -j yes -q sub.q sub.sh  # run on sub.q (self-defined queue)


