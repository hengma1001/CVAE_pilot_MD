# Running the code

1. Prior to every run, if you want to clean up the jobs on the GPU, and stop the running `rabbitmq-server` and `celery` workers, run the bash script 

   ```bash 
   bash prerun_clean.sh
   ```

2. The simulation will reuse the trajectories from the directories, of which the name starts with `omm_run`. You need to move or remove them if you don't want to use them. There is short script will take some information from previous log file, which recorded the length of the trajectories under those folders. It can remove those short trajectories, which have less than 1000 frames.  

   ```python 
   python clean_dir.py
   ```

3. After everything is clear, the program can be run with 

   ```python 
   python cvae_md.py
   ```


