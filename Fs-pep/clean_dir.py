import shutil, glob 

omm_list = open('scheduler_logs/openmm_log.txt', 'r') 
file_list = glob.glob('omm_run*') 
omm_info = [line.split() for line in omm_list.readlines()]

for info in omm_info: 
    if int(info[1]) < 1000: 
        if info[0][:-13] in file_list: 
            print 'deleting', info[0][:-13] 
            try: 
                shutil.rmtree(info[0][:-13]) 
            except OSError: 
                print 'Couldn\'t delete %s, passing' % info[0][:-13]
