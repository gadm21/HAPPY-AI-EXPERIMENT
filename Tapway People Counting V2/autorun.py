from subprocess import Popen
import sys
import datetime
import os

filename = sys.argv[1]

try:
    while True:
        # print("\nStarting " + filename)
        # p = Popen("python3 " + filename, shell=True)    
        # timeout
        print('start')
        p = Popen([sys.executable, filename])
        print('wait')
        p.wait()
        print('end')
except:
    print('Terminate long running script')
p.terminate()