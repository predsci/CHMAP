import subprocess
from subprocess import PIPE
import threading
import queue
import time



"""
# THESE commands illustrate how to monitor a subprocess and interact with it
# I learned this from the nice example given by this blog by Eli Bendersky:
# https://eli.thegreenplace.net/2017/interacting-with-a-long-running-child-process-in-python/

def output_reader(proc, outq):
    for line in iter(proc.stdout.readline, b''):
        outq.put(line.decode('utf-8'))


proc = subprocess.Popen(["/Applications/exelis/idl85/bin/idl"],
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=PIPE)

outq = queue.Queue()
t = threading.Thread(target=output_reader, args=(proc, outq))
t.start()

proc.stdin.write(b'print, findgen(10)\n')
proc.stdin.flush()  # without this it hangs...
time.sleep(0.2)

empty=0
while empty == 0:
    try:
        line = outq.get(block=False)
        print('got line from outq: {0}'.format(line), end='')
    except queue.Empty:
        print('could not get line from queue')
        empty=1







proc = subprocess.Popen(["/Users/cdowns/ssw.csh"],universal_newlines=True)




def output_reader(proc):
    for line in iter(proc.stdout.readline, b''):
        print('got line: {0}'.format(line.decode('utf-8')), end='')



output_reader(proc)


print(proc.stdout.readline())

proc.stdin.write(b'print, 2+2\n')
proc.stdin.flush()  # without this it hangs...
print(proc.stdout.readline())
"""



"""
p = subprocess.Popen(["ls", "-l"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
p.communicate()

p = subprocess.Popen(["ls", "-l"], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

p = subprocess.Popen("ls -l", shell=True)
"""
