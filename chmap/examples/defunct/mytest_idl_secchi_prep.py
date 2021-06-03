from chmap.utilities.file_io import io_helpers
from chmap.utilities.idl_connect import idl_helper
import time
import os.path
from chmap.settings.app import App

fits_compressed = os.path.join(App.APP_HOME,'reference_data','sta_euvi_20130203T000530_195.fits')
fits_uncompressed = os.path.join(App.TMP_HOME, 'tmp_euvi_uncompressed.fits')
fits_prepped = os.path.join(App.TMP_HOME, 'tmp_euvi_prepped.fits')
hdf_prepped = os.path.join(App.TMP_HOME, 'tmp_euvi_prepped.hdf')


print(fits_compressed)
print(fits_uncompressed)
print(hdf_prepped)

io_helpers.uncompress_compressed_fits_image(fits_compressed, fits_uncompressed, int=True)

idl_session = idl_helper.Session()

t1 = time.perf_counter()
idl_session.secchi_prep(fits_uncompressed, fits_prepped)
t2 = time.perf_counter()
print(t2 - t1)

idl_session.test()

idl_session.run('wait, 0.1')

idl_session.end()

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
