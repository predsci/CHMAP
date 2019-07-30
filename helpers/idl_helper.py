"""
Helper module to start and interact with an SSW IDL process for instrument specific data
reduction routines.

- I use the basic functionality of the subprocess module and my routines ARE NOT ROBUST AT ALL

- Eventually I'd like to phase out SSW completely, but for now I need it for the SECCHI routines.

- SSW must be invoked by calling a script (ssw_exe) that invokes SSWIDL in command line mode

- The subprocess interaction is based off of a nice example given by this blog by Eli Bendersky:
  https://eli.thegreenplace.net/2017/interacting-with-a-long-running-child-process-in-python/

- If something goes wrong, the threading stuff can cause python to hang, just hit CTRL-C to
  interrupt those kind of processes. As far as i can tell once the python exe dies it kills
  the related routines.

"""
import subprocess
from subprocess import PIPE
import threading
import queue
import time
import os
import signal

# ssw_exe='/Applications/exelis/idl85/bin/idl'
ssw_exe = '/Users/cdowns/ssw.csh'

sleep_count = 0.1

# maximum run time in seconds for any command
max_cmdtime = 600.0


def output_reader(process, output_queue):
    """
    Read the pipe of the subprocess (a POPEN class)
    This pipe is interacted with via the queue/threading modules (queue variable: output_queue)
    This routine looks through all unread output lines and places them in output_queue
    """
    for line in iter(process.stdout.readline, b''):
        output_queue.put(line.decode('utf-8'))


def error_reader(process, error_queue):
    """
    Read the pipe of the subprocess (a POPEN class)
    This pipe is interacted with via the queue/threading modules (queue variable: output_queue)
    This routine looks through all unread output lines and places them in output_queue
    """
    for line in iter(process.stderr.readline, b''):
        error_queue.put(line.decode('utf-8'))


class Session:
    """
    Class to start and interact with an IDL session
    """

    def __init__(self, verbose=False):

        print("### IDL: Starting up IDL process!")
        print("  exe: ", ssw_exe)

        self.proc = subprocess.Popen([ssw_exe], preexec_fn=os.setsid,
                                     stdin=PIPE,
                                     stdout=PIPE,
                                     stderr=PIPE)

        # setup monitoring stdout
        self.outq = queue.Queue()
        self.outthread = threading.Thread(target=output_reader, args=(self.proc, self.outq))
        self.outthread.start()

        # setup monitoring stderr
        self.errq = queue.Queue()
        self.errthread = threading.Thread(target=error_reader, args=(self.proc, self.errq))
        self.errthread.start()

        self.lastline = None
        self.lastline_err = None

        self.startkey = 'PYTHON_IDL_CMD: START'
        self.startcmd = 'print, "' + self.startkey + '"'
        self.startstring = self.startkey + '\n'

        self.donekey = 'PYTHON_IDL_CMD: FINISHED'
        self.donecmd = 'print, "' + self.donekey + '"'
        self.donestring = self.donekey + '\n'

        time.sleep(sleep_count)
        if self.proc.poll() is None:
            print("  pid: ", self.proc.pid)
        else:
            print(" UNSUCCESSFUL! return code: " + self.proc.poll())

        # now make sure it started up correctly
        self.run('print, "Done with startup"', quiet=True)
        # self.monitor()

        # flush the error buffer
        self.flush_err(quiet=True)

    def run(self, cmd, asynchronous=False, quiet=False):
        """
        Run a single IDL command in the session.
        cmd: is the string with the command text.
        """

        if not quiet:
            print('### IDL: Running Command: ' + cmd)

        # Send the command (flush is needed to make sure it goes through to IDL)
        self.proc.stdin.write((self.startcmd + '\n ' + cmd + '\n ' + self.donecmd + '\n').encode('utf-8'))
        self.proc.stdin.flush()
        time.sleep(sleep_count)

        # Reset the mechanism for tracking command completion
        self.lastline = False

        # Monitor the status of the command, wait for it to finish
        if not asynchronous:
            self.monitor(quiet=quiet)

        # self.proc.stdin.write((self.startcmd+'\n').encode('utf-8'))
        # self.proc.stdin.flush()
        # self.proc.stdin.write((cmd+'\n').encode('utf-8'))
        # self.proc.stdin.flush()
        # self.proc.stdin.write((self.donecmd+'\n').encode('utf-8'))
        # self.proc.stdin.flush()

    def monitor(self, quiet=False):
        """
        Monitor a process by checking buffers for new information while a process is running.
        Exit when the command appears to have been completed or hits max wait time.
        """
        # dump out the error buffer in case there is defunct info here
        self.flush_err(quiet=quiet)

        # dump out remaining stdout buffer
        self.flush(quiet=quiet)

        # wait for last line of output to be command completion
        t_start = time.perf_counter()
        while self.lastline != self.donestring:
            self.flush(quiet=quiet)
            time.sleep(sleep_count)
            if max_cmdtime < time.perf_counter() - t_start:
                raise Exception(" This IDL command exceeded the maximum time allowed\n'+"
                                "  it might have hung? use the end() method to kill it")

        # dump out the error buffer in case there is new info here
        self.flush_err(quiet=quiet)

    def flush(self, quiet=False, loud=False):
        """
        print the buffered output of the running IDL process.
        """
        empty = 0
        while empty == 0:
            try:
                self.lastline = self.outq.get(block=False)
                if not quiet:
                    print('IDL> {0}'.format(self.lastline), end='')
            except queue.Empty:
                empty = 1
                if loud:
                    print('could not get line from queue')

    def flush_err(self, quiet=False, loud=False):
        """
        Print the buffered error output of the running IDL process.
        """
        empty = 0
        while empty == 0:
            try:
                self.lastline_err = self.errq.get(block=False)
                if not quiet:
                    print('IDL> {0}'.format(self.lastline_err), end='')
            except queue.Empty:
                empty = 1
                if loud:
                    print('could not get line from queue')

    def test(self):
        """
        Test the idl session with a simple print command
        """
        self.proc.stdin.write(b'print, "BEGIN"\n print, findgen(5)\n print, "END"\n')
        self.proc.stdin.flush()  # without this it hangs...
        time.sleep(sleep_count)

        self.flush()

    def end(self):
        """
        End the IDL process.
        """
        if self.proc.poll() is None:
            print("### IDL: Killing Process")

            # first try to exit gracefully
            self.run('exit', asynchronous=True)
            time.sleep(sleep_count)

            if self.proc.poll() is None:
                # basic terminate command if no children
                # self.proc.terminate()

                # Use this command to kill the entire group of processes if SSW launched from a script
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                time.sleep(sleep_count)

            print("  Return code", self.proc.poll())
        else:
            print("  IDL Process already dead, return code: ", self.proc.poll())

    def secchi_prep(self, infile, outfile):
        """
        Run the SSW IDL commands necessary for prepping a STEREO EUVI image from lvl 0.5 to 1.0

        - Because PSF deconvolution needs to happen AFTER calibration but BEFORE rotation and/or scaling
          we do not rotate the image with secchi_prep. This will be done in using sunpy map objects.

        - This can take a while for some dates because the SPICE kernels are sometimes needed to be initialized.
          --> it may be better down the road to call arrays of images (10-100+) vs one at a time.
        """
        # set the IDL variables
        self.run('InFile="' + infile + '"')
        self.run('OutFile="' + outfile + '"')

        # prep the EUVI image file
        self.run('secchi_prep, InFile, Hdr, Im')

        # write out the uncompressed result.
        self.run('mwritefits, OutFile=OutFile, Hdr, Im')

        # use these commands if just testing reading/writing.
        #self.run('mreadfits, InFile, Hdr, Im')
        #self.run('writefits, OutFile, Im, Hdr')
        #self.run('mwritefits, OutFile=OutFile, Hdr, Im')
        #self.run('write_hdf_2d, OutFile, findgen(2048), findgen(2048), Im, iErr')
