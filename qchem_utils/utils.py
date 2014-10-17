#
# Copyright 2014 Lee-Ping Wang
#

import os
import re
import sys
from select import select
import subprocess

__all__ = ['_exec']


def _exec(command, print_to_screen=False, outfnm=None, logfnm=None, stdin="",
          print_command=True, copy_stdout=True, copy_stderr=False, persist=False,
          expand_cr=False, print_error=True, rbytes=1, cwd=None, **kwargs):
    """Runs command line using subprocess, optionally returning stdout.
    Options:
    command (required) = Name of the command you want to execute
    outfnm (optional) = Name of the output file name (overwritten if exists)
    logfnm (optional) = Name of the log file name (appended if exists)
    stdin (optional) = A string to be passed to stdin, as if it were typed (use newline character to mimic Enter key)
    print_command = Whether to print the command.
    copy_stdout = Copy the stdout stream; can set to False in strange situations
    copy_stderr = Copy the stderr stream to the stdout stream; useful for GROMACS which prints out everything to stderr (argh.)
    expand_cr = Whether to expand carriage returns into newlines (useful for GROMACS mdrun).
    print_error = Whether to print error messages on a crash. Should be true most of the time.
    persist = Continue execution even if the command gives a nonzero return code.
    rbytes = Number of bytes to read from stdout and stderr streams at a time.  GMX requires rbytes = 1 otherwise streams are interleaved.  Higher values for speed.
    """

    # Dictionary of options to be passed to the Popen object.
    cmd_options = {
        'shell': (type(command) is str),
        'stdin': subprocess.PIPE,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'universal_newlines': expand_cr,
        'cwd': cwd
    }

    # If the current working directory is provided, the outputs will be
    # written to there as well.
    if cwd is not None:
        if outfnm is not None:
            outfnm = os.path.abspath(os.path.join(cwd, outfnm))
        if logfnm is not None:
            logfnm = os.path.abspath(os.path.join(cwd, logfnm))

    # "write to file" : Function for writing some characters to the log and/or output files.
    def wtf(out):
        if logfnm != None:
            with open(logfnm, 'a+') as f:
                f.write(out)
                f.flush()
        if outfnm != None:
            with open(outfnm, 'w+' if wtf.first else 'a+') as f:
                f.write(out)
                f.flush()
        wtf.first = False
    wtf.first = True

    # Preserve backwards compatibility; sometimes None gets passed to stdin.
    if stdin == None:
        stdin = ""

    if print_command:
        print("Executing process: \x1b[92m%-50s\x1b[0m%s%s%s\n" % (' '.join(command) if type(command) is list else command,
                                                                   " Output: %s" % outfnm if outfnm != None else "",
                                                                   " Append: %s" % logfnm if logfnm != None else "",
                                                                   (" Stdin: %s" % stdin.replace('\n', '\\n')) if stdin else ""))
        wtf("Executing process: %s%s\n" % (
            command, (" Stdin: %s" % stdin.replace('\n', '\\n')) if stdin else ""))

    cmd_options.update(kwargs)
    p = subprocess.Popen(command, **cmd_options)

    # Write the stdin stream to the process.
    p.stdin.write(stdin)
    p.stdin.close()

    #===============================================================#
    #| Read the output streams from the process.  This is a bit    |#
    #| complicated because programs like GROMACS tend to print out |#
    #| stdout as well as stderr streams, and also carriage returns |#
    #| along with newline characters.                              |#
    #===============================================================#
    # stdout and stderr streams of the process.
    streams = [p.stdout, p.stderr]
    # These are functions that take chunks of lines (read) as inputs.

    def process_out(read):
        if print_to_screen:
            sys.stdout.write(read)
        if copy_stdout:
            process_out.stdout.append(read)
            wtf(read)
    process_out.stdout = []

    def process_err(read):
        if print_to_screen:
            sys.stderr.write(read)
        process_err.stderr.append(read)
        if copy_stderr:
            process_out.stdout.append(read)
            wtf(read)
    process_err.stderr = []
    # This reads the streams one byte at a time, and passes it to the LineChunker
    # which splits it by either newline or carriage return.
    # If the stream has ended, then it is removed from the list.
    with LineChunker(process_out) as out_chunker, LineChunker(process_err) as err_chunker:
        while True:
            to_read, _, _ = select(streams, [], [])
            for fh in to_read:
                if fh is p.stdout:
                    read = fh.read(rbytes)
                    if not read:
                        streams.remove(p.stdout)
                        p.stdout.close()
                    else:
                        out_chunker.push(read)
                elif fh is p.stderr:
                    read = fh.read(rbytes)
                    if not read:
                        streams.remove(p.stderr)
                        p.stderr.close()
                    else:
                        err_chunker.push(read)
                else:
                    raise RuntimeError
            if len(streams) == 0:
                break

    p.wait()

    process_out.stdout = ''.join(process_out.stdout)
    process_err.stderr = ''.join(process_err.stderr)

    if p.returncode != 0:
        if process_err.stderr and print_error:
            print("Received an error message:\n")
            print("\n[====] \x1b[91mError Message\x1b[0m [====]\n")
            print(process_err.stderr)
            print("[====] \x1b[91mEnd o'Message\x1b[0m [====]\n")
        if persist:
            if print_error:
                print("%s gave a return code of %i (it may have crashed) -- carrying on\n" %
                      (command, p.returncode))
        else:
            # This code (commented out) would not throw an exception, but instead exit with the returncode of the crashed program.
            # sys.stderr.write("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n" % (command, p.returncode))
            # sys.exit(p.returncode)
            raise RuntimeError("\x1b[1;94m%s\x1b[0m gave a return code of %i (\x1b[91mit may have crashed\x1b[0m)\n\n" % (
                command, p.returncode))

    # Return the output in the form of a list of lines, so we can loop over it
    # using "for line in output".
    Out = process_out.stdout.split('\n')
    if Out[-1] == '':
        Out = Out[:-1]
    return Out



class LineChunker(object):
    # Thanks to cesarkawakami on #python (IRC freenode) for this code.

    def __init__(self, callback):
        self.callback = callback
        self.buf = ""

    def push(self, data):
        self.buf += data
        self.nomnom()

    def close(self):
        if self.buf:
            self.callback(self.buf + "\n")

    def nomnom(self):
        # Splits buffer by new line or carriage return, and passes
        # the splitted results onto processing.
        while "\n" in self.buf or "\r" in self.buf:
            chunk, sep, self.buf = re.split(r"(\r|\n)", self.buf, maxsplit=1)
            self.callback(chunk + sep)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
