################################################################################
# Copyright (c) 2014, Lee-Ping Wang and the Authors
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""
Python interface to Q-Chem, very useful for automating Q-Chem calculations.

Contains a QChem class representing a Q-Chem calculation, where calling
methods like sp() and make_stable() produces Q-Chem results wrapped up in
a Molecule object.

Also contains a number of functions to wrap around TS/IRC calculations and
make the results easier to use.
"""


import os, sys, shutil, glob
import traceback
import time
import numpy as np
from molecule import Molecule, Elements
from utils import _exec
from collections import defaultdict, OrderedDict
from copy import deepcopy


# Default Q-Chem input file to be used when QChem class is initialized
# from a .xyz file.  Written mainly for HF and DFT calculations; note
# the relatively conservative settings.
qcrem_default = """
$molecule
{chg} {mult}
$end

$rem
method              {method}
basis               {basis}
symmetry            off
incdft              false
incfock             0
sym_ignore          true
unrestricted        true
scf_convergence     8
thresh              14
$end
"""

vib_top = """#==========================================#
#| File containing vibrational modes from |#
#|           Q-Chem calculation           |#
#|                                        |#
#| Octothorpes are comments               |#
#| This file should be formatted like so: |#
#| (Full XYZ file for the molecule)       |#
#| Number of atoms                        |#
#| Comment line                           |#
#| a1 x1 y1 z1 (xyz for atom 1)           |#
#| a2 x2 y2 z2 (xyz for atom 2)           |#
#|                                        |#
#| These coords will be actually used     |#
#|                                        |#
#| (Followed by vibrational modes)        |#
#| Do not use mass-weighted coordinates   |#
#| ...                                    |#
#| v (Eigenvalue in wavenumbers)          |#
#| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
#| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
#| ...                                    |#
#| (Empty line is optional)               |#
#| v (Eigenvalue)                         |#
#| dx1 dy1 dz1 (Eigenvector for atom 1)   |#
#| dx2 dy2 dz2 (Eigenvector for atom 2)   |#
#| ...                                    |#
#| and so on                              |#
#|                                        |#
#| Please list freqs in increasing order  |#
#==========================================#
"""

# Handle known errors.
# Reaction path errors culled from Q-Chem source.
# Spelling errors are probably not mine.
erroks = defaultdict(list)
erroks['rpath'] = ['Bad Hessian -- imaginary mode too soft', 'Bad Hessian -- no negative eigenvalue',
                   'Bad initial gradient', 'Failed line search', 'First_IRC_step: Illegal value of coordinates',
                   'First_IRC_step: Internal programming error.', 'IRC backup failure', 'IRC failed bisector line search',
                   'IRC failed final bisector step', 'IRC --- Failed line search', 'IRC internal programming error',
                   'Maxium number of steps reached.', 'NAtom.GT.NAtoms', 'RPATH_ITER_MAX reached.',
                   'rpath_new: EWCs not yet implemented', 'rpath_new: Unimplemented coordinates',
                   'RPath_new: unimplemented coordinates.', 'rpath: no hessian at the start.',
                   'rpath: Starting Geometry Does NOT Correspond to TS']
erroks['opt'] = ['OPTIMIZE fatal error']

def tarexit(exitstat=0):
    """
    Archive files and quit.  It's helpful for remote scripts to call this.
    Note that tarexit.tarfnm and tarexit.include need to be set.

    Fields (basically globals)
    ------
    tarfnm : str
        Name of the archive to create.
    include : list or str
        Files to be archived; each entry is expanded using glob.
    exclude : list or str
        Exclude files from being archived; these are expanded using glob.
    save : list or str
        Do not remove these files even when remove_files is set to True
    archive_dirs : bool
        If set to True, directories will be archived as well.
    remove_files : bool
        If set to True, everything that is archived will be removed using --remove-files.
    remove_dirs : bool
        If set to True, all subdirectories ending in '.d' (generated by Q-Chem) will be removed.

    Parameters
    ----------
    exitstat : int
        Use this exit status.  Note that exit status > 0 indicates error.
    """
    # Type checking, make everything into a list
    if isinstance(tarexit.include, str):
        tarexit.include = [tarexit.include]
    if isinstance(tarexit.exclude, str):
        tarexit.exclude = [tarexit.exclude]
    # Remove .btr files created by OpenMPI (I think?) as well as existing .tar file.
    for f in glob.glob('*.btr'):
        os.remove(f)
    # Expand each term in "exclude" and remove from the list of files.
    excludes = sum([glob.glob(g) for g in tarexit.exclude], [])
    # Expand each term in "include" and add them to the list of files.
    include_files = []
    for g in tarexit.include:
        for f in glob.glob(g):
            # Conditions for including paths in archive list:
            # 1) Path isn't added already
            # 2) Path isn't in the list of exclusions
            # 3) Either archive_dirs is set or path is not a folder
            if f not in include_files and f not in excludes and (tarexit.archive_dirs or (not os.path.isdir(f))):
                include_files.append(f)
    # Files ending in .log are never deleted
    if tarexit.remove_files:
        saved = [f for f in sum([glob.glob(g) for g in tarexit.save], []) if f in include_files]
        if not os.path.exists('saved'):
            os.makedirs('saved')
        for f in saved:
            shutil.copy2(f, 'saved/%s' % f)
    # Actually execute the tar command.
    # If the tar file exists, then extract / delete it.
    if os.path.exists(tarexit.tarfnm):
        _exec("tar xjf %s" % tarexit.tarfnm, print_command=True)
        _exec("rm -f %s" % tarexit.tarfnm, print_command=True)
    _exec("tar cjf %s %s%s" % (tarexit.tarfnm, ' '.join(include_files), ' --remove-files' if tarexit.remove_files else ''), print_command=True)
    # Touch the file to ensure that something is created (even zero bytes).
    _exec("touch %s" % tarexit.tarfnm)
    if tarexit.remove_files:
        for f in saved:
            shutil.copy2('saved/%s' % f, f)
        shutil.rmtree('saved')
    # Delete directories that end in .d if desired.
    for f in os.listdir('.'):
        if tarexit.remove_dirs and os.path.isdir(f) and ('.d' in f):
            shutil.rmtree(f)
    sys.exit(exitstat)
tarexit.tarfnm = 'default.tar.bz2'
tarexit.include = []
tarexit.exclude = []
tarexit.save = ['*.log']
tarexit.archive_dirs=False
tarexit.remove_files=True
tarexit.remove_dirs=True

# Basis set combinations which may be provided as an argument to "basis".
# Provides rudimentary basis set mixing functionality.  You may define
# a mapping from element to basis set here.
basdict = OrderedDict([('%s_lanl2dz' % bas, OrderedDict([(Elements[i], 'lanl2dz' if i > 10 else bas) for i in range(1, 94)])) for
                       bas in ['3-21g', '3-21+g', '3-21g*', 
                               '6-31g', '6-31g*', '6-31g(d)', '6-31g**', '6-31g(d,p)',
                               '6-31+g', '6-31+g*', '6-31+g(d)', '6-31+g**', '6-31+g(d,p)',
                               '6-31++g', '6-31++g*', '6-31++g(d)', '6-31++g**', '6-31++g(d,p)',
                               '6-311g', '6-311g*', '6-311g(d)', '6-311g**', '6-311g(d,p)',
                               '6-311+g', '6-311+g*', '6-311+g(d)', '6-311+g**', '6-311+g(d,p)',
                               '6-311++g', '6-311++g*', '6-311++g(d)', '6-311++g**', '6-311++g(d,p)']])

# In most cases, the ECP can be determined from the basis
ecpdict = OrderedDict([('lanl2dz', 'lanl2dz')] + 
                      [('%s_lanl2dz' % bas, 'lanl2dz') for
                       bas in ['3-21g', '3-21+g', '3-21g*', 
                               '6-31g', '6-31g*', '6-31g(d)', '6-31g**', '6-31g(d,p)',
                               '6-31+g', '6-31+g*', '6-31+g(d)', '6-31+g**', '6-31+g(d,p)',
                               '6-31++g', '6-31++g*', '6-31++g(d)', '6-31++g**', '6-31++g(d,p)',
                               '6-311g', '6-311g*', '6-311g(d)', '6-311g**', '6-311g(d,p)',
                               '6-311+g', '6-311+g*', '6-311+g(d)', '6-311+g**', '6-311+g(d,p)',
                               '6-311++g', '6-311++g*', '6-311++g(d)', '6-311++g**', '6-311++g(d,p)']])

class QChem(object):
    """
    Class for facilitating Q-Chem calculations.  I wrote this
    because it was helpful to execute a chain of calculations on a
    single geometry.
    """
    def __init__(self, fin, ftype=None, charge=None, mult=None, method=None,
                 basis=None, qcin=None, qcout=None, qcdir=None, readsave=None,
                 readguess=True, clean=False, qcsave=True):
        """
        Create a QChem object.

        Parameters
        ----------
        fin : str
            Name of input .xyz or .in file (other coordinate files in
            molecule.py supported as well).  The .in file will contain
            settings for the Q-Chem calculation whereas the .xyz file
            does not; in this case, default fields are filled in, but
            the user must procide charge / mult / method / basis.
        ftype : str, optional
            Force the Molecule class to read the input file as this format.
        charge : int, optional
            Net charge.  Required if xyz coordinates are provided.  If
            Q-Chem input and charge are both provided, this will override.
        mult : int, optional
            Spin multiplicity.  Required if xyz coordinates are provided.
            If Q-Chem input and mult are both provided, this will override.
        method : str, optional
            Electronic structure method (e.g. b3lyp).  This is written
            to the "method" field in the Q-Chem input file - supported
            by Q-Chem 4.2 and later.  Required if xyz coordinate are
            provided.  If Q-Chem input and method are both provided, this
            will override.
        basis : str, optional
            Gaussian basis set (e.g. 6-31g*).  This is written to the
            "basis" field in the Q-Chem input file.  Required if xyz
            coordinate are provided.  If Q-Chem input and basis
            are both provided, this will override.
        qcin : str, optional
            Base name of Q-Chem input files to be written.  If not provided,
            will use "fin" (extension will be changed to ".qcin" if necessary
            to avoid overwriting input file.)
        qcout : str, optional
            Base name of Q-Chem output files.  If not provided, will use "qcin"
            base name and ".out" extension.
        qcdir : str, optional
            Base name of Q-Chem temporary folder.  If not provided, will use "qcin"
            base name and ".d" extension.  NOTE: This folder will be removed prior
            to calculation start!  Also, will create a ".dsav" folder used to recover
            from failed calculations, think of it as a "save game file".
        readsave : str or bool, optional
            If set to True, the "qcdsav" (i.e. qcdir+"sav") folder will automatically
            be used to initialize this calculation.
            If string, this is a folder containing Q-Chem files used to initialize
            this calculation.  This folder will be copied to "self.qcdsav" and self.qcdsav
            will NOT be removed prior to calculation start.
            If not provided, self.qcdsav will be removed prior to calculation start.
        readguess : bool, optional
            Write "scf_guess read" to Q-Chem input file.  If readsave is provided,
            then the very first calculation will read the SCF guess as well.
        clean : bool, optional
            When set, this calculation never updates qcdsav.  However, if readsave is
            provided it will still be used to initialize each calculation.  Use this
            if you never want the calculation result to depend on the previous state.
            Note that this is relatively uncommon (e.g. if we want to run a series
            of calculations without reading the SCF guess from the previous one.)
        qcsave : bool, optional
            Append the "-save" argument to the system call to Q-Chem.  This results
            in more files being saved to qcdir.
        """
        # Name of the input file.
        self.fin = fin
        # Molecule object from loading the input file.
        self.M = Molecule(fin, ftype)
        if 'elem' not in self.M.Data.keys():
            raise RuntimeError('Input file contains no atoms')
        # Q-Chem input file that will be written for each Q-Chem execution.
        # If the original input file happens to also be a Q-Chem input file,
        # then use the suffix 'qcin' add an underscore so we
        # don't accidentally overwrite our original file.
        if qcin == None:
            qcin = os.path.splitext(fin)[0]+'.in'
        if qcin == fin and fin.endswith('.in'):
            self.qcin = os.path.splitext(fin)[0]+'.qcin'
        elif qcin == fin:
            raise RuntimeError('Please do not provide a file with extension .qcin')
        else:
            self.qcin = qcin
        # Keep track of the number of calculations done.
        self.ncalc = 0
        # Whether a Hessian calculation has been done.
        self.haveH = 0
        # Set Q-Chem calculation options ($rem variables).
        elems = sorted(list(set(self.M.elem)))
        elemsort = np.argsort(np.array([Elements.index(i) for i in elems]))
        elems = [elems[i] for i in elemsort]
        # Treat custom basis and ECP.
        # Basis set can either be a string or a dictionary.
        # ECP can also be a string or a dictionary and it is keyed using the basis.
        if basis != None:
            basisval = basdict.get(basis.lower(), basis)
            if isinstance(basisval, dict):
                basisname = 'gen'
                basissect = sum([[e, basisval[e], '****'] for e in elems], [])
            else:
                basisname = basisval.lower()
                basissect = None
            ecp = ecpdict.get(basis.lower(), None)
            ecpname = None
            if ecp != None:
                if isinstance(ecp, dict):
                    ecpname = 'gen'
                    ecpsect = sum([[e, ecp[e], '****'] for e in elems], [])
                else:
                    ecpname = ecp.lower()
                    ecpsect = None
        if 'qcrems' not in self.M.Data.keys():
            if method == None or basis == None or charge == None or mult == None:
                raise RuntimeError('Must provide charge/mult/method/basis!')
            # Print a Q-Chem template file.
            with open('.qtemp.in','w') as f: print >> f, \
                    qcrem_default.format(chg=charge, mult=mult, method=method, basis=(basisname + '\necp                 %s' % (ecpname if ecp != None else '')))
            # Print general basis and ECP sections to the Q-Chem template file.
            if basisname == 'gen':
                with open('.qtemp.in','a') as f:
                    print >> f
                    print >> f, '$basis'
                    print >> f, '\n'.join(basissect)
                    print >> f, '$end'
            if ecpname == 'gen':
                with open('.qtemp.in','a') as f:
                    print >> f
                    print >> f, '$ecp'
                    print >> f, '\n'.join(ecpsect)
                    print >> f, '$end'
            self.M.add_quantum('.qtemp.in')
        else:
            if charge != None:
                self.M.charge = charge
            if mult != None:
                self.M.mult = mult
            if method != None:
                self.M.edit_qcrems({'method' : method})
            if basis != None:
                self.M.edit_qcrems({'basis' : basisname})
                if basisname == 'gen':
                    self.M.qctemplate['basis'] = basissect
            if ecp != None:
                self.M.edit_qcrems({'ecp' : ecpname})
                if ecpname == 'gen':
                    self.M.qctemplate['ecp'] = ecpsect
        # The current job type, which we can set using
        # different methods for job types.
        self.jobtype = 'sp'
        # Rem dictionary for SCF convergence.
        self.remscf = OrderedDict()
        # Extra rem variables for a given job type.
        self.remextra = OrderedDict()
        # Default name of Q-Chem output file
        self.qcout = os.path.splitext(self.qcin)[0]+".out" if qcout == None else qcout
        self.qcerr = os.path.splitext(self.qcin)[0]+".err"
        # Saved Q-Chem calculations if there is more than one.
        self.qcins = []
        self.qcouts = []
        self.qcerrs = []
        # Specify whether to tack "-save" onto the end of each Q-Chem call.
        self.qcsave = qcsave
        # Q-Chem scratch directory
        self.qcdir = os.path.splitext(self.qcin)[0]+".d" if qcdir == None else qcdir
        # Flag to read SCF guess at the first calculation
        self.readguess = readguess
        # Without guess to read from, use "scf_guess core"
        # and "scf_guess_mix 5" which allows us to find broken
        # symmetry states.
        self.coreguess = True
        # Error message if the calculation failed for a known reason
        self.errmsg = ''
        # qcdsav is "known-good qcdir for this object",
        # used to restore from failed calcs (e.g. SCF failure)
        self.qcdsav = self.qcdir+'sav'
        #--------
        # The clean option makes sure nothing on the disk influences this calculation.
        # This can be a bit confusing.  There are two modes of usage:
        # 1) Clean OFF.  Calculation uses whatever is in qcdir and backs it up to qcdsav on successful calcs.
        # 2) Clean ON.  qcdir is always cleared, and copied over from qcdsav (if exist) prior to calling Q-Chem.
        # This allows us to save the state of a good calculation without worrying about outside interference.
        # - Use case 1: AnalyzeReaction.py does not like to read SCF guesses from previous calculations so we use clean = True.
        # - Use case 2: Growing string does like to read SCF guesses so we use clean = False.
        # - Use case 3: IRC calculation requires Hessian from a previous calculation, so again we use clean = False.
        self.clean = clean
        # If readsave is set, then copy it to self.qcdsav and it will be used
        # to initialize this calculation.  Otherwise self.qcdsav will be removed.
        self.readsave = readsave
        if isinstance(self.readsave, str):
            if not os.path.isdir(self.readsave):
                raise RuntimeError('Tried to initialize Q-Chem reading from a save folder but does not exist')
	    if self.readsave == self.qcdsav: pass
            elif os.path.exists(self.qcdsav):
                shutil.rmtree(self.qcdsav)
                shutil.copytree(self.readsave, self.qcdsav)
        elif isinstance(self.readsave, int) and self.readsave: pass
        elif os.path.exists(self.qcdsav): shutil.rmtree(self.qcdsav)
        # Remove self.qcdir; it will be restored from self.qcdsav right before calling Q-Chem.
        if os.path.exists(self.qcdir): shutil.rmtree(self.qcdir)

    def write(self, *args, **kwargs):
        """ Write the Molecule object to a file. """
        self.M.write(*args, **kwargs)

    def write_qcin(self):
        """ Write Q-Chem input file. """
        rems = OrderedDict([('jobtype', self.jobtype)])
        rems['scf_convergence'] = 8
        # If not the first calculation, read SCF guess from the first calculation.
        if self.readguess and os.path.exists(self.qcdsav):
            rems['scf_guess'] = 'read'
            rems['scf_guess_mix'] = None
        elif self.coreguess:
            rems['scf_guess'] = 'core'
            rems['scf_guess_mix'] = 5
        # Add SCF convergence rem variables.
        rems.update(self.remscf)
        # Add job-related rem variables.
        rems.update(self.remextra)
        # If doing stability analysis, loosen SCF convergence tolerance by 1.
        # This is a bootleg solution to our workflow hanging indefinitely
        # when Q-Chem crashes.
        if 'stability_analysis' in rems.keys():
            rems['scf_convergence'] -= 2
        # Create copy of stored Molecule object, update
        # Q-Chem rem variables and write Q-Chem input file.
        M1 = deepcopy(self.M)
        M1.edit_qcrems(rems)
        M1.write(self.qcin, ftype="qcin")

    def DIE(self, errmsg):
        """ Does what it says. """
        raise RuntimeError("Error: Q-Chem calculation failed! (%s)" % errmsg)

    def load_qcout(self):
        """
        Return Molecule object corresponding to Q-Chem output
        file. SCF convergence failures and maximum optimization cycles
        reached will not trigger a parser error.
        """
        try:
            return Molecule(self.qcout, errok=erroks[self.jobtype.lower()] +
                            ['SCF failed to converge', 'Maximum optimization cycles reached'])
        except RuntimeError:
            tarexit.include=['*']
            tarexit(1)

    def call_qchem(self, debug=False):
        """
        Call Q-Chem.  There are several functions that wrap
        around this innermost call.  Assumes that Q-Chem input
        file has been written.

        Determine whether to run in serial, OpenMP-parallel or
        MPI-parallel mode. Restore qcdir from qcdsav. Execute
        Q-Chem executable but don't copy qcdir back to qcdsav
        (outer wrapper functions should do this).
        """
        if debug:
            print "Calling Q-Chem with jobtype", self.jobtype
            for line in open(self.qcin).readlines():
                print line,

        # Figure out whether to use OpenMP or MPI.
        mode = "openmp"
        M1 = Molecule(self.qcin)
        for qcrem in M1.qcrems:
            for key in qcrem.keys():
                if key == 'stability_analysis' and qcrem[key].lower() == 'true':
                    mode = "mpi"
                if key == 'jobtype' and qcrem[key].lower() == 'freq':
                    mode = "mpi"

        # Set commands to run Q-Chem.
        # The OMP_NUM_THREADS environment variable shall be used to determine
        # the number of processors.  The environment variable is then unset.
        # If not set, default to one.
        if 'OMP_NUM_THREADS' in os.environ:
            cores=int(os.environ['OMP_NUM_THREADS'])
            del os.environ['OMP_NUM_THREADS']
        else:
            cores=1
        # Q-Chem parallel (OpenMP), serial, and parallel (MPI) commands.
        # The MPI command is useful for jobs that aren't OpenMP-parallel,
        # such as stability analysis (which uses TDDFT/CIS).
        if 'QCCMD' in os.environ:
            qccmd = os.environ['QCCMD']
        else:
            qccmd = "qchem42 -nt %i" % cores
        # Command for serial jobs
        qc1cmd = qccmd.split()[0]
        # Command for MPI jobs
        qcmpi = qccmd.replace('-nt', '-np')
        # I believe this saves more scratch files from Q-Chem.
        if self.qcsave:
            qccmd += ' -save'
            qc1cmd += ' -save'
            qcmpi += ' -save'

        # Frequency calculations with less atoms than the # of cores should be serial.
        if M1.na < cores and self.jobtype.lower() == 'freq': mode = "serial"

        # I don't remember why this is here.  Something about "Recomputing EXC"?
        # if 'scf_algorithm' in self.remscf and self.remscf['scf_algorithm'] == 'rca_diis': mode = "serial"

        #----
        # Note that on some clusters I was running into random
        # crashes, which led to this code becoming more complicated.
        # The code is now cleaned up because I haven't seen the errors
        # in a while .. but if they come back, make sure to look back
        # in the commit history.
        #----

        # When "clean mode" is on, we always start from a clean slate
        # (restore qcdir from qcdsav if exist; otherwise delete)
        if (self.clean or os.path.exists(self.qcdsav)) and os.path.exists(self.qcdir):
            shutil.rmtree(self.qcdir)
        # If qcdsav exists, we restore from it
        if os.path.exists(self.qcdsav):
            _exec("rsync -a --delete %s/ %s/" % (self.qcdsav, self.qcdir), print_command=False)
        # Execute Q-Chem.
        if mode == "openmp":
            qccmd_ = qccmd
        elif mode == "mpi":
            qccmd_ = qcmpi
            # Force BW compute node to use single processor instead of MPI.
            if 'nid' in os.environ.get('HOSTNAME', 'None'):
                qccmd_ = qc1cmd
        elif mode == "serial":
            qccmd_ = qc1cmd
        try:
            _exec('%s %s %s %s &> %s' % (qccmd_, self.qcin, self.qcout, self.qcdir, self.qcerr), print_command=False)
        except:
            tarexit.include=['*']
            tarexit(1)
        # Catch known Q-Chem crashes. :(
        # I've run into a lot of TCP socket errors and OpenMP segfaults on Blue Waters.
        for line in open(self.qcerr):
            if 'Unable to open a TCP socket for out-of-band communications' in line:
                with open(self.qcerr, 'a') as f: print >> f, 'TCP socket failure :('
                tarexit.include=['*']
                tarexit(1)
        # Note that we do NOT copy qcdir to qcdsav here, because we don't know whether the calculation is good.
        # Delete the strange .btr files that show up on some clusters.
        _exec('rm -rf *.btr', print_command=False)
        # Reset the OMP_NUM_THREADS environment variable.
        os.environ['OMP_NUM_THREADS'] = str(cores)

    def scf_tactic(self, attempt=1):
        """
        Set the SCF convergence strategy.

        First attempt uses 100 SCF iterations,
        all subsequent attempts use 300 SCF iterations.

        Attempt 1: DIIS with core / read guess.
        Attempt 2: RCA with SAD guess.
        Attempt 3: GDM with core / read guess.
        Attempts 4-6: Sleazy SCF convergence with the above strategies.

        Note that readguess and coreguess are not explicitly set in
        self.remscf because their activation depends on the existence
        of self.qcdsav.
        """
        self.remscf = OrderedDict()
        # Set SCF convergence algorithm.
        if attempt in [1, 4]:
            self.readguess = True
            self.coreguess = True
            self.remscf['scf_algorithm'] = 'diis'
        if attempt in [2, 5]:
            print "RCA..",
            self.readguess = False
            self.coreguess = False
            self.remscf['scf_algorithm'] = 'rca_diis'
            self.remscf['thresh_rca_switch'] = 4
        if attempt in [3, 6]:
            print "GDM..",
            self.readguess = True
            self.coreguess = True
            self.remscf['scf_algorithm'] = 'diis_gdm'
        # Set SCF convergence criterion.
        if attempt <= 3:
            self.remscf['scf_convergence'] = 8
        else:
            if attempt == 4:
                print "Relax convergence criterion..",
            self.remscf['scf_convergence'] = 6
        # Set SCF max number of cycles.
        if attempt > 1:
            self.remscf['max_scf_cycles'] = 300
        else:
            self.remscf['max_scf_cycles'] = 100

    def converge(self, attempt=1):
        """ Attempt to converge the SCF. """
        while True:
            self.scf_tactic(attempt)
            self.write_qcin()
            # Note to self: Within this approach, each SCF algorithm
            # starts from either (1) the initial guess or (2) the MOs
            # from qcdsav.  That is to say, the subsequent attempts
            # do NOT read partially converged solutions from the previous
            # attempts.
            self.call_qchem()
            if all(["failed to converge" not in line and \
                        "Convergence failure" not in line \
                        for line in open(self.qcout)]): break
            attempt += 1
            if attempt > 6:
                self.DIE("SCF convergence failure")
        # Reset the SCF tactic back to 1 after convergence.
        # Note: This is a bit controversial. :)
        self.scf_tactic(1)
        # If not running in "clean mode", the qcdsav folder is updated.
        if not self.clean:
            _exec("rsync -a --delete %s/ %s/" % (self.qcdir, self.qcdsav), print_command=False)
        return attempt

    def converge_opt(self):
        """
        SCF convergence forcing for geometry optimization jobs.
        This function exists because SCF convergence may fail for a geometry optimization.
        When this happens, we run SP calculations with different SCF algorithms, and then
        we may either do the geometry optimization with this new SCF algorithm or revert
        back to the first one.
        """
        optouts = []
        thisopt = 1
        # SCF tactic for the optimization itself.
        # If any point in the optimization requires an alternate
        # algorithm, we continue the optimization using that.
        attempt = 1
        while True:
            self.scf_tactic(attempt)
            self.write_qcin()
            self.call_qchem()
            M1 = self.load_qcout()
            # For any optimization with at least one step,
            # we copy it to a temporary file to be joined at the end.
            if len(M1.qm_energies) >= (2 if (M1.qcerr == 'SCF failed to converge') else 1):
                optouts.append('.opt%i.out' % thisopt)
                _exec('cp %s .opt%i.out' % (self.qcout, thisopt), print_command=False)
                thisopt += 1
            if M1.qcerr in ['SCF failed to converge', 'killed']:
                # If SCF fails to converge, try different algorithms to enforce convergence.
                self.M.xyzs = [M1.xyzs[-1]]
                jobtype0 = self.jobtype
                self.jobtype = 'sp'
                attempt = self.converge(attempt)
                # If we were running an IRC calculation, we must revert
                # to geometry optimization because not at the TS anymore.
                if jobtype0 == 'rpath':
                    self.jobtype = 'opt'
                else:
                    self.jobtype = jobtype0
            else:
                # Optimization is finished; concatenate output files.
                _exec('cat %s > %s' % (' '.join(optouts), self.qcout), print_command=False)
                break

    def calculate(self, converge=True):
        """
        Perform Q-Chem calculation.

        This is a higher-level function that wraps around converge()
        and converge_opt(), which themselves wrap around call_qchem().
        """
        if converge:
            if self.jobtype in ['opt', 'ts', 'rpath']:
                self.converge_opt()
            else:
                self.converge()
        else:
            self.call_qchem()
        # Update qcdsav (if not using clean option).
        if not self.clean:
            _exec("rsync -a --delete %s/ %s/" % (self.qcdir, self.qcdsav), print_command=False)
        # Save the sequence of Q-Chem input and output files.
        jobsuf = self.jobsuf if hasattr(self, 'jobsuf') else self.jobtype
        this_qcin = os.path.splitext(self.qcin)[0] + '.%02i.%s.in' % (self.ncalc, jobsuf)
        this_qcout = os.path.splitext(self.qcout)[0] + '.%02i.%s.out' % (self.ncalc, jobsuf)
        this_qcerr = os.path.splitext(self.qcout)[0] + '.%02i.%s.err' % (self.ncalc, jobsuf)
        _exec("cp %s %s" % (self.qcin, this_qcin), print_command=False)
        _exec("cp %s %s" % (self.qcout, this_qcout), print_command=False)
        _exec("cp %s %s" % (self.qcerr, this_qcerr), print_command=False)
        self.qcins.append(this_qcin)
        self.qcouts.append(this_qcout)
        self.qcerrs.append(this_qcerr)
        self.ncalc += 1

    def sp(self):
        """ Q-Chem single point calculation. """
        self.jobtype = 'sp'
        # Clear dictionary of extra rem variables.
        self.remextra = OrderedDict()
        self.calculate()

    def stab(self):
        """ Q-Chem stability analysis calculation. """
        self.jobtype = 'sp'
        self.jobsuf = 'stb'
        self.remextra = OrderedDict([('stability_analysis', 'true'), ('max_cis_cycles', '100'), ('cis_n_roots', '4')])
        self.calculate()
        delattr(self, 'jobsuf')

    def make_stable(self, maxstab=3):
        """ Repeat stability analysis calculation until stable. """
        self.nstab = 1
        self.stable = False
        while not self.stable:
            self.readguess = True
            self.sp()
            self.readguess = True
            self.stab()
            # Parse Q-Chem output file for stability.
            stab2 = 0
            for line in open(self.qcout):
                if "UHF-> UHF   stable" in line:
                    stab2 = 1
                if "UKS-> UKS   stable" in line:
                    stab2 = 1
            if stab2:
                self.stable = True
                if self.nstab > 1:
                    print "HF/KS stable %s" % (("at attempt %i" % self.nstab) if self.nstab > 1 else "")
                break
            else:
                self.nstab += 1
            if self.nstab > maxstab:
                print "Warning: Stability analysis could not find HF/KS stable state"
                break

    def force(self):
        """ Q-Chem gradient calculation. """
        self.jobtype = 'force'
        self.remextra = OrderedDict()
        self.calculate()

    def freq(self):
        """ Q-Chem frequency and Hessian calculation. """
        self.jobtype = 'freq'
        self.remextra = OrderedDict()
        self.calculate()
        self.haveH = 1

    def write_vdata(self, fout):
        """ Write vibrational data to an easy-to-use text file. """
        M = self.load_qcout()
        with open(fout, 'w') as f:
            print >> f, vib_top
            print >> f, M.na
            print >> f, "Coordinates and vibrations calculated from %s" % self.qcout
            for e, i in zip(M.elem, M.xyzs[0]):
                print >> f, "%2s % 8.3f % 8.3f % 8.3f" % (e, i[0], i[1], i[2])
            for frq, mode in zip(M.freqs, M.modes):
                print >> f
                print >> f, "%.4f" % frq
                for i in mode:
                    print >> f, "% 8.3f % 8.3f % 8.3f" % (i[0], i[1], i[2])

    def opt(self):
        """
        Q-Chem geometry optimization.

        Updates the geometry in the object, so subsequent
        calculations use the optimized geometry.
        """
        self.jobtype = 'opt'
        self.remextra = OrderedDict([('geom_opt_max_cycles', '300')])
        self.calculate()
        M1 = self.load_qcout()
        self.M.comms = [M1.comms[-1]]
        self.M.xyzs = [M1.xyzs[-1]]

    def ts(self):
        """
        Q-Chem transition state calculation.

        Updates the geometry in the object, so subsequent
        calculations use the optimized TS geometry.
        """
        self.jobtype = 'ts'
        self.remextra = OrderedDict([('geom_opt_max_cycles', '500'),
                                     ('geom_opt_dmax', '100'),
                                     ('geom_opt_tol_gradient', '10')])
        if self.haveH:
            self.remextra['geom_opt_hessian'] = 'read'
        self.calculate()
        M1 = self.load_qcout()
        self.M.comms = [M1.comms[-1]]
        self.M.xyzs = [M1.xyzs[-1]]

    def fsm(self, nnode=21):
        """
        Q-Chem freezing string calculation.

        Updates the geometry in the object, so subsequent
        calculations use the transition state guess geometry.
        """
        self.jobtype = 'fsm'
        self.remextra = OrderedDict([('fsm_nnode', nnode),
                                     ('fsm_ngrad', 3),
                                     ('fsm_mode', 2),
                                     ('fsm_opt_mode', 2)])
        self.calculate()
        M1 = self.load_qcout()
        self.M.comms = [M1.comms[-1]]
        self.M.xyzs = [M1.xyzs[-1]]
