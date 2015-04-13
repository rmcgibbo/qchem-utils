from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from six.moves import input
#======================================================================#
#|              Chemical file format conversion module                |#
#|                                                                    |#
#|         Copyright (c) Lee-Ping Wang (leeping@stanford.edu)         |#
#|                                                                    |#
#|   This is free software released under version 2 of the GNU GPL,   |#
#|   please use or redistribute as you see fit under the terms of     |#
#|   this license. (http://www.gnu.org/licenses/gpl-2.0.html)         |#
#|                                                                    |#
#|   This program is distributed in the hope that it will be useful,  |#
#|   but without any warranty; without even the implied warranty of   |#
#|   merchantability or fitness for a particular purpose.  See the    |#
#|   GNU General Public License for more details.                     |#
#|                                                                    |#
#=====================================================================|#

#=========================================#
#|     DECLARE VARIABLE NAMES HERE       |#
#|                                       |#
#|  Any member variable in the Molecule  |#
#| class must be declared here otherwise |#
#| the Molecule class won't recognize it |#
#=========================================#
#| Data attributes in FrameVariableNames |#
#| must be a list along the frame axis,  |#
#| and they must have the same length.   |#
#=========================================#
# xyzs       = List of arrays of atomic xyz coordinates
# comms      = List of comment strings
# boxes      = List of 3-element or 9-element arrays for periodic boxes
# qm_grads   = List of arrays of gradients (i.e. negative of the atomistic forces) from QM calculations
# qm_espxyzs = List of arrays of xyz coordinates for ESP evaluation
# qm_espvals = List of arrays of ESP values
FrameVariableNames = set(['xyzs', 'comms', 'boxes', 'qm_hessians', 'qm_grads', 'qm_energies', 'qm_interaction',
                          'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins'])
#=========================================#
#| Data attributes in AtomVariableNames  |#
#| must be a list along the atom axis,   |#
#| and they must have the same length.   |#
#=========================================#
# elem       = List of elements
# partial_charge = List of atomic partial charges
# atomname   = List of atom names (can come from MM coordinate file)
# atomtype   = List of atom types (can come from MM force field)
# tinkersuf  = String that comes after the XYZ coordinates in TINKER .xyz or .arc files
# resid      = Residue IDs (can come from MM coordinate file)
# resname    = Residue names
# terminal   = List of true/false denoting whether this atom is followed
# by a terminal group.
AtomVariableNames = set(['elem', 'partial_charge', 'atomname', 'atomtype', 'tinkersuf',
                         'resid', 'resname', 'qcsuf', 'qm_ghost', 'chain', 'altloc', 'icode', 'terminal'])
#=========================================#
#| This can be any data attribute we     |#
#| want but it's usually some property   |#
#| of the molecule not along the frame   |#
#| atom axis.                            |#
#=========================================#
# bonds      = A list of 2-tuples representing bonds.  Carefully pruned when atom subselection is done.
# fnm        = The file name that the class was built from
# qcrems     = The Q-Chem 'rem' variables stored as a list of OrderedDicts
# qctemplate = The Q-Chem template file, not including the coordinates or rem variables
# charge     = The net charge of the molecule
# mult       = The spin multiplicity of the molecule
MetaVariableNames = set(['fnm', 'ftype', 'qcrems', 'qctemplate',
                         'qcerr', 'charge', 'mult', 'bonds', 'topology', 'molecules'])
# Variable names relevant to quantum calculations explicitly
QuantumVariableNames = set(
    ['qcrems', 'qctemplate', 'charge', 'mult', 'qcsuf', 'qm_ghost', 'qm_bondorder'])
# Superset of all variable names.
AllVariableNames = QuantumVariableNames | AtomVariableNames | MetaVariableNames | FrameVariableNames

# OrderedDict requires Python 2.7 or higher
import os
import sys
import re
import copy
import numpy as np
from numpy import sin, cos, arccos
import itertools
from collections import OrderedDict, namedtuple
from warnings import warn

__all__ = ['Molecule']

module_name = __name__.replace('.molecule', '')

# Covalent radii from Cordero et al. 'Covalent radii revisited' Dalton
# Transactions 2008, 2832-2838.
Radii = [0.31, 0.28,  # H and He
         1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,  # First row elements
         1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,  # Second row elements
         2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50,
         # Third row elements, K through Kr
         1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
         2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42,
         # Fourth row elements, Rb through Xe
         1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
         2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98,
         # Fifth row elements, s and f blocks
         1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
         1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36,
         # Fifth row elements, d and p blocks
         1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
         2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69]  # Sixth row elements

# A list that gives you the element if you give it the atomic number,
# hence the 'none' at the front.
Elements = ["None", 'H', 'He',
            'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt']

# Dictionary of atomic masses ; also serves as the list of elements
# (periodic table)
PeriodicTable = OrderedDict([('H', 1.0079), ('He', 4.0026),
                             ('Li', 6.941), ('Be', 9.0122), ('B', 10.811), ('C', 12.0107), ('N',
                                                                                            14.0067), ('O', 15.9994), ('F', 18.9984), ('Ne', 20.1797),
                             ('Na', 22.9897), ('Mg', 24.305), ('Al', 26.9815), ('Si', 28.0855), (
                                 'P', 30.9738), ('S', 32.065), ('Cl', 35.453), ('Ar', 39.948),
                             ('K', 39.0983), ('Ca', 40.078), ('Sc', 44.9559), ('Ti', 47.867), (
                                 'V', 50.9415), ('Cr', 51.9961), ('Mn', 54.938), ('Fe', 55.845), ('Co', 58.9332),
                             ('Ni', 58.6934), ('Cu', 63.546), ('Zn', 65.39), ('Ga', 69.723), (
                                 'Ge', 72.64), ('As', 74.9216), ('Se', 78.96), ('Br', 79.904), ('Kr', 83.8),
                             ('Rb', 85.4678), ('Sr', 87.62), ('Y', 88.9059), ('Zr', 91.224), (
                                 'Nb', 92.9064), ('Mo', 95.94), ('Tc', 98), ('Ru', 101.07), ('Rh', 102.9055),
                             ('Pd', 106.42), ('Ag', 107.8682), ('Cd', 112.411), ('In', 114.818), (
                                 'Sn', 118.71), ('Sb', 121.76), ('Te', 127.6), ('I', 126.9045), ('Xe', 131.293),
                             ('Cs', 132.9055), ('Ba', 137.327), ('La', 138.9055), ('Ce', 140.116), (
                                 'Pr', 140.9077), ('Nd', 144.24), ('Pm', 145), ('Sm', 150.36),
                             ('Eu', 151.964), ('Gd', 157.25), ('Tb', 158.9253), ('Dy', 162.5), (
                                 'Ho', 164.9303), ('Er', 167.259), ('Tm', 168.9342), ('Yb', 173.04),
                             ('Lu', 174.967), ('Hf', 178.49), ('Ta', 180.9479), ('W', 183.84), (
                                 'Re', 186.207), ('Os', 190.23), ('Ir', 192.217), ('Pt', 195.078),
                             ('Au', 196.9665), ('Hg', 200.59), ('Tl', 204.3833), ('Pb',
                                                                                  207.2), ('Bi', 208.9804), ('Po', 209), ('At', 210), ('Rn', 222),
                             ('Fr', 223), ('Ra', 226), ('Ac', 227), ('Th', 232.0381), ('Pa',
                                                                                       231.0359), ('U', 238.0289), ('Np', 237), ('Pu', 244),
                             ('Am', 243), ('Cm', 247), ('Bk', 247), ('Cf',
                                                                     251), ('Es', 252), ('Fm', 257), ('Md', 258), ('No', 259),
                             ('Lr', 262), ('Rf', 261), ('Db', 262), ('Sg', 266), ('Bh', 264), ('Hs', 277), ('Mt', 268)])


def getElement(mass):
    return list(PeriodicTable.keys())[np.argmin([np.abs(m - mass) for m in list(PeriodicTable.values())])]

def elem_from_atomname(atomname):
    """ Given an atom name, attempt to get the element in most cases. """
    return re.search('[A-Z][a-z]*', atomname).group(0)

#===========================#
#| Convenience subroutines |#
#===========================#

# One bohr equals this many angstroms
bohrang = 0.529177249


def unmangle(M1, M2):
    """
    Create a mapping that takes M1's atom indices to M2's atom indices based on position.

    If we start with atoms in molecule "PDB", and the new molecule "M"
    contains re-numbered atoms, then this code works:

    M.elem = list(np.array(PDB.elem)[unmangled])
    """
    if M1.na != M2.na:
        raise RuntimeError("Unmangler only deals with same number of atoms")
    unmangler = {}
    for i in range(M1.na):
        for j in range(M2.na):
            if np.linalg.norm(M1.xyzs[0][i] - M2.xyzs[0][j]) < 0.1:
                unmangler[j] = i
    unmangled = [unmangler[i] for i in sorted(unmangler.keys())]
    if len(unmangled) != M1.na:
        raise RuntimeError("Unmangler failed (different structures?)")
    return unmangled


def isint(word):
    """ONLY matches integers! If you have a decimal point? None shall pass!"""
    return re.match('^[-+]?[0-9]+$', word)


def isfloat(word):
    """Matches ANY number; it can be a decimal, scientific notation, integer, or what have you"""
    return re.match('^[-+]?[0-9]*\.?[0-9]*([eEdD][-+]?[0-9]+)?$', word)

# Used to get the white spaces in a split line.
splitter = re.compile(r'(\s+|\S+)')

# Container for Bravais lattice vector.  Three cell lengths, three angles,
# three vectors, volume, and TINKER trig functions.
Box = namedtuple(
    'Box', ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'A', 'B', 'C', 'V'])
radian = 180. / np.pi


def BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma):
    """ This function takes in three lattice lengths and three lattice angles, and tries to return a complete box specification. """
    alph = alpha * np.pi / 180
    bet = beta * np.pi / 180
    gamm = gamma * np.pi / 180
    v = np.sqrt(1 - cos(alph) ** 2 - cos(bet) ** 2 - cos(gamm)
                ** 2 + 2 * cos(alph) * cos(bet) * cos(gamm))
    Mat = np.matrix([[a, b * cos(gamm), c * cos(bet)],
                     [0, b * sin(gamm), c *
                      ((cos(alph) - cos(bet) * cos(gamm)) / sin(gamm))],
                     [0, 0, c * v / sin(gamm)]])
    L1 = Mat * np.matrix([[1], [0], [0]])
    L2 = Mat * np.matrix([[0], [1], [0]])
    L3 = Mat * np.matrix([[0], [0], [1]])
    return Box(a, b, c, alpha, beta, gamma, np.array(L1).flatten(), np.array(L2).flatten(), np.array(L3).flatten(), v * a * b * c)


def BuildLatticeFromVectors(v1, v2, v3):
    """ This function takes in three lattice vectors and tries to return a complete box specification. """
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)
    alpha = arccos(
        np.dot(v2, v3) / np.linalg.norm(v2) / np.linalg.norm(v3)) * radian
    beta = arccos(
        np.dot(v1, v3) / np.linalg.norm(v1) / np.linalg.norm(v3)) * radian
    gamma = arccos(
        np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)) * radian
    alph = alpha * np.pi / 180
    bet = beta * np.pi / 180
    gamm = gamma * np.pi / 180
    v = np.sqrt(1 - cos(alph) ** 2 - cos(bet) ** 2 - cos(gamm)
                ** 2 + 2 * cos(alph) * cos(bet) * cos(gamm))
    Mat = np.matrix([[a, b * cos(gamm), c * cos(bet)],
                     [0, b * sin(gamm), c *
                      ((cos(alph) - cos(bet) * cos(gamm)) / sin(gamm))],
                     [0, 0, c * v / sin(gamm)]])
    L1 = Mat * np.matrix([[1], [0], [0]])
    L2 = Mat * np.matrix([[0], [1], [0]])
    L3 = Mat * np.matrix([[0], [0], [1]])
    return Box(a, b, c, alpha, beta, gamma, np.array(L1).flatten(), np.array(L2).flatten(), np.array(L3).flatten(), v * a * b * c)


def format_xyz_coord(element, xyz, tinker=False):
    """ Print a line consisting of (element, x, y, z) in accordance with .xyz file format

    @param[in] element A chemical element of a single atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    if tinker:
        return "%-3s % 13.8f % 13.8f % 13.8f" % (element, xyz[0], xyz[1], xyz[2])
    else:
        return "%-5s % 15.10f % 15.10f % 15.10f" % (element, xyz[0], xyz[1], xyz[2])


def format_gro_coord(resid, resname, aname, seqno, xyz):
    """ Print a line in accordance with .gro file format, with six decimal points of precision

    Nine decimal points of precision are necessary to get forces below 1e-3 kJ/mol/nm.

    @param[in] resid The number of the residue that the atom belongs to
    @param[in] resname The name of the residue that the atom belongs to
    @param[in] aname The name of the atom
    @param[in] seqno The sequential number of the atom
    @param[in] xyz A 3-element array containing x, y, z coordinates of that atom

    """
    return "%5i%-5s%5s%5i % 13.9f % 13.9f % 13.9f" % (resid, resname, aname, seqno, xyz[0], xyz[1], xyz[2])


def format_xyzgen_coord(element, xyzgen):
    """ Print a line consisting of (element, p, q, r, s, t, ...) where
    (p, q, r) are arbitrary atom-wise data (this might happen, for
    instance, with atomic charges)

    @param[in] element A chemical element of a single atom
    @param[in] xyzgen A N-element array containing data for that atom

    """
    return "%-5s" + ' '.join(["% 15.10f" % i] for i in xyzgen)


def format_gro_box(box):
    """ Print a line corresponding to the box vector in accordance with .gro file format

    @param[in] box Box NamedTuple

    """
    if box.alpha == 90.0 and box.beta == 90.0 and box.gamma == 90.0:
        return ' '.join(["% 13.9f" % (i / 10) for i in [box.a, box.b, box.c]])
    else:
        return ' '.join(["% 13.9f" % (i / 10) for i in [box.A[0], box.B[1], box.C[2], box.A[1], box.A[2], box.B[0], box.B[2], box.C[0], box.C[1]]])


def is_gro_coord(line):
    """ Determines whether a line contains GROMACS data or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) == 6:
        return all([isint(sline[2]), isfloat(sline[3]), isfloat(sline[4]), isfloat(sline[5])])
    elif len(sline) == 5:
        return all([isint(line[15:20]), isfloat(sline[2]), isfloat(sline[3]), isfloat(sline[4])])
    else:
        return 0


def is_charmm_coord(line):
    """ Determines whether a line contains CHARMM data or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) >= 7:
        return all([isint(sline[0]), isint(sline[1]), isfloat(sline[4]), isfloat(sline[5]), isfloat(sline[6])])
    else:
        return 0


def is_gro_box(line):
    """ Determines whether a line contains a GROMACS box vector or not

    @param[in] line The line to be tested

    """
    sline = line.split()
    if len(sline) == 9 and all([isfloat(i) for i in sline]):
        return 1
    elif len(sline) == 3 and all([isfloat(i) for i in sline]):
        return 1
    else:
        return 0


def add_strip_to_mat(mat, strip):
    out = list(mat)
    if out == [] and strip != []:
        out = list(strip)
    elif out != [] and strip != []:
        for (i, j) in zip(out, strip):
            i += list(j)
    return out


def pvec(vec):
    return ''.join([' % .10e' % i for i in list(vec.flatten())])


def grouper(n, iterable):
    """ Groups a big long iterable into groups of ten or what have you. """
    args = [iter(iterable)] * n
    return list([e for e in t if e != None] for t in itertools.zip_longest(*args))


def even_list(totlen, splitsize):
    """ Creates a list of number sequences divided as evenly as possible.  """
    joblens = np.zeros(splitsize, dtype=int)
    subsets = []
    for i in range(totlen):
        joblens[i % splitsize] += 1
    jobnow = 0
    for i in range(splitsize):
        subsets.append(list(range(jobnow, jobnow + joblens[i])))
        jobnow += joblens[i]
    return subsets


def both(A, B, key):
    return key in A.Data and key in B.Data


def diff(A, B, key):
    if not (key in A.Data and key in B.Data):
        return False
    else:
        if type(A.Data[key]) is np.ndarray:
            return (A.Data[key] != B.Data[key]).any()
        elif key == 'tinkersuf':
            return [sorted([int(j) for j in i.split()]) for i in A.Data[key]] != \
                [sorted([int(j) for j in i.split()]) for i in B.Data[key]]
        else:
            return A.Data[key] != B.Data[key]


def either(A, B, key):
    return key in A.Data or key in B.Data



class Molecule(object):

    """ Lee-Ping's general file format conversion class.

    The purpose of this class is to read and write chemical file formats in a
    way that is convenient for research.  There are highly general file format
    converters out there (e.g. catdcd, openbabel) but I find that writing
    my own class can be very helpful for specific purposes.  Here are some things
    this class can do:

    - Convert a .gro file to a .xyz file, or a .pdb file to a .dcd file.
    Data is stored internally, so any readable file can be converted into
    any writable file as long as there is sufficient information to write
    that file.

    - Accumulate information from different files.  For example, we may read
    A.gro to get a list of coordinates, add quantum settings from a B.in file,
    and write A.in (this gives us a file that we can use to run QM calculations)

    - Concatenate two trajectories together as long as they're compatible.  This
    is done by creating two Molecule objects and then simply adding them.  Addition
    means two things:  (1) Information fields missing from each class, but present
    in the other, are added to the sum, and (2) Appendable or per-frame fields
    (i.e. coordinates) are concatenated together.

    - Slice trajectories using reasonable Python language.  That is to
    say, MyMolecule[1:10] returns a new Molecule object that contains
    frames 1 through 9 (inclusive and numbered starting from zero.)

    Special variables:  These variables cannot be set manually because
    there is a special method associated with getting them.

    na = The number of atoms.  You'll get this if you use MyMol.na or MyMol['na'].
    ns = The number of snapshots.  You'll get this if you use MyMol.ns or MyMol['ns'].

    Unit system:  Angstroms.

    """

    def __len__(self):
        """ Return the number of frames in the trajectory. """
        L = -1
        klast = None
        Defined = False
        for key in self.FrameKeys:
            Defined = True
            if L != -1 and len(self.Data[key]) != L:
                self.repair(key, klast)
            L = len(self.Data[key])
            klast = key
        if not Defined:
            return 0
        return L

    def __getattr__(self, key):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        if key == 'qm_forces':
            warn(
                'qm_forces is a deprecated keyword because it actually meant gradients; setting to qm_grads.')
            key = 'qm_grads'
        if key == 'ns':
            return len(self)
        elif key == 'na':  # The 'na' attribute is the number of atoms.
            L = -1
            klast = None
            Defined = False
            for key in self.AtomKeys:
                Defined = True
                if L != -1 and len(self.Data[key]) != L:
                    self.repair(key, klast)
                L = len(self.Data[key])
                klast = key
            if Defined:
                return L
            elif 'xyzs' in self.Data:
                return len(self.xyzs[0])
            else:
                return 0
            #raise RuntimeError('na is ill-defined if the molecule has no AtomKeys member variables.')
        # These attributes return a list of attribute names defined in this class that belong in the chosen category.
        # For example: self.FrameKeys should return set(['xyzs','boxes']) if
        # xyzs and boxes exist in self.Data
        elif key == 'FrameKeys':
            return set(self.Data) & FrameVariableNames
        elif key == 'AtomKeys':
            return set(self.Data) & AtomVariableNames
        elif key == 'MetaKeys':
            return set(self.Data) & MetaVariableNames
        elif key == 'QuantumKeys':
            return set(self.Data) & QuantumVariableNames
        elif key in self.Data:
            return self.Data[key]
        return getattr(super(Molecule, self), key)

    def __setattr__(self, key, value):
        """ Whenever we try to get a class attribute, it first tries to get the attribute from the Data dictionary. """
        # These attributes return a list of attribute names defined in this class, that belong in the chosen category.
        # For example: self.FrameKeys should return set(['xyzs','boxes']) if
        # xyzs and boxes exist in self.Data
        if key == 'qm_forces':
            warn(
                'qm_forces is a deprecated keyword because it actually meant gradients; setting to qm_grads.')
            key = 'qm_grads'
        if key in AllVariableNames:
            self.Data[key] = value
        return super(Molecule, self).__setattr__(key, value)

    def __getitem__(self, key):
        """
        The Molecule class has list-like behavior, so we can get slices of it.
        If we say MyMolecule[0:10], then we'll return a copy of MyMolecule with frames 0 through 9.
        """
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, np.ndarray):
            if isinstance(key, int):
                key = [key]
            New = Molecule()
            for k in self.FrameKeys:
                if k == 'boxes':
                    New.Data[k] = [
                        j for i, j in enumerate(self.Data[k]) if i in np.arange(len(self))[key]]
                else:
                    New.Data[k] = list(np.array(self.Data[k])[key])
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = copy.deepcopy(self.Data[k])
            return New
        else:
            raise RuntimeError(
                'getitem is not implemented for keys of type %s\n' % str(key))

    def __delitem__(self, key):
        """
        Similarly, in order to delete a frame, we simply perform item deletion on
        framewise variables.
        """
        for k in self.FrameKeys:
            del self.Data[k][key]

    def __iter__(self):
        """ List-like behavior for looping over trajectories. Note that these values are returned by reference.
        Note that this is intended to be more efficient than __getitem__, so when we loop over a trajectory,
        it's best to go "for m in M" instead of "for i in range(len(M)): m = M[i]"
        """
        for frame in range(self.ns):
            New = Molecule()
            for k in self.FrameKeys:
                New.Data[k] = self.Data[k][frame]
            for k in self.AtomKeys | self.MetaKeys:
                New.Data[k] = self.Data[k]
            yield New

    def __add__(self, other):
        """ Add method for Molecule objects. """
        # Check type of other
        if not isinstance(other, Molecule):
            raise TypeError(
                'A Molecule instance can only be added to another Molecule instance\n')
        # Create the sum of the two classes by copying the first class.
        Sum = Molecule()
        for key in AtomVariableNames | MetaVariableNames:
            # Because a molecule object can only have one 'file name' or 'file type' attribute,
            # we only keep the original one.  This isn't perfect, but that's
            # okay.
            if key in ['fnm', 'ftype', 'bonds', 'molecules', 'topology'] and key in self.Data:
                Sum.Data[key] = self.Data[key]
            elif diff(self, other, key):
                for i, j in zip(self.Data[key], other.Data[key]):
                    print((i, j, i == j))
                raise RuntimeError(
                    'The data member called %s is not the same for these two objects\n' % key)
            elif key in self.Data:
                Sum.Data[key] = copy.deepcopy(self.Data[key])
            elif key in other.Data:
                Sum.Data[key] = copy.deepcopy(other.Data[key])
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    raise RuntimeError(
                        'Key %s in self is a FrameKey, it must be a list\n' % key)
                if type(other.Data[key]) is not list:
                    raise RuntimeError(
                        'Key %s in other is a FrameKey, it must be a list\n' % key)
                Sum.Data[key] = list(self.Data[key] + other.Data[key])
            elif either(self, other, key):
                # TINKER 6.3 compatibility - catch the specific case that one
                # has a periodic box and the other doesn't.
                if key == 'boxes':
                    if key in self.Data:
                        other.Data['boxes'] = [self.Data['boxes'][0]
                                               for i in range(len(other))]
                    elif key in other.Data:
                        self.Data['boxes'] = [other.Data['boxes'][0]
                                              for i in range(len(self))]
                else:
                    raise RuntimeError(
                        'Key %s is a FrameKey, must exist in both self and other for them to be added (for now).\n' % key)
        return Sum

    def __iadd__(self, other):
        """ Add method for Molecule objects. """
        # Check type of other
        if not isinstance(other, Molecule):
            raise TypeError(
                'A Molecule instance can only be added to another Molecule instance\n')
        # Create the sum of the two classes by copying the first class.
        for key in AtomVariableNames | MetaVariableNames:
            if key in ['fnm', 'ftype', 'bonds']:
                pass
            elif diff(self, other, key):
                for i, j in zip(self.Data[key], other.Data[key]):
                    print((i, j, i == j))
                raise RuntimeError(
                    'The data member called %s is not the same for these two objects\n' % key)
            # Information from the other class is added to this class (if said
            # info doesn't exist.)
            elif key in other.Data:
                self.Data[key] = copy.deepcopy(other.Data[key])
        # FrameKeys must be a list.
        for key in FrameVariableNames:
            if both(self, other, key):
                if type(self.Data[key]) is not list:
                    raise RuntimeError(
                        'Key %s in self is a FrameKey, it must be a list\n' % key)
                if type(other.Data[key]) is not list:
                    raise RuntimeError(
                        'Key %s in other is a FrameKey, it must be a list\n' % key)
                self.Data[key] += other.Data[key]
            elif either(self, other, key):
                # TINKER 6.3 compatibility - catch the specific case that one
                # has a periodic box and the other doesn't.
                if key == 'boxes':
                    if key in self.Data:
                        other.Data['boxes'] = [self.Data['boxes'][0]
                                               for i in range(len(other))]
                    elif key in other.Data:
                        self.Data['boxes'] = [other.Data['boxes'][0]
                                              for i in range(len(self))]
                else:
                    raise RuntimeError(
                        'Key %s is a FrameKey, must exist in both self and other for them to be added (for now).\n' % key)
        return self

    def repair(self, key, klast):
        """ Attempt to repair trivial issues that would otherwise break the object. """
        kthis = key if len(self.Data[key]) < len(self.Data[klast]) else klast
        kother = klast if len(self.Data[key]) < len(self.Data[klast]) else key
        diff = abs(len(self.Data[key]) - len(self.Data[klast]))
        if kthis == 'comms':
            # Append empty comments if necessary because this causes way too
            # many crashes.
            if diff > 0:
                for i in range(diff):
                    self.Data['comms'].append('')
            else:
                for i in range(-1 * diff):
                    self.Data['comms'].pop()
        elif kthis == 'boxes' and len(self.Data['boxes']) == 1:
            # If we only have one box then we can fill in the rest of the
            # trajectory.
            for i in range(diff):
                self.Data['boxes'].append(self.Data['boxes'][-1])
        else:
            raise RuntimeError('The keys %s and %s have different lengths (%i %i)'
                               '- this isn\'t supposed to happen for two AtomKeys member variables.'
                               % (key, klast, len(self.Data[key]), len(self.Data[klast])))

    def reorder_according_to(self, other):
        """

        Reorder atoms according to some other Molecule object.  This
        happens when we run a program like pdb2gmx or pdbxyz and it
        scrambles our atom ordering, forcing us to reorder the atoms
        for all frames in the current Molecule object.

        Directions:
        (1) Load up the scrambled file as a new Molecule object.
        (2) Call this function: Original_Molecule.reorder_according_to(scrambled)
        (3) Save Original_Molecule to a new file name.

        """

        M = self[0]
        N = other
        unmangled = unmangle(M, N)
        NewData = {}
        for key in self.AtomKeys:
            NewData[key] = list(np.array(M.Data[key])[unmangled])
        for key in self.FrameKeys:
            if key in ['xyzs', 'qm_grads', 'qm_mulliken_charges', 'qm_mulliken_spins']:
                NewData[key] = list(
                    [self.Data[key][i][unmangled] for i in range(len(self))])
        for key in NewData:
            setattr(self, key, copy.deepcopy(NewData[key]))

    def reorder_indices(self, other):
        """

        Return the indices that would reorder atoms according to some
        other Molecule object.  This happens when we run a program
        like pdb2gmx or pdbxyz and it scrambles our atom ordering.

        Directions:
        (1) Load up the scrambled file as a new Molecule object.
        (2) Call this function: Original_Molecule.reorder_indices(scrambled)

        """
        return unmangle(self[0], other)

    def append(self, other):
        self += other

    def __init__(self, fnm=None, ftype=None, **kwargs):
        """
        Create a Molecule object.

        Parameters
        ----------
        fnm : str, optional
            File name to create the Molecule object from.  If provided,
            the file will be parsed and used to fill in the fields such as
            elem (elements), xyzs (coordinates) and so on.  If ftype is not
            provided, will automatically try to determine file type from file
            extension.  If not provided, will create an empty object.
        ftype : str, optional
            File type, corresponding to an entry in the internal table of known
            file types.  Provide this if you have a nonstandard file extension
            or if you wish to force to invoke a particular parser.
        Fac : float, optional
            Multiplicative factor to covalent radii criterion for deciding whether two atoms are bonded
            Default value of 1.2 is reasonable, 1.4 will produce lots of bonds
        positive_resid : bool, optional
            If provided, enforce all positive resIDs.
        """
        #=========================================#
        #|           File type tables            |#
        #|    Feel free to edit these as more    |#
        #|      readers / writers are added      |#
        #=========================================#
        # The table of file readers
        self.Read_Tab = {'gaussian': self.read_com,
                         'gromacs': self.read_gro,
                         'charmm': self.read_charmm,
                         'mdcrd': self.read_mdcrd,
                         'inpcrd': self.read_inpcrd,
                         'xyz': self.read_xyz,
                         'mol2': self.read_mol2,
                         'qcin': self.read_qcin,
                         'qcout': self.read_qcout,
                         'qcesp': self.read_qcesp,
                         'qdata': self.read_qdata,
                         'tinker': self.read_arc}
        # The table of file writers
        self.Write_Tab = {'gromacs': self.write_gro,
                          'xyz': self.write_xyz,
                          'molproq': self.write_molproq,
                          'inpcrd': self.write_inpcrd,
                          'mdcrd': self.write_mdcrd,
                          'qcin': self.write_qcin,
                          'qdata': self.write_qdata,
                          'tinker': self.write_arc}
        # A funnel dictionary that takes redundant file types
        # and maps them down to a few.
        self.Funnel = {'gromos': 'gromacs',
                       'gro': 'gromacs',
                       'g96': 'gromacs',
                       'gmx': 'gromacs',
                       'in': 'qcin',
                       'qcin': 'qcin',
                       'com': 'gaussian',
                       'rst': 'inpcrd',
                       'out': 'qcout',
                       'esp': 'qcesp',
                       'txt': 'qdata',
                       'crd': 'charmm',
                       'cor': 'charmm',
                       'arc': 'tinker'}
        # Creates entries like 'gromacs' : 'gromacs' and 'xyz' : 'xyz'
        # in the Funnel
        self.positive_resid = kwargs.get('positive_resid', 0)
        self.built_bonds = False
        # Topology settings
        self.top_settings = {
            'Fac': kwargs.get('Fac', 1.2),
            'read_bonds': False}

        for i in set(list(self.Read_Tab.keys()) + list(self.Write_Tab.keys())):
            self.Funnel[i] = i
        # Data container.  All of the data is stored in here.
        self.Data = {}
        # Read in stuff if we passed in a file name, otherwise return an empty
        # instance.
        if fnm != None:
            self.Data['fnm'] = fnm
            if ftype == None:
                # Try to determine from the file name using the extension.
                ftype = os.path.splitext(fnm)[1][1:]
            if not os.path.exists(fnm):
                raise IOError(
                    'Tried to create Molecule object from a file that does not exist: %s\n' % fnm)
            self.Data['ftype'] = ftype
            # Actually read the file.
            Parsed = self.Read_Tab[self.Funnel[ftype.lower()]](fnm, **kwargs)
            # Set member variables.
            for key, val in Parsed.items():
                self.Data[key] = val
            # Create a list of comment lines if we don't already have them from
            # reading the file.
            if 'comms' not in self.Data:
                self.comms = ['Generated by ForceBalance from %s: Frame %i of %i' % (
                    fnm, i + 1, self.ns) for i in range(self.ns)]
            else:
                self.comms = [i.expandtabs() for i in self.comms]

    #=====================================#
    #|     Core read/write functions     |#
    #| Hopefully we won't have to change |#
    #|         these very often!         |#
    #=====================================#

    def require(self, *args):
        for arg in args:
            if arg not in self.Data:
                raise RuntimeError(
                    "%s is a required attribute for writing this type of file but it's not present\n" % arg)

    # def read(self, fnm, ftype = None):
    #     """ Read in a file. """
    #     if ftype == None:
    # Try to determine from the file name using the extension.
    #         ftype = os.path.splitext(fnm)[1][1:]
    # This calls the table of reader functions and prints out an error message if it fails.
    # 'Answer' is a dictionary of data that is returned from the reader function.
    #     Answer = self.Read_Tab[self.Funnel[ftype.lower()]](fnm)
    #     return Answer

    def write(self, fnm=None, ftype=None, append=False, select=None, **kwargs):
        if fnm == None and ftype == None:
            raise RuntimeError(
                "Output file name and file type are not specified.\n")
        elif ftype == None:
            ftype = os.path.splitext(fnm)[1][1:]
        # Fill in comments.
        if 'comms' not in self.Data:
            self.comms = ['Generated by ForceBalance from %s: Frame %i of %i' % (
                fnm, i + 1, self.ns) for i in range(self.ns)]
        if 'xyzs' in self.Data and len(self.comms) < len(self.xyzs):
            for i in range(len(self.comms), len(self.xyzs)):
                self.comms.append("Frame %i: generated by ForceBalance" % i)
        # I needed to add in this line because the DCD writer requires the file name,
        # but the other methods don't.
        self.fout = fnm
        if type(select) in [int, np.int64, np.int32]:
            select = [select]
        if select == None:
            select = list(range(len(self)))
        Answer = self.Write_Tab[self.Funnel[ftype.lower()]](select, **kwargs)
        # Any method that returns text will give us a list of lines, which we
        # then write to the file.
        if Answer != None:
            if fnm == None or fnm == sys.stdout:
                outfile = sys.stdout
            elif append:
                # Writing to symbolic links risks unintentionally overwriting the source file -
                # thus we delete the link first.
                if os.path.islink(fnm):
                    os.unlink(fnm)
                outfile = open(fnm, 'a')
            else:
                if os.path.islink(fnm):
                    os.unlink(fnm)
                outfile = open(fnm, 'w')
            for line in Answer:
                print(line, file=outfile)
            outfile.close()

    def load_frames(self, fnm):
        NewMol = Molecule(fnm)
        if NewMol.na != self.na:
            raise RuntimeError(
                'When loading frames, don\'t change the number of atoms.\n')
        for key in NewMol.FrameKeys:
            self.Data[key] = NewMol.Data[key]

    def edit_qcrems(self, in_dict, subcalc=None):
        """ Edit Q-Chem rem variables with a dictionary.  Pass a value of None to delete a rem variable. """
        if subcalc == None:
            for qcrem in self.qcrems:
                for key, val in in_dict.items():
                    if val == None:
                        qcrem.pop(key, None)
                    else:
                        qcrem[key] = val
        else:
            for key, val in in_dict.items():
                if val == None:
                    self.qcrems[subcalc].pop(key, None)
                else:
                    self.qcrems[subcalc][key] = val

    def add_quantum(self, other):
        if type(other) is Molecule:
            OtherMol = other
        elif type(other) is str:
            OtherMol = Molecule(other)
        for key in OtherMol.QuantumKeys:
            if key in AtomVariableNames and len(OtherMol.Data[key]) != self.na:
                raise RuntimeError(
                    'The quantum-key %s is AtomData, but it doesn\'t have the same number of atoms as the Molecule object we\'re adding it to.')
            self.Data[key] = copy.deepcopy(OtherMol.Data[key])

    def replace_peratom(self, key, orig, want):
        """ Replace all of the data for a certain attribute in the system from orig to want. """
        if key in self.Data:
            for i in range(self.na):
                if self.Data[key][i] == orig:
                    self.Data[key][i] = want
        else:
            raise RuntimeError(
                'The key that we want to replace (%s) doesn\'t exist.\n' % key)

    def replace_peratom_conditional(self, key1, cond, key2, orig, want):
        """ Replace all of the data for a attribute key2 from orig to want, contingent on key1 being equal to cond.
        For instance: replace H1 with H2 if resname is SOL."""
        if key2 in self.Data and key1 in self.Data:
            for i in range(self.na):
                if self.Data[key2][i] == orig and self.Data[key1][i] == cond:
                    self.Data[key2][i] = want
        else:
            raise RuntimeError(
                'Either the comparison or replacement key (%s, %s) doesn\'t exist.\n' % (key1, key2))

    def atom_select(self, atomslice):
        """ Return a copy of the object with certain atoms selected.  Takes an integer, list or array as argument. """
        if isinstance(atomslice, int):
            atomslice = [atomslice]
        if isinstance(atomslice, list):
            atomslice = np.array(atomslice)
        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])
        for key in self.AtomKeys:
            if key == 'tinkersuf':  # Tinker suffix is a bit tricky
                Map = dict([(a + 1, i + 1) for i, a in enumerate(atomslice)])
                CopySuf = list(np.array(self.Data[key])[atomslice])
                NewSuf = []
                for line in CopySuf:
                    whites = re.split('[^ ]+', line)
                    s = line.split()
                    if len(s) > 1:
                        for i in range(1, len(s)):
                            s[i] = str(Map[int(s[i])])
                    sn = [int(i) for i in s[1:]]
                    s = [s[0]] + list(np.array(s[1:])[np.argsort(sn)])
                    NewSuf.append(
                        ''.join([whites[j] + s[j] for j in range(len(s))]))
                New.Data['tinkersuf'] = NewSuf[:]
            else:
                New.Data[key] = list(np.array(self.Data[key])[atomslice])
        for key in self.FrameKeys:
            if key in ['xyzs', 'qm_grads', 'qm_mulliken_charges', 'qm_mulliken_spins']:
                New.Data[key] = [self.Data[key][i][atomslice]
                                 for i in range(len(self))]
        if 'bonds' in self.Data:
            New.Data['bonds'] = [(list(atomslice).index(b[0]), list(atomslice).index(
                b[1])) for b in self.bonds if (b[0] in atomslice and b[1] in atomslice)]
        New.top_settings = self.top_settings
        return New

    def atom_stack(self, other):
        """ Return a copy of the object with another molecule object appended.  WARNING: This function may invalidate stuff like QM energies. """
        if len(other) != len(self):
            raise RuntimeError(
                'The number of frames of the Molecule objects being stacked are not equal.\n')

        New = Molecule()
        for key in self.FrameKeys | self.MetaKeys:
            New.Data[key] = copy.deepcopy(self.Data[key])

        # This is how we're going to stack things like Cartesian coordinates
        # and QM forces.
        def FrameStack(k):
            if k in self.Data and k in other.Data:
                New.Data[k] = [np.vstack((s, o))
                               for s, o in zip(self.Data[k], other.Data[k])]
        for i in ['xyzs', 'qm_grads', 'qm_espxyzs', 'qm_espvals', 'qm_extchgs', 'qm_mulliken_charges', 'qm_mulliken_spins']:
            FrameStack(i)

        # Now build the new atom keys.
        for key in self.AtomKeys:
            if key not in other.Data:
                raise RuntimeError(
                    'Trying to stack two Molecule objects - the first object contains %s and the other does not\n' % (key))
            if key == 'tinkersuf':  # Tinker suffix is a bit tricky
                NewSuf = []
                for line in other.Data[key]:
                    whites = re.split('[^ ]+', line)
                    s = line.split()
                    if len(s) > 1:
                        for i in range(1, len(s)):
                            s[i] = str(int(s[i]) + self.na)
                    NewSuf.append(
                        ''.join([whites[j] + s[j] for j in range(len(s))]))
                New.Data[key] = copy.deepcopy(self.Data[key]) + NewSuf
            else:
                if type(self.Data[key]) is np.ndarray:
                    New.Data[key] = np.concatenate(
                        (self.Data[key], other.Data[key]))
                elif type(self.Data[key]) is list:
                    New.Data[key] = self.Data[key] + other.Data[key]
                else:
                    raise RuntimeError(
                        'Cannot stack %s because it is of type %s\n' % (key, str(type(New.Data[key]))))
        if 'bonds' in self.Data and 'bonds' in other.Data:
            New.Data['bonds'] = self.bonds + \
                [(b[0] + self.na, b[1] + self.na) for b in other.bonds]
        return New

    def get_populations(self):
        """ Return a cloned molecule object but with X-coordinates set
        to Mulliken charges and Y-coordinates set to Mulliken
        spins. """
        QS = copy.deepcopy(self)
        QSxyz = np.array(QS.xyzs)
        QSxyz[:, :, 0] = self.qm_mulliken_charges
        QSxyz[:, :, 1] = self.qm_mulliken_spins
        QSxyz[:, :, 2] *= 0.0
        QS.xyzs = list(QSxyz)
        return QS

    def load_popxyz(self, fnm):
        """ Given a charge-spin xyz file, load the charges (x-coordinate) and spins (y-coordinate) into internal arrays. """
        QS = Molecule(fnm, ftype='xyz')
        self.qm_mulliken_charges = list(np.array(QS.xyzs)[:, :, 0])
        self.qm_mulliken_spins = list(np.array(QS.xyzs)[:, :, 1])

    def split(self, fnm=None, ftype=None, method="chunks", num=None):
        """ Split the molecule object into a number of separate files
        (chunks), either by specifying the number of frames per chunk
        or the number of chunks.  Only relevant for "trajectories".
        The type of file may be specified; if they aren't specified
        then the original file type is used.

        The output file names are [name].[numbers].[extension] where
        [name] can be specified by passing 'fnm' or taken from the
        object's 'fnm' attribute by default.  [numbers] are integers
        ranging from the lowest to the highest chunk number, prepended
        by zeros.

        If the number of chunks / frames is not specified, then one file
        is written for each frame.

        @return fnms A list of the file names that were written.
        """
        raise NotImplementedError(
            'Apparently this function has not been implemented!!')

    def without(self, *args):
        """
        Return a copy of the Molecule object but with certain fields
        deleted if they exist.  This is useful for when we'd like to
        add Molecule objects that don't share some fields that we know
        about.
        """
        # Create the sum of the two classes by copying the first class.
        New = Molecule()
        for key in AllVariableNames:
            if key in self.Data and key not in args:
                New.Data[key] = copy.deepcopy(self.Data[key])
        return New

    def read_comm_charge_mult(self, verbose=False):
        """ Set charge and multiplicity from reading the comment line, formatted in a specific way. """
        q, sz = extract_qsz(self, verbose=verbose)
        self.charge = q
        self.mult = abs(sz) + 1

    #=====================================#
    #|         Reading functions         |#
    #=====================================#
    def read_xyz(self, fnm, **kwargs):
        """ .xyz files can be TINKER formatted which is why we have the try/except here. """
        try:
            return self.read_xyz0(fnm, **kwargs)
        except:
            return self.read_arc(fnm, **kwargs)

    def read_xyz0(self, fnm, **kwargs):
        """ Parse a .xyz file which contains several xyz coordinates, and return their elements.

        @param[in] fnm The input file name
        @return elem  A list of chemical elements in the XYZ file
        @return comms A list of comments.
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)

        """
        xyz = []
        xyzs = []
        comms = []
        elem = []
        an = 0
        na = 0
        ln = 0
        absln = 0
        for line in open(fnm):
            line = line.strip().expandtabs()
            if ln == 0:
                # Skip blank lines.
                if len(line.strip()) > 0:
                    na = int(line.strip())
            elif ln == 1:
                comms.append(line.strip())
            else:
                line = re.sub(r"([0-9])(-[0-9])", r"\1 \2", line)
                sline = line.split()
                xyz.append([float(i) for i in sline[1:]])
                if len(elem) < na:
                    elem.append(sline[0])
                an += 1
                if an == na:
                    xyzs.append(np.array(xyz))
                    xyz = []
                    an = 0
            if ln == na + 1:
                # Reset the line number counter when we hit the last line in a
                # block.
                ln = -1
            ln += 1
            absln += 1
        Answer = {'elem': elem,
                  'xyzs': xyzs,
                  'comms': comms}
        return Answer

    def read_mdcrd(self, fnm, **kwargs):
        """ Parse an AMBER .mdcrd file.  This requires at least the number of atoms.
        This will FAIL for monatomic trajectories (but who the heck makes those?)

        @param[in] fnm The input file name
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)
        @return boxes Boxes (if present.)

        """
        self.require('na')
        xyz = []
        xyzs = []
        boxes = []
        ln = 0
        for line in open(fnm):
            sline = line.split()
            if ln == 0:
                pass
            else:
                if xyz == [] and len(sline) == 3:
                    a, b, c = (float(i) for i in line.split())
                    boxes.append(
                        BuildLatticeFromLengthsAngles(a, b, c, 90.0, 90.0, 90.0))
                else:
                    xyz += [float(i) for i in line.split()]
                    if len(xyz) == self.na * 3:
                        xyzs.append(np.array(xyz).reshape(-1, 3))
                        xyz = []
            ln += 1
        Answer = {'xyzs': xyzs}
        if len(boxes) > 0:
            Answer['boxes'] = boxes
        return Answer

    def read_inpcrd(self, fnm, **kwargs):
        """ Parse an AMBER .inpcrd or .rst file.

        @param[in] fnm The input file name
        @return xyzs  A list of XYZ coordinates (number of snapshots times number of atoms)
        @return boxes Boxes (if present.)

        """
        xyz = []
        xyzs = []
        # We read in velocities but never use them.
        vel = []
        vels = []
        boxes = []
        ln = 0
        an = 0
        mode = 'x'
        for line in open(fnm):
            line = line.replace('\n', '')
            if ln == 0:
                comms = [line]
            elif ln == 1:
                na = int(line[:5])
            elif mode == 'x':
                xyz.append(
                    [float(line[:12]), float(line[12:24]), float(line[24:36])])
                an += 1
                if an == na:
                    xyzs.append(np.array(xyz))
                    mode = 'v'
                    an = 0
                if len(line) > 36:
                    xyz.append(
                        [float(line[36:48]), float(line[48:60]), float(line[60:72])])
                    an += 1
                    if an == na:
                        xyzs.append(np.array(xyz))
                        mode = 'v'
                        an = 0
            elif mode == 'v':
                vel.append(
                    [float(line[:12]), float(line[12:24]), float(line[24:36])])
                an += 1
                if an == na:
                    vels.append(np.array(vel))
                    mode = 'b'
                    an = 0
                if len(line) > 36:
                    vel.append(
                        [float(line[36:48]), float(line[48:60]), float(line[60:72])])
                    an += 1
                    if an == na:
                        vels.append(np.array(vel))
                        mode = 'b'
                        an = 0
            elif mode == 'b':
                a, b, c = (
                    float(line[:12]), float(line[12:24]), float(line[24:36]))
                boxes.append(
                    BuildLatticeFromLengthsAngles(a, b, c, 90.0, 90.0, 90.0))
            ln += 1
        # If there is only one velocity, then it should actually be a periodic
        # box.
        if len(vel) == 1:
            a, b, c = vel[0]
            boxes.append(
                BuildLatticeFromLengthsAngles(a, b, c, 90.0, 90.0, 90.0))
        Answer = {'xyzs': xyzs, 'comms': comms}
        if len(boxes) > 0:
            Answer['boxes'] = boxes
        return Answer

    def read_qdata(self, fnm, **kwargs):
        xyzs = []
        energies = []
        forces = []
        espxyzs = []
        espvals = []
        interaction = []
        for line in open(fnm):
            line = line.strip().expandtabs()
            if 'COORDS' in line:
                xyzs.append(
                    np.array([float(i) for i in line.split()[1:]]).reshape(-1, 3))
            elif 'FORCES' in line:
                forces.append(
                    np.array([float(i) for i in line.split()[1:]]).reshape(-1, 3))
            elif 'ESPXYZ' in line:
                espxyzs.append(
                    np.array([float(i) for i in line.split()[1:]]).reshape(-1, 3))
            elif 'ESPVAL' in line:
                espvals.append(np.array([float(i) for i in line.split()[1:]]))
            elif 'ENERGY' in line:
                energies.append(float(line.split()[1]))
            elif 'INTERACTION' in line:
                interaction.append(float(line.split()[1]))
        Answer = {}
        if len(xyzs) > 0:
            Answer['xyzs'] = xyzs
        if len(energies) > 0:
            Answer['qm_energies'] = energies
        if len(interaction) > 0:
            Answer['qm_interaction'] = interaction
        if len(forces) > 0:
            Answer['qm_grads'] = forces
        if len(espxyzs) > 0:
            Answer['qm_espxyzs'] = espxyzs
        if len(espvals) > 0:
            Answer['qm_espvals'] = espvals
        return Answer

    def read_mol2(self, fnm, **kwargs):
        xyz = []
        charge = []
        atomname = []
        atomtype = []
        elem = []
        data = Mol2.mol2_set(fnm)
        if len(data.compounds) > 1:
            sys.stderr.write(
                "Not sure what to do if the MOL2 file contains multiple compounds\n")
        for i, atom in enumerate(data.compounds.items()[0][1].atoms):
            xyz.append([atom.x, atom.y, atom.z])
            charge.append(atom.charge)
            atomname.append(atom.atom_name)
            atomtype.append(atom.atom_type)
            thiselem = atom.atom_name
            if len(thiselem) > 1:
                thiselem = thiselem[0] + re.sub('[A-Z0-9]', '', thiselem[1:])
            elem.append(thiselem)

        resname = [list(data.compounds.items())[0][0] for i in range(len(elem))]
        resid = [1 for i in range(len(elem))]

        # Deprecated 'abonds' format.
        # bonds    = [[] for i in range(len(elem))]
        # for bond in data.compounds.items()[0][1].bonds:
        #     a1 = bond.origin_atom_id - 1
        #     a2 = bond.target_atom_id - 1
        #     aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
        #     bonds[aL].append(aH)

        bonds = []
        for bond in data.compounds.items()[0][1].bonds:
            a1 = bond.origin_atom_id - 1
            a2 = bond.target_atom_id - 1
            aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
            bonds.append((aL, aH))

        self.top_settings["read_bonds"] = True
        Answer = {'xyzs': [np.array(xyz)],
                  'partial_charge': charge,
                  'atomname': atomname,
                  'atomtype': atomtype,
                  'elem': elem,
                  'resname': resname,
                  'resid': resid,
                  'bonds': bonds
                  }

        return Answer

    def read_com(self, fnm, **kwargs):
        """ Parse a Gaussian .com file and return a SINGLE-ELEMENT list of xyz coordinates (no multiple file support)

        @param[in] fnm The input file name
        @return elem   A list of chemical elements in the XYZ file
        @return comms  A single-element list for the comment.
        @return xyzs   A single-element list for the  XYZ coordinates.
        @return charge The total charge of the system.
        @return mult   The spin multiplicity of the system.

        """
        elem = []
        xyz = []
        ln = 0
        absln = 0
        comfile = open(fnm).readlines()
        inxyz = 0
        for line in comfile:
            line = line.strip().expandtabs()
            # Everything after exclamation point is a comment
            sline = line.split('!')[0].split()
            if len(sline) == 2:
                if isint(sline[0]) and isint(sline[1]):
                    charge = int(sline[0])
                    mult = int(sline[1])
                    title_ln = ln - 2
            elif len(sline) == 4:
                inxyz = 1
                if sline[0].capitalize() in PeriodicTable and isfloat(sline[1]) and isfloat(sline[2]) and isfloat(sline[3]):
                    elem.append(sline[0])
                    xyz.append(
                        np.array([float(sline[1]), float(sline[2]), float(sline[3])]))
            elif inxyz:
                break
            ln += 1
            absln += 1

        Answer = {'xyzs': [np.array(xyz)],
                  'elem': elem,
                  'comms': [comfile[title_ln].strip()],
                  'charge': charge,
                  'mult': mult}
        return Answer

    def read_arc(self, fnm, **kwargs):
        """ Read a TINKER .arc file.

        @param[in] fnm  The input file name
        @return xyzs    A list for the  XYZ coordinates.
        @return boxes   A list of periodic boxes (newer .arc files have these)
        @return resid   The residue ID numbers.  These are not easy to get!
        @return elem    A list of chemical elements in the XYZ file
        @return comms   A single-element list for the comment.
        @return tinkersuf  The suffix that comes after lines in the XYZ coordinates; this is usually topology info

        """
        tinkersuf = []
        boxes = []
        xyzs = []
        xyz = []
        resid = []
        elem = []
        comms = []
        thisres = set([])
        forwardres = set([])
        title = True
        nframes = 0
        thisresid = 1
        ln = 0
        thisatom = 0
        for line in open(fnm):
            line = line.strip().expandtabs()
            sline = line.split()
            if len(sline) == 0:
                continue
            # The first line always contains the number of atoms
            # The words after the first line are comments
            if title:
                na = int(sline[0])
                comms.append(' '.join(sline[1:]))
                title = False
            elif len(sline) >= 5:
                # Newer .arc files have a .box line.
                if len(sline) == 6 and isfloat(sline[1]) and all([isfloat(i) for i in sline]):
                    a, b, c, alpha, beta, gamma = (float(i) for i in sline[:6])
                    boxes.append(
                        BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma))
                # A line of data better look like this
                elif isint(sline[0]) and isfloat(sline[2]) and isfloat(sline[3]) and isfloat(sline[4]):
                    if nframes == 0:
                        elem.append(elem_from_atomname(sline[1]))
                        resid.append(thisresid)
                        whites = re.split('[^ ]+', line)
                        if len(sline) > 5:
                            s = sline[5:]
                            if len(s) > 1:
                                sn = [int(i) for i in s[1:]]
                                s = [s[0]] + \
                                    list(np.array(s[1:])[np.argsort(sn)])
                            tinkersuf.append(
                                ''.join([whites[j] + s[j - 5] for j in range(5, len(sline))]))
                        else:
                            tinkersuf.append('')
                    # LPW Make sure ..
                    thisatom += 1
                    #thisatom = int(sline[0])
                    thisres.add(thisatom)
                    forwardres.add(thisatom)
                    if len(sline) >= 6:
                        forwardres.update([int(j) for j in sline[6:]])
                    if thisres == forwardres:
                        thisres = set([])
                        forwardres = set([])
                        thisresid += 1
                    xyz.append(
                        [float(sline[2]), float(sline[3]), float(sline[4])])
                    if thisatom == na:
                        thisatom = 0
                        nframes += 1
                        title = True
                        xyzs.append(np.array(xyz))
                        xyz = []
            ln += 1
        Answer = {'xyzs': xyzs,
                  'resid': resid,
                  'elem': elem,
                  'comms': comms,
                  'tinkersuf': tinkersuf}
        if len(boxes) > 0:
            Answer['boxes'] = boxes
        return Answer

    def read_gro(self, fnm, **kwargs):
        """ Read a GROMACS .gro file.

        """
        xyzs = []
        # The element, most useful for quantum chemistry calculations
        elem = []
        atomname = []  # The atom name, for instance 'HW1'
        comms = []
        resid = []
        resname = []
        boxes = []
        xyz = []
        ln = 0
        frame = 0
        absln = 0
        na = -10
        for line in open(fnm):
            sline = line.split()
            if ln == 0:
                comms.append(line.strip())
            elif ln == 1:
                na = int(line.strip())
            elif ln == na + 2:
                box = [float(i) * 10 for i in sline]
                if len(box) == 3:
                    a = box[0]
                    b = box[1]
                    c = box[2]
                    alpha = 90.0
                    beta = 90.0
                    gamma = 90.0
                    boxes.append(
                        BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma))
                elif len(box) == 9:
                    v1 = np.array([box[0], box[3], box[4]])
                    v2 = np.array([box[5], box[1], box[6]])
                    v3 = np.array([box[7], box[8], box[2]])
                    boxes.append(BuildLatticeFromVectors(v1, v2, v3))
                xyzs.append(np.array(xyz) * 10)
                xyz = []
                ln = -1
                frame += 1
            else:
                coord = []
                # Create the list of residues, atom names etc. only if it's the
                # first frame.
                if frame == 0:
                    # Name of the residue, for instance '153SOL1 -> SOL1' ;
                    # strips leading numbers
                    thisresid = int(line[0:5].strip())
                    resid.append(thisresid)
                    thisresname = line[5:10].strip()
                    resname.append(thisresname)
                    thisatomname = line[10:15].strip()
                    atomname.append(thisatomname)

                    pdeci = [i for i, x in enumerate(line) if x == '.']
                    ndeci = pdeci[1] - pdeci[0] - 5

                    thiselem = sline[1]
                    if len(thiselem) > 1:
                        thiselem = thiselem[
                            0] + re.sub('[A-Z0-9]', '', thiselem[1:])
                    elem.append(thiselem)

                for i in range(1, 4):
                    try:
                        thiscoord = float(
                            line[(pdeci[0] - 4) + (5 + ndeci) * (i - 1):(pdeci[0] - 4) + (5 + ndeci) * i].strip())
                    except:  # Attempt to read incorrectly formatted GRO files.
                        thiscoord = float(line.split()[i + 2])
                    coord.append(thiscoord)
                xyz.append(coord)

            ln += 1
            absln += 1
        Answer = {'xyzs': xyzs,
                  'elem': elem,
                  'atomname': atomname,
                  'resid': resid,
                  'resname': resname,
                  'boxes': boxes,
                  'comms': comms}
        return Answer

    def read_charmm(self, fnm, **kwargs):
        """ Read a CHARMM .cor (or .crd) file.

        """
        xyzs = []
        # The element, most useful for quantum chemistry calculations
        elem = []
        atomname = []  # The atom name, for instance 'HW1'
        comms = []
        resid = []
        resname = []
        xyz = []
        thiscomm = []
        ln = 0
        frame = 0
        an = 0
        for line in open(fnm):
            line = line.strip().expandtabs()
            sline = line.split()
            if re.match('^\*', line):
                if len(sline) == 1:
                    comms.append(';'.join(list(thiscomm)))
                    thiscomm = []
                else:
                    thiscomm.append(' '.join(sline[1:]))
            elif re.match('^ *[0-9]+ +(EXT)?$', line):
                na = int(sline[0])
            elif is_charmm_coord(line):
                # Create the list of residues, atom names etc. only if it's the
                # first frame.
                if frame == 0:
                    resid.append(sline[1])
                    resname.append(sline[2])
                    atomname.append(sline[3])
                    thiselem = sline[3]
                    if len(thiselem) > 1:
                        thiselem = thiselem[
                            0] + re.sub('[A-Z0-9]', '', thiselem[1:])
                    elem.append(thiselem)
                xyz.append([float(i) for i in sline[4:7]])
                an += 1
                if an == na:
                    xyzs.append(np.array(xyz))
                    xyz = []
                    an = 0
                    frame += 1
            ln += 1
        Answer = {'xyzs': xyzs,
                  'elem': elem,
                  'atomname': atomname,
                  'resid': resid,
                  'resname': resname,
                  'comms': comms}
        return Answer

    def read_qcin(self, fnm, **kwargs):
        """ Read a Q-Chem input file.

        These files can be very complicated, and I can't write a completely
        general parser for them.  It is important to keep our goal in
        mind:

        1) The main goal is to convert a trajectory to Q-Chem input
        files with identical calculation settings.

        2) When we print the Q-Chem file, we should preserve the line
        ordering of the 'rem' section, but also be able to add 'rem'
        options at the end.

        3) We should accommodate the use case that the Q-Chem file may have
        follow-up calculations delimited by '@@@@'.

        4) We can read in all of the xyz's as a trajectory, but only the
        Q-Chem settings belonging to the first xyz will be saved.

        """

        qcrem = OrderedDict()
        qcrems = []
        xyz = []
        xyzs = []
        elem = []
        section = None
        # The Z-matrix printing in new versions throws me off.
        zmatrix = False
        template = []
        fff = False
        inside_section = False
        reading_template = True
        charge = 0
        mult = 0
        Answer = {}
        SectionData = []
        template_cut = 0
        readsuf = True
        # The suffix, which comes after every atom line in the $molecule
        # section, is for determining the MM atom type and topology.
        suffix = []
        # If the element in the $molecule section is preceded by an '@' sign,
        # it's a ghost atom for counterpoise calculations.
        ghost = []
        infsm = False

        for line in open(fnm).readlines():
            line = line.strip().expandtabs()
            sline = line.split()
            dline = line.split('!')[0].split()
            if "Z-matrix Print" in line:
                zmatrix = True
            if re.match('^\$', line):
                wrd = re.sub('\$', '', line)
                if wrd == 'end':
                    zmatrix = False
                    inside_section = False
                    if section == 'molecule':
                        if len(xyz) > 0:
                            xyzs.append(np.array(xyz))
                        xyz = []
                        fff = True
                        if suffix != []:
                            readsuf = False
                    elif section == 'rem':
                        if reading_template:
                            qcrems.append(qcrem)
                            qcrem = OrderedDict()
                    if reading_template:
                        # Ignore the external charges section because it varies
                        # from frame to frame.
                        if section != 'external_charges':
                            template.append((section, SectionData))
                    SectionData = []
                else:
                    section = wrd
                    inside_section = True
            elif inside_section:
                if section == 'molecule' and not zmatrix:
                    if line.startswith("*"):
                        infsm = True
                    if (not infsm) and (len(dline) >= 4 and all([isfloat(dline[i]) for i in range(1, 4)])):
                        if fff:
                            reading_template = False
                            template_cut = list(
                                i for i, dat in enumerate(template) if dat[0] == '@@@@')[-1]
                        else:
                            # This is a ghost atom
                            if re.match('^@', sline[0]):
                                ghost.append(True)
                            else:
                                ghost.append(False)
                            elem.append(int(re.sub('@', '', sline[0])))
                        xyz.append([float(i) for i in sline[1:4]])
                        if readsuf and len(sline) > 4:
                            whites = re.split('[^ ]+', line)
                            suffix.append(
                                ''.join([whites[j] + sline[j] for j in range(4, len(sline))]))
                    elif re.match("[+-]?[0-9]+ +[0-9]+$", line.split('!')[0].strip()):
                        if not fff:
                            charge = int(sline[0])
                            mult = int(sline[1])
                    else:
                        SectionData.append(line)
                elif reading_template and not zmatrix:
                    if section == 'basis':
                        SectionData.append(line.split('!')[0])
                    elif section == 'rem':
                        S = splitter.findall(line)
                        if len(S) > 0:
                            if S[0] == '!':
                                qcrem[''.join(S[0:3]).lower()] = ''.join(S[4:])
                            else:
                                qcrem[S[0].lower()] = ''.join(S[2:])
                    else:
                        SectionData.append(line)
            elif re.match('^@+$', line) and reading_template:
                template.append(('@@@@', []))
            elif re.match('Welcome to Q-Chem', line) and reading_template and fff:
                template.append(('@@@@', []))

        if template_cut != 0:
            template = template[:template_cut]

        Answer = {'qctemplate': template,
                  'qcrems': qcrems,
                  'charge': charge,
                  'mult': mult,
                  }
        if suffix != []:
            Answer['qcsuf'] = suffix

        if len(xyzs) > 0:
            Answer['xyzs'] = xyzs
        else:
            Answer['xyzs'] = [np.array([])]
        if len(elem) > 0:
            Answer['elem'] = elem
        if len(ghost) > 0:
            Answer['qm_ghost'] = ghost
        return Answer

    def read_qcesp(self, fnm, **kwargs):
        espxyz = []
        espval = []
        for line in open(fnm):
            line = line.strip().expandtabs()
            sline = line.split()
            if len(sline) == 4 and all([isfloat(sline[i]) for i in range(4)]):
                espxyz.append([float(sline[i]) for i in range(3)])
                espval.append(float(sline[3]))
        Answer = {'qm_espxyzs': [np.array(espxyz) * bohrang],
                  'qm_espvals': [np.array(espval)]
                  }
        return Answer

    def read_qcout(self, fnm, errok=[], **kwargs):
        """ Q-Chem output file reader, adapted for our parser.

        Q-Chem output files are very flexible and there's no way I can account for all of them.  Here's what
        I am able to account for:

        A list of:
        - Coordinates
        - Energies
        - Forces

        Calling with errok will proceed with reading file even if the specified error messages are encountered.

        Note that each step in a geometry optimization counts as a frame.

        As with all Q-Chem output files, note that successive calculations can have different numbers of atoms.

        """

        Answer = {}
        xyzs = []
        xyz = []
        elem = []
        elemThis = []
        mkchg = []
        mkspn = []
        mkchgThis = []
        mkspnThis = []
        frqs = []
        modes = []
        XMode = 0
        MMode = 0
        VMode = 0
        conv = []
        convThis = 0
        readChargeMult = 0
        energy_scf = []
        float_match = {'energy_scfThis': ("^[1-9][0-9]* +[-+]?([0-9]*\.)?[0-9]+ +[-+]?([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)[A-Za-z0 ]*$", 1),
                       'energy_opt': ("^Final energy is +[-+]?([0-9]*\.)?[0-9]+$", -1),
                       'charge': ("Sum of atomic charges", -1),
                       'mult': ("Sum of spin +charges", -1),
                       'energy_mp2': ("^(ri)*(-)*mp2 +total energy += +[-+]?([0-9]*\.)?[0-9]+ +au$", -2),
                       'energy_ccsd': ("^CCSD Total Energy += +[-+]?([0-9]*\.)?[0-9]+$", -1),
                       'energy_ccsdt': ("^CCSD\(T\) Total Energy += +[-+]?([0-9]*\.)?[0-9]+$", -1),
                       }
        matrix_match = {'analytical_grad': 'Full Analytical Gradient',
                        'gradient_scf': 'Gradient of SCF Energy',
                        'gradient_mp2': 'Gradient of MP2 Energy',
                        'gradient_dualbas': 'Gradient of the Dual-Basis Energy',
                        'hessian_scf': 'Hessian of the SCF Energy',
                        'mayer': 'Mayer SCF Bond Order'
                        }
        qcrem = OrderedDict()

        matblank = {'match': '', 'All': [], 'This': [], 'Strip': [], 'Mode': 0}
        Mats = {}
        Floats = {}
        for key, val in matrix_match.items():
            Mats[key] = copy.deepcopy(matblank)
        for key, val in float_match.items():
            Floats[key] = []

        # Detect freezing string
        FSM = False
        # Intrinsic reaction coordinate stuff
        IRCDir = 0
        RPLine = False
        #---- Intrinsic reaction coordinate data.
        # stat: Status, X : Coordinates, E : Energies, Q : Charges, Sz: Spin-Z
        # Explanation of Status:
        # -1 : IRC calculation does not exist in this direction.
        #  0 : IRC calculation finished successfully.
        #  1 : IRC calculation did not finish but we can start a geometry optimization from the final point.
        #  2 : IRC calculation failed in this direction (i.e. set to 2 once we encounter first_irc_step).
        # Two dictionaries of coordinates, energies, Mulliken Charges and Spin
        # Populations.
        IRCData = [OrderedDict(
            [('stat', -1), ('X', []), ('E', []), ('Q', []), ('Sz', [])]) for i in range(2)]

        Answer['qcerr'] = ''
        fatal = 0
        for line in open(fnm):
            line = line.strip().expandtabs()
            if 'Welcome to Q-Chem' in line:
                Answer['qcerr'] = ''
            if 'total processes killed' in line:
                Answer['qcerr'] = 'killed'
            if fatal and len(line.split()) > 0:
                # Print the error message that comes after the "fatal error"
                # line.
                if line in errok:
                    Answer['qcerr'] = line.strip()
                    fatal = 0
                else:
                    raise RuntimeError(
                        'Calculation encountered a fatal error! (%s)\n' % line)
            if 'Q-Chem fatal error' in line:
                fatal = 1
            if XMode >= 1:
                # Perfectionist here; matches integer, element, and three
                # floating points
                if re.match("^[0-9]+ +[A-Z][A-Za-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", line):
                    XMode = 2
                    sline = line.split()
                    elemThis.append(sline[1])
                    xyz.append([float(i) for i in sline[2:]])
                # Break out of the loop if we encounter anything other than
                # atomic data
                elif XMode == 2:
                    if elem == []:
                        elem = elemThis
                    elif elem != elemThis:
                        raise RuntimeError(
                            'Q-Chem output parser will not work if successive calculations have different numbers of atoms!\n')
                    elemThis = []
                    xyzs.append(np.array(xyz))
                    xyz = []
                    XMode = 0
            elif re.match("Standard Nuclear Orientation".lower(), line.lower()):
                XMode = 1
            if MMode >= 1:
                # Perfectionist here; matches integer, element, and two
                # floating points
                if re.match("^[0-9]+ +[A-Z][a-z]?( +[-+]?([0-9]*\.)?[0-9]+){2}$", line):
                    MMode = 2
                    sline = line.split()
                    mkchgThis.append(float(sline[2]))
                    mkspnThis.append(float(sline[3]))
                elif re.match("^[0-9]+ +[A-Z][a-z]?( +[-+]?([0-9]*\.)?[0-9]+){1}$", line):
                    MMode = 2
                    sline = line.split()
                    mkchgThis.append(float(sline[2]))
                    mkspnThis.append(0.0)
                # Break out of the loop if we encounter anything other than
                # Mulliken charges
                elif MMode == 2:
                    mkchg.append(mkchgThis[:])
                    mkspn.append(mkspnThis[:])
                    mkchgThis = []
                    mkspnThis = []
                    MMode = 0
            elif re.match("Ground-State Mulliken Net Atomic Charges".lower(), line.lower()):
                MMode = 1
            for key, val in float_match.items():
                if re.match(val[0].lower(), line.lower()):
                    Floats[key].append(float(line.split()[val[1]]))
            #----- Begin Intrinsic reaction coordinate stuff
            if line.startswith('IRC') and IRCData[IRCDir]['stat'] == -1:
                IRCData[IRCDir]['stat'] = 2
            if "Reaction path following." in line:
                RPLine = True
                IRCData[IRCDir]['X'].append(xyzs[-1])
            # Assumes the IRC energy comes right after the coordinates.
            elif RPLine:
                RPLine = False
                IRCData[IRCDir]['E'].append(float(line.split()[3]))
                IRCData[IRCDir]['Q'].append(mkchg[-1])
                IRCData[IRCDir]['Sz'].append(mkspn[-1])
            # Geometry optimization info can also get appended to IRC data.
            # This is because my qchem.py script recovers IRC jobs
            # that have failed from SCF convergence failures with geometry
            # optimizations.
            if "GEOMETRY OPTIMIZATION" in line:
                IRCData[IRCDir]['X'].append(xyzs[-1])
                IRCData[IRCDir]['E'].append(energy_scf[-1])
                IRCData[IRCDir]['Q'].append(mkchg[-1])
                IRCData[IRCDir]['Sz'].append(mkspn[-1])
            # Determine whether we are in the forward or the backward part of
            # the IRC.
            if "IRC -- convergence criterion reached." in line or "OPTIMIZATION CONVERGED" in line:
                IRCData[IRCDir]['stat'] = 0
                IRCDir = 1
            if "MAXIMUM OPTIMIZATION CYCLES REACHED" in line:
                IRCData[IRCDir]['stat'] = 1
            # Output file indicates whether we can start a geometry
            # optimization from this point.
            if "geom opt from" in line:
                IRCData[IRCDir]['stat'] = 1
                IRCDir = 1
            #----- End IRC stuff
            # Look for SCF energy
            # Note that COSMO has two SCF energies per calculation so this parser won't work.
            # Need to think of a better way.
            if re.match(".*Convergence criterion met$".lower(), line.lower()):
                conv.append(1)
                energy_scf.append(Floats['energy_scfThis'][-1])
                Floats['energy_scfThis'] = []
            elif re.match(".*Including correction$".lower(), line.lower()):
                energy_scf[-1] = Floats['energy_scfThis'][-1]
                Floats['energy_scfThis'] = []
            elif re.match(".*Convergence failure$".lower(), line.lower()):
                conv.append(0)
                Floats['energy_scfThis'] = []
                energy_scf.append(0.0)
            #----- If doing freezing string calculation, do NOT treat as a geometry optimization.
            if 'Starting FSM Calculation' in line:
                FSM = True
            #----- Vibrational stuff
            VModeNxt = None
            if 'VIBRATIONAL ANALYSIS' in line:
                VMode = 1
            if VMode > 0 and line.strip().startswith('Mode:'):
                VMode = 2
            if VMode == 2:
                s = line.split()
                if 'Frequency:' in line:
                    nfrq = len(s) - 1
                    frqs += [float(i) for i in s[1:]]
                if re.match('^X +Y +Z', line):
                    VModeNxt = 3
                    readmodes = [[] for i in range(nfrq)]
                if 'Imaginary Frequencies' in line:
                    VMode = 0
            if VMode == 3:
                s = line.split()
                if len(s) != nfrq * 3 + 1:
                    VMode = 2
                    modes += readmodes[:]
                elif 'TransDip' not in s:
                    for i in range(nfrq):
                        readmodes[i].append(
                            [float(s[j]) for j in range(1 + 3 * i, 4 + 3 * i)])
            if VModeNxt != None:
                VMode = VModeNxt
            for key, val in matrix_match.items():
                if Mats[key]["Mode"] >= 1:
                    # Match any number of integers on a line.  This signifies a
                    # column header to start the matrix
                    if re.match("^[0-9]+( +[0-9]+)*$", line):
                        Mats[key]["This"] = add_strip_to_mat(
                            Mats[key]["This"], Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                        Mats[key]["Mode"] = 2
                    # Match a single integer followed by any number of floats.
                    # This is a strip of data to be added to the matrix
                    elif re.match("^[0-9]+( +[-+]?([0-9]*\.)?[0-9]+)+$", line):
                        Mats[key]["Strip"].append(
                            [float(i) for i in line.split()[1:]])
                    # In any other case, the matrix is terminated.
                    elif Mats[key]["Mode"] >= 2:
                        Mats[key]["This"] = add_strip_to_mat(
                            Mats[key]["This"], Mats[key]["Strip"])
                        Mats[key]["Strip"] = []
                        Mats[key]["All"].append(np.array(Mats[key]["This"]))
                        Mats[key]["This"] = []
                        Mats[key]["Mode"] = 0
                elif re.match(val.lower(), line.lower()):
                    Mats[key]["Mode"] = 1

        if len(Floats['mult']) == 0:
            Floats['mult'] = [0]

        # Copy out the coordinate lists; Q-Chem output cannot be trusted to get
        # the chemical elements
        Answer['xyzs'] = xyzs
        Answer['elem'] = elem
        # Read the output file as an input file to get a Q-Chem template.
        Aux = self.read_qcin(fnm)
        for i in ['qctemplate', 'qcrems', 'elem', 'qm_ghost', 'charge', 'mult']:
            if i in Aux:
                Answer[i] = Aux[i]
        # Copy out the charge and multiplicity
        if len(Floats['charge']) > 0:
            Answer['charge'] = int(Floats['charge'][0])
        if len(Floats['mult']) > 0:
            Answer['mult'] = int(Floats['mult'][0]) + 1
        # Copy out the energies and forces
        # Q-Chem can print out gradients with several different headings.
        # We start with the most reliable heading and work our way down.
        if len(Mats['analytical_grad']['All']) > 0:
            Answer['qm_grads'] = Mats['analytical_grad']['All']
        elif len(Mats['gradient_mp2']['All']) > 0:
            Answer['qm_grads'] = Mats['gradient_mp2']['All']
        elif len(Mats['gradient_dualbas']['All']) > 0:
            Answer['qm_grads'] = Mats['gradient_dualbas']['All']
        elif len(Mats['gradient_scf']['All']) > 0:
            Answer['qm_grads'] = Mats['gradient_scf']['All']
        # Mayer bond order matrix from SCF_FINAL_PRINT=1
        if len(Mats['mayer']['All']) > 0:
            Answer['qm_bondorder'] = Mats['mayer']['All'][-1]
        if len(Mats['hessian_scf']['All']) > 0:
            Answer['qm_hessians'] = Mats['hessian_scf']['All']
        # else:
        #    raise RuntimeError('There are no forces in %s' % fnm)
        # Also work our way down with the energies.
        if len(Floats['energy_ccsdt']) > 0:
            Answer['qm_energies'] = Floats['energy_ccsdt']
        elif len(Floats['energy_ccsd']) > 0:
            Answer['qm_energies'] = Floats['energy_ccsd']
        elif len(Floats['energy_mp2']) > 0:
            Answer['qm_energies'] = Floats['energy_mp2']
        elif len(energy_scf) > 0:
            if 'correlation' in Answer['qcrems'][0] and Answer['qcrems'][0]['correlation'].lower() in ['mp2', 'rimp2', 'ccsd', 'ccsd(t)']:
                raise RuntimeError(
                    "Q-Chem was called with a post-HF theory but we only got the SCF energy\n")
            Answer['qm_energies'] = energy_scf
        elif 'SCF failed to converge' not in errok:
            raise RuntimeError('There are no energies in %s\n' % fnm)

        # Sanity checks
        # We currently don't have a graceful way of dealing with SCF convergence failures in the output file.
        # For instance, a failed calculation will have elem / xyz but no
        # forces. :/
        if 0 in conv and 'SCF failed to converge' not in errok:
            raise RuntimeError(
                'SCF convergence failure encountered in parsing %s\n' % fnm)
        elif (0 not in conv):
            # The molecule should have only one charge and one multiplicity
            if len(set(Floats['charge'])) != 1 or len(set(Floats['mult'])) != 1:
                raise RuntimeError(
                    'Unexpected number of charges or multiplicities in parsing %s\n' % fnm)

        # If we have any QM energies (not the case if SCF convergence failure)
        if 'qm_energies' in Answer:
            # Catch the case of failed geometry optimizations.
            if len(Answer['xyzs']) == len(Answer['qm_energies']) + 1:
                Answer['xyzs'] = Answer['xyzs'][:-1]
            # Catch the case of freezing string method, it prints out two extra
            # coordinates.
            if len(Answer['xyzs']) == len(Answer['qm_energies']) + 2:
                for i in range(2):
                    Answer['qm_energies'].append(0.0)
                    mkchg.append([0.0 for j in mkchg[-1]])
                    mkspn.append([0.0 for j in mkchg[-1]])
            lens = [len(i) for i in (Answer['qm_energies'], Answer['xyzs'])]
            if len(set(lens)) != 1:
                raise RuntimeError(
                    'The number of energies and coordinates in %s are not the same : %s\n' % (fnm, str(lens)))

        # The number of atoms should all be the same
        if len(set([len(i) for i in Answer['xyzs']])) > 1:
            raise RuntimeError(
                'The numbers of atoms across frames in %s are not all the same\n' % (fnm))

        if 'qm_grads' in Answer:
            for i, frc in enumerate(Answer['qm_grads']):
                Answer['qm_grads'][i] = frc.T
            for i in np.where(np.array(conv) == 0)[0]:
                Answer['qm_grads'].insert(i, Answer['qm_grads'][0] * 0.0)
            if len(Answer['qm_grads']) != len(Answer['qm_energies']):
                warn(
                    "Number of energies and gradients is inconsistent (composite jobs?)  Deleting gradients.")
                del Answer['qm_grads']
        # A strange peculiarity; Q-Chem sometimes prints out the final Mulliken
        # charges a second time, after the geometry optimization.
        if mkchg != []:
            Answer['qm_mulliken_charges'] = list(np.array(mkchg))
            for i in np.where(np.array(conv) == 0)[0]:
                Answer['qm_mulliken_charges'].insert(
                    i, np.array([0.0 for i in mkchg[-1]]))
            Answer['qm_mulliken_charges'] = Answer[
                'qm_mulliken_charges'][:len(Answer['qm_energies'])]
        if mkspn != []:
            Answer['qm_mulliken_spins'] = list(np.array(mkspn))
            for i in np.where(np.array(conv) == 0)[0]:
                Answer['qm_mulliken_spins'].insert(
                    i, np.array([0.0 for i in mkspn[-1]]))
            Answer['qm_mulliken_spins'] = Answer[
                'qm_mulliken_spins'][:len(Answer['qm_energies'])]

        Answer['Irc'] = IRCData
        if len(modes) > 0:
            unnorm = [np.array(i) for i in modes]
            Answer['freqs'] = np.array(frqs)
            Answer['modes'] = [i / np.linalg.norm(i) for i in unnorm]

        return Answer

    #=====================================#
    #|         Writing functions         |#
    #=====================================#

    def write_qcin(self, select, **kwargs):
        self.require('qctemplate', 'qcrems', 'charge', 'mult')
        out = []
        if 'read' in kwargs:
            read = kwargs['read']
        else:
            read = False
        for SI, I in enumerate(select):
            fsm = False
            remidx = 0
            molecule_printed = False
            # Each 'extchg' has number_of_atoms * 4 elements corresponding to
            # x, y, z, q.
            if 'qm_extchgs' in self.Data:
                extchg = self.qm_extchgs[I]
                out.append('$external_charges')
                for i in range(len(extchg)):
                    out.append("% 15.10f % 15.10f % 15.10f %15.10f" % (
                        extchg[i, 0], extchg[i, 1], extchg[i, 2], extchg[i, 3]))
                out.append('$end')
            for SectName, SectData in self.qctemplate:
                if 'jobtype' in self.qcrems[remidx] and self.qcrems[remidx]['jobtype'].lower() == 'fsm':
                    fsm = True
                    if len(select) != 2:
                        raise RuntimeError(
                            'For freezing string method, please provide two structures only.\n')
                if SectName != '@@@@':
                    out.append('$%s' % SectName)
                    for line in SectData:
                        out.append(line)
                    if SectName == 'molecule':
                        if molecule_printed == False:
                            molecule_printed = True
                            if read:
                                out.append("read")
                            elif self.na > 0:
                                out.append("%i %i" % (self.charge, self.mult))
                                an = 0
                                for e, x in zip(self.elem, self.xyzs[I]):
                                    pre = '@' if (
                                        'qm_ghost' in self.Data and self.Data['qm_ghost'][an]) else ''
                                    suf = self.Data['qcsuf'][
                                        an] if 'qcsuf' in self.Data else ''
                                    out.append(
                                        pre + format_xyz_coord(e, x) + suf)
                                    an += 1
                                if fsm:
                                    out.append("****")
                                    an = 0
                                    for e, x in zip(self.elem, self.xyzs[select[SI + 1]]):
                                        pre = '@' if (
                                            'qm_ghost' in self.Data and self.Data['qm_ghost'][an]) else ''
                                        suf = self.Data['qcsuf'][
                                            an] if 'qcsuf' in self.Data else ''
                                        out.append(
                                            pre + format_xyz_coord(e, x) + suf)
                                        an += 1
                    if SectName == 'rem':
                        for key, val in self.qcrems[remidx].items():
                            out.append("%-21s %-s" % (key, str(val)))
                    if SectName == 'comments' and 'comms' in self.Data:
                        out.append(self.comms[I])
                    out.append('$end')
                else:
                    remidx += 1
                    out.append('@@@@')
                out.append('')
            # if I < (len(self) - 1):
            if fsm:
                break
            if I != select[-1]:
                out.append('@@@@')
                out.append('')
        return out

    def write_xyz(self, select, **kwargs):
        self.require('elem', 'xyzs')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            out.append("%-5i" % self.na)
            out.append(self.comms[I])
            for i in range(self.na):
                out.append(format_xyz_coord(self.elem[i], xyz[i]))
        return out

    def write_molproq(self, select, **kwargs):
        self.require('xyzs', 'partial_charge')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            # Comment comes first, then number of atoms.
            out.append(self.comms[I])
            out.append("%-5i" % self.na)
            for i in range(self.na):
                out.append("% 15.10f % 15.10f % 15.10f % 15.10f   0" % (
                    xyz[i, 0], xyz[i, 1], xyz[i, 2], self.partial_charge[i]))
        return out

    def write_mdcrd(self, select, **kwargs):
        self.require('xyzs')
        # In mdcrd files, there is only one comment line
        out = ['mdcrd file generated using ForceBalance']
        for I in select:
            xyz = self.xyzs[I]
            out += [''.join(["%8.3f" % i for i in g])
                    for g in grouper(10, list(xyz.flatten()))]
            if 'boxes' in self.Data:
                out.append(''.join(
                    ["%8.3f" % i for i in [self.boxes[I].a, self.boxes[I].b, self.boxes[I].c]]))
        return out

    def write_inpcrd(self, select, sn=None, **kwargs):
        self.require('xyzs')
        if len(self.xyzs) != 1 and sn == None:
            raise RuntimeError(
                "inpcrd can only be written for a single-frame trajectory\n")
        if sn != None:
            self.xyzs = [self.xyzs[sn]]
            self.comms = [self.comms[sn]]
        # In inp files, there is only one comment line
        # I believe 20A4 means 80 characters.
        out = [self.comms[0][:80], '%5i' % self.na]
        xyz = self.xyzs[0]
        strout = ''
        for ix, x in enumerate(xyz):
            strout += "%12.7f%12.7f%12.7f" % (x[0], x[1], x[2])
            if ix % 2 == 1 or ix == (len(xyz) - 1):
                out.append(strout)
                strout = ''
        # From reading the AMBER file specification I am not sure if this is
        # correct.
        if 'boxes' in self.Data:
            out.append(''.join(
                ["%12.7f" % i for i in [self.boxes[0].a, self.boxes[0].b, self.boxes[0].c]]))
        return out

    def write_arc(self, select, **kwargs):
        self.require('elem', 'xyzs')
        out = []
        if 'tinkersuf' not in self.Data:
            sys.stderr.write(
                "Beware, this .arc file contains no atom type or topology info\n")
        for I in select:
            xyz = self.xyzs[I]
            out.append("%6i  %s" % (self.na, self.comms[I]))
            if 'boxes' in self.Data:
                b = self.boxes[I]
                out.append(" %11.6f %11.6f %11.6f %11.6f %11.6f %11.6f" % (
                    b.a, b.b, b.c, b.alpha, b.beta, b.gamma))
            for i in range(self.na):
                out.append("%6i  %s%s" % (i + 1, format_xyz_coord(self.elem[i], xyz[
                           i], tinker=True), self.tinkersuf[i] if 'tinkersuf' in self.Data else ''))
        return out

    def write_gro(self, select, **kwargs):
        out = []
        if sys.stdin.isatty():
            self.require('elem', 'xyzs')
            self.require_resname()
            self.require_resid()
            self.require_boxes()
        else:
            self.require('elem', 'xyzs', 'resname', 'resid', 'boxes')

        if 'atomname' not in self.Data:
            count = 0
            resid = -1
            atomname = []
            for i in range(self.na):
                if self.resid[i] != resid:
                    count = 0
                count += 1
                resid = self.resid[i]
                atomname.append("%s%i" % (self.elem[i], count))
        else:
            atomname = self.atomname

        for I in select:
            xyz = self.xyzs[I]
            xyzwrite = xyz.copy()
            xyzwrite /= 10.0  # GROMACS uses nanometers
            out.append(self.comms[I])
            #out.append("Generated by ForceBalance from %s" % self.fnm)
            out.append("%5i" % self.na)
            for an, line in enumerate(xyzwrite):
                out.append(format_gro_coord(
                    self.resid[an], self.resname[an], atomname[an], an + 1, xyzwrite[an]))
            out.append(format_gro_box(self.boxes[I]))
        return out

    def write_qdata(self, select, **kwargs):
        """ Text quantum data format. """
        # self.require('xyzs','qm_energies','qm_grads')
        out = []
        for I in select:
            xyz = self.xyzs[I]
            out.append("JOB %i" % I)
            out.append("COORDS" + pvec(xyz))
            if 'qm_energies' in self.Data:
                out.append("ENERGY % .12e" % self.qm_energies[I])
            if 'mm_energies' in self.Data:
                out.append("EMD0   % .12e" % self.mm_energies[I])
            if 'qm_grads' in self.Data:
                out.append("FORCES" + pvec(self.qm_grads[I]))
            if 'qm_espxyzs' in self.Data and 'qm_espvals' in self.Data:
                out.append("ESPXYZ" + pvec(self.qm_espxyzs[I]))
                out.append("ESPVAL" + pvec(self.qm_espvals[I]))
            if 'qm_interaction' in self.Data:
                out.append("INTERACTION % .12e" % self.qm_interaction[I])
            out.append('')
        return out

    def require_resid(self):
        if 'resid' not in self.Data:
            na_res = int(input(
                "Enter how many atoms are in a residue, or zero as a single residue -> "))
            if na_res == 0:
                self.resid = [1 for i in range(self.na)]
            else:
                self.resid = [1 + i / na_res for i in range(self.na)]

    def require_resname(self):
        if 'resname' not in self.Data:
            resname = input(
                "Enter a residue name (3-letter like 'SOL') -> ")
            self.resname = [resname for i in range(self.na)]

    def require_boxes(self):
        def buildbox(line):
            s = [float(i) for i in line.split()]
            if len(s) == 1:
                a = s[0]
                b = s[0]
                c = s[0]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 3:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = 90.0
                beta = 90.0
                gamma = 90.0
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 6:
                a = s[0]
                b = s[1]
                c = s[2]
                alpha = s[3]
                beta = s[4]
                gamma = s[5]
                return BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma)
            elif len(s) == 9:
                v1 = np.array([s[0], s[3], s[4]])
                v2 = np.array([s[5], s[1], s[6]])
                v3 = np.array([s[7], s[8], s[2]])
                return BuildLatticeFromVectors(v1, v2, v3)
            else:
                raise RuntimeError(
                    "Not sure what to do since you gave me %i numbers\n" % len(s))

        if 'boxes' not in self.Data or len(self.boxes) != self.ns:
            sys.stderr.write("Please specify the periodic box using:\n")
            sys.stderr.write("1 float (cubic lattice length in Angstrom)\n")
            sys.stderr.write(
                "3 floats (orthogonal lattice lengths in Angstrom)\n")
            sys.stderr.write(
                "6 floats (triclinic lattice lengths and angles in degrees)\n")
            sys.stderr.write(
                "9 floats (triclinic lattice vectors v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y) in Angstrom)\n")
            sys.stderr.write(
                "Or: Name of a file containing one of these lines for each frame in the trajectory\n")
            boxstr = input("Box Vector Input: -> ")
            if os.path.exists(boxstr):
                boxfile = open(boxstr).readlines()
                if len(boxfile) != len(self):
                    raise RuntimeError(
                        'Tried to read in the box file, but it has a different length from the number of frames.\n')
                else:
                    self.boxes = [buildbox(line) for line in boxfile]
            else:
                mybox = buildbox(boxstr)
                self.boxes = [mybox for i in range(self.ns)]
