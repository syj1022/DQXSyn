import numpy as np
import glob
from ase import io
from ase import Atoms
from espresso import espresso

bulk =  io.read(glob.glob("*.cif")[0])
bulk.set_pbc([True,True,True])

calc = espresso(mode='relax',
                pw=500,
                dw=5000,
                xc='BEEF-vdW',
                kpts=(4,4,4),
                nbands=-30,
                sigma=0.1,
                opt_algorithm = 'bfgs',
                fmax = 0.03,
		nstep=500,
		nosym=True,
                convergence= {'energy':1e-5,
                    'mixing':0.1,
                    'nmix':10,
                    'mix':4,
                    'maxsteps':2000,
                    'diag':'david'
                    },
                 dipole={'status':False},
                 output = {'avoidio':False,
                    'removewf':True,
                    'wf_collect':False},
                 spinpol=False,
                 parflags='-npool 1',
                 onlycreatepwinp='pw.in',
                 psppath='/scratch/alevoj1/esp-psp/v2/',
		 outdir='calcdir')

bulk.set_calculator(calc)
calc.initialize(bulk)

