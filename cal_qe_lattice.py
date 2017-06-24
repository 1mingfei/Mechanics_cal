#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yangchaoming
# @Date:   2017-06-13 15:37:47
# @Last Modified by:   chaomy
# @Last Modified time: 2017-06-24 12:19:00

import os
import numpy as np
import gn_qe_inputs
import md_pot_data
import gn_config
import get_data
import output_data
import glob
import plt_drv
from optparse import OptionParser
from scipy.interpolate import InterpolatedUnivariateSpline


class cal_lattice(gn_config.bcc,
                  gn_config.fcc,
                  gn_config.hcp,
                  get_data.get_data,
                  output_data.output_data,
                  gn_qe_inputs.gn_qe_infile,
                  plt_drv.plt_drv):

    def __init__(self, inpot=None):
        self.pot = md_pot_data.qe_pot.vca_W75Re25
        output_data.output_data.__init__(self)
        get_data.get_data.__init__(self)
        gn_qe_inputs.gn_qe_infile.__init__(self, self.pot)
        plt_drv.plt_drv.__init__(self)
        self.alat0 = self.pot['lattice']
        self.element = self.pot['element']
        self.mass = self.pot['mass']
        self.kpnts = [46, 46, 46]
        self.root = os.getcwd()
        return

    def cal_bcc_lattice(self):
        bcc_drv = gn_config.bcc(self.pot)
        for i in range(-10, 10):
            if i >= 0:
                dirname = "dir-p-{:03d}".format(i)
            else:
                dirname = "dir-n-{:03d}".format(abs(i))
            self.mymkdir(dirname)
            alat = self.alat0 + i * 0.006
            bcc_drv.set_lattce_constant(alat)
            atoms = bcc_drv.set_bcc_primitive((1, 1, 1))
            self.gn_qe_bcc_lattice_infile(atoms)
            os.system("mv qe.in {}".format(dirname))
            os.system('cp $POTDIR/{} {}'.format(self.pot['file'],
                                                dirname))
        return

    def loop_pots(self):
        count = 0
        potlist = None
        for pot in potlist:
            count += 1
            latticeList = []
            energy = []
            stress = []
            for i in range(-10, 10):
                alat = self.alat0 + i * 0.005
                latticeList.append(alat)
                self.gnInfile(alat,
                              self._kpoint,
                              pot)
                data = self.qe_get_energy_stress()
                energy.append(data[0])
                stress.append(data[1])
                self.output_delta_energy(delta=latticeList,
                                         energy=energy,
                                         stress=stress,
                                         file_name='lat.txt')
        return

    def loop_kpoints(self):
        bcc_drv = gn_config.bcc(self.pot)
        bcc_drv.set_lattce_constant(self.alat0)
        self.set_ecut('{}'.format(48))
        self.set_degauss('0.02D0')
        self.set_disk_io('none')
        for kpts in range(32, 48):
            self.set_kpnts((kpts, kpts, kpts))
            dirname = 'dir-kpt-{}'.format(kpts)
            self.mymkdir(dirname)
            atoms = bcc_drv.set_bcc_primitive((1, 1, 1))
            self.gn_qe_bcc_lattice_infile(atoms)
            os.system('mv qe.in {}'.format(dirname))
            os.system('cp $POTDIR/{} {}'.format(self.pot['file'],
                                                dirname))
        return

    def loop_ecut(self):
        bcc_drv = gn_config.bcc(self.pot)
        bcc_drv.set_lattce_constant(self.alat0)
        for ecut in range(30, 50):
            self.set_ecut('{}'.format(ecut))
            dirname = 'dir-ecut-{}'.format(ecut)
            self.mymkdir(dirname)
            atoms = bcc_drv.set_bcc_primitive((1, 1, 1))
            self.gn_qe_bcc_lattice_infile(atoms)
            os.system('mv qe.in {}'.format(dirname))
            os.system('cp $POTDIR/{} {}'.format(self.pot['file'],
                                                dirname))
        return

    def clc_data(self, tag='ecut'):
        dirlist = glob.glob("dir-*")
        npts = len(dirlist)
        ecutlist = np.zeros(npts)
        engylist = np.zeros(npts)
        for i in range(npts):
            print dirlist[i]
            os.chdir(dirlist[i])
            ecutlist[i] = int(dirlist[i][-2:])
            (engylist[i], stress) = self.qe_get_energy_stress()
            os.chdir(self.root)
        np.savetxt('{}.txt'.format(tag), [ecutlist, engylist])
        return

    def clc_lattice(self, tag='bcc'):
        rng = [-10, 10]
        cnt = 0
        data = np.zeros([rng[1] - rng[0], 2])
        for i in range(rng[0], rng[1]):
            if i >= 0:
                dirname = "dir-p-{:03d}".format(i)
            else:
                dirname = "dir-n-{:03d}".format(abs(i))
            os.chdir(dirname)
            (energy, vol, stress) = self.qe_get_energy_stress('qe.out')
            cellmtx = self.qe_get_cell('qe.in')
            if (tag == 'fcc') or (tag == 'bcc'):
                data[cnt, 0] = 2 * cellmtx[0, 1]
                data[cnt, 1] = (energy)
            cnt += 1
            os.chdir(self.root)
        print data
        np.savetxt('lat.dat', data)
        return

    def plt_data(self, tag='ecut', data=None):
        if data is None:
            [val, engy] = np.loadtxt('{}.txt'.format(tag))
        else:
            [val, engy] = data[:, 0], data[:, 1]
        engy = engy[np.argsort(val)]
        self.set_111plt()
        self.set_keys()
        self.ax.plot(np.sort(val), engy)
        self.fig.savefig('{}.png'.format(tag))
        return

    def loop_run(self):
        dirlist = glob.glob("dir-*")
        for dirname in dirlist:
            print dirname
            os.chdir(dirname)
            if not os.path.isfile('qe.out'):
                os.system("mpirun -n 24 pw.x < qe.in > qe.out")
            os.chdir(self.root)
        return

    def gn_qe_bcc_lattice_infile(self, atoms):
        self.set_thr('1.0D-6')
        with open('qe.in', 'w') as fid:
            fid = self.qe_write_control(fid, atoms)
            fid = self.qe_write_system(fid, atoms)
            fid = self.qe_write_electrons(fid)
            fid = self.qe_write_cell(fid, atoms.get_cell())
            fid = self.qe_write_species(fid, atoms, self.pot)
            fid = self.qe_write_pos(fid, atoms)
            fid = self.qe_write_kpts(fid, self.kpnts)
            fid.close()
        return

    def find_lattice(self, data=None):
        if data is None:
            data = np.loadtxt('lat.txt')
            data[:, 0] = np.abs(data[:, 0])  # in case
        interps = np.linspace(data[0, 0], data[-1, 0], 201)
        print interps
        spl = InterpolatedUnivariateSpline(data[:, 0], data[:, 1])
        print "min lat", interps[np.argmin(spl(interps))]
        return data

if __name__ == '__main__':
    usage = "usage:%prog [options] arg1 [options] arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-t",
                      "--mtype",
                      action="store",
                      type="string",
                      dest="mtype",
                      help="",
                      default="prp_r")

    (options, args) = parser.parse_args()
    drv = cal_lattice()

    if options.mtype.lower() in ['kpts', 'prepkpts', 'loopkpts']:
        drv.loop_kpoints()

    elif options.mtype.lower() == 'ecut':
        drv.loop_ecut()

    elif options.mtype.lower() == 'pot':
        drv.loop_pots()

    elif options.mtype.lower() == 'bcc':
        drv.cal_bcc_lattice()

    elif options.mtype.lower() == 'run':
        drv.loop_run()

    elif options.mtype.lower() == 'clccut':
        drv.clc_data(tag='ecut')

    elif options.mtype.lower() == 'clckpts':
        drv.clc_data(tag='kpts')

    elif options.mtype.lower() == 'clclat':
        drv.clc_lattice()

    elif options.mtype.lower() == 'pltcut':
        drv.plt_data(tag='ecut')

    elif options.mtype.lower() == 'pltkpts':
        drv.plt_data(tag='kpts')

    elif options.mtype.lower() == 'pltlat':
        data = drv.find_lattice()
        drv.plt_data(tag='lat', data=data)
