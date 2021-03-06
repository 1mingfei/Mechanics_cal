#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yang37
# @Date:   2017-06-21 18:42:47
# @Last Modified by:   chaomy
# @Last Modified time: 2018-05-08 13:49:22


import glob
import os
import ase.io
import numpy as np
import md_pot_data
from optparse import OptionParser
from scipy.optimize import leastsq
from utils import cal_add_strain
import gn_config
import get_data
import gn_incar
import gn_pbs
import output_data


class cal_cij(get_data.get_data, gn_incar.gn_incar,
              gn_pbs.gn_pbs, output_data.output_data,
              cal_add_strain.cal_add_strain, gn_config.gnStructure):

    def __init__(self):
        self.pot = md_pot_data.va_pot.Nb_pbe
        get_data.get_data.__init__(self)
        gn_incar.gn_incar.__init__(self)
        output_data.output_data.__init__(self)
        cal_add_strain.cal_add_strain.__init__(self)
        gn_config.gnStructure.__init__(self, self.pot)

        # self.unit_delta = 0.002 near equilibrim
        self.unit_delta = 0.0003
        self.npts = 8
        self.cij_type_list = ['c11', 'c12', 'c44']

    def fit_para(self, delta_list, energy_list):
        xdata = delta_list
        ydata = energy_list - self.energy0

        def residuals(p):
            a, c = p
            return ydata - (a * (0.5 * xdata**2) + c)
        r = leastsq(residuals, [1, 0])
        a, c = r[0]
        a = a / self.volume * self.ev_angstrom3_to_GPa
        return a

    def collect_data_cij(self):
        for mtype in ['c11', 'c12', 'c44']:
            data = np.ndarray([2 * self.npts, 2])
            cn = 0
            for j in range(-self.npts, self.npts):
                delta = self.unit_delta * j
                if j >= 0:
                    dirname = "dir-%s-p%03d" % (mtype, j)
                else:
                    dirname = "dir-%s-n%03d" % (mtype, np.abs(j))
                os.chdir(dirname)
                (energy, stress, vol) = self.vasp_energy_stress_vol()  # in eV
                os.chdir(os.pardir)
                data[cn, 0], data[cn, 1] = delta, energy
                cn += 1
            np.savetxt("data_{}.txt".format(mtype), data)

    def obtain_cij(self, opt='np'):
        self.set_volume_energy0()
        raw1 = np.loadtxt("data_c11.txt")
        raw2 = np.loadtxt("data_c12.txt")
        raw3 = np.loadtxt("data_c44.txt")

        delta_list, energy_list = raw1[:, 0], raw1[:, 1]
        c11_plus_c12 = self.fit_para(delta_list, energy_list)

        delta_list, energy_list = raw2[:, 0], raw2[:, 1]
        c11_minus_c12 = self.fit_para(delta_list, energy_list)

        delta_list, energy_list = raw3[:, 0], raw3[:, 1]
        c44 = self.fit_para(delta_list, energy_list)

        c11 = (0.111111111111111 * c11_plus_c12 +
               0.333333333333333 * c11_minus_c12)
        c12 = (0.111111111111111 * c11_plus_c12 -
               0.166666666666667 * c11_minus_c12)
        c44 = 0.25 * c44

        with open("cij.dat", 'w') as fout:
            fout.write("C11\t%f\t\nC12\t%f\t\nC44\t%f\t\n" % (c11, c12, c44))
            fout.close()

    def set_volume_energy0(self):
        self.volume = self.pot["lattice"] ** 3 / 2.
        self.energy0 = -10.09207329

    def loop_prepare_cij_otho(self):
        for kk in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']:
            for j in list(range(-100, -50)) + list(range(50, 100)):
                delta = self.unit_delta * j
                if j >= 0:
                    mdir = "dir-%s-p%03d" % (kk, j)
                else:
                    mdir = "dir-%s-n%03d" % (kk, -j)
                self.mymkdir(mdir)
                atoms = self.set_bcc_primitive((1, 1, 1))
                atoms = self.strain_for_otho(delta, atoms, kk)
                ase.io.write("POSCAR", images=atoms, format="vasp")

                os.system("cp va.pbs {}".format(mdir))
                os.system("cp POTCAR {}".format(mdir))
                os.system("cp POSCAR {}".format(mdir))
                os.system("cp KPOINTS {}".format(mdir))
                os.system("cp INCAR {}".format(mdir))

    def loop_prepare_cij(self):
        for mtype in self.cij_type_list:
            self.cij_type = mtype
            for j in range(-self.npts, self.npts):
                delta = self.unit_delta * j
                if j >= 0:
                    mdir = "dir-%s-p%03d" % (mtype, j)
                else:
                    mdir = "dir-%s-n%03d" % (mtype, np.abs(j))
                self.mymkdir(mdir)
                atoms = self.write_bcc_primitive_with_strain(delta=delta,
                                                             in_tag=mtype,
                                                             write=False)
                ase.io.write("POSCAR", images=atoms, format="vasp")
                os.system("cp va.pbs {}".format(mdir))
                os.system("cp POTCAR {}".format(mdir))
                os.system("cp POSCAR {}".format(mdir))
                os.system("cp KPOINTS {}".format(mdir))
                os.system("cp INCAR {}".format(mdir))

    def ouput_data_cij(self):
        for i in range(['c11', 'c12', 'c44']):
            out_file_name = "data_%s.txt" % (mtype)
            for j in range(-self.npts, self.npts):
                delta = self.unit_delta * j
                if j >= 0:
                    mdir = "dir-%s-delta-%03d" % (mtype, j)
                else:
                    mdir = "dir-%s-delta-n%03d" % (mtype, np.abs(j))
                os.chdir(mdir)
                print("i am in ", mdir)
                (energy, volume) = self.vasp_energy_stress_vol_quick()
                os.chdir(self.root_dir)
                self.output_delta_energy(delta, energy, out_file_name)


if __name__ == "__main__":
    usage = "usage:%prog [options] arg1 [options] arg2"
    parser = OptionParser(usage=usage)
    parser.add_option('-t', "--mtype", action="store",
                      type="string", dest="mtype")
    parser.add_option('-p', "--param", action="store",
                      type='string', dest="fargs")
    (options, args) = parser.parse_args()

    drv = cal_cij()
    dispatcher = {'prep': drv.loop_prepare_cij,
                  'cal': drv.obtain_cij,
                  'clc': drv.collect_data_cij,
                  'otho': drv.loop_prepare_cij_otho}

    if options.fargs is not None:
        dispatcher[options.mtype.lower()](options.fargs)
    else:
        dispatcher[options.mtype.lower()]()
