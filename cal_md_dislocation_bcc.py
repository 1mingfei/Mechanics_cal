#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2017-07-05 08:12:30
# @Last Modified by:   chaomy
# @Last Modified time: 2017-09-28 14:02:14


import os
from ase import io
from numpy import array


class md_dislocation_bcc(object):

    def __init__(self):
        self.bccneigh = array([[0.5, 0.5, 0.5],
                               [-0.5, 0.5, 0.5],
                               [0.5, -0.5, 0.5],
                               [0.5, 0.5, -0.5],
                               [-0.5, -0.5, 0.5],
                               [0.5, -0.5, -0.5],
                               [-0.5, 0.5, -0.5],
                               [-0.5, -0.5, -0.5]])
        return

    def cal_single_edge_dislocations_read(self):
        atoms = io.read('edgeinit.dump', format='lammps-dump')
        atoms = self.cut_y_normal_atoms(atoms)

        atoms = self.assign_ynormal_fixatoms(atoms)
        atoms = self.intro_dipole_edge_with_image_atoms(atoms)

        atoms = self.cut_x_normal_atoms(atoms, self.pot['lattice'])
        self.write_lmp_config_data_charge(atoms, 'init.edge')
        self.write_lmp_config_data(atoms, 'init.edge.atoms')
        return

    def cal_single_screw_dislocatoins_read(self):
        atoms = io.read('screwinit.dump', format='lammps-dump')
        atoms = self.cut_y_normal_atoms(atoms)
        atoms = self.assign_ynormal_fixatoms(atoms)

        atoms = self.intro_single_screw_atoms(atoms)
        atoms = self.cut_z_normal_atoms(atoms)

        self.write_lmp_config_data_charge(atoms, 'init.screw')
        self.write_lmp_config_data(atoms, 'init.screw.atoms')
        return

    def cal_single_edge_dislocations(self):
        e1 = [1., 1., 1.]
        e2 = [-1., 1., 0.]
        e3 = [-1., -1., 2.]

        # atoms = self.set_bcc_convention([e1, e2, e3], (140, 80, 6))
        atoms = self.set_bcc_convention([e1, e2, e3], (140, 80, 1))
        atoms = self.cut_y_normal_atoms(atoms)
        atoms = self.assign_ynormal_fixatoms(atoms)
        # atoms = self.intro_dipole_edge_with_image_atoms(atoms)
        atoms = self.intro_single_edge_atoms(atoms)
        # if we don't cut a layer of atoms, will generate two dislocations
        # cut a layer normal the burger direction
        atoms = self.cut_x_normal_atoms(atoms)
        atoms = self.make_sphere(atoms)
        self.write_lmp_config_data(atoms, 'init.edge.atoms')
        self.write_lmp_config_data_charge(atoms, 'init.edge')
        return

    def cal_single_screw_dislocations(self):
        e1 = [1., -1., 0]
        e2 = [1., 1., -2]
        e3 = [0.5, 0.5, 0.5]

        # atoms = self.set_bcc_convention(
        #     [e1, e2, e3], (100, 60, 12))  # z periodic 12
        atoms = self.set_bcc_convention(
            [e1, e2, e3], (100, 60, 40))  # z periodic 12
        atoms = self.cut_y_normal_atoms(atoms)
        atoms = self.assign_ynormal_fixatoms(atoms)

        io.write("lmp_perf.cfg", atoms, "cfg")
        atoms = self.intro_single_screw_atoms(atoms)
        #  if we don't cut a layer of atoms,
        #  it will generate two screw dislocations
        atoms = self.cut_z_normal_atoms(atoms)
        # for Mike, to generate the cluster
        atoms = self.make_sphere(atoms)
        self.write_lmp_config_data(atoms, 'init.screw.atoms')
        self.write_lmp_config_data_charge(atoms, 'init.screw')
        return

    # calculate the nano particle  #
    def cal_non_periodic_screw_xdislocation(self):
        e1 = array([-1., 1., 0])
        e2 = array([-1., -1., 2])
        e3 = array([1., 1., 1.])
        # atoms = self.set_bcc_convention([e1, e2, e3], (60, 40, 3))  # z peri
        # 18
        atoms = self.set_bcc_convention(
            [e1, e2, e3], (60, 40, 80))  # z peri 18

        atoms = self.intro_single_screw_atoms(atoms)
        #  atoms = self.intro_dipole_screw_with_image_atoms(atoms);

        atoms = self.cut_y_normal_atoms(atoms)

        self.write_lmp_config_data(atoms)

        if not os.path.isdir("restart"):
            os.mkdir("restart")
            os.mkdir("cfg")

        #  self.gn_md_nano_tensile(
            #  potential_file = "Nb.eam.alloy.webarchive",
            #  element = "Nb",
            #  temperature = "20",
            #  deform_direction = "xz",
            #  deform_rate = "5e-7")

        self.gn_md_minimize_cfg("lmp_init.txt",
                                "W.set.txt",    # "Nb.eam.alloy.webarchive",
                                "W")
        os.system("rm cfg/* ; mpirun -n 4 lmp_mpi -in in.minimize")
        # write pbs
        return