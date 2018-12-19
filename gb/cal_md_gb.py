#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2017-07-05 08:12:30
# @Last Modified by:   chaomy
# @Last Modified time: 2018-11-21 23:36:46


import cal_md_gb_hcp
import cal_md_gb_pre
import cal_md_gb_pos
import cal_md_gb_run
#import cal_md_gb_lmp
import cal_md_gb_indx
import cal_md_gb_del
import cal_md_gb_hcp_0001
import cal_md_gb_hcp_1100
import cal_md_gb_hcp_1210
import cal_md_gb_ase_1100
import cal_md_gb_ase_1210
#add FCC 100 modules -1mingfei 12/19/2018
import cal_md_gb_fcc_100
import cal_md_gb_ase_fcc_100

import cal_md_edge_shift
import get_data
import plt_drv
from utils import Intro_vasp
from md_pot_data import md_pot
from optparse import OptionParser
import gn_config


class md_gb(cal_md_gb_pre.md_gb_pre,
            cal_md_gb_run.md_gb_run,
            cal_md_gb_pos.md_gb_pos,
            #cal_md_gb_lmp.md_gb_lmp,
            cal_md_gb_indx.md_gb_indx,
            cal_md_gb_hcp_0001.md_gb_loop,
            cal_md_gb_hcp.md_gb_hcp,
            cal_md_gb_del.md_gb_del,
            cal_md_gb_hcp_1100.md_gb_hcp_1100,
            cal_md_gb_hcp_1210.md_gb_hcp_1210,
            cal_md_gb_ase_1100.md_gb_ase_1100,
            cal_md_gb_ase_1210.md_gb_ase_1210,
            #add fcc moduli -1mingfei 12/19/2018
            cal_md_gb_fcc_100.md_gb_fcc_100,
            cal_md_gb_ase_fcc_100.md_gb_ase_fcc_100,

            Intro_vasp.vasp_change_box,
            cal_md_edge_shift.md_edge_shift,
            plt_drv.plt_drv,
            get_data.get_data,
            gn_config.gnStructure):

    def __init__(self):
        # self.pot = md_pot.mg_Poco
        # self.pot = md_pot.Ti_Ackland
        #self.pot = md_pot.ti_zope
        self.pot = md_pot.Ag_Williams
        # self.pot = md_pot.mg_sun
        cal_md_gb_pre.md_gb_pre.__init__(self)
        cal_md_gb_pos.md_gb_pos.__init__(self)
        cal_md_gb_run.md_gb_run.__init__(self)
        #cal_md_gb_lmp.md_gb_lmp.__init__(self)
        cal_md_gb_del.md_gb_del.__init__(self)
        cal_md_gb_indx.md_gb_indx.__init__(self)
        cal_md_gb_hcp_0001.md_gb_loop.__init__(self)
        cal_md_gb_hcp.md_gb_hcp.__init__(self)
        cal_md_gb_hcp_1100.md_gb_hcp_1100.__init__(self)
        cal_md_gb_fcc_100.md_gb_fcc_100.__init__(self) #-1mingfei
        cal_md_gb_hcp_1210.md_gb_hcp_1210.__init__(self)
        cal_md_gb_ase_1100.md_gb_ase_1100.__init__(self)
        cal_md_gb_ase_fcc_100.md_gb_ase_fcc_100.__init__(self) #-1mingfei
        cal_md_gb_ase_1210.md_gb_ase_1210.__init__(self)
        cal_md_edge_shift.md_edge_shift.__init__(self)
        Intro_vasp.vasp_change_box.__init__(self)
        plt_drv.plt_drv.__init__(self)
        get_data.get_data.__init__(self)
        gn_config.gnStructure.__init__(self, self.pot)


if __name__ == "__main__":
    usage = "usage:%prog [options] arg1 [options] arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-t", "--mtype", action="store",
                      type="string", dest="mtype")
    parser.add_option('-p', "--param", action="store",
                      type='string', dest="fargs")
    (options, args) = parser.parse_args()
    drv = md_gb()
    dispatcher = {'gblist': drv.loop_gb_list,
                  'run': drv.loop_gb_run,
                  'index': drv.hcp_tilt_index,
                  'plt': drv.loop_plt_angle,
                  '0001': drv.loop_angle_0001,
                  '1100lmp': drv.loop_angle_1100,
                  '1100lmpg': drv.give_angle_1100,
                  '1100lg': drv.build_hcp_ase_1100_3ABA,
                  '1210': drv.loop_angle_1210,
                  '1210sm': drv.build_hcp_ase_1210_small,
                  '1100e': drv.build_hcp_ase_1100_with_edge,
                  '1100edp': drv.intro_edge_dipole,
                  '1100sm': drv.build_hcp_ase_1100_small,
                  'hcp': drv.make_perf_HCP,
                  'edge': drv.intro_edge,
                  'cut': drv.make_gb_cut,
                  'shft': drv.shift_make_edge,
                  'fml': drv.formular_make_edge,
                  'scnt': drv.shift_make_edge_cnt,
                  'shear': drv.add_shear,
                  'dup': drv.duplicate_make_edge,
                  'rep': drv.make_repeat,
                  'loop': drv.loop_init_1100,
                  'loop_fcc_100': drv.loop_init_fcc100,  #-1mingfei 
                  'del': drv.analysize_atomic_strain,
                  'usp': drv.loop_set_usp_run,
                  'init': drv.loop_grand,
                  'dft': drv.make_DFT,
                  'loopprint': drv.print_angles,
                  'loopclc': drv.loop_collect_energy,
                  'looppre': drv.loop_clc_init_structures,
                  'loopplt': drv.loop_plot_energy,
                  'plteach': drv.loop_plt_each,
                  'loopcmb': drv.loop_combine,
                  'loopdup': drv.loop_dup_structures,
                  'makedup': drv.extend_along_y}

    if options.fargs is not None:
        dispatcher[options.mtype.lower()](options.fargs)
    else:
        dispatcher[options.mtype.lower()]()
