# -*- coding: utf-8 -*-
# @Author: 1mingfei
# @Date:   2018-12-18 18:07:29
# @Last Modified by:   1mingfei
# @Last Modified time: 2018-12-18 18:14:21

import os
from numpy import sqrt, arccos, arcsin, rad2deg
from numpy import ndarray, loadtxt, savetxt
from numpy import mat, unique


class md_gb_fcc_100(object):

    def find_angles_100(self, il=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
                         jl=[1, 2, 3, 4]):
        ags = []
        res = []
        ux = self.pot["lattice"]
        uy = self.pot["lattice"]
        cnt = 0
        
        for j in jl:
            for i in il[j - 1]:
                idx = i
                idy = j
                lx = idx * ux
                ly = idy * uy

                ll = sqrt(lx**2 + ly**2)
                ang = rad2deg(arccos(lx / ll))

                if (ang <= 90):
                    cnt += 1
                    ags.append(ang)
                    while (ll < 0):
                        ll *= 2
                    res.append([ang, ll, i, j])

        print(len(ags))
        (a, b) = unique(ags, return_index=True)

        print(len(a), len(b))
        self.ag = []

        for ee in b:
            self.ag.append(res[ee])

    def give_angle_100(self, opt='run'):
        self.find_angles_100(il=[[], [1]], jl=[2])   # 72.877
        # self.find_angles_100(il=[[1], [1]], jl=[1])    # 58.361

        npts = 2
        j = 0
        self.gbdat = ndarray([2, npts])
        self.gbdat[0, 0], self.gbdat[0, 1] = 0, 0

        ux = self.pot["lattice"]
        uz = sqrt(3) * self.pot["lattice"]

        for m in range(npts):
            for n in range(npts):
                for ee in self.ag:
                    dx = m / (npts + 1) * ux
                    dz = n / (npts + 1) * uz
                    mdir = "dir_{:02.03f}_{:02d}_{:02d}_{:02d}_{:02d}".format(
                        ee[0], ee[2], ee[3], m, n)
                    '''
                    if opt in ['prep']:
                        self.mymkdir(mdir)
                        self.build_hcp_gb_lmp(                             #need to change to fcc     
                            ee[0], ee[1], (50, 60), (dx, 0.0, dz), '1100') #need to change to fcc
                        os.system('mv in.gb {}'.format(mdir))
                        self.mymkdir('{}/out'.format(mdir))

                    elif opt in ['run']:
                        os.chdir(mdir)
                        os.system('mpirun lmp_mpi -i in.gb')
                        os.chdir(os.pardir)

                    elif opt in ['read']:
                        os.chdir(mdir)
                        self.gbdat[0, j] = j
                        self.gbdat[1, j] = loadtxt('lmp.dat')
                        j += 1
                        os.chdir(os.pardir)

                    elif opt in ["move"]:
                        os.system(
                            "cp {}/out/rel.chkpt.0 hcp.gb.{:03}".format(mdir, j))
                        j += 1
                    '''

        if opt in ['read']:
            self.set_111plt()
            print(self.gbdat[1, :])
            self.ax.plot(self.gbdat[0, :], self.gbdat[1, :])
            self.fig.savefig("fig_gb.png")

    def loop_angle_100(self, opt='prep'):
        self.find_angles_100()

        npts = len(self.ag)

        j = 0
        self.gbdat = ndarray([2, npts + 1])
        self.gbdat[0, 0], self.gbdat[0, 1] = 0, 0

        for ee in self.ag:
            mdir = "dir_{:02.03f}_{:02d}_{:02d}".format(ee[0], ee[2], ee[3])

            '''
            if opt in ['prep']:
                self.mymkdir(mdir)
                self.build_hcp_gb_lmp(                               #need to change to fcc
                    ee[0], ee[1], (40, 50), (0.0, 0.0, 0.0), '1100') #need to change to fcc
                os.system('mv in.gb {}'.format(mdir))
                self.mymkdir('{}/out'.format(mdir))

            elif opt in ['run']:
                os.chdir(mdir)
                os.system('mpirun -n 4 lmp_mpi -i in.gb')
                os.chdir(os.pardir)

            elif opt in ['read']:
                os.chdir(mdir)
                j += 1
                self.gbdat[0, j] = ee[0]
                self.gbdat[1, j] = loadtxt('lmp.dat')
                os.chdir(os.pardir)
            '''

