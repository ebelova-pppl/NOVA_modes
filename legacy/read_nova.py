# read NOVA AE mode structures from NOVA output files as egn02w.7439E+00
# in /u/ngorelen/work/nova/default_tae23/
#flnm1 = 'egn02w.5937E+00'
#
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) <= 1:
    print('Arguments should include the name of the AE structure file, i.e. use it like this: python ~/work/exe/plotNOVAegn.py egn02w.5937E+00')
    sys.exit('exiting due to no arguments given')
else:
    flnm1 = sys.argv[1]
    print('opening file', flnm1, 'for normal processing')

f1 = np.fromfile(flnm1)
"""
    f1 is 1D array of size 3*nr*nhar + 4,
    it contains 3 perturbations: xi_psi, delta_p, xi_surf,
    and 4 scalar parameters: 
    f1[0]= omega, 
    f1[-3]= nr (=number of radial points = sqrt(psi)),
    f1[-2]= gamma_d of continuum decay, 
    f1[-1]= ntor
"""
omega = f1[0]
nr = int(f1[-3])
gamma_d = f1[-2]
ntor = int(f1[-1])
nhar = int((f1.size-4)/(3*nr))  # this is number of poloidal harmonics

#print(f1.size)
print('')
print('omega=',omega,' nr=',nr,' gamma_d=',gamma_d,' ntor=',ntor,' nhar=',nhar)
print('=========================================================')

xxy = np.linspace(0.,1.,nr)
xh = np.linspace(0.,nhar,nhar)
xpl = []
ypl = []
for km in range(nhar):
         for j in range(nr):
                  xpl.append(xxy[j])
                  ypl.append(f1[1+j+km*nr])   # plots only xi_psi
#print('len of basearray',len(xpl),len(ypl))

f11 =  f1[1:-3].reshape(3,nhar,nr)
mode = f11[0,:,:].reshape(nhar,nr)
maxh = np.zeros([nhar])

for km in range(nhar):
         maxh[km] = np.max(np.absolute(f11[:,km,0]))


# Plots

fig,ax=plt.subplots(1,1,sharex=True,figsize=[7,6])
fig.subplots_adjust(wspace=0,hspace=0,left=0.15,right=0.98,top=0.9,bottom=0.15)
plt.grid(True)
plt.xlabel('$\\sqrt{\\psi_\\varphi/\\psi_{\\varphi 1}}$',fontsize=16)
plt.ylabel('$\\xi_\\psi$(a.u.)',fontsize=16)

ax.plot(xpl, ypl,'-k')          # plot all poloidal harmonics
#ax.plot(xxy,mode[7,:])          # plot selected harmonic
#ax.plot(xh,maxh)                # plot max amplitude vs m
plt.show()

