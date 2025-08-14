import sys
import glob
import h5py
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

fileInd=int(sys.argv[1])

tStepMax=200
gridNum=128
tGrid=0.005

storeFieldList=sorted(glob.glob('FiveHWRunner/storeField_0*'))\
              +sorted(glob.glob('FiveHWRunner/storeField_1*'))\
              +sorted(glob.glob('FiveHWRunner/storeField_2*'))

def MP(taskid_lst=None,func=None,Nprocs=8):
    def worker(taskid_lst,out_q):
        outdict={}
        for tid in taskid_lst:
            outdict[tid]=func(tid)
        out_q.put(outdict)
    out_q=multiprocessing.Queue()
    chunksize=int(np.ceil(len(taskid_lst)/float(Nprocs)))
    procs=[]
    for i in range(Nprocs):
        p=multiprocessing.Process(target=worker,args=(taskid_lst[chunksize*i:chunksize*(i+1)],out_q)) 
        procs.append(p)
        p.start()
    resultdict={}
    for i in range(Nprocs):
        resultdict.update(out_q.get())
    for p in procs:
        p.join()
    return resultdict

def constrAFile(sfNameOne):
    with h5py.File(sfNameOne) as h5File:
        keyLength=len(h5File.keys())
    if keyLength<12:
        print('failed file, omitted')
        return np.zeros([0,3,gridNum,gridNum]),np.zeros([0,3,gridNum,gridNum]),np.zeros([0,7]),np.zeros([0,7]),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)

    with h5py.File(sfNameOne) as h5File:
        dens=np.array(h5File['density'])
        phi=np.array(h5File['phi'])
        omega=np.array(h5File['omega'])
        enstrophy=np.array(h5File['enstrophy'])
        enstrophyPhi=np.array(h5File['enstrophy_phi'])
        gammac=np.array(h5File['gamma_c'])
        gamman=np.array(h5File['gamma_n'])
        gammaNSp=np.array(h5File['gamma_n_spectral'])
        kineEng=np.array(h5File['kinetic_energy'])
        therEng=np.array(h5File['thermal_energy'])
        quantiList=np.array([enstrophy,enstrophyPhi,gammac,gamman,gammaNSp,kineEng,therEng]).T
        
    ind1=int(sfNameOne.split('/')[-1].split('.h5')[0].split('_')[1])
    ind2=int(sfNameOne.split('/')[-1].split('.h5')[0].split('_')[2])
    c1=np.load('FiveHWRunner/c1List_'+str(ind1)+'.npy')[ind2]
    k0=np.load('FiveHWRunner/k0List_'+str(ind1)+'.npy')[ind2]
    kappa=np.load('FiveHWRunner/kappaList_'+str(ind1)+'.npy')[ind2]
    pbc=np.load('FiveHWRunner/pbcList_'+str(ind1)+'.npy')[ind2]
    
    timeList=np.arange(len(dens),dtype=int)
    mask=(dens.max(axis=(1,2))>2)&(dens.max(axis=(1,2))<20)
    dens=dens[mask]
    phi=phi[mask]
    omega=omega[mask]
    timeList=timeList[mask]
    quantiList=quantiList[mask]
    if (len(timeList)-100)<1:
        print('failed file, omitted')
        return np.zeros([0,3,gridNum,gridNum]),np.zeros([0,3,gridNum,gridNum]),np.zeros([0,7]),np.zeros([0,7]),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)
    chosenMask=np.zeros(len(timeList),dtype=bool)
    omegaPhiDenInOne=[]
    omegaPhiDenOutOne=[]
    quantiInOne=[]
    quantiOutOne=[]
    tDiffer=[]
    while True:
        if np.mean(chosenMask)==1:break
        chosStarter=np.random.choice(timeList[chosenMask==False])
        chosStartInd=np.where(chosStarter==timeList)[0][0]
        if chosStarter==timeList[chosenMask==False][-1]:
            chosenMask[chosStartInd]=True
            continue
        chosEndInLi=timeList[chosenMask==False]
        chosEndInLi=chosEndInLi[((chosEndInLi-chosStarter)<tStepMax)&((chosEndInLi-chosStarter)>0)]
        if len(chosEndInLi)==0:
            chosenMask[chosStartInd]=True
            continue
        chosEnder=np.random.choice(chosEndInLi)
        chosEndInd=np.where(chosEnder==timeList)[0][0]
        omegaPhiDen=np.stack([omega[chosStartInd],phi[chosStartInd],dens[chosStartInd]])
        omegaPhiDenInOne.append(omegaPhiDen)
        omegaPhiDen=np.stack([omega[chosEndInd],phi[chosEndInd],dens[chosEndInd]])
        omegaPhiDenOutOne.append(omegaPhiDen)
        quantiInOne.append(quantiList[chosStartInd])
        quantiOutOne.append(quantiList[chosEndInd])
        tDiffer.append((chosEnder-chosStarter)*tGrid)
        chosenMask[chosStartInd]=True
        chosenMask[chosEndInd]=True
    omegaPhiDenInOne=np.stack(omegaPhiDenInOne)
    omegaPhiDenOutOne=np.stack(omegaPhiDenOutOne)
    quantiInOne=np.stack(quantiInOne)
    quantiOutOne=np.stack(quantiOutOne)
    tDiffer=np.array(tDiffer)
    return omegaPhiDenInOne,\
    omegaPhiDenOutOne,\
    quantiInOne,\
    quantiOutOne,\
    tDiffer,\
    c1*np.ones([len(omegaPhiDenInOne)]),\
    k0*np.ones([len(omegaPhiDenInOne)]),\
    kappa*np.ones([len(omegaPhiDenInOne)]),\
    pbc*np.ones([len(omegaPhiDenInOne)])

storeFieldChos=np.array_split(np.array(storeFieldList),40)[fileInd]
MPout=MP(storeFieldChos,constrAFile)
omegaPhiDenInList=np.concatenate([MPout[i][0] for i in storeFieldChos])
omegaPhiDenOutList=np.concatenate([MPout[i][1] for i in storeFieldChos])
quantiInList=np.concatenate([MPout[i][2] for i in storeFieldChos])
quantiOutList=np.concatenate([MPout[i][3] for i in storeFieldChos])
tDifferList=np.concatenate([MPout[i][4] for i in storeFieldChos])
c1List=np.concatenate([MPout[i][5] for i in storeFieldChos])
k0List=np.concatenate([MPout[i][6] for i in storeFieldChos])
kappaList=np.concatenate([MPout[i][7] for i in storeFieldChos])
pbcList=np.concatenate([MPout[i][8] for i in storeFieldChos])
shuArr=np.arange(len(omegaPhiDenInList))
np.random.shuffle(shuArr)

omegaPhiDenInList=omegaPhiDenInList[shuArr]
omegaPhiDenOutList=omegaPhiDenOutList[shuArr]
quantiInList=quantiInList[shuArr]
quantiOutList=quantiOutList[shuArr]
tDifferList=tDifferList[shuArr]
c1List=c1List[shuArr]
k0List=k0List[shuArr]
kappaList=kappaList[shuArr]
pbcList=pbcList[shuArr]

np.save('Cache/omegaPhiDenInList_'+str(fileInd)+'.npy',omegaPhiDenInList)
np.save('Cache/omegaPhiDenOutList_'+str(fileInd)+'.npy',omegaPhiDenOutList)
np.save('Cache/quantiInList_'+str(fileInd)+'.npy',quantiInList)
np.save('Cache/quantiOutList_'+str(fileInd)+'.npy',quantiOutList)
np.save('Cache/tDifferList_'+str(fileInd)+'.npy',tDifferList)
np.save('Cache/c1List_'+str(fileInd)+'.npy',c1List)
np.save('Cache/k0List_'+str(fileInd)+'.npy',k0List)
np.save('Cache/kappaList_'+str(fileInd)+'.npy',kappaList)
np.save('Cache/pbcList_'+str(fileInd)+'.npy',pbcList)



