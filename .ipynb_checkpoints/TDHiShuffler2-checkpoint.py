import sys
import glob
import h5py
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

horiInd=int(sys.argv[1])

omegaDenInList=[]
omegaDenOutList=[]
tDifferList=[]
c1List=[]
k0List=[]
kappaList=[]
pbcList=[]
for i in range(40):
    omegaDenInLoad=np.load('Cache/omegaPhiDenInList_'+str(i)+'.npy')
    omegaDenInList.append(np.array_split(omegaDenInLoad,100)[horiInd].copy())
    del omegaDenInLoad
    omegaDenOutLoad=np.load('Cache/omegaPhiDenOutList_'+str(i)+'.npy')
    omegaDenOutList.append(np.array_split(omegaDenOutLoad,100)[horiInd].copy())
    del omegaDenOutLoad
    tDifferList.append(np.array_split(np.load('Cache/tDifferList_'+str(i)+'.npy'),100)[horiInd])
    c1List.append(np.array_split(np.load('Cache/c1List_'+str(i)+'.npy'),100)[horiInd])
    k0List.append(np.array_split(np.load('Cache/k0List_'+str(i)+'.npy'),100)[horiInd])
    kappaList.append(np.array_split(np.load('Cache/kappaList_'+str(i)+'.npy'),100)[horiInd])
    pbcList.append(np.array_split(np.load('Cache/pbcList_'+str(i)+'.npy'),100)[horiInd])
    print(i)
omegaDenInList=np.concatenate(omegaDenInList)
omegaDenOutList=np.concatenate(omegaDenOutList)
tDifferList=np.concatenate(tDifferList)
c1List=np.concatenate(c1List)
k0List=np.concatenate(k0List)
kappaList=np.concatenate(kappaList)
pbcList=np.concatenate(pbcList)

np.save('TDHiData/omegaPhiDenInList_'+str(horiInd)+'.npy',np.float32(omegaDenInList))
np.save('TDHiData/omegaPhiDenOutList_'+str(horiInd)+'.npy',np.float32(omegaDenOutList))
np.save('TDHiData/tDifferList_'+str(horiInd)+'.npy',np.float32(tDifferList))
np.save('TDHiData/c1List_'+str(horiInd)+'.npy',np.float32(c1List))
np.save('TDHiData/k0List_'+str(horiInd)+'.npy',np.float32(k0List))
np.save('TDHiData/kappaList_'+str(horiInd)+'.npy',np.float32(kappaList))
np.save('TDHiData/pbcList_'+str(horiInd)+'.npy',np.float32(pbcList))







