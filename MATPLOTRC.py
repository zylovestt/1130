import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

colors_=cycler('color',['#002bff','#3399BB','#9988DD','#EECC55','#88BB44','#FFBBBB','#EE6666'])
# plt.rc('axes',facecolor='#fbf9f9',edgecolor='k',axisbelow=True,grid=True,prop_cycle=colors_)
plt.rc('axes',facecolor='w',edgecolor='k',axisbelow=True,grid=True,prop_cycle=colors_)
plt.rc('grid',color='#d0cccc',linestyle='solid')
plt.rc('xtick',direction='out',color='k')
plt.rc('ytick',direction='out',color='k')
plt.rc('patch',edgecolor='k')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams["font.size"] = 20
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['legend.edgecolor']='#868383'
plt.rcParams['legend.columnspacing']=10
plt.rcParams['legend.handletextpad']=0.3
plt.rcParams['legend.handlelength']=0.7
plt.rcParams['legend.framealpha']=1
plt.rcParams['legend.fancybox']=True
ppm_o_c='#002bff'
ppm_p_c='#804000'
deg_o_c='#002bff'
deg_p_c='#804000'
insert_c='#804000'
o_label='Original Data'
p_label='Predicted Data'
line_c='#f29644'
point_size=14
tar_year=np.array([2004,2022,2050,2100])