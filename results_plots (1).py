#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
#%matplotlib qt  
#ouvre une fenêtre à chaque nouveau output

import numpy as np
#This cell is used to transform the values in a column into a list
cfos_ventral_CP ="""18
36
26
3
0
2
17
46
46
9
18
13
5
8
6
32
42
50
2
1
1
12
33
32
9
19
21
35
36
31"""
cfos_ventral_CP= cfos_ventral_CP.replace('\n', ",")
print(cfos_ventral_CP)


# In[52]:


#cfos ventral counting

cfos_ventral_FJ = [14,26,15,13,11,16,7,9,8,9,13,8,4,8,8,15,13,22,8,7,6,11,13,4,7,11,17,16,14,15]
cfos_ventral_CP =[328,219,248,431,345,326,588,626,577,551,501,458,727,557,612,324,356,249,711,759,629,464,271,286,456,382,296,231,230,291]
cfos_ventral_CPbis = [32,55,54,27,44,41,6,2,3,12,22,19,6,10,10,45,55,64,6,2,3,18,43,39,12,21,25,48,41,39]  #cpt=4
cfos_ventral_CP5=[18,36,26,3,0,2,17,46,46,9,18,13,5,8,6,32,42,50,2,1,1,12,33,32,9,19,21,35,36,31]


# In[89]:


#results plots for cfos ventral mettre fiji  à gauche
fig1, ax1 = plt.subplots()
ax1.set_title('cfos ventral counting', color="teal")
ax1.set_ylabel('Cell count')
boxp1 = ax1.boxplot([cfos_ventral_FJ, cfos_ventral_CP5], labels = [ 'FIJI method \n "ground truth"', 'Cellpose Method \n with c_p_t=5'], patch_artist=True)
colors = ['chartreuse', 'mediumpurple']
for patch, color in zip(boxp1['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('boxplot_cfos_ventral.png')


# In[90]:


#w/ different values of cpt, influence on cfos ventral

fig_all, ax_all = plt.subplots()
ax_all.set_title('Influence of cellprob on cfos ventral', color="teal")
ax_all.set_ylabel('Cell count')
boxp_all = ax_all.boxplot([cfos_ventral_FJ,cfos_ventral_CP, cfos_ventral_CP5,cfos_ventral_CPbis],
                          labels =['FIJI method \n "ground truth"', 'Cellpose \n with c_p_t =-6', 'Cellpose \n with c_p_t=5','Cellpose \n with c_p_t=4'],
                          patch_artist = True)
colors = [ 'chartreuse', 'skyblue', 'mediumpurple','darkturquoise']
for patch, color in zip(boxp_all['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig('boxplot_influence_cpt_cfos_ventr.png')


# In[91]:


"influence of cellprob on cfos ventral"
import numpy as np
#np.random.seed(19680801)

fig, ax = plt.subplots()
ax.scatter(range(30), cfos_ventral_FJ, c= 'chartreuse', label = 'fj_cfos')
ax.scatter(range(30), cfos_ventral_CPbis, c = 'lightcoral', label = 'c_threshold=4')
ax.scatter(range(30), cfos_ventral_CP5, c = 'mediumpurple', label = 'c_threshold=5')
ax.scatter(range(30), cfos_ventral_CP, c = 'skyblue', label = 'c_threshold=-6')
ax.set_title('Influence of the cell probability threshold on cfos-ventral', color ='teal')
ax.set_ylabel('Cell Count')
ax.set_xlabel('#image')
ax.legend()
ax.grid(True)
plt.show()
plt.savefig('scatterplot_infl_cpt_cfos_ventr.png')


# In[92]:


#cfos dorsal counting
cfos_dorsal_CP=[396,274,300,359,293,274,315,366,323,487,384,449,310,317,407,287,257,369,384,329,356,407,378,485,370,275,321,285,316,296,379,287,116,195,253]
cfos_dorsal_FJ=[18,22,22,4,8,9,5,6,4,3,6,4,5,3,11,13,17,8,4,9,4,2,4,8,8,11,6,7,6,2,1,2,8,10,17]
cfos_dorsal_CP2 = [0,8,10,3,13,12,1,1,2,2,2,3,2,4,3,7,19,6,2,41,1,2,12,3,9,8,19,21,21,3,2,10,16,29,27]  #(w/ cell prob = 5 instead of 3 or -6)

#standard deviation
sd_cfos=[]
for i in range(35):
    sd_cfos.append(abs(cfos_dorsal_FJ[i]-cfos_dorsal_CP2[i]))
print(sd_cfos)


# In[93]:


#cfos dorsal results plots (with cell prob = 5 instead of 3 or -6)
fig2, ax2 = plt.subplots()
ax2.set_title('Cfos dorsal counting', color="teal")
#ax2.set_xlabel('Methods')
ax2.set_ylabel('Cell count')
boxp2 = ax2.boxplot([cfos_dorsal_FJ, cfos_dorsal_CP2], labels = ['FIJI method \n "ground truth"', 'Cellpose method \n with c_p_t=5'], patch_artist=True)
colors = [ 'chartreuse', 'darkturquoise']
for patch, color in zip(boxp2['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('boxplot_cfos_dorsal.png')


# In[94]:


#w/ different values of cpt, influence on cfos dorsal
cpthreshold3 = [396,274,300,359,293,274,315,366,323,487,384,449,310,317,407,287,257,369,384,329,356,407,378,485,370,275,321,285,316,296,379,287,116,195,253]
cpthreshold5 = [0,8,10,3,13,12,1,1,2,2,2,3,2,4,3,7,19,6,2,41,1,2,12,3,9,8,19,21,21,3,2,10,16,29,27]
fj_cfos = [18,22,22,4,8,9,5,6,4,3,6,4,5,3,11,13,17,8,4,9,4,2,4,8,8,11,6,7,6,2,1,2,8,10,17]
fig_all, ax_all = plt.subplots()
ax_all.set_title('Influence of cellprob on cfos-dorsal.tif', color ="teal")
ax_all.set_ylabel('Cell count')
boxp_all = ax_all.boxplot([fj_cfos, cpthreshold3, cpthreshold5],
                          labels =['FIJI method \n "ground truth"', 'Cellpose \n with c_p_t =3', 'Cellpose \n with c_p_t=5'],
                          patch_artist = True)
colors = [ 'chartreuse', 'teal','darkturquoise']
for patch, color in zip(boxp_all['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig('boxplot_influence_cpt_cfos_dors.png')


# In[97]:


#influence of cellprob on cfos dorsal
import numpy as np
#np.random.seed(19680801)

fig, ax = plt.subplots()
ax.scatter(range(35), fj_cfos, c= 'chartreuse', label = 'fj_cfos')
ax.scatter(range(35), cpthreshold3, c = 'lightcoral', label = 'c_threshold=3')
ax.scatter(range(35), cpthreshold5, c= 'mediumpurple', label= 'c_threshold=5 ')

ax.set_title('Influence of the cell probability threshold on cfos-dorsal', color= "teal")
ax.set_ylabel('Cell Count')
ax.set_xlabel('#image')
ax.legend()
ax.grid(True)
plt.show()
plt.savefig('scatterplot_infl_cpt_cfos_dors.png')


# In[13]:


#dapi ventral counting
dapi_ventral_CP = [493,475,442,314,299,315,446,441,439,512,544,479,399,480,414,619,478,350,473,534,509,569,590,583,525,526,872,290,317,321,586,445,397]
dapi_ventral_FJ = [689,748,609,570,643,736,763,619,587,559,731,703,514,525,596,784,730,505]

#dapis dorsal counting
dapi_dorsal_CP = [359,190,241,341,261,260,232,262,297,172,252,266,185,202,382,451,470,234,374,355,202,251,223,177,157,167,152,200,133,331,386,365,234,295,364]
dapi_dorsal_FJ = [502,442,429,613,426,383,326,357,352,274,313,522,330,352,339,366]


# In[14]:


#dapi ventral plots

fig3, ax3 = plt.subplots()
ax3.set_title("DAPI ventral counting",color="teal")
#ax3.set_xlabel('Methods')
ax3.set_ylabel('Cell count')
boxp3 = ax3.boxplot([dapi_ventral_FJ, dapi_ventral_CP], labels = ['FIJI method \n "ground truth"','Cellpose Method \n with c_p_t=5'], patch_artist=True)
colors = ['chartreuse', 'darkturquoise']
for patch, color in zip(boxp3['boxes'], colors):
    patch.set_facecolor(color)
plt.savefig('boxplot_dapi_ventral.png')

#dapi dorsal results plots (where cells have been counted accordingly to the fact that a cell transcends many slices)
fig4, ax4 = plt.subplots()
ax4.set_title('DAPI dorsal counting', color="teal")
ax4.set_ylabel('Cell count')
boxp4 = ax4.boxplot([dapi_dorsal_FJ, dapi_dorsal_CP], labels = ['FIJI Method \n "ground truth"', ' Cellpose method \ with c_p_t=5'], patch_artist=True)
colors = ['chartreuse', 'darkturquoise']
for patch, color in zip(boxp4['boxes'], colors):
    patch.set_facecolor(color)
    
plt.savefig('boxplot_dapi_dorsal.png')


# In[100]:


#influence of the cell prob threshold
c1 = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
nb1_CP =[474,474,474,474,474,474,474,465,457,433,382,279,154]
nb1bis_CP =[434,434,434,434,434,434,434,418,391,327,245,144,83]
fig, ax = plt.subplots()
plt.scatter(c1, nb1_CP, c = 'yellow')
plt.scatter(c1, nb1bis_CP, c = 'lightblue')
plt.hlines(730, -6,6, color = 'yellow')   
plt.hlines(748,-6,6, color="lightblue")
plt.title('Influence of the cell probability threshold parameter for DAPI', color="teal")
plt.xlabel('Cell_prob_threshold')
plt.ylabel('Cell count')
plt.savefig('scatterplot_cpt_influence_DAPI.png')


# It seems that for dapis, in order to obtain suitable/ closer to the ground truth, the cell probability threshold has to be very weak. (~-6/-4)

# In[9]:


#influence of the cell prob threshold for cfos
c2 = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
nb2_CP = [52,52,52,52,52,52,52,29,11,6,2,1,1]
nb3_CP = [60,60,60,60,60,60,60,32,20,10,4,2,1]
fig, ax = plt.subplots()
plt.scatter(c2, nb2_CP, c = 'lightgreen')
plt.hlines(5, -6,6, color ='lightgreen')
plt.scatter(c2, nb3_CP, c = 'darkturquoise')
plt.hlines(6, -6,6, color ="darkturquoise")
plt.title('Influence of the cell probability threshold on cfos', color = "teal")
plt.xlabel('cell_prob_threshold')
plt.ylabel('Cell count')
plt.savefig('cfos_cpt_influence.png')


# In opposition to the dapis, the computing of cfos has to be made with a very high threshold for the cell probability. >3 &<5

# In[17]:


#influence of the diameter on dapis
c3 = [12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13]
nb3_CP = [481,478,477,475,469,450,457,458,448,447]

fig, ax = plt.subplots()
plt.scatter(c3, nb3_CP, c = 'purple')
plt.hlines(500, 12,13.1, color ='purple')#some dapis are to be recounted
plt.title('Influence of the diameter for DAPI')
plt.xlabel('diameter in pixel')
plt.ylabel('Cell count')


# In[102]:


c4 = [12.1,12.2,12.3,12.4,12.5,12.6,12.7]
nb4_CP = [72,67,68,63,67,76,74]
plt.scatter(c4, nb4_CP, c = 'pink')
plt.hlines(11, 12,13.1, color ='pink')
plt.title('Influence of the diameter for cfos')
plt.xlabel('Diameter in pixel')
plt.ylabel('Cell count')


# There are still too many differences between the CP counting and the ground truth (=FIJI).

# In[6]:


# Mann-Whitney U test for FIJI between cfos and DAPI
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
# seed the random number generator
seed(1)
#samples
FJ_cfos_dorsal = [18,22,22,4,8,9,5,6,4,3,6,4,5,3,11,13,17,8,4,9,4,2,4,8,8,11,6,7,6,2,1,2,8,10,17]
FJ_cfos_ventral = [14,26,15,13,11,16,7,9,8,9,13,8,4,8,8,15,13,22,8,7,6,11,13,4,7,11,17,16,14,15]
FJ_DAPI_dorsal = [502,442,429,613,426,383,326,357,352,274,313,522,330,352,339,366]
FJ_DAPI_ventral = [689,748,609,570,643,736,763,619,587,559,731,703,514,525,596,784,730,505]

# compare samples
statc, p_c = mannwhitneyu(FJ_cfos_ventral, FJ_cfos_dorsal)
print('Statistics  for cfos=%.3f, p_cfos=%.3f' % (statc, p_c))

statD, p_D = mannwhitneyu(FJ_DAPI_ventral, FJ_DAPI_dorsal)
print('Statistics for DAPI=%.3f, p_DAPI=%.3f' % (statD, p_D))

# interpret
alpha = 0.05
if p_c > alpha:
	print('With FIJI, Cfos dorsal and ventral have same distribution (fail to reject H0)')
else:
	print('With FIJI, Cfos dorsal and ventral have different distribution (reject H0)')

if p_D > alpha:
	print('With FIJI, DAPI dorsal and ventral have same distribution (fail to reject H0)')
else:
	print('With FIJI, DAPI dorsal and ventral have different distribution (reject H0)')
    


# In[8]:


# Mann-Whitney U test for Cellpose between cfos and DAPI
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
# seed the random number generator
seed(1)
#samples
CP_cfos_dorsal = [0,8,10,3,13,12,1,1,2,2,2,3,2,4,3,7,19,6,2,41,1,2,12,3,9,8,19,21,21,3,2,10,16,29,27]
CP_cfos_ventral = [18,36,26,3,0,2,17,46,46,9,18,13,5,8,6,32,42,50,2,1,1,12,33,32,9,19,21,35,36,31]
CP_DAPI_dorsal = [359,190,241,341,261,260,232,262,297,172,252,266,185,202,382,451,470,234,374,355,202,251,223,177,157,167,152,200,133,331,386,365,234,295,364]
CP_DAPI_ventral = [493,475,442,314,299,315,446,441,439,512,544,479,399,480,414,619,478,350,473,534,509,569,590,583,525,526,872,290,317,321,586,445,397]

# compare samples
statc, p_c = mannwhitneyu(CP_cfos_ventral, CP_cfos_dorsal)
print('Statistics  for cfos=%.3f, p_cfos=%.3f' % (statc, p_c))

statD, p_D = mannwhitneyu(CP_DAPI_ventral, CP_DAPI_dorsal)
print('Statistics for DAPI=%.3f, p_DAPI=%.3f' % (statD, p_D))

# interpret
alpha = 0.05
if p_c > alpha:
	print('With Cellpose, Cfos dorsal and ventral have same distribution (fail to reject H0)')
else:
	print('With Cellpose, Cfos dorsal and ventral have different distribution (reject H0)')

if p_D > alpha:
	print('With Cellpose, DAPI dorsal and ventral have same distribution (fail to reject H0)')
else:
	print('With Cellpose, DAPI dorsal and ventral have different distribution (reject H0)')
    


# The non-parametric Mann-Whitney U test rejected H0 for both of the methods therefore it confirmed that dorsal and ventral have different distributions. 
