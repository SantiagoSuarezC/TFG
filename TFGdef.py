import pandas as pd
import numpy as np

#cargamos el excel y lo manipulamos:

path="D:/Escritorio/datos.xlsx"
df_1 = pd.read_excel(path,header=None,sheet_name="coordenadas")
df_2 = pd.read_excel(path,sheet_name="lista")
df_1.columns=['NdCi','x','y','z']
df_1.drop([0],axis=0,inplace=True)
df_1.reset_index(drop=True, inplace=True)

#Creo un índice para identificar cada caso

df_1["index"]=0
label=0
for i in range(0,len(df_1)):
    if i%9==0 and i!=0:
        label=label+1
    df_1["index"].iloc[i]=label
index=df_1["index"].unique()
aux=df_1[df_1["index"]==index[0]]

df=df_1.merge(df_2,on="index")

#ajuste plano puntos totales

v_norm=pd.DataFrame()
df["xf"]=0
df["yf"]=0
df["zf"]=0
df["index_2"]=[0,1,2,3,4,5,6,7,8]*431
for i in range(0,len(df)):
    if df["index_2"].iloc[i]<=5:
        df["xf"].iloc[i]=512-df["y"].iloc[i]*df["dimX"].iloc[i]
        df["yf"].iloc[i]=512-df["x"].iloc[i]*df["dimX"].iloc[i]
        df["zf"].iloc[i]=df["NdCi"].iloc[i]*df["ST"].iloc[i]-df["z"].iloc[i]*df["ST"].iloc[i]
    else:
        df["xf"].iloc[i]=512-df["x"].iloc[i]*df["dimX"].iloc[i]
        df["yf"].iloc[i]=512-df["z"].iloc[i]*df["dimX"].iloc[i]
        df["zf"].iloc[i]=df["y"].iloc[i]*df["ST"].iloc[i]


for i in index:
    aux=df[df["index"]==i]
    points=np.array([aux["xf"],aux["yf"],aux["zf"]])
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    left = svd[0]
    n=left[:, -1]
    v_norm=v_norm.append(pd.Series(np.append(n,i)),ignore_index=True)

v_norm.columns=['xn','yn','zn','cason']

#Ajuste puntos arriba

v_normup=pd.DataFrame()
v_nu=pd.DataFrame()


for i in range(0,len(df)):
    if df["index_2"].iloc[i]>=2 and df["index_2"].iloc[i]<=8:
        new_row = {"xu": df["xf"].iloc[i],"yu":df["yf"].iloc[i],"zu":df["zf"].iloc[i]}
        v_normup = v_normup.append(new_row, ignore_index=True)

v_normup["indexu"]=0
label=0
for i in range(0,len(v_normup)):
    if i%7==0 and i!=0:
        label=label+1
    v_normup["indexu"].iloc[i]=label
index=v_normup["indexu"].unique()
auxu=v_normup[v_normup["indexu"]==index[0]]
        
for i in index:
    auxu=v_normup[v_normup["indexu"]==i]
    pointsu=np.array([auxu["xu"],auxu["yu"],auxu["zu"]])
    svdu = np.linalg.svd(pointsu - np.mean(pointsu, axis=1, keepdims=True))
    leftu = svdu[0]
    nu=leftu[:, -1]
    v_nu=v_nu.append(pd.Series(np.append(nu,i)),ignore_index=True)       
    
        
v_normup.columns=['xu','yu','zu','casou']
v_nu.columns=['xnu','ynu','znu','casou']

#Ajuste puntos abajo

v_normd=pd.DataFrame()
v_nd=pd.DataFrame()


for i in range(0,len(df)):
    if df["index_2"].iloc[i]>=0 and df["index_2"].iloc[i]<=3:
        new_row = {"xd": df["xf"].iloc[i],"yd":df["yf"].iloc[i],"zd":df["zf"].iloc[i]}
        v_normd = v_normd.append(new_row, ignore_index=True)

v_normd["indexd"]=0
label=0
for i in range(0,len(v_normd)):
    if i%4==0 and i!=0:
        label=label+1
    v_normd["indexd"].iloc[i]=label
index=v_normd["indexd"].unique()
auxd=v_normd[v_normd["indexd"]==index[0]]
        
for i in index:
    auxd=v_normd[v_normd["indexd"]==i]
    pointsd=np.array([auxd["xd"],auxd["yd"],auxd["zd"]])
    svdd = np.linalg.svd(pointsd - np.mean(pointsd, axis=1, keepdims=True))
    leftd = svdd[0]
    nd=leftd[:, -1]
    v_nd=v_nd.append(pd.Series(np.append(nd,i)),ignore_index=True)       
    
        
v_normd.columns=['xd','yd','zd','casod']
v_nd.columns=['xnd','ynd','znd','casod']    

#Cálculo de ángulos

ang=pd.DataFrame()
#ángulo entre planos
ang["angp"]=[1]*431
#ángulo plano total
ang["angyz"]=[1]*431
ang["angxz"]=[1]*431
ang["angxy"]=[1]*431
for i in range(0,len(v_nd)): 
    a=np.arccos((v_nu["xnu"].iloc[i]*v_nd["xnd"].iloc[i]+v_nu["ynu"].iloc[i]*v_nd["ynd"].iloc[i]+v_nu["znu"].iloc[i]*v_nd["znd"].iloc[i])/np.sqrt((np.square(v_nu["znu"].iloc[i])+np.square(v_nu["xnu"].iloc[i])+np.square(v_nu["ynu"].iloc[i]))*(np.square(v_nd["znd"].iloc[i])+np.square(v_nd["xnd"].iloc[i])+np.square(v_nd["ynd"].iloc[i]))))*(180/np.pi)
    ang["angp"].iloc[i]=min(a,(180-a))
    c=np.arccos((v_norm["xn"].iloc[i])/np.sqrt(np.square(v_norm["xn"].iloc[i])+np.square(v_norm["yn"].iloc[i])+np.square(v_norm["zn"].iloc[i])))*(180/np.pi)
    ang["angyz"].iloc[i]=min(c,(180-c))
    c=np.arccos((v_norm["yn"].iloc[i])/np.sqrt(np.square(v_norm["xn"].iloc[i])+np.square(v_norm["yn"].iloc[i])+np.square(v_norm["zn"].iloc[i])))*(180/np.pi)
    ang["angxz"].iloc[i]=min(c,(180-c))
    c=np.arccos((v_norm["zn"].iloc[i])/np.sqrt(np.square(v_norm["xn"].iloc[i])+np.square(v_norm["yn"].iloc[i])+np.square(v_norm["zn"].iloc[i])))*(180/np.pi)
    ang["angxy"].iloc[i]=min(c,(180-c))

#Cálculo distancias
prueba=pd.DataFrame()
ang["H"]=[0]*431
ang["W"]=[0]*431
ang["H/W"]=[0]*431
dfproy=pd.DataFrame()
dfproyx=pd.DataFrame()
dfproyy=pd.DataFrame()
dfproyz=pd.DataFrame()
dfproyf=pd.DataFrame()
for i in index:
    auxu=v_normup[v_normup["casou"]==i]
    D=(np.mean(auxu["xu"])*v_nu["xnu"].iloc[i]+np.mean(auxu["yu"])*v_nu["ynu"].iloc[i]+np.mean(auxu["zu"])*v_nu["znu"].iloc[i])
    l=-1*(v_nu["xnu"].iloc[i]*auxu["xu"]+v_nu["ynu"].iloc[i]*auxu["yu"]+v_nu["znu"].iloc[i]*auxu["zu"]-D)/(np.square(v_nu["xnu"].iloc[i])+np.square(v_nu["ynu"].iloc[i])+np.square(v_nu["znu"].iloc[i]))
    l=l.to_frame()
    xproy=v_nu["xnu"].iloc[i]*l[0]+auxu["xu"]
    xproy=xproy.to_frame()
    yproy=v_nu["ynu"].iloc[i]*l[0]+auxu["yu"]
    yproy=yproy.to_frame()
    zproy=v_nu["znu"].iloc[i]*l[0]+auxu["zu"]
    zproy=zproy.to_frame()
    dfproy=dfproy.append(l,ignore_index=True)
    dfproyx=dfproyx.append(xproy,ignore_index=True)
    dfproyy=dfproyy.append(yproy,ignore_index=True)
    dfproyz=dfproyz.append(zproy,ignore_index=True)
    ang["W"].iloc[i]=np.linalg.norm(np.array([xproy[0].iloc[0],yproy[0].iloc[0],zproy[0].iloc[0]])-np.array([xproy[0].iloc[1],yproy[0].iloc[1],zproy[0].iloc[1]]))
    ang["H"].iloc[i]=max(np.linalg.norm(np.array([xproy[0].iloc[6]-0.5*(xproy[0].iloc[0]+xproy[0].iloc[1]),yproy[0].iloc[6]-0.5*(yproy[0].iloc[0]+yproy[0].iloc[1]),zproy[0].iloc[6]-0.5*(zproy[0].iloc[0]+zproy[0].iloc[1])])),np.linalg.norm(np.array([xproy[0].iloc[5]-0.5*(xproy[0].iloc[0]+xproy[0].iloc[1]),yproy[0].iloc[5]-0.5*(yproy[0].iloc[0]+yproy[0].iloc[1]),zproy[0].iloc[5]-0.5*(zproy[0].iloc[0]+zproy[0].iloc[1])])),np.linalg.norm(np.array([xproy[0].iloc[4]-0.5*(xproy[0].iloc[0]+xproy[0].iloc[1]),yproy[0].iloc[4]-0.5*(yproy[0].iloc[0]+yproy[0].iloc[1]),zproy[0].iloc[4]-0.5*(zproy[0].iloc[0]+zproy[0].iloc[1])])))
    
    ang["H/W"].iloc[i]=ang["H"].iloc[i]/ang["W"].iloc[i]



dfproyf=pd.concat([dfproy,dfproyx,dfproyy,dfproyz],axis=1)
dfproyf.columns=["l","x","y","z"]

resultados=pd.DataFrame()
resultados=pd.concat([df_2["Sex"],df_2["Age"],ang],axis=1)
resultados.to_excel("resultados.xlsx")


    
#gráficas
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


df["index"]=0
label=0
for i in range(0,len(df)):
    if i%9==0 and i!=0:
        label=label+1
    df["index"].iloc[i]=label
index=df["index"].unique()
aux=df[df["index"]==index[0]]  

for i in index:
    aux=df[df["index"]==i]
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(aux["xf"],aux["yf"],aux["zf"], label="datos caso")
    
    


#Correlograma

import seaborn as sns


corrdf = resultados.drop(["angyz"], axis=1)
corrdf['Sex'].replace({'M': 1, 'F': 0}, inplace=True)
corr_matrix = corrdf.corr()

# Generar el gráfico de calor
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, cmap='RdYlGn', annot=True, fmt='.2f', mask=mask, center=0)

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()





























    



