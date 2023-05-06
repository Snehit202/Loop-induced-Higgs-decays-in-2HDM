import Higgs.predictions as HP
import Higgs.bounds as HB
import Higgs.signals as HS
import numpy as np
import pandas as pd
from tqdm import tqdm

pred = HP.Predictions() # create the model predictions
bounds = HB.Bounds('/home/snehit/higgstools-main/hbdataset') # load HB dataset
signals = HS.Signals('/home/snehit/higgstools-main/hsdataset') # load HS dataset


# In[88]:


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import rundec
#constants
Gf = 1.1663787e-5 #fremi constant
sw = 0.472122 #sin(mixing angle)
cw=np.sqrt(1-sw**2)
#mass of elementary particles in GeV
#leptons
m_e = 0.000511
m_tau = 1.777
m_mu = 0.1057 
#u-type quarks
m_uu = 0.0019
m_cc = 1.6
m_tt = 178.2
#d-type quarks 
m_dd = 0.0044
m_ss = 0.093 
m_bb = 4.78 
#vector bosons
m_w = 80.433 
m_z = 91.1876 

#2HDM parameters alpha and beta
#A = np.linspace(-np.pi/2,np.pi/2,10001)

def fcp(stg,x,A,B):
    if(x==1):
        return np.cos(A)/np.sin(B)
    if(x==2):
        if(stg=='u'):
            return np.cos(A)/np.sin(B)
        else:
            return -np.sin(A)/np.cos(B)
    if(x==3):
        if(stg=='l'):
            return -np.sin(A)/np.cos(B)
        else:
            return np.cos(A)/np.sin(B)
    if(x==4):
        if(stg=='d'):
            return -np.sin(A)/np.cos(B)
        else:
            return np.cos(A)/np.sin(B)


def hfp2(A,B):  # particles that couple to phi2 for light higgs
    return np.cos(A)/np.sin(B)

def hfp1(A,B):  # particles that couple to phi2 for light higgs
    return -np.sin(A)/np.cos(B)
def hv(A,B):
    return np.sin(B-A)
def Hv(A,B):
    return np.cos(B-A)
def hpm(A,B):
    return np.sin(B-A) + np.cos(2*B)*np.sin(A+B)/(2*cw**2)
def Hpm(A,B):
    return np.cos(B-A) - np.cos(2*B)*np.sin(A+B)/(2*cw**2)
#calculating Higgs decay widths:
crd = rundec.CRunDec()
def coupling(mh):
	return crd.AlphasExact(.1185,91.18,mh,5,4)

def qmass(mq,mh):
	return crd.mOS2mMS(mq,crd.mq,coupling(mh),mh,5,4)

gs = coupling(125.09)
print(gs)
m_t = qmass(m_tt,125.09)
m_b = qmass(m_bb,125.09)
m_c = qmass(m_cc,125.09)
m_u = m_uu
m_d = m_dd
m_s = m_ss
#calculating Higgs decay widths:
#1. H->ff~/qq~
def G_Hff(mf,mH):
    G = np.where(mH>=2*mf,(Gf*mH*(mf**2.)*(1.-(4.*(mf**2)/(mH**2)))**1.5)/(4.*np.sqrt(2)*np.pi),0)
    return G

#2 H->WW/ZZ
def G_Hww(mw,mh):#Z decay will be half of this
    G2=np.where(mh>=2*mw,((Gf*mh**3)/(8.*np.sqrt(2)*np.pi))*np.sqrt(1-(4*mw**2)/mh**2)*(1-(4.*mw**2)/(mh**2) +(12.*mw**4)/mh**4),0)
    return G2

#3 H->Vf~f
def G_Vff(mv,mf,mh):
    dw =1
    dz = ((7./12.)-(10./9.)*sw**2+(40./9.)*sw**4)
    x= (mv**2)/(mh**2)
    R = np.where(abs((mh/mv)-1.5)<0.5,(-abs(1-x)/(2.*x))*(2.-13.*x+47.*x**2)-1.5*(1.-6.*x+4.*x**2)*(np.log(x))+(3.*(1.-8.*x+20.*x**2)/np.sqrt(4.*x-1.))*np.arccos((3.*x-1.)/(2.*x**1.5)),0)
    if(mv==m_w):
        G3=3*(Gf**2)*(mv**4)*mh*dw*R/(16.*np.pi**3)
    if(mv==m_z):
        G3=3*(Gf**2)*(mv**4)*mh*dz*R/(16.*np.pi**3)
    return G3 

#5 H-> gg
def f(x):
    return np.where(x<=1,np.arcsin(np.sqrt(x))**2,-0.25*(np.log((1+np.sqrt(1-x**(-1)))/(1-np.sqrt(1-x**(-1))))-complex(0,1)*np.pi)**2)


def As(x):
    return -(x-f(x))/x**2
def Af(x):
    return (2./x**2)*(x+(x-1.)*f(x))
def Av(x):
    return -(2.*x**2 + 3.*x +3.*(2*x-1.)*f(x))/x**2


#6 H-> 2 photons

#7 H-> Z photon
def g(x):
    return np.where(x<1,(np.sqrt(x**(-1.)-1.))*np.arcsin(np.sqrt(x)),0.5*np.sqrt(1.-x**(-1.))*(np.log((1+np.sqrt(1-x**(-1)))/(1-np.sqrt(1-x**(-1))))-complex(0,1)*np.pi))

def I1(x,y):
    return 0.5*((x*y/(x-y))+((x*y)**2)*(f(1./x)-f(1./y))/(x-y)**2) + (g(1./x)-g(1./y))*(x**2)*y/(x-y)**2

def I2(x,y):
    return -0.5*(x*y/(x-y))*(f(1./x)-f(1./y))

def A2f(x,y):
    return I1(x,y)-I2(x,y)

dqq = lambda a,mq,mh : 1+5.67*a/np.pi +(35.94+1.57-1.36*5-.6777*2*np.log(mh/m_t)+(1./9.)*(2*np.log(mq/mh)**2))*(a/np.pi)**2 +(164.14-25.77*5+.26*25)*(a/np.pi)**3

def A2v(x,y):
    return cw*(4.*(3.-(sw/cw)**2.)*I2(x,y) + I1(x,y)*(((1+(2./x))*(sw/cw)**2)-(5.+(2./x))))
def hup(m_u,m_h,A,B,x):
    return 3*G_Hff(m_u,m_h)*(dqq(gs,m_u,125.09))*fcp('u',x,A,B)**2
def hdown(m_d,m_h,A,B,x):
    return 3*G_Hff(m_d,m_h)*(dqq(gs,m_d,125.09))*fcp('d',x,A,B)**2
def hlept(m_e,m_h,A,B,x):
    return G_Hff(m_e,m_h)*fcp('l',x,A,B)**2
def hW(m_w,m_h,A,B):
    return (G_Vff(m_w, m_d, m_h)+G_Hww(m_w,m_h))*hv(A,B)**2
def hZ(m_z,m_h,A,B):
    return (G_Vff(m_z, m_d, m_h)+0.5*G_Hww(m_z,m_h))*hv(A,B)**2
def G_Hgg(mh,A,B,x):
    k= (Gf/(36.*np.sqrt(2)*np.pi**3))
    #gs = (12.*np.pi)/(21*np.log((mh**2)/0.004))
    cnt = 1*k*(gs**2)*mh**3
    E = (95./4 - 7.)*gs/np.pi
    t = (mh**2)/(4*m_t**2)
    b = (mh**2)/(4*m_b**2)
    c = (mh**2)/(4*m_c**2)
    s = (mh**2)/(4*m_s**2)
    u = (mh**2)/(4*m_u**2)
    d = (mh**2)/(4*m_d**2)
    S = (Af(t)+Af(u)+Af(c))*fcp('u',x,A,B) + (Af(b)+Af(s)+Af(d))*fcp('d',x,A,B)
    return (1.+E)*cnt*abs(0.75*S)**2
def G_Hpp(mh,A,B,x,m_pm):
    #gs= (12.*np.pi)/(21*np.log((mh**2)/0.004))
    a= 1./137.5
    k= (Gf*(a**2)/(128.*np.sqrt(2)*np.pi**3))
    q1=fcp('u',x,A,B)*4./9.
    q2=fcp('d',x,A,B)*1./9.
    cnt = k*mh**3
    t = (mh**2)/(4*m_tt**2)
    b = (mh**2)/(4*m_bb**2)
    c = (mh**2)/(4*m_cc**2)
    s = (mh**2)/(4*m_ss**2)
    u = (mh**2)/(4*m_u**2)
    d = (mh**2)/(4*m_d**2)
    w = (mh**2)/(4*m_w**2)
    e = (mh**2)/(4*m_e**2)
    mu = (mh**2)/(4*m_mu**2)
    tau = (mh**2)/(4*m_tau**2)
    pm = (mh**2)/(4*m_pm**2)
    S = 3*(Af(t)*q1+Af(b)*q2+Af(c)*q1+Af(s)*q2+Af(u)*q1+Af(d)*q2)*(1.-gs/np.pi)+(Af(e)+Af(mu)+Af(tau))*fcp('l',x,A,B)+Av(w)*hv(A,B)+As(pm)*hpm(A,B)*(m_w/m_pm)**2
    return(cnt*abs(S)**2)
def G_Hzp(m_h,A,B,x,m_pm):
    a= 1./137.5
    const = (a*(m_w**2)*(Gf**2)*(m_h**3)*(1-(m_z/m_h)**2)**3)/(64*np.pi**4)
    e1 = 4*(m_e/m_h)**2
    e2 = 4*(m_e/m_z)**2
    mu1 = 4*(m_mu/m_h)**2
    mu2 = 4*(m_mu/m_z)**2
    tau1 = 4*(m_tau/m_h)**2
    tau2 = 4*(m_tau/m_z)**2
    u1 = 4*(m_u/m_h)**2
    u2 = 4*(m_u/m_z)**2
    d1 = 4*(m_d/m_h)**2
    d2 = 4*(m_d/m_z)**2
    b1 = 4*(m_bb/m_h)**2
    b2 = 4*(m_bb/m_z)**2
    t1 = 4*(m_tt/m_h)**2
    t2 = 4*(m_tt/m_z)**2
    c1 = 4*(m_cc/m_h)**2
    c2 = 4*(m_cc/m_z)**2
    s1 = 4*(m_ss/m_h)**2
    s2 = 4*(m_ss/m_z)**2
    w1 = 4*(m_w/m_h)**2
    w2 = 4*(m_w/m_z)**2
    pm1 = 4*(m_pm/m_h)**2
    pm2 = 4*(m_pm/m_z)**2
    S = A2v(w1,w2)*hv(A,B)+(3./(sw*cw))*(0.5-(4./3.)*sw**2)*(A2f(u1,u2)+A2f(c1,c2)+A2f(t1,t2)*(1-gs/np.pi))*fcp('u',x,A,B)+(0.5/(sw*cw))*(-0.5+(2./3.)*(sw**2))*(A2f(d1,d2)+A2f(s1,s2)+A2f(b1,b2))*fcp('d',x,A,B) + (0.25/(sw*cw))*(-0.5 + 2*sw**2)*(A2f(e1,e2)+A2f(mu1,mu2)+A2f(tau1,tau2))*fcp('l',x,A,B) + ((1-2*sw**2)/(sw*cw))*I1(pm1,pm2)*hpm(A,B)*(m_w/m_pm)**2
    return const*abs(S)**2
def Gamma(m_h,A,B,x,m_pm):
    return hup(m_u, m_h, A, B,x)+hup(m_t, m_h, A, B,x)+hup(m_c, m_h, A, B,x)+hdown(m_d, m_h, A, B,x)+hdown(m_b, m_h, A, B,x)+hdown(m_s, m_h, A, B,x)+hlept(m_e, m_h, A, B,x)+hlept(m_mu, m_h, A, B,x)+hlept(m_tau, m_h, A, B,x)+hW(m_w,m_h,A,B)+hZ(m_z,m_h,A,B)+G_Hgg(m_h,A,B,x)+G_Hpp(m_h,A,B,x,m_pm)+G_Hzp(m_h, A, B,x,m_pm)
print(Gamma(125,0.1,0.5,1,200))
#Heavy Higgs
def Fcp(stg,x,A,B):
    if(x==1):
        return np.sin(A)/np.sin(B)
    if(x==2):
        if(stg=='u'):
            return np.sin(A)/np.sin(B)
        else:
            return np.cos(A)/np.cos(B)
    if(x==3):
        if(stg=='l'):
            return np.cos(A)/np.cos(B)
        else:
            return np.sin(A)/np.sin(B)
    if(x==4):
        if(stg=='d'):
            return np.cos(A)/np.cos(B)
        else:
            return np.sin(A)/np.sin(B)
def Hup(m_u,m_h,A,B,x):
    if(m_u==m_tt or m_u==m_cc):
        mu=qmass(m_u,m_h)
    else:
        mu=m_u
    Gs = coupling(m_h)
    return 3*G_Hff(mu,m_h)*(dqq(Gs,mu,m_h))*Fcp('u',x,A,B)**2
def Hdown(m_d,m_h,A,B,x):
    if(m_d==m_bb):
        md=qmass(m_d,m_h)
    else:
        md=m_d
    Gs = coupling(m_h)
    return 3*G_Hff(m_d,m_h)*(dqq(Gs,md,m_h))*Fcp('d',x,A,B)**2
def Hlept(m_e,m_h,A,B,x):
    return G_Hff(m_e,m_h)*Fcp('l',x,A,B)**2
def HW(m_w,m_h,A,B):
    return (G_Vff(m_w, m_d, m_h)+G_Hww(m_w,m_h))*Hv(A,B)**2
def HZ(m_z,m_h,A,B):
    return (G_Vff(m_z, m_d, m_h)+0.5*G_Hww(m_z,m_h))*Hv(A,B)**2
def Hhh(m_H,m_h,A,B):
    coeff = np.cos(2*A)*np.cos(A+B)-2*np.sin(2*A)*np.sin(B+A)
    if m_H>=2*m_h:
        f = np.sqrt(1-4*(m_h/m_H)**2)
    else:
        f = 0
    cnt = Gf*((m_z)**4)/(16*np.sqrt(2)*np.pi*m_H)
    return cnt*f*(coeff)**2
def HAA(m_H,m_h,A,B):
    coeff = -np.cos(2*B)*np.cos(A+B)
    if m_H>=2*m_h:
        f = np.sqrt(1-4*(m_h/m_H)**2)
    else:
        f = 0
    cnt = Gf*((m_z)**4)/(16*np.sqrt(2)*np.pi*m_H)
    return cnt*f*(coeff)**2
def G_HHgg(m_h,A,B,x):
    mh=m_h
    k= (Gf/(36.*np.sqrt(2)*np.pi**3))
    gs = (12.*np.pi)/(21*np.log((mh**2)/0.004))
    cnt = 1*k*(gs**2)*mh**3
    E = (95./4 - 7.)*gs/np.pi
    m_d=m_dd
    m_u=m_uu
    m_s=m_ss
    m_t=qmass(m_tt,m_h)
    m_c=qmass(m_cc,m_h)
    m_b=qmass(m_bb,m_h)
    t = (mh**2)/(4*m_t**2)
    b = (mh**2)/(4*m_b**2)
    c = (mh**2)/(4*m_c**2)
    s = (mh**2)/(4*m_s**2)
    u = (mh**2)/(4*m_u**2)
    d = (mh**2)/(4*m_d**2)
    S = (Af(t)+Af(u)+Af(c))*Fcp('u',x,A,B) + (Af(b)+Af(s)+Af(d))*Fcp('d',x,A,B)
    return((1+E)*cnt*abs(0.75*S)**2)
def G_HHpp(mh,A,B,x,m_pm):
    gs= (12.*np.pi)/(21*np.log((mh**2)/0.004))
    a= 1./127.5
    k= 2*(Gf*(a**2)/(128.*np.sqrt(2)*np.pi**3))
    q1=Fcp('u',x,A,B)*4./9.
    q2=Fcp('d',x,A,B)*1./9.
    m_d=m_dd
    m_u=m_uu
    m_s=m_ss
    m_t=qmass(m_tt,mh)
    m_c=qmass(m_cc,mh)
    m_b=qmass(m_bb,mh)    
    cnt = k*mh**3
    t = (mh**2)/(4*m_t**2)
    b = (mh**2)/(4*m_b**2)
    c = (mh**2)/(4*m_c**2)
    s = (mh**2)/(4*m_s**2)
    u = (mh**2)/(4*m_u**2)
    d = (mh**2)/(4*m_d**2)
    w = (mh**2)/(4*m_w**2)
    e = (mh**2)/(4*m_e**2)
    mu = (mh**2)/(4*m_mu**2)
    tau = (mh**2)/(4*m_tau**2)
    pm = (mh**2)/(4*m_pm**2)
    S = 3*(Af(t)*q1+Af(b)*q2+Af(c)*q1+Af(s)*q2+Af(u)*q1+Af(d)*q2)*(1.-gs/np.pi)+(Af(e)+Af(mu)+Af(tau))*Fcp('l',x,A,B)+Av(w)*Hv(A,B)+As(pm)*Hpm(A,B)*(m_w/m_pm)**2
    return(cnt*abs(S)**2)
def G_HHzp(m_h,A,B,x,m_pm):
    Gs = coupling(m_h)
    a= 1./127.5
    m_d=m_dd
    m_u=m_uu
    m_s=m_ss
    m_t=qmass(m_tt,m_h)
    m_c=qmass(m_cc,m_h)
    m_b=qmass(m_bb,m_h)    
    const = 2*(a*(m_w**2)*(Gf**2)*(m_h**3)*(1-(m_z/m_h)**2)**3)/(64*np.pi**4)
    e1 = 4*(m_e/m_h)**2
    e2 = 4*(m_e/m_z)**2
    mu1 = 4*(m_mu/m_h)**2
    mu2 = 4*(m_mu/m_z)**2
    tau1 = 4*(m_tau/m_h)**2
    tau2 = 4*(m_tau/m_z)**2
    u1 = 4*(m_u/m_h)**2
    u2 = 4*(m_u/m_z)**2
    d1 = 4*(m_d/m_h)**2
    d2 = 4*(m_d/m_z)**2
    b1 = 4*(m_b/m_h)**2
    b2 = 4*(m_b/m_z)**2
    t1 = 4*(m_t/m_h)**2
    t2 = 4*(m_t/m_z)**2
    c1 = 4*(m_c/m_h)**2
    c2 = 4*(m_c/m_z)**2
    s1 = 4*(m_s/m_h)**2
    s2 = 4*(m_s/m_z)**2
    w1 = 4*(m_w/m_h)**2
    w2 = 4*(m_w/m_z)**2
    pm1 = 4*(m_pm/m_h)**2
    pm2 = 4*(m_pm/m_z)**2
    S = A2v(w1,w2)*Hv(A,B)+(3./(sw*cw))*(0.5-(4./3.)*sw**2)*(A2f(u1,u2)+A2f(c1,c2)+A2f(t1,t2))*(1-Gs/np.pi)*Fcp('u',x,A,B)+(0.5/(sw*cw))*(-0.5+(2./3.)*(sw**2))*(A2f(d1,d2)+A2f(s1,s2)+A2f(b1,b2))*(1-Gs/np.pi)*Fcp('d',x,A,B) + (0.25/(sw*cw))*(-0.5 + 2*sw**2)*(A2f(e1,e2)+A2f(mu1,mu2)+A2f(tau1,tau2))*Fcp('l',x,A,B) + ((1-2*sw**2)/(sw*cw))*I1(pm1,pm2)*Hpm(A,B)*(m_w/m_pm)**2
    return const*abs(S)**2
def GammaH(m_h,A,B,x,m_pm):
    return Hhh(m_h,125.09,A,B)+HAA(m_h,m_A,A,B)+Hup(m_uu, m_h, A, B,x)+Hup(m_tt, m_h, A, B,x)+Hup(m_cc, m_h, A, B,x)+Hdown(m_dd, m_h, A, B,x)+Hdown(m_bb, m_h, A, B,x)+Hdown(m_ss, m_h, A, B,x)+Hlept(m_e, m_h, A, B,x)+Hlept(m_mu, m_h, A, B,x)+Hlept(m_tau, m_h, A, B,x)+HW(m_w,m_h,A,B)+HZ(m_z,m_h,A,B)+G_HHgg(m_h,A,B,x)+G_HHpp(m_h,A,B,x,m_pm)+G_HHzp(m_h, A, B,x,m_pm)
print(GammaH(800,0.1,0.5,1,800))
#CP odd Higgs

#1. A->ff~/qq~
def G_Aff(mf,mH):
	G = np.where(np.greater(mH,2*mf),(Gf*mH*(mf**2.)*(1.+(4.*(mf**2)/(mH**2)))*(1.-(4.*(mf**2)/(mH**2)))**0.5)/(4.*np.sqrt(2)*np.pi),0)
	return G


def Af2(x):
    return -2*f(x)/x
    
def Acp(stg,x,B):
    if(x==1):
    	if(stg == 'u'):
    		return 1./np.tan(B)
    	else:
    		return -1./np.tan(B)
    elif(x==2):
        if(stg == 'u'):
        	return 1./np.tan(B)
        else:
    	    	return np.tan(B)
    elif(x==3):
        if(stg=='d'):
            return -1./np.tan(B)
        elif(stg == 'u'):
        	return 1./np.tan(B)
        else:
            return np.tan(B)
    elif(x==4):
        if(stg=='l'):
            return -1./np.tan(B)
        elif(stg == 'u'):
        	return 1./np.tan(B)
        else:
            return np.tan(B)
    else:
    	print('Invalid value')
    	return 0
            
dqqA = lambda a,mq,mh : 1+5.67*a/np.pi +(35.94+3.83-1.36*5-2*np.log(mh/m_t)+(1./6.)*(2*np.log(mq/mh)**2))*(a/np.pi)**2 +(164.14-25.77*5+.26*25)*(a/np.pi)**3            
def Aup(m_u,m_H,B,x):
        if(m_u==m_tt or m_u==m_cc):
            mu=qmass(m_u,m_H)
        else:
            mu=m_u
        Gs = coupling(m_H)
        return 3*G_Aff(mu,m_H)*(dqqA(Gs,mu,m_H))*Acp('u',x,B)**2
def Adown(m_d,m_H,B,x):
        if(m_d==m_bb):
            md=qmass(m_d,m_H)
        else:
            md=m_d
        Gs = coupling(m_H)
        return 3*G_Aff(md,m_H)*(dqqA(Gs,md,m_H))*Acp('d',x,B)**2
def Alept(m_e,m_H,B,x):
        return G_Aff(m_e,m_H)*Acp('l',x,B)**2
def AZh(m_A,m_z,m_h,A,B):
    coef = np.cos(B-A)**2
    if(m_A>=m_z+m_h):
        cnt = (Gf*m_w**2)/(8*np.sqrt(2)*np.pi*m_z*m_z*m_A*m_A*m_A*cw**2)
    else:
        cnt=0
    lam = (m_A**2 - m_h**2 + m_z**2)**2-4*(m_A*m_z)**2
    return cnt*coef*lam**1.5
def AZH(m_A,m_z,m_h,A,B):
    coef = np.sin(B-A)**2
    if(m_A>=m_z+m_h):
        cnt = (Gf*m_w**2)/(8*np.sqrt(2)*np.pi*m_z*m_z*m_A*m_A*m_A*cw**2)
    else:
        cnt=0
    lam = (m_A**2 - m_h**2 + m_z**2)**2-4*(m_A*m_z)**2
    return cnt*coef*lam**1.5

def G_Agg(mh,B,x):
        k= (Gf/(36.*np.sqrt(2)*np.pi**3))
        gs = (12.*np.pi)/(21*np.log((mh**2)/0.004))
        cnt = 1*k*(gs**2)*mh**3
        E = (95./4 - 7.)*gs/np.pi
        m_d=m_dd
        m_u=m_uu
        m_s=m_ss
        m_t=qmass(m_tt,mh)
        m_c=qmass(m_cc,mh)
        m_b=qmass(m_bb,mh) 
        t = (mh**2)/(4*m_t**2)
        b = (mh**2)/(4*m_b**2)
        c = (mh**2)/(4*m_c**2)
        s = (mh**2)/(4*m_s**2)
        u = (mh**2)/(4*m_u**2)
        d = (mh**2)/(4*m_d**2)
        S = (Af2(t)+Af2(c)+Af2(u))*Acp('u',x,B)+(Af2(b)+Af2(s)+Af2(d))*Acp('d',x,B)
        return((1+E)*cnt*abs(0.75*S)**2)
def G_App(mh,B,x):
        gs = (12.*np.pi)/(21*np.log((mh**2)/0.004))
        a= 1./128.5
        k= 2*(Gf*(a**2)/(128.*np.sqrt(2)*np.pi**3))
        q1=Acp('u',x,B)*4./9.
        q2=Acp('d',x,B)*1./9.
        m_d=m_dd
        m_u=m_uu
        m_s=m_ss
        m_t=qmass(m_tt,mh)
        m_c=qmass(m_cc,mh)
        m_b=qmass(m_bb,mh)        
        cnt = k*mh**3
        t = (mh**2)/(4*m_t**2)
        b = (mh**2)/(4*m_b**2)
        c = (mh**2)/(4*m_c**2)
        s = (mh**2)/(4*m_s**2)
        u = (mh**2)/(4*m_u**2)
        d = (mh**2)/(4*m_d**2)
        w = (mh**2)/(4*m_w**2)
        e = (mh**2)/(4*m_e**2)
        mu = (mh**2)/(4*m_mu**2)
        tau = (mh**2)/(4*m_tau**2)
        S = 3*(Af2(t)*q1+Af2(b)*q2+Af2(c)*q1+Af2(s)*q2+Af2(u)*q1+Af2(d)*q2)*(1-gs/np.pi)+(Af2(e)+Af2(mu)+Af2(tau))*Acp('l',x,B)
        return(cnt*abs(S)**2)
def G_Azp(m_h,B,x):
        gs = (12.*np.pi)/(21*np.log((m_h**2)/0.004))
        a= 1./127.5
        m_d=m_dd
        m_u=m_uu
        m_s=m_ss
        m_t=qmass(m_tt,m_h)
        m_c=qmass(m_cc,m_h)
        m_b=qmass(m_bb,m_h)        
        const = 2*(a*(m_w**2)*(Gf**2)*(m_h**3)*(1-(m_z/m_h)**2)**3)/(64*np.pi**4)
        e1 = 4*(m_e/m_h)**2
        e2 = 4*(m_e/m_z)**2
        mu1 = 4*(m_mu/m_h)**2
        mu2 = 4*(m_mu/m_z)**2
        tau1 = 4*(m_tau/m_h)**2
        tau2 = 4*(m_tau/m_z)**2
        u1 = 4*(m_u/m_h)**2
        u2 = 4*(m_u/m_z)**2
        d1 = 4*(m_d/m_h)**2
        d2 = 4*(m_d/m_z)**2
        b1 = 4*(m_b/m_h)**2
        b2 = 4*(m_b/m_z)**2
        t1 = 4*(m_t/m_h)**2
        t2 = 4*(m_t/m_z)**2
        c1 = 4*(m_c/m_h)**2
        c2 = 4*(m_c/m_z)**2
        s1 = 4*(m_s/m_h)**2
        s2 = 4*(m_s/m_z)**2
        
        S = (3./(sw*cw))*(0.5-(4./3.)*sw**2)*(I2(u1,u2)+I2(c1,c2)+I2(t1,t2))*(1-gs/np.pi)*Acp('u',x,B)+(0.5/(sw*cw))*(-0.5+(2./3.)*(sw**2))*(I2(d1,d2)+I2(s1,s2)+I2(b1,b2))*(1-gs/np.pi)*Acp('d',x,B)+ (0.25/(sw*cw))*(-0.5 + 2*sw**2)*(I2(e1,e2)+I2(mu1,mu2)+I2(tau1,tau2))*Acp('l',x,B)
        return const*abs(S)**2
def GammaA(m_A,A,B,x):
        return AZh(m_A,m_z,125.09,A,B)+AZH(m_A,m_z,m_H,A,B)+Aup(m_uu, m_A, B,x)+Aup(m_tt, m_A, B,x)+ Aup(m_cc, m_A, B,x)+Adown(m_dd, m_A, B,x)+Adown(m_bb, m_A, B,x)+Adown(m_ss, m_A, B,x)+Alept(m_e, m_A, B,x)+Alept(m_mu, m_A, B,x)+Alept(m_tau, m_A, B,x)+G_Agg(m_A,B,x)+G_App(m_A,B,x)+G_Azp(m_A, B,x)

# In[89]:

m_H=int(input("mass of Heavy scalar Higgs(H) in GeV: "))
m_A=int(input("mass of pseudoscalar Higgs(A) in GeV: "))
h = pred.addParticle(HP.BsmParticle("h", "neutral", "even"))
h.setMass(125.09)

H = pred.addParticle(HP.BsmParticle("H", "neutral", "even"))
H.setMass(m_H)

A = pred.addParticle(HP.BsmParticle("A", "neutral", "odd"))
A.setMass(m_A)

X = pred.addParticle(HP.BsmParticle("X", "single"))
X.setMass(800)


# In[90]:


cs = np.cos
ss = np.sin

# run HB and HS for one parameter point pt with couplings cpl
def run_higgstools(cpl, pt):
    set_h_properties(cpl[0], pt)
    set_H_properties(cpl[1], pt)
    set_A_properties(cpl[2], pt)
    #set_X_properties(pt)
    res = bounds(pred)
    chisq = signals(pred)
    return res, chisq

# set properties of the h boson
def set_h_properties(dc, pt):
    cpls = HP.NeutralEffectiveCouplings()
    cpls.tt = dc['tt']
    cpls.bb = dc['bb']
    cpls.ZZ = dc['ZZ']
    cpls.WW = dc['WW']
    HP.effectiveCouplingInput(
        h,
        cpls,
        reference=HP.ReferenceModel.SMHiggsEW)
    w = pt['Wh']
    h.setDecayWidth('gg', pt['BRh2gg'] * w)
    h.setDecayWidth('bb', pt['BRh2bb'] * w)
    h.setDecayWidth('tautau', pt['BRh2ll'] * w)
    h.setDecayWidth('cc', pt['BRh2cc'] * w)
    h.setDecayWidth('ss', pt['BRh2ss'] * w)
    h.setDecayWidth('mumu', pt['BRh2mm'] * w)
    h.setDecayWidth('gamgam', pt['BRh2yy'] * w)
    h.setDecayWidth('Zgam', pt['BRh2Zy'] * w)
    h.setDecayWidth('WW', pt['BRh2WW'] * w)
    h.setDecayWidth('ZZ', pt['BRh2ZZ'] * w)
    #print(h.totalWidth(), w)
    if abs(h.totalWidth() - w) > 1e-3:
        raise DecayError("Missing decay channel for particle h.")

# set properties of the H boson
def set_H_properties(dc, pt):
    cpls = HP.NeutralEffectiveCouplings()
    cpls.tt = dc['tt']
    cpls.bb = dc['bb']
    cpls.ZZ = dc['ZZ']
    cpls.WW = dc['WW']
    HP.effectiveCouplingInput(H,cpls,reference=HP.ReferenceModel.SMHiggs)
    # H.setCxn('LHC13', 'ggH', pt['XSHgg'])
    # H.setCxn('LHC13', 'bbH', pt['XSHbb'])
    w = pt['WH']
    H.setDecayWidth('gg', pt['BRH2gg'] * w)
    H.setDecayWidth('WW', pt['BRH2WW'] * w)
    H.setDecayWidth('ZZ', pt['BRH2ZZ'] * w)
    H.setDecayWidth('gamgam', pt['BRH2yy'] * w)
    H.setDecayWidth('tt', pt['BRH2tt'] * w)
    H.setDecayWidth('bb', pt['BRH2bb'] * w)
    H.setDecayWidth('cc', pt['BRH2cc'] * w)
    H.setDecayWidth('ss', pt['BRH2ss'] * w)
    H.setDecayWidth('tautau', pt['BRH2ll'] * w)
    H.setDecayWidth('h', 'h', pt['BRH2hh'] * w)
    H.setDecayWidth('A', 'A', pt['BRH2AA'] * w)
    if abs(H.totalWidth() - w) > 1e-1:
        print(H.totalWidth())
        print(w)
        raise DecayError("Missing decay channel for particle H.")

# set properties of the A boson
def set_A_properties(dc, pt):
    cpls = HP.NeutralEffectiveCouplings()
    cpls.tt = 1j*dc['tt']
    cpls.bb = 1j*dc['bb']
    HP.effectiveCouplingInput(
        A,
        cpls,
        reference=HP.ReferenceModel.SMHiggs)
    # A.setCxn('LHC13', 'ggH', pt['XSAgg'])
    # A.setCxn('LHC13', 'bbH', pt['XSAbb'])
    w = pt['WA']
    A.setDecayWidth('gg', pt['BRA2gg'] * w)
    A.setDecayWidth('gamgam', pt['BRA2yy'] * w)
    A.setDecayWidth('tt', pt['BRA2tt'] * w)
    A.setDecayWidth('bb', pt['BRA2bb'] * w)
    A.setDecayWidth('cc', pt['BRA2cc'] * w)
    A.setDecayWidth('ss', pt['BRA2ss'] * w)
    A.setDecayWidth('tautau', pt['BRA2ll'] * w)
    A.setDecayWidth('Z', 'h', pt['BRA2Zh'] * w)
    A.setDecayWidth('Z', 'H', pt['BRA2ZH'] * w)
    if abs(A.totalWidth() - w) > .15:
        print(A.totalWidth())
        print(w)
        raise DecayError("Missing decay channel for particle A.")
# read ScannerS output file
def read_scanners_output(x):
    a = np.linspace(-np.pi/2,np.pi/2,200)
    a = np.tile(a,200)
    tb = np.linspace(.1,10,200)
    tb= np.repeat(tb,200)
    b=np.arctan(tb)
    G = Gamma(125.09,a,b,x,800)
    print(G)
    hbb = hdown(m_b,125.09,a,b,x)/G
    htau= hlept(m_tau,125.09,a,b,x)/G
    hmu=hlept(m_mu,125.09,a,b,x)/G
    hcc=hup(m_c,125.09,a,b,x)/G
    hss=hdown(m_s,125.09,a,b,x)/G
    hzz=hZ(m_z,125.09,a,b)/G
    hww=hW(m_w,125.09,a,b)/G
    hgg=G_Hgg(125.09,a,b,x)/G
    hpp=G_Hpp(125.09,a,b,x,800)/G
    hzp=G_Hzp(125.09,a,b,x,800)/G
    GH = GammaH(m_H,a,b,x,800)
    print(GH)
    Hbb = Hdown(m_bb,m_H,a,b,x)/GH
    Htau= Hlept(m_tau,m_H,a,b,x)/GH
    Hmu=Hlept(m_mu,m_H,a,b,x)/GH
    Hcc=Hup(m_cc,m_H,a,b,x)/GH
    Htt=Hup(m_tt,m_H,a,b,x)/GH
    Hss=Hdown(m_ss,m_H,a,b,x)/GH
    Hzz=HZ(m_z,m_H,a,b)/GH
    Hww=HW(m_w,m_H,a,b)/GH
    Hgg=G_HHgg(m_H,a,b,x)/GH
    Hpp=G_HHpp(m_H,a,b,x,800)/GH
    Hzp=G_HHzp(m_H,a,b,x,800)/GH
    H2hh = Hhh(m_H,125.09,a,b)/GH
    H2AA = HAA(m_H,m_A,a,b)/GH
    GA = GammaA(m_A,a,b,x)
    print(GA)
    Abb = Adown(m_bb,m_A,b,x)/GA
    Atau= Alept(m_tau,m_A,b,x)/GA
    Ahmu=Alept(m_mu,m_A,b,x)/GA
    Acc=Aup(m_cc,m_A,b,x)/GA
    Att=Aup(m_tt,m_A,b,x)/GA
    Ass=Adown(m_ss,m_A,b,x)/GA
    Agg=G_Agg(m_A,b,x)/GA
    App=G_App(m_A,b,x)/GA
    Azp=G_Azp(m_A,b,x)/GA
    Azh = AZh(m_A,m_z,125.09,a,b)/GA
    AzH=AZH(m_A,m_z,m_H,a,b)/GA
    dcs = []
    for i in range(len(tb)):
        dc = {}
        dc['tb'] = tb[i]
        dc['al'] = a[i]
        dc['BRh2bb'] = np.abs(hbb[i])
        dc['BRh2ll'] = np.abs(htau[i])
        dc['BRh2cc'] = np.abs(hcc[i])
        dc['BRh2ss'] = np.abs(hss[i])
        dc['BRh2mm'] = np.abs(hmu[i])
        dc['BRh2ZZ'] = np.abs(hzz[i])
        dc['BRh2WW'] = np.abs(hww[i])
        dc['BRh2gg'] = np.abs(hgg[i])
        dc['BRh2yy'] = np.abs(hpp[i])
        dc['BRh2Zy'] = np.abs(hzp[i])
        dc['Wh'] = np.abs(G[i])
        dc['BRA2gg'] = np.abs(Agg[i])
        dc['BRA2yy'] = np.abs(App[i])
        dc['BRA2tt'] = np.abs(Att[i])
        dc['BRA2bb'] = np.abs(Abb[i])
        dc['BRA2cc'] = np.abs(Acc[i])
        dc['BRA2ss'] = np.abs(Ass[i])
        dc['BRA2ll'] = np.abs(Atau[i])
        dc['BRA2Zh'] = np.abs(Azh[i])
        dc['BRA2ZH'] = np.abs(AzH[i])
        dc['BRH2hh'] = np.abs(H2hh[i])
        dc['BRH2AA'] = np.abs(H2AA[i])
        dc['BRH2WW'] = np.abs(Hww[i])
        dc['BRH2ZZ'] = np.abs(Hzz[i])
        dc['BRH2bb'] = np.abs(Hbb[i])
        dc['BRH2cc'] = np.abs(Hcc[i])
        dc['BRH2ss'] = np.abs(Hcc[i])
        dc['BRH2yy'] = np.abs(Hpp[i])
        dc['BRH2ll'] = np.abs(Htau[i])
        dc['BRH2tt'] = np.abs(Htt[i])
        dc['BRH2gg'] = np.abs(Hgg[i])
        dc['WH'] = np.abs(GH[i])
        dc['WA'] = np.abs(GA[i])
        dcs.append(dc)
    dt = pd.DataFrame(dcs)
    dt.to_csv(f'new_fr_calcdata_ytype{x}.csv')
    return dcs

# calculate effective couplings
def calc_effective_couplings(al, be, yuktype):
    if yuktype == 1:
        dcs = calc_effective_cpls_type1(al, be)
    elif yuktype == 2:
        dcs = calc_effective_cpls_type2(al, be)
    elif yuktype == 3:
        dcs = calc_effective_cpls_type3(al, be)
    elif yuktype == 4:
        dcs = calc_effective_cpls_type4(al, be)
    else:
        raise RuntimeError
    return dcs

# calculate effective couplings in type 1
def calc_effective_cpls_type1(a, b):
    uu = cs(a) / ss(b)
    dd = uu
    vv = ss(b - a)
    cplh = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = ss(a) / ss(b)
    dd = uu
    vv = cs(b - a)
    cplH = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = 1 / np.tan(b)
    dd = -1 / np.tan(b)
    cplA = {
        'tt': uu,
        'bb': dd}
    return [cplh, cplH, cplA]

# calculate effective couplings in type 2
def calc_effective_cpls_type2(a, b):
    uu = cs(a) / ss(b)
    dd = -ss(a) / cs(b)
    vv = ss(b - a)
    cplh = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = ss(a) / ss(b)
    dd = cs(a) / cs(b)
    vv = cs(b - a)
    cplH = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = 1 / np.tan(b)
    dd = np.tan(b)
    cplA = {
        'tt': uu,
        'bb': dd}
    return [cplh, cplH, cplA]

# calculate effective couplings in type 3
def calc_effective_cpls_type3(a, b):
    uu = cs(a) / ss(b)
    dd = uu
    vv = ss(b - a)
    cplh = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = ss(a) / ss(b)
    dd = uu
    vv = cs(b - a)
    cplH = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = 1 / np.tan(b)
    dd = -1 / np.tan(b)
    cplA = {
        'tt': uu,
        'bb': dd}
    return [cplh, cplH, cplA]

# calculate effective couplings in type 4
def calc_effective_cpls_type4(a, b):
    uu = cs(a) / ss(b)
    dd = -ss(a) / cs(b)
    vv = ss(b - a)
    cplh = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = ss(a) / ss(b)
    dd = cs(a) / cs(b)
    vv = cs(b - a)
    cplH = {
        'tt': uu,
        'bb': dd,
        'ZZ': vv,
        'WW': vv}
    uu = 1 / np.tan(b)
    dd = np.tan(b)
    cplA = {
        'tt': uu,
        'bb': dd}
    return [cplh, cplH, cplA]

# process dataset and save output to file
def process_data(dataset, yuktype):
    data = []
    for point in tqdm(dataset):
        tb = point['tb']
        al = point['al'] #important 
        be = np.arctan(tb)
        cba = cs(be - al)
        cpl = calc_effective_couplings(al, be, yuktype)
        reshb, Chisq = run_higgstools(cpl, point)
        try:
            data.append({
            'be': tb,
            'al': al,
            'chisq': Chisq,
            'cba':cba,
            #'hexp': reshb.selectedLimits['h'].expRatio(),
            'hobs': reshb.selectedLimits['h'].obsRatio(),
            'Hobs': reshb.selectedLimits['H'].obsRatio(),    
            #'hcha': reshb.selectedLimits['h'].limit().citeKey(),
            #'aexp': reshb.selectedLimits['A'].expRatio(),
            'aobs': reshb.selectedLimits['A'].obsRatio(),
            #'acha': reshb.selectedLimits['A'].limit().citeKey(),
            #'xexp': reshb.selectedLimits['X'].expRatio(),
            #'xobs': reshb.selectedLimits['X'].obsRatio(),
            #'xcha': reshb.selectedLimits['X'].limit().citeKey()
        })
        except KeyError:
            data.append({
            'be': tb,
            'al': al,
            'chisq': Chisq,
            #'hexp': reshb.selectedLimits['h'].expRatio(),
            #'hobs': reshb.selectedLimits['h'].obsRatio(),
            #'hcha': reshb.selectedLimits['h'].limit().citeKey(),
            #'aexp': reshb.selectedLimits['A'].expRatio(),
            #'aobs': reshb.selectedLimits['A'].obsRatio(),
            #'acha': reshb.selectedLimits['A'].limit().citeKey(),
            #'xexp': reshb.selectedLimits['X'].expRatio(),
            #'xobs': reshb.selectedLimits['X'].obsRatio(),
            #'xcha': reshb.selectedLimits['X'].limit().citeKey()
        })
    df = pd.DataFrame(data)
    df.to_csv(f'new_result_type{yuktype}_H{m_H}A{m_A}.csv')


# In[91]:


type1_dataset = read_scanners_output(1)
type2_dataset = read_scanners_output(2)
type3_dataset = read_scanners_output(3)
type4_dataset = read_scanners_output(4)


# In[92]:


process_data(type1_dataset, 1)
process_data(type2_dataset, 2)
process_data(type4_dataset, 4)
process_data(type3_dataset, 3)


# In[93]:


from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D


# In[94]:


def plot_2HDM_bounds(yuktype):
    df  = pd.read_csv(f'new_result_type{yuktype}_H{m_H}A{m_A}.csv')

    x = np.array(df['chisq'].tolist())
    m = np.min(x)
    x = x - m
    y = [z if z < 6.001 else 6.001 for z in x]
    df['col'] = y

    x1 = []
    x2 = []
    for i, r in df.iterrows():
        if (r['Hobs'] < 1) and (r['aobs'] < 1):
            x1.append(0)
        else:
            x1.append(1)
    #for i, r in df2.iterrows():
        #if (r['hb_Hh_obsratio'] < 1) and (r['hb_A_obsratio'] < 1):#and (r['hb_Hp_obsratio'] < 1):
            #x2.append(0)
        #else:
            #x2.append(0.1)
    df['col2'] = x1
    #df['col3'] = x2

    xscale = np.max(df['cba']) - np.min(df['cba'])
    yscale = np.max(df['be']) - np.min(df['be'])
    scale = np.array([xscale, yscale])

    mat = np.array([
        df['cba'].to_numpy(),
        df['be'].to_numpy(),
        df['col'].to_numpy()]).T
    x = np.linspace(-.5, .5, 1000)
    y = np.linspace(1, 10, 1000)
    X, Y = np.meshgrid(x, y)
    Z = interpolate.griddata(
        mat[:,0:2] / scale,
        mat[:,2],
        (X / xscale, Y / yscale),
        method='nearest')

    fig, ax = plt.subplots(
        figsize=(3.4, 3.4),
        constrained_layout=True)

    sc = ax.scatter(
        df['cba'],
        df['be'],
        c=df['col'],
        s=0.4,
        cmap='GnBu_r',
        rasterized=True)

    ax.set_xlim(-.5, .5)
    ax.set_ylim(1, 10)

    clb = fig.colorbar(
        sc,
        ax=ax,
        label=r"$\Delta\chi^2$",
        ticks=np.arange(0, 6.1, 1),
        pad=0,
        fraction=0.1,
        location='top',
        aspect=40
    )
    clb.ax.minorticks_on()

    ax.contour(
        X,
        Y,
        Z,
        levels=[2.3, 5.99],
        colors="black",
        linestyles=["-", "--"],
    )

    bfp = np.unravel_index(
        np.argmin(Z), Z.shape)
    ax.plot(
        x[bfp[1]], y[bfp[0]],
        marker="*", ls="none", c="r")

    sc2 = ax.scatter(
        df['cba'],
        df['be'],
        c='lightgray',
        s=df['col2'],
        alpha=1.0,
        rasterized=True)

    percent = "\%" if plt.rcParams["text.usetex"] else "%"
    hsLegend = ax.legend(
        handles=[
            Line2D([0], [0], color="k", ls="-", label=f"HS 68{percent} CL"),
            Line2D([0], [0], color="k", ls="--", label=f"HS 95{percent} CL"),
            Line2D([0], [0], color="k", ls="none", marker="*", c="r", label="HS BFP"),
        ],
        loc="upper right",
        frameon=False,
    )
    ax.add_artist(hsLegend)

    majorLocator = MultipleLocator(1)
    minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(
        axis="y",
        direction="in",
        which="both",
        right=True)

    majorLocator = MultipleLocator(0.2)
    minorLocator = MultipleLocator(0.02)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(
        axis="x",
        direction="in",
        which="both",
        top=True)

    ax.set_xlabel(r'cos$\beta-\alpha$')
    ax.set_ylabel(r'tan$\beta$')

    ax.text(
        0.97,
        0.03,
        f'2HDM Type {yuktype}',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,
        )

    plt.savefig(f'new_results_type{yuktype}_H{m_H}A{m_A}.pdf')


# In[95]:


plot_2HDM_bounds(1)


# In[96]:


plot_2HDM_bounds(2)


# In[97]:


plot_2HDM_bounds(3)


# In[98]:


plot_2HDM_bounds(4)
