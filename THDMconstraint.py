import numpy as np
import pandas as pd
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
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D


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
    return (x-f(x))/x**2
def Af(x):
    return -(2./x**2)*(x+(x-1.)*f(x))
def Av(x):
    return (2.*x**2 + 3.*x +3.*(2*x-1.)*f(x))/x**2


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
    return 3*G_Hff(m_d,m_h)*(dqq(gs,m_u,125.09))*fcp('d',x,A,B)**2
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


print("\nlight Higgs\n")
a = int(input("Enter model type: "))
m_H = 800
m_A = 800
df  = pd.read_csv(f'datafiles/new_result_type{a}_H{m_H}A{m_A}.csv')
tb = df['be'].to_numpy()
step = tb[250]-tb[1]
chi = df['chisq'].to_numpy()
m = np.min(chi)
findmin=np.where(chi==m)[0][0]
hb = df['Hobs'].to_numpy()
ab = df['aobs'].to_numpy()
hba = np.where((hb<1) & (ab<1),0,1)
print(hba)
m_h = 125.07
m_pm =300

A=np.linspace(-np.pi/2,np.pi/2,1000)
A = df['al'].to_numpy()
y=np.where(np.around(tb,3)==np.around(tb[findmin],3))[0]
A = A[y]
tb_in = np.arctan(tb[findmin])
yax = np.linspace(0,1,200)
X,Y = np.meshgrid(A,yax)
# Create the figure and the line that we will manipulate
fig,ax= plt.subplots()
line3, = ax.semilogy(A, G_Hpp(m_h,A,tb_in,a,m_pm)/Gamma(m_h,A,tb_in,a,m_pm),label=r'h->$\gamma$ $\gamma$')
line4, = ax.semilogy(A,G_Hzp(m_h,A,tb_in,a,m_pm)/Gamma(m_h,A,tb_in,a,m_pm),label=r"h->Z$\gamma$")
xxx = chi[y]-m
#print(xxx)
yyy = [i if i<6.001 else 6.001 for i in xxx]
z = np.array([yyy[i] for j in yax for i in range(0,200)])
z2 = hba[y]
z2 = np.array([z2[i] for j in yax for i in range(0,200)])
Z=z.reshape(200,200)
ZZ=Z
Z2=z2.reshape(200,200)
cmin=0
cmax=6
levelss = np.linspace(cmin,cmax,100)
s=ax.contourf(X, Y, Z,levels =levelss,vmax = cmax,vmin=cmin)
bou=ax.fill_between(A,0,1,where=hba[y]==1, facecolor='lightgray')

fig.colorbar(s,label=r"$\Delta\chi^2$",ticks=np.arange(0, 6.1, 1),pad=0,fraction=0.1,location='right',aspect=40)
ax.set_title("Light Higgs decay \n 2HDM Type "+str(a))
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Branching ratio')
ax.set_ylim(1e-5,1)
percent = "\%" if plt.rcParams["text.usetex"] else "%"
ax.legend(handles=[
	    line3,
            line4,
            Line2D([0], [0], color="k", ls="-", label=f"HS 68{percent} CL"),
            Line2D([0], [0], color="k", ls="--", label=f"HS 95{percent} CL"),
            Line2D([0], [0], color="k", ls="-", c="r", label="HS BFP"),
            ],
        loc="upper right",
        frameon=False,
         )
text = ax.text(.5,.8,r"tan$\beta$="+str(np.around(np.tan(tb_in),3)))
text2=ax.text(1.2,.08,r"H mass="+str(m_H))
text2=ax.text(1.2,.06,r"A mass="+str(m_A))
#ax.legend()
# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
b_slider = Slider(
    ax=axfreq,
    label=r'tan($\beta$)',
    valmin=0.1,
    valmax=10,
    valinit=np.tan(tb_in),
    valstep=step
)


axmpm= fig.add_axes([0.1, 0.25, 0.0225, 0.63])
mpm_slider = Slider(
    ax=axmpm,
    label=r'charged Higgs mass(GeV)',
    valmin=50,
    valmax=500,
    valinit=m_pm,
    orientation="vertical"
)
test=ax.contour(X,Y,ZZ,levels=[0,2.3, 5.99],colors=["red","black","black"],linestyles=["-","-","--"],)
# The function to be called anytime a slider's value changes
def update(val):
    fig.canvas.update()
    global test, s, text, bou
    new = np.around(b_slider.val,3)
    #print(new)
    y=np.where(np.around(tb,3)==new)[0]
    #print(y)
    xxx = chi[y]-m
    #print(xxx)
    yyy = [i if i<6.001 else 6.001 for i in xxx]   
    z = np.array([yyy[i] for j in yax for i in range(0,200)])
    z2 = hba[y]
    z2 = np.array([z2[i] for j in yax for i in range(0,200)])
    Z=z.reshape(200,200)
    Z2=z2.reshape(200,200)
    for csc in s.collections:
    	csc.remove()
    bou.remove()
    s=ax.contourf(X, Y, Z,levels =levelss,vmax = cmax,vmin=cmin)
    bou=ax.fill_between(A,0,1,where=hba[y]==1, facecolor='lightgray')
    #bou=ax.contourf(X, Y, Z2,levels=[0,1],colors=['yellow','lightgrey'])
    ZZ=Z
    lvl = [0,2.3, 5.99]
    for csc in test.collections:
    	csc.remove()
    test=ax.contour(X,Y,ZZ,levels=lvl,colors=["red","black","black"],linestyles=["-","-","--"],)
    line3.set_ydata(G_Hpp(m_h,A,np.arctan(b_slider.val),a,mpm_slider.val)/Gamma(m_h, A,np.arctan(b_slider.val),a,mpm_slider.val))
    line4.set_ydata(G_Hzp(m_h,A,np.arctan(b_slider.val),a,mpm_slider.val)/Gamma(m_h, A,np.arctan(b_slider.val),a,mpm_slider.val))
    text.remove()
    text=ax.text(1.2,.00002,r"tan$\beta$="+str(np.around(b_slider.val,3)))
    #del(test)
    fig.canvas.draw_idle()
    #t.clear()
    #ax.cla()

# register the update function with each slider
b_slider.on_changed(update)
mpm_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    b_slider.reset()
    mpm_slider.reset()
button.on_clicked(reset)



#fig.colorbar(s,label=r"$\Delta\chi^2$",ticks=np.arange(0, 6.1, 1),pad=0,fraction=0.1,location='top',aspect=40)

plt.legend()
plt.show()
