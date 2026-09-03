# Rabi in case of chirp absence but doppler shift changing is taken into account
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


def Rabifit(t, w, A, B, tc):
    return -A*np.exp(-(t/tc))*np.cos(w*t) + B


def Rabifit_t(t, w0, A, B, tc, a):
    return -A*np.exp(-(t/tc))*np.cos((w0+a*t)*t)+B


def ph(t, v0, Dw):
    return -keff*(v0 + g*t/2)*t +  Dw*t # sign of g is under question

def equa(t, B, v0, Dw):
    B1, B2 = B[0], B[1]

    dB1 = 1j*W1**2/D * B1 + 1j*W1*W2/D * np.exp(1j*ph(t,v0,Dw)) * B2
    dB2 = 1j*W2**2/D * B2 + 1j*W1*W2/D * np.exp(-1j*ph(t,v0,Dw)) * B1

    return [dB1, dB2]


def Rabi_osc(Dw, v0, B0, t1, t_steps):


    t_eval = np.linspace(0, t1, t_steps)


    sol = solve_ivp(
        equa,
        (0, t1),
        B0,
        args=(v0, Dw,),
        t_eval=t_eval,
        method='DOP853'
    )


    B1 = sol.y[0]
    B2 = sol.y[1]

    P1 = abs(B1)**2
    P2 = abs(B2)**2

    # plt.figure()
    # plt.title("Rabi osc")
    # #plt.plot(t_eval*1e6, P1, label=r'$|B_1|^2$')
    # plt.plot(t_*1e6, P2, label=r'$P_2$')
    # print(f"WGcounted = {np.sqrt((W1**2/D - W2**2/D - (-keff*v0 + Dw))**2 + 4*W1**2*W2**2/D**2)*1e-3} kHz (no g acceleration)")
    # print(f"WGequa = {np.pi/t_[np.argmax(P2)]*1e-3} kHz")
    
    # plt.xlabel("t (µs)")
    # plt.ylabel("Probability")
    # plt.legend()
    # plt.grid()
    return P2


def Rabi_osc_group(v0, TK, Dw, B0, t1, t_steps, N):
    sigma_v = np.sqrt(kb*TK/m_Rb)
    v_range = np.linspace(-4*sigma_v, 4*sigma_v, N)
    dv = v_range[1] - v_range[0]
    P = np.zeros(t_steps)
    
    for v_i in v_range:
        P += Rabi_osc(Dw, v_i, B0, t1, t_steps) * np.exp(-(v_i-v0)**2/(2*sigma_v**2))*dv / (np.sqrt(2*np.pi)*sigma_v)

    plt.figure()
    plt.title("Rabi osc group")

    t_eval = np.linspace(0, t1, t_steps)
    plt.plot(t_eval*1e6, P)

    plt.xlabel("time (µs)")
    plt.ylabel("Amplitude")
    #plt.legend()
    plt.grid()
    print(f"WGequa = {np.pi/t_eval[np.argmax(P)]*1e-3} kHz")
    print(f"WGcounted = {np.sqrt((W1**2/D - W2**2/D - (-keff*v0 + Dw))**2 + 4*W1**2*W2**2/D**2)*1e-3} kHz (no g acceleration)")

    # comparison 
    # np.save("t_eval.npy", t_eval)
    # np.save("Pg-.npy", P)

    # fit
    p0 = [np.pi/t_eval[np.argmax(P)],(np.max(P) - np.min(P))/2, (np.max(P) + np.min(P))/2, 2*t_eval[np.argmax(P)]]
    popt, pcov = curve_fit(Rabifit, t_eval, P, p0=p0, maxfev=10000)
    w, A, B, tc = popt
    plt.plot(t_eval*1e6, Rabifit(t_eval, w, A, B, tc), label="fit $\Omega=const$")
    print(f"Weffconst={w*1e-3} kHz")
    print(f"typconst = {np.pi/w}")

    # fit_t
    tau_peak2 = t_eval[np.argmax(P)]**2*1e12
    p0 = [w, A, B, tc, w/tau_peak2]
    popt, pcov = curve_fit(Rabifit_t, t_eval, P, p0=p0, maxfev=10000)
    w0, A, B, tc, a = popt

    plt.plot(t_eval*1e6, Rabifit_t(t_eval, w0, A, B, tc, a), label="fit $\Omega(t)$")



    plt.figure()

    plt.plot(t_eval*1e6, (w0+a*t_eval)*1e-3)


    return 1

def spectrum(Dw1, Dw2, v0, B0, t1, t_steps, N):

    Dw_range = np.linspace(Dw1, Dw2, N)
    P = []

    t_eval = np.linspace(0, t1, t_steps)

    for Dw_i in Dw_range:

        sol = solve_ivp(
            equa,
            (0, t1),
            B0,
            args=(v0, Dw_i,),
            t_eval=t_eval,
            method='DOP853'
        )
        B1 = sol.y[0]
        B2 = sol.y[1]

        P1 = abs(B1)**2
        P2 = abs(B2)**2
        P.append(P2[-1])

    P = np.array(P)
    # plt.figure()
    # plt.title("Raman spectrum")
    # plt.plot(Dw_range*1e-3, P)

    # plt.xlabel("freq (kHz)")
    # plt.ylabel("Probability")
    # #plt.legend()
    # plt.grid()


    return P

def spectrum_group(v0, T, Dw1, Dw2, B0, t1, t_steps, N):

    sigma_v = np.sqrt(kb*T/m_Rb)
    v_range = np.linspace(-4*sigma_v, 4*sigma_v, N)
    dv = v_range[1] - v_range[0]
    
    Dw_range = np.linspace(Dw1, Dw2, N)
    t_eval = np.linspace(0, t1, t_steps)

    P = np.zeros(N)
    for v_i in v_range:
        P += spectrum(Dw1, Dw2, v_i, B0, t1, t_steps, N) * np.exp(-(v_i-v0)**2/(2*sigma_v**2))*dv / (np.sqrt(2*np.pi)*sigma_v)

        
    plt.figure()
    plt.title("Raman spectrum")
    plt.plot(Dw_range*1e-3, P)

    plt.xlabel("freq (kHz)")
    plt.ylabel("Probability")
    #plt.legend()
    plt.grid()

def spectrumF(Dw1, Dw2, v0, B0, t1, Ndw=200, n_steps=400):

    Dw = np.linspace(Dw1, Dw2, Ndw)

    dt = t1 / n_steps

    B1 = np.full(Ndw, B0[0], dtype=np.complex128)
    B2 = np.full(Ndw, B0[1], dtype=np.complex128)

    t = 0.0

    for _ in range(n_steps):

        phi = -keff*(v0 + g*t/2)*t + Dw*t

        eplus = np.exp(1j*phi)
        eminus = np.conj(eplus)

        # ---------- k1 ----------
        k1_B1 = 1j*W1**2/D*B1 + 1j*W1*W2/D*eplus*B2
        k1_B2 = 1j*W2**2/D*B2 + 1j*W1*W2/D*eminus*B1

        # ---------- k2 ----------
        B1t = B1 + dt*k1_B1/2
        B2t = B2 + dt*k1_B2/2

        phi = -keff*(v0 + g*(t+dt/2)/2)*(t+dt/2) + Dw*(t+dt/2)

        eplus = np.exp(1j*phi)
        eminus = np.conj(eplus)

        k2_B1 = 1j*W1**2/D*B1t + 1j*W1*W2/D*eplus*B2t
        k2_B2 = 1j*W2**2/D*B2t + 1j*W1*W2/D*eminus*B1t

        # ---------- k3 ----------
        B1t = B1 + dt*k2_B1/2
        B2t = B2 + dt*k2_B2/2

        k3_B1 = 1j*W1**2/D*B1t + 1j*W1*W2/D*eplus*B2t
        k3_B2 = 1j*W2**2/D*B2t + 1j*W1*W2/D*eminus*B1t

        # ---------- k4 ----------
        B1t = B1 + dt*k3_B1
        B2t = B2 + dt*k3_B2

        phi = -keff*(v0 + g*(t+dt)/2)*(t+dt) + Dw*(t+dt)

        eplus = np.exp(1j*phi)
        eminus = np.conj(eplus)

        k4_B1 = 1j*W1**2/D*B1t + 1j*W1*W2/D*eplus*B2t
        k4_B2 = 1j*W2**2/D*B2t + 1j*W1*W2/D*eminus*B1t

        B1 += dt*(k1_B1 + 2*k2_B1 + 2*k3_B1 + k4_B1)/6
        B2 += dt*(k1_B2 + 2*k2_B2 + 2*k3_B2 + k4_B2)/6

        t += dt

    return np.abs(B2)**2

def spectrum_groupF(v0, T, Dw1, Dw2, B0, t1,
                   Nv=200, Ndw=200, n_steps=400):

    sigma_v = np.sqrt(kb*T/m_Rb)

    v_range = np.linspace(v0-4*sigma_v,
                          v0+4*sigma_v,
                          Nv)

    dv = v_range[1]-v_range[0]

    norm = dv/(np.sqrt(2*np.pi)*sigma_v)

    P = np.zeros(Ndw)
    i = 0
    for v in v_range:
        print(i)
        w = np.exp(-(v-v0)**2/(2*sigma_v**2))*norm

        P += w*spectrumF(Dw1, Dw2, v, B0, t1,
                        Ndw=Ndw,
                        n_steps=n_steps)
        i += 1
    Dw = np.linspace(Dw1, Dw2, Ndw)

    plt.figure()
    plt.plot(Dw*1e-3, P)
    plt.grid()
    plt.xlabel("detun (kHz)")
    plt.ylabel("Probability")


# constants
m_Rb = 1.44e-25
kb = 1.38e-23
lam = 780e-9
keff = 4*np.pi/lam
w0 = 6.834682611e9*2*np.pi
g = 9.8*1

# Rabi param
D = 1e9
W1 = 8e6
W2 = 8e6
Weff = 2*W1*W2/D
print(f"Weffcounted = {Weff*1e-3} kHz")


# Rabi Osc single
Dw = W1**2/D - W2**2/D
print(f"Raman detuning = {Dw*1e-3} kHz")
B0 = [1+0j, 0+0j]
t_steps = 2000
v0 = 0
t1 = 200e-6          
#Rabi_osc(Dw, v0, B0, t1, t_steps)

# spectrum signle
Dw1 = -300e3
Dw2 = 600e3
v0 = 0
B0 = [1+0j, 0+0j]
t1 = 15e-6
N = 200
#spectrum(Dw1, Dw2, v0, B0, t1, t_steps, N)

# Rabi Osc group
v0 = 0
TK = 6e-6
Dw = W1**2/D - W2**2/D
B0 = [1+0j, 0+0j]
v0 = 0
t1 = 200e-6          
N = 200
Rabi_osc_group(v0, TK, Dw, B0, t1, t_steps, N)



# spectrum group
v0 = 0
T = 6e-6
Dw1 = -600e3
Dw2 = 600e3
B0 = [1+0j, 0+0j]
v0 = 0
t1 = 15e-6          
#spectrum_groupF(v0, T, Dw1, Dw2, B0, t1, Nv=101, Ndw=301, n_steps=400)


plt.show()







