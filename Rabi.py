# Rabi in case of chirp absence but doppler shift changing is taken into account
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp


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


def Rabi_osc_group(v0, T, Dw, B0, t1, t_steps, N):
    sigma_v = np.sqrt(kb*T/m_Rb)
    v_range = np.linspace(-4*sigma_v, 4*sigma_v, N)
    P = np.zeros(2000)
    for v_i in v_range:
        P += Rabi_osc(Dw, v0, B0, t1, t_steps)*np.exp((v_i-v0)**2/2/sigma_v**2)

    plt.figure()
    plt.title("Rabi osc group")

    t_eval = np.linspace(0, t1, t_steps)
    plt.plot(t_eval*1e6, P)

    plt.xlabel("time (µs)")
    plt.ylabel("Probability")
    #plt.legend()
    plt.grid()
    print(f"WGequa = {np.pi/t_eval[np.argmax(P)]*1e-3} kHz")
    print(f"WGcounted = {np.sqrt((W1**2/D - W2**2/D - (-keff*v0 + Dw))**2 + 4*W1**2*W2**2/D**2)*1e-3} kHz (no g acceleration)")


    return 1

def spectrum(Dw1, Dw2, v0, B0, t1, t_steps):

    Dw_range = np.linspace(Dw1, Dw2, 201)
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
        t_ = sol.t

        P1 = abs(B1)**2
        P2 = abs(B2)**2
        P.append(P2[-1])

    P = np.array(P)
    plt.figure()
    plt.title("Raman spectrum")
    plt.plot(Dw_range*1e-3, P)

    plt.xlabel("freq (kHz)")
    plt.ylabel("Probability")
    #plt.legend()
    plt.grid()

    return 1

# constants
m_Rb = 1.44e-25
kb = 1.38e-23
lam = 780e-9
keff = 4*np.pi/lam
w0 = 6.834682611e9*2*np.pi
g = 9.8

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
Rabi_osc(Dw, v0, B0, t1, t_steps)

# spectrum signle
Dw1 = -300e3
Dw2 = 600e3
v0 = 0
B0 = [1+0j, 0+0j]
t1 = 15e-6
spectrum(Dw1, Dw2, v0, B0, t1, t_steps)

# Rabi Osc group
v0 = 0
T = 6e-6
Dw = W1**2/D - W2**2/D
print(f"Raman detuning = {Dw*1e-3} kHz")
B0 = [1+0j, 0+0j]
v0 = 0
t1 = 200e-6          
N = 200
Rabi_osc_group(v0, T, Dw, B0, t1, t_steps, N)


plt.show()







