import numpy
import math
from matplotlib import pyplot, cm

# All values in SI units
# Except length which is in mm

#-----Data-----

def getVal(vals, param, default):
    if param in vals:
        return float(vals[param])
    else:
        return default

vals = {}

#if 'e' in g: print (g['e']) 
#else: print ("no")

with open("input.in", "r") as f:
    for line in f:
        param = line.split(' ')[0]
        val = line.split(' ')[1]
        vals[param] = val.rstrip("\n")

nx = int(getVal(vals, 'nx', 41))
ny = int(getVal(vals, 'ny', 41))
nit = int(getVal(vals, 'nit', 50))
nt = int(getVal(vals, 'nt_vel', 5000))
nt_ = int(getVal(vals, 'nt_temp', 5000))
dump_rate = int(getVal(vals, 'dump_rate_vel', 1000))
dump_rate_ = int(getVal(vals, 'dump_rate_temp', 1000))

xlen = int(getVal(vals, 'xlen', 5))
ylen = int(getVal(vals, 'ylen', 5))
dx = xlen/(nx-1)
dy = ylen/(ny-1)
dt = getVal(vals, 'dt_vel', .0001)
dt_ = getVal(vals, 'dt_temp', .001)
x = numpy.linspace(0,xlen,nx)
y = numpy.linspace(0,ylen,ny)
X, Y = numpy.meshgrid(x, y)

Uinlet = getVal(vals, 'Uinlet', 2)
Tinlet = getVal(vals, 'Tinlet', 298)

rho = getVal(vals, 'rho', 995.1e-9)
k = getVal(vals, 'k', 651.400)
nu = getVal(vals, 'nu', .801627)
alpha = getVal(vals, 'alpha', .148020)

#Calculating mean free path of water
Patm = getVal(vals, 'Patm', 1.0135e5) #Pressure in Pa
dia = getVal(vals, 'mol_dia', 2.75e-10) #Water molecular diameter in meters
lam = (8.314*Tinlet)/(1.414*3.14*dia*dia*6.022e23*Patm) #mean free path in meters
lam = lam*math.pow(10, 3) #convert to mm

#Calcuation of jump temperature distance
acc_coef = getVal(vals, 'acc_coef', 0.94)
gamma = getVal(vals, 'gamma', 1)
Cv = getVal(vals, 'Cv', 4186) #Cv of water
Pr = k*pow(10, -6)/(nu*rho*Cv)
sig = ((2-acc_coef)*2*gamma*Pr*lam)/(acc_coef*(gamma+1))

micro = int(getVal(vals, 'microfluidic', 0))

if micro == 0: #if microfludic disabled
    lam = 0
    sig = 0

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx)) 
b = numpy.zeros((ny, nx))
T = numpy.zeros((ny,nx))

T1 = getVal(vals, 'T1', 340)
T2 = getVal(vals, 'T2', 298)
Uwall = 0 

udiff = 1
Tdiff = 1

Tcell_final = 0

#-----Functions-----

def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b
    
def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        
        #Pressure poisson equation        
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        #pressure driven flow
        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]
        
        #dp/dy = 0 at walls        
        p[0, :] = p[1, :]        
        p[-1, :] = p[-2, :]
        
    return p

def flow(nt, u, v, Uwall, dt, dx, dy, p, rho, nu, lam):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny, nx))
    
    n = 1    
    
    for n in range(nt+1):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        #Navier STokes equation
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        #wall conditions
        u[0,:] = (dx*Uwall + lam*u[1,:])/(dx+lam)
        u[-1,:] = (dx*Uwall + lam*u[-2,:])/(dx+lam)
        v[0,:] = (dx*Uwall + lam*v[1,:])/(dx+lam)
        v[-1,:] = (dx*Uwall + lam*v[-2,:])/(dx+lam)
        
        #inlet and outlet conditions
        #all derivates are 0 at exit
        u[:, 0] = Uinlet
        u[:, -1] = u[:, -2]        
        v[:, 0] = 0
        v[:, -1] = v[:, -2]
        
        udiff = (numpy.sum(u)-numpy.sum(un))/numpy.sum(u)
            
        if n % dump_rate == 0:
            print('%d: %f' % (n, udiff))
    
    return u, v, p, udiff

def heat(nt, u, v, T, T1, T2, dt, dx, dy, alpha, sig):
    Tn = numpy.empty_like(T)
            
    n = 0
    
    for n in range(nt+1):
        Tn = T.copy();
    
        #Energy Equation
        T[1:-1,1:-1] = (Tn[1:-1,1:-1]-
                        u[1:-1,1:-1]*dt/dx*(Tn[1:-1,1:-1]-Tn[1:-1,0:-2])-
                        v[1:-1,1:-1]*dt/dy*(Tn[1:-1,1:-1]-Tn[0:-2,1:-1])+\
                        alpha*(dt/dx**2*(Tn[1:-1,2:]-2*Tn[1:-1,1:-1]+Tn[1:-1,0:-2])+\
                        dt/dy**2*(Tn[2:,1:-1]-2*Tn[1:-1,1:-1]+Tn[0:-2,1:-1])))
                        
        
        #wall conditions
        T[-1,:] = (dx*T2+sig*T[-2,:])/(dx+sig) #the casing    
        T[0,:] = (dx*T1+sig*T[1,:])/(dx+sig) #the cell
        
        #inlet and outlet conditions
        T[:, 0] = Tinlet
        T[:, -1] = T[:, -2] #dT/dx = 0 at the exit
            
        Tdiff = (numpy.sum(T)-numpy.sum(Tn))/numpy.sum(u)
            
        if n % dump_rate == 0:
            print('%d: %f' % (n, Tdiff))
    
    return T, T1
    
#-----Main-----

print("*****MFlowSim: Microfluidic Pipe Flow Simulator*****")

print("Velocity Convergence:")
u, v, p, udiff = flow(nt, u, v, Uwall, dt, dx, dy, p, rho, nu, lam)

print("\nTemperature Convergence:")
T, Tcell_final = heat(nt_, u, v, T, T1, T2, dt_, dx, dy, alpha, sig)

fig = pyplot.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
Gx, Gy = numpy.gradient(u) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
N = G/G.max()  # normalize 0..1
surf = ax.plot_surface(X,Y,u[:], rstride=1, cstride=1,
    facecolors=cm.jet(N),
    linewidth=0, antialiased=False, shade=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('X Velocity')

fig2 = pyplot.figure(figsize=(11,7), dpi=100)
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(X,Y,T[:])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Temperature')



#-----Bulk Temperature Calculation-----

num = 0
dem = 0
n = 0

for n in range(ny):
    if n == 0 or n == ny:
        num += u[n, -1]*T[n, -1]
    elif n%2 == 0 :
        num += 2*u[n, -1]*T[n, -1]
    elif n%2 != 0:
        num += 4*u[n, -1]*T[n, -1]

for n in range(ny):
    if n == 0 or n == ny:
        dem += u[n, -1]
    elif n%2 == 0 :
        dem += 2*u[n, -1]
    elif n%2 != 0:
        dem += 4*u[n, -1]

Tbulk = num/dem

#-----Heat Transfer Coefficient Calculation-----

mid_val = int(nx/2)

q = k*(T[0, mid_val] - T[1, mid_val])/(dt*Tbulk)
lnT = (Tbulk - Tinlet)/(math.log(T1 - Tinlet)-math.log(T1 - Tbulk))
h = q/lnT

print('Inner Side Heat Transfer Coefficient: %f' % h)

#-----Printing X Velocity for at a specific x-----

#for n in range(ny):
#    print('%f: %f' % (n*dy, u[n, -1]))