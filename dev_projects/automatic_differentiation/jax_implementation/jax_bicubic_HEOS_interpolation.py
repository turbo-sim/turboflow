#%% INITIALIZATION

import os,psutil
#Look at the number of physical cores.
#NOTE: On modern machines might want to use the number of high performance
#cores, because different core architectures can cause problems in parallel
#processing
NCORES=psutil.cpu_count(logical=False)
#Before loading jax, force it to see the CPU count we want
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=%d"%NCORES
import jax
#By default jax uses 32 bit, for scientific computing we need 64 bit precisin
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
#To read more regarding the automatic parallelization in jax
#https://jax.readthedocs.io/en/latest/sharded-computation.html
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec , NamedSharding

import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
import numpy as np
import time

float64=jnp.dtype('float64')
complex128=jnp.dtype('complex128')

#%% ################# USER SETUP #########################
#Select the property to interpolate. For example iDmass, iSmass and iT
#The variable names use the symbol D, but can represent any required 
#keyed output by changing this line.
iD=cp.iSmass
#Grid size in h direction
N=50
#Grid size in log(P) direction
M=50
#Fluid selection
name="CO2"
#Grid boundaries
hmin=1e5
hmax=10e5
Pmin=1e6
Pmax=1e8
#Number of random points to test the error
Npoints=50000
#Number of times to repeat the computation. used to stabilize the timing of  
#very fast computations when using time.time()
Nrepeats=int(1e6/Npoints)
##########################################################
#Full evaluation of Helmholtz equation of state
f=cp.AbstractState("HEOS",name)
#Use bicubic table
ftab=cp.AbstractState("BICUBIC&HEOS",name)
#%% FUNCTION TO COMPUTE INTERPOLATION COEFFICIENTS
#For more details, see:https://en.wikipedia.org/wiki/Bicubic_interpolation
#In bicubic interpolation, approximate a generic function with a piecewise
#cubic polynomial in the form:
#z= aij * x**i * y**j with i,y =[0,1,2,3]
#The 16 aij coefficients can be computed from the 4 corners of the cell using:
#   -The value of the function f
#   -The derivative in the x direction fx
#   -The derivative in the y direction fy
#   -The cross derivative fxy
#We can compute on the fly the coefficients, or compute them in advance for all
#the cells in the grid. If we use the compuation in advance we can save time 
#during execution, but the storage and memory requirement increases by a factor
#of 4x. For an uncompressed 64bit 100*100 grid, storing the function and
#derivatives takes 320kB per variable, while storing the cubic coefficients
#takes 1.28MB.

@jax.jit
def compute_bicubic_coefficients_of_ij(i,j,f,fx,fy,fxy):
    #xx=f(0,0)&f(1,0)&f(0,1)&f(1,1)&f_x(0,0)&f_x(1,0)&f_x(0,1)&f_x(1,1)&f_y(0,0)&f_y(1,0)&f_y(0,1)&f_y(1,1)&f_{xy}(0,0)&f_{xy}(1,0)&f_{xy}(0,1)&f_{xy}(1,1)
    xx=[f[i,j],f[i+1,j],f[i,j+1],f[i+1,j+1],fx[i,j],fx[i+1,j],fx[i,j+1],fx[i+1,j+1],fy[i,j],fy[i+1,j],fy[i,j+1],fy[i+1,j+1],fxy[i,j],fxy[i+1,j],fxy[i,j+1],fxy[i+1,j+1]]
    A=[ [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ -3., 3., 0., 0., -2., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 2., -2., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., -3., 3., 0., 0., -2., -1., 0., 0 ],
        [ 0., 0., 0., 0., 0., 0., 0., 0., 2., -2., 0., 0., 1., 1., 0., 0 ],
        [ -3., 0., 3., 0., 0., 0., 0., 0., -2., 0., -1., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., -3., 0., 3., 0., 0., 0., 0., 0., -2., 0., -1., 0 ],
        [ 9., -9., -9., 9., 6., 3., -6., -3., 6., -6., 3., -3., 4., 2., 2., 1 ],
        [ -6., 6., 6., -6., -3., -3., 3., 3., -4., 4., -2., 2., -2., -2., -1., -1 ],
        [ 2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0 ],
        [ 0., 0., 0., 0., 2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0 ],
        [ -6., 6., 6., -6., -4., -2., 4., 2., -3., 3., -3., 3., -2., -1., -2., -1 ],
        [ 4., -4., -4., 4., 2., 2., -2., -2., 2., -2., 2., -2., 1., 1., 1., 1]]
    return jnp.matmul(jnp.array(A,dtype=f.dtype),jnp.array(xx,dtype=f.dtype))

#%%
#Vector initialization
Lmin=jnp.log(Pmin)
Lmax=jnp.log(Pmax)
h=jnp.linspace(hmin,hmax,num=N,dtype=float64)
L=jnp.linspace(Lmin,Lmax,num=M,dtype=float64)
P=jnp.exp(L)
PP,hh=jnp.meshgrid(P,h)
deltah=h[1]-h[0]
deltaL=L[1]-L[0]
D=jnp.zeros((N,M),dtype=float64)
dDdL=jnp.zeros_like(D)
dDdh=jnp.zeros_like(D)
d2DdhdL=jnp.zeros_like(D)
bicubic_coefficients=jnp.zeros((N,M,16),dtype=float64)


#First, get the property values and derivatives from the equation of state.
#The syntax is: first_partial_deriv(parameters OF, parameters WRT, parameters CONSTANT)
#The different properties are keyed to integer values
progress_checkpoints=10
t0=time.time()
for i,hi in enumerate(h):
    if i%(N/progress_checkpoints)<1:print('Progress: %f %% done in %f s'%(i/N*100,time.time()-t0))
    for j,Pj in enumerate(P):
        try:
            f.update(cp.HmassP_INPUTS,hi,Pj)
            D=D.at[i,j].set(f.keyed_output(iD))
            #For some reason some output work better using two phase derivative
            #while other with normal derivatives
            #TODO: investigate this further to understand how implemented
            #in CoolProp
            if iD==cp.iDmass or iD==cp.iT:
                use_two_phase_deriv=True
            elif iD==cp.iSmass:
                use_two_phase_deriv=False
            else:
                print("WARNING!!: Defaulting to not using two phase derivatives as the current property is not specified. Adjoust the code above this print.")
            if use_two_phase_deriv and f.phase()==cp.iphase_twophase:
                dDdP=f.first_two_phase_deriv(iD,cp.iP,cp.iHmass)
                dDdhtemp=f.first_two_phase_deriv(iD,cp.iHmass,cp.iP)
                d2DdPdh=f.second_two_phase_deriv(iD,cp.iP,cp.iHmass,cp.iHmass,cp.iP)
            else:
                dDdP=f.first_partial_deriv(iD,cp.iP,cp.iHmass)
                dDdhtemp=f.first_partial_deriv(iD,cp.iHmass,cp.iP)
                d2DdPdh=f.second_partial_deriv(iD,cp.iP,cp.iHmass,cp.iHmass,cp.iP)
            
            dPdL=1*Pj #P=exp(L) ->dP/dL=exp(L)->dP/dL=P
            dDdL=dDdL.at[i,j].set(dDdP*dPdL)
            dDdh=dDdh.at[i,j].set(dDdhtemp)
            #second_partial_deriv(parameters OF, parameters WRT1, parameters CONSTANT1, parameters WRT2, parameters CONSTANT2)
            d2DdhdL=d2DdhdL.at[i,j].set(d2DdPdh*dPdL)
        except:
            pass

#Because the derivative are not on an unitary grid, they must be rescaled when
#computing the bicubic coefficients            
t0=time.time()
for i,hi in enumerate(h):
    if i%(N/progress_checkpoints)<1:print('Progress: %f %% done in %f s'%(i/N*100,time.time()-t0))
    for j,Pj in enumerate(P):
        temp=compute_bicubic_coefficients_of_ij(i,j,D,dDdh*deltah,
                                                      dDdL*deltaL,
                                                      d2DdhdL*deltah*deltaL)
        bicubic_coefficients=bicubic_coefficients.at[i,j,:].set(temp)

#%% JIT INTERPOLANT specialized with respect to a fixed bicubic coefficient map
# This can be made more flexible using the static arguments capability of jax,
# the values passed from the global variables become hard coded during the
# compilation. This approach was used because is simpler for a proof of concept
@jax.jit
def bicubic_interpolant(h,P):
    ii=((h-hmin)/(hmax-hmin)*(N-1))
    i=ii.astype(int)
    x=ii-i
    L=jnp.log(P)
    jj=((L-Lmin)/(Lmax-Lmin)*(M-1))
    j=jj.astype(int)
    y=jj-j
    
    xth=jnp.ones_like(h) #x to 0 power
    D=0*h#initialize density - can be arbitrary shaped array
    #NOTE: for loops get unrolled by jax. For small loops such as these
    #this result in performance improvements, but for large loops might
    #increase the computation time
    for m in range(4):
        xthyth=1.0*xth #x to m-th power and y to 0-th power
        for n in range(4):
            c=bicubic_coefficients[i,j,4*n+m]
            D+=c*xthyth
            xthyth=xthyth*y #increase y degree for next iter
        xth=xth*x #increase x degree for next iter
    return D
#%% INVERSE INTERPOLANT - to find one of inputs given output
#TODO: this will be more efficient if it is natively vecotrized. 
#The problem encountered was broadcasting the correct dimensions in D_nodal
@jax.jit
def inverse_interpolant_scalar(h,D):
    #Find the real(float) index
    ii=((h-hmin)/(hmax-hmin)*(N-1))
    #The integer part is the cell index
    i=ii.astype(int)
    #The remainder (for numerical stability better to use the difference)
    #is instead the position within our interpolation cell.
    x=ii-i
    #find interval that contains the solution
    xth=jnp.ones_like(h) #initialize x to the 0th power
    #First we compute the nodal values, that is the values of D(h,P) where
    #h is the actual enthalpy and P are grid values.
    #TODO: instead of computing all the nodal values and then use sortedsearch
    #to find the correct interval, we could do a binary search. This would
    #constraint M to be a power of 2.
    #Possible example (to be refined) to compute the node. Start with the node
    #corresponding to j=M/2, then compute new index j=j+M/4*(2*(Dj>D)-1)
    #then j=j+M/8*(2*(Dj>D)-1) and so on ...
    #after log2(M) iteration we converged to the index j.
    D_nodal=jnp.zeros(M)
    for m in range(4):
        D_nodal+=bicubic_coefficients[i,:,m]*xth
        xth=xth*x
    #We search more efficiently in which interval we have the solution
    #if we assume a sorted vector.
    #TODO: This assumes that P has a monotonic trend with respect to D
    #at fixed h. This causes some problems and needs further investigation
    if iD==cp.iSmass:
        j=jax.numpy.searchsorted(-D_nodal,-D).astype(int)-1
    else:
        j=jax.numpy.searchsorted(D_nodal,D).astype(int)-1

    #After we are in the unit square, that is for known i and j
    #compute 1D cubic coefficients (as complex numbers to avoid promotion)
    #Each coefficient is bj=sum(aij*x**i)
    #Leading to the equation D=b0 + b1*y + b2*y**2 + b3*y**3
    xth=jnp.ones_like(h)
    b0 =jnp.zeros_like(h,dtype=complex128)
    b1 =jnp.zeros_like(h,dtype=complex128)
    b2 =jnp.zeros_like(h,dtype=complex128)
    b3 =jnp.zeros_like(h,dtype=complex128)
    for m in range(4):
        b0 +=bicubic_coefficients[i,j,4*0+m]*xth
        b1 +=bicubic_coefficients[i,j,4*1+m]*xth
        b2 +=bicubic_coefficients[i,j,4*2+m]*xth
        b3 +=bicubic_coefficients[i,j,4*3+m]*xth
        xth=xth*x
    #solve cubic equation - all three solutions
    #TODO: if necessary, add solution for degenerate (quadratic and linear)
    #For more information:https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    D0=b2*b2-3*b3*b1
    D1=2*b2*b2*b2-9*b3*b2*b1+27*b3*b3*(b0-D)
    C=((D1+(D1*D1-4*D0*D0*D0)**0.5)/2)**(1/3)
    D0C=jax.lax.select(C==(0+0j),0+0j,D0/C)
    z=jnp.array([1,-0.5+0.8660254037844386j,-0.5-0.8660254037844386j])
    y=-1/(3*b3)*(b2+C*z+D0C/z)
    #To find our solution we have two criteria:
    #   -0 imaginary part
    #   -real part between 0 and 1, that are the bounds of our cell
    # We define a "badness" as the deviation from these critera, and pick the
    # solution with the lowest badness
    badness=jax.nn.relu(4*(jnp.real(y)-0.5)**2-1)+jnp.imag(y)**2
    yreal=jnp.real(y[jnp.argmin(badness)])
    jj=j+yreal
    L=Lmin+jj*(Lmax-Lmin)/(M-1)
    P=jnp.exp(L)
    return P
#Use the jax vectorization for this function
inverse_interpolant=jax.vmap(inverse_interpolant_scalar)
#%% CHECK CORRECTNESS OF BICUBIC INTERPOLATION PROCESS
checks=[(h[-2]+1e-4      ,P[-3]+1e-4      ,'Check at node N-1+eps,M-2+eps'),
        (h[-2]-1e-4      ,P[-3]-1e-4      ,'Check at node N-1-eps,N-2-eps'),
        (.5*(h[-2]+h[-3]),.5*(P[-3]+P[-4]),'Check midnode N-1.5,M-2.5')]
for h0,P0,check_name in checks:
    f.update(cp.HmassP_INPUTS,h0,P0)
    D0=f.keyed_output(iD)
    if f.phase()==cp.iphase_twophase:
        dDdP0=f.first_two_phase_deriv(iD,cp.iP,cp.iHmass)
        dDdh0=f.first_two_phase_deriv(iD,cp.iHmass,cp.iP)
        d2DdhdP0=f.second_two_phase_deriv(iD,cp.iP,cp.iHmass,cp.iHmass,cp.iP)
    else:
        dDdP0=f.first_partial_deriv(iD,cp.iP,cp.iHmass)
        dDdh0=f.first_partial_deriv(iD,cp.iHmass,cp.iP)
        d2DdhdP0=f.second_partial_deriv(iD,cp.iP,cp.iHmass,cp.iHmass,cp.iP)
    Di=bicubic_interpolant(h0,P0)
    dDdhi,dDdPi=jax.grad(bicubic_interpolant,argnums=(0,1))(h0,P0)     
    d2DdhdPi=jax.grad(jax.grad(bicubic_interpolant,argnums=1),argnums=0)(h0,P0)  
    print("\n###############################\n")
    print(check_name)
    print("Quantity | HEOS         | Interpolated | Relative error")
    print("D        | %+.4e  | %+.4e  | %+.4e"%(D0,Di,(Di/D0-1)))
    print("dDdP     | %+.4e  | %+.4e  | %+.4e"%(dDdP0,dDdPi,(dDdPi/dDdP0-1)))
    print("dDdh     | %+.4e  | %+.4e  | %+.4e"%(dDdh0,dDdhi,(dDdhi/dDdh0-1)))
    print("d2DdhdP  | %+.4e  | %+.4e  | %+.4e"%(d2DdhdP0,d2DdhdPi,(d2DdhdPi/d2DdhdP0-1)))

#%% Test random points
#NOTE: Jax uses deterministic random numbers. This is a good thing for 
#reproducibility, but must be taken into account
#For more info:https://jax.readthedocs.io/en/latest/random-numbers.html
key=jax.random.key(0)
rand=jax.random.uniform(jax.random.key(0),(2,Npoints))
hrand=hmin+(hmax-hmin)*rand[0]
Prand=jnp.exp(Lmin+(Lmax-Lmin)*rand[1])

#Direct jax interpolation
print("\n###############################\n")
print("Evaluation of batches of %d points. Jax evaluations are vectorized."%Npoints)
t0=time.time()
for irepeat in range(Nrepeats):
    Drand_interp= bicubic_interpolant(hrand,Prand)
print("Direct (h-P) evaluation with jax cubic in %.4e seconds/point"%((time.time()-t0)/Nrepeats/Npoints))

#Reverse jax interpolation
t0=time.time()
for irepeat in range(Nrepeats):
    P_rev=inverse_interpolant(hrand,Drand_interp)
print("Inverse (h-D) evaluation with jax cubic in %.4e seconds/point"%((time.time()-t0)/Nrepeats/Npoints))
err_rev=(P_rev/Prand-1)#error of reverse interpolation


#reverse jax interpolation on multi-core
#First we create the mesh of processors. It is important to note that the 
#shape of the arrays we subdivide must be a multiple of the number of 
#devices in the mesh.
jaxmesh = Mesh(devices=mesh_utils.create_device_mesh((NCORES,)),
            axis_names=('x'))
#We subdivide the array in shards, with each shard assigned to a specific processor  
h_mesh = jax.device_put(hrand, NamedSharding(jaxmesh, PartitionSpec('x')))
D_mesh = jax.device_put(Drand_interp, NamedSharding(jaxmesh, PartitionSpec('x')))
#This will lead to each core receiving and working only on a part of the problem.
P_rev=inverse_interpolant(h_mesh,D_mesh)#jit compile and push shards to devices
t0=time.time()
for irepeat in range(Nrepeats):
    dummy1=inverse_interpolant(h_mesh,D_mesh)
print("Inverse multi-core with jax cubic in %.4e seconds/point"%((time.time()-t0)/Nrepeats/Npoints))
err_rev=(P_rev/Prand-1)#error of reverse interpolation

#Direct EOS evaluation
#Note: for a fair comparison converting into native types to remove overhead
hrandlist=hrand.tolist()
Prandlist=Prand.tolist()
iterator=list(range(Npoints))
Drand_HEOS_numpy=0*np.array(hrand)
ziplist=zip(iterator,hrandlist,Prandlist)
t0=time.time()
for i,hn,Pn in ziplist:
    try:
        f.update(cp.HmassP_INPUTS,hn,Pn)
        Drand_HEOS_numpy[i]=f.keyed_output(iD)
    except:
        Drand_HEOS_numpy[i]=np.nan
print("Direct evaluation with coolprop in %.4e seconds/point"%((time.time()-t0)/Npoints))
Drand_HEOS=jnp.array(Drand_HEOS_numpy,dtype=float64)

#Coolprop bicubic evaluation
Drand_BIC_numpy=0**np.array(hrand)
ziplist=zip(iterator,hrandlist,Prandlist)
t0=time.time()
for irepeat in range(Nrepeats):
    for i,hn,Pn in ziplist:
        try:
            ftab.update(cp.HmassP_INPUTS,hn,Pn)
            Drand_BIC_numpy[i]=ftab.keyed_output(iD) 
        except Exception:
            Drand_BIC_numpy[i]=np.nan 
print("Direct evaluation with coolprop cubic in %.4e seconds/point"%((time.time()-t0)/Nrepeats/Npoints))        
Drand_BIC=jnp.array(Drand_BIC_numpy,dtype=float64)  
     
#test reverse interpolation
bad_interpolation=jnp.sum(jnp.abs(err_rev)>1e-9)
print("\n###############################\n")
print("Inverse interpolation test:")
print("Quantity | Interpolated | Reversed     | Relative error")
print("P        | %+.4e  | %+.4e  | %+.4e"%(Prand[0],P_rev[0],(P_rev[0]/Prand[0]-1)))
if bad_interpolation:
    i=jnp.nanargmax(jnp.abs(err_rev))
    print("P        | %+.4e  | %+.4e  | %+.4e"%(Prand[i],P_rev[i],(P_rev[i]/Prand[i]-1)))
    print("WARNING: For %d out of %d tested points, reverse interpolation did not converge (relative error>1e-9).The worst point is shown.NOTE: This is likely caused by the existence of multiple solution due to poor quality of the underlying grid in some regions, where the density trend is not monotonic"%(bad_interpolation,Npoints))
if any(jnp.isnan(err_rev)):
    print("WARNING: For some points the reverse interpolation returned nan. This might be due to bad cell values in the grid")
error_interp=Drand_interp/Drand_HEOS-1
error_BIC=Drand_BIC/Drand_HEOS-1
#%% PLOTS
#plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(hh, jnp.log10(PP), D)
fig2=plt.figure()
ax2=fig2.add_subplot(111)
ax2.semilogy(jnp.sort(abs(error_interp)),label='own interpolant on %dx%d grid'%(N,M))
ax2.semilogy(jnp.sort(abs(error_BIC)),label='CoolProp interpolant')
ax2.legend()
if iD==cp.iDmass:
    ax2.set_title("Relative error on density")
elif iD==cp.iSmass:
    ax2.set_title("Relative error on entropy")
elif iD==cp.iT:
    ax2.set_title("Relative error on Temperature")
plt.grid(True)
fig3=plt.figure()
ax3=fig3.add_subplot(111)#, projection='3d')           
scplt=ax3.scatter(hrand, jnp.log10(Prand),c=jnp.log10(0.5+(jnp.abs(error_interp)>1e-5)))
fig3.colorbar(scplt)
ax3.set_xlabel('Enthalpy [J/kg]')
ax3.set_ylabel('log10(Pressure / (1 [Pa]))')
if iD==cp.iDmass:
    ax3.set_title("log10 of relative error on density")
elif iD==cp.iSmass:
    ax3.set_title("log10 of relative error on entropy")
elif iD==cp.iT:
    ax3.set_title("log10 of relative error on Temperature")
plt.grid(True)