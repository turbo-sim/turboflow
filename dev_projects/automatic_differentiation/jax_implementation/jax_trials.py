# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:28:25 2024

@author: sipar
"""

import jax
import jax.numpy as jnp
import CoolProp.CoolProp as cp

DT=cp.DmassT_INPUTS

#%% PART 1: Analytical and callback equations of state

#By default jax uses 32 bit float. Force 64 bit.
jax.config.update("jax_enable_x64", True)

#Define fluid in CoolProp
rhoT0=jnp.array([100.0,310.0])
fluid=cp.AbstractState('HEOS','CO2')
fluid.update(DT,rhoT0[0],rhoT0[1])

#Parameters for Peng Robinson
fld={'acentric':fluid.acentric_factor(),
     'R':fluid.gas_constant(),
     'Tc':fluid.T_critical(),
     'Pc':fluid.p_critical(),
     'Pr':fluid.Prandtl(),
     'k' :fluid.conductivity(),
     'mu':fluid.viscosity(),
     'cp0':fluid.cp0molar(),
     'MM':fluid.molar_mass()}

#define Peng-Robinson EOS (note: copied from Wikipedia, do not use for anything serious)
def PR(rhoT):
    rho=rhoT[0]
    T=rhoT[1]
    R=fld['R']
    Tc=fld['Tc']
    Vm=fld['MM']/rho
    ηc=(1+(4-8**(1/2))**(1/3)+(4+8**(1/2))**(1/3))**-1
    Ωa=(8+40*ηc)/(49-37*ηc)
    a=Ωa*(R*Tc)**2/fld['Pc']
    Ωb=ηc/(3+ηc)
    b=Ωb*(R*Tc)/fld['Pc']
    Tr=T/fld['Tc']
    κ=0.37464+1.54226*fld['acentric']-0.26992*fld['acentric']**2
    α=(1+κ*(1-Tr**(1/2)))**2
    P=(fld['R']*T)/(Vm-b)-a*α/(Vm**2+2*b*Vm-b**2)
    Pr=P/fld['Pc']
    hid=T*fld['cp0']/fld['MM']
    Z=P*Vm/(R*T)
    B=0.07780*Pr/Tr
    h=hid+R/fld['MM']*Tc*(Tr*(Z-1)-2.078*(1+κ)*α**0.5*jnp.log((Z+2.414*B)/(Z-0.414*B)))
    sid=fld['cp0']/fld['MM']*jnp.log(T)-fld['R']/fld['MM']*jnp.log(P)
    s=sid+R/fld['MM']*(jnp.log(Z-B)-2.078*κ*((1+κ)/Tr**0.5-κ)*jnp.log((Z+2.414*B)/(Z-0.414*B)))
    return jnp.array([P,h,s,fld['k'],fld['mu']])

#define HEOS function calling CoolProp
def HEOS_callback(rhoT):
    fluid.update(cp.DmassT_INPUTS,rhoT[0].astype(float),rhoT[1].astype(float))
    mu=fld['mu']+0*rhoT[1] #This could be e.g. Sutherland like
    return jnp.array([fluid.p(),fluid.hmass(),fluid.smass(),fld['k'],mu])

#wrap the callback in jax format. The decorator jax.custom_jvp is for custom gradients
@jax.custom_jvp
def HEOS(rhoT):
    result_shape=jax.ShapeDtypeStruct((5,),jnp.dtype('float'))
    return jax.pure_callback(HEOS_callback, result_shape, rhoT)

#define the custom jacobian calling CoolProp
def HEOS_fwd_callback(rhoT):
    primal_out=HEOS(rhoT)
    dpdD=fluid.first_partial_deriv(cp.iP,cp.iDmass,cp.iT)
    dhdD=fluid.first_partial_deriv(cp.iHmass,cp.iDmass,cp.iT)
    dsdD=fluid.first_partial_deriv(cp.iSmass,cp.iDmass,cp.iT)
    dkdD=0.0
    dmudD=0.0
    dpdT=fluid.first_partial_deriv(cp.iP,cp.iT,cp.iDmass)
    dhdT=fluid.first_partial_deriv(cp.iHmass,cp.iT,cp.iDmass)
    dsdT=fluid.first_partial_deriv(cp.iSmass,cp.iT,cp.iDmass)
    dkdT=0.0
    dmudT=0.0
    fluid.update(cp.DmassT_INPUTS,rhoT[0],rhoT[1])
    return jnp.array([[dpdD,dpdT],[dhdD,dhdT],[dsdD,dsdT],[dkdD,dkdT],[dmudD,dmudT]])

#wrap the callback in jax format.
#https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
def HEOS_fwd(rhoT,tangents):
    primal_out=HEOS(rhoT[0])
    result_shape=jax.ShapeDtypeStruct((5,2),jnp.dtype('float'))
    jacobian=jax.pure_callback(HEOS_fwd_callback, result_shape, rhoT[0])
    tangent_out=jnp.dot(jacobian,tangents[0])
    return (primal_out,tangent_out)

#attach the forward mode jacobian to the main function
HEOS.defjvp(HEOS_fwd)

#make jit version of PR
PRjit=jax.jit(PR)
yPR=PRjit(rhoT0)
#make forward jacobian
PRgrad_nonjit=jax.jacfwd(PR)
PRgrad=jax.jit(PRgrad_nonjit)
dydxPR=PRgrad(rhoT0)

#same thing for HEOS, in the case the jacobian is the custom one
HEOSjit=jax.jit(HEOS)
yHEOS=HEOSjit(rhoT0)
HEOSgrad_nonjit=jax.jacfwd(HEOS)
HEOSgrad=jax.jit(HEOSgrad_nonjit)
dydxHEOS=HEOSgrad(rhoT0)

print('Jacobian from PR EOS:')
print(dydxPR)
print('Jacobian from HEOS:')
print(dydxHEOS)
print('\n\nInternal code represenation of Peng-Robinson')
print(jax.make_jaxpr(PR)(rhoT0))

# #%% PART 2: Implicit function differentiation (AD for nested solvers)
# import jaxopt    
# #y=f(a,x) subject to g(a,x)=0. Find y=F(x) and dydx=dF/dx
# #This is implemented in jaxopt: https://jaxopt.github.io/stable/implicit_diff.html
# #g(a,x)=0 -> ∂g/∂x*dx+∂g/∂a*da=0 -> da=-(∂g/∂x)/(∂g/∂a)*dx
# #dy=∂f/∂x*dx+∂f/∂a*da -> dy/dx=∂f/∂x+∂f/∂a*da/dx
# #dy/dx=∂f/∂x-∂f/∂a*(∂g/∂x)/(∂g/∂a)
# #Bisection from jaxopt cannot be jit-ed.
# def f(a,x):
#     return x**5+a*x+1.0
# def g(a,x):
#     return x**3+a*x+1.0
# def F(x):
#     bisec=jaxopt.Bisection(optimality_fun=g,lower=-10,upper=10)
#     a=bisec.run(x=x).params
#     return f(a,x)

# x=2.7

# #solve for a explcitly
# bisec=jaxopt.Bisection(optimality_fun=g,lower=-10,upper=10)
# a=bisec.run(x=x).params

# #function value
# y=F(x)
# y_verify=f(a,x)

# #gradient with implicit theorem
# dFdx=jax.grad(F)(x)

# dfda=jax.grad(f,argnums=0)(a,x)
# dfdx=jax.grad(f,argnums=1)(a,x)
# dgda=jax.grad(g,argnums=0)(a,x)
# dgdx=jax.grad(g,argnums=1)(a,x)
# #manually applied implicit theorem
# dFdx_verify=dfdx-dfda*dgdx/dgda

# #finite difference
# Dx=1e-8
# Da=-dgdx/dgda*Dx
# DgDx=(g(a+Da,x+Dx)-g(a,x))/Dx #should be close to 0
# dFdx_finite=(f(a+Da,x+Dx)-f(a,x))/Dx