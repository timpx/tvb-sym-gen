# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Generating OpenCL for integrating differential equation systems


#%pylab inline
from pylab import *


from sympy import *
from sympy.tensor import IndexedBase, Idx


symbols('dt')==symbols('dt')


# Start with a simple oscillator model


x, y, dt, tau, a = symbols('x y dt tau a')

ddt = {
    x: tau*(x - x**3/3 + y),
    y: (a - x)/tau
}
param = [a, tau]

# Form the "half" step, and then the full step for the Heun method


half = {u: u + dt*du for u, du in ddt.iteritems()}
heun = [u + (du + du.subs(half))*dt/2 for u, du in half.iteritems()]


# Then, ask sympy to perform common subexpression elimination, or `cse`, so that we don't compute the same thing multiple times


aux, exs = cse(heun)
for l, r in aux:
    print l, '<-', r
for u, ex in zip(ddt.keys(), exs):
    print u, '<-', ex


# Then we can generate a looping kernel for OpenCL like this


template = '''
cdef void integration_loop(float X, float DX, float dt, dict par)

    cdef:
        {cdef}
    {eq_def}
    {parameters}
    {equations}
'''
cdef = 'float ' + ', '.join([str(ke) for ke in ddt.keys()]) + '\n        ' + 'float ' + ', '.join([str(au) for au, _ in aux])
eq_def = ('; '.join(['%s=X[%d]' % (str(u), i) for i, u in enumerate(ddt.keys())]) + '\n    ' +
         ''.join(['%s = %s;\n    ' % (l, str(r)) for l, r in aux]) + '\n')
parameters =  ', '.join([str(parv) for parv in param]) + ' = ' + ', '.join(["param['%s']" % parv for parv in param])
equations = '\n    '.join('DX[%d] = %s' % (i, str(ex)) for i, ex in enumerate(exs))
#loop = ('; '.join(['%s=X[%d]' % (ccode(u), i) for i, u in enumerate(ddt.keys())]) + ';\n\n'
#    + ''.join(['%s = %s;\n' % (l, ccode(r)) for l, r in aux]) + '\n'
#    + ''.join(['DX[%d] = %s;\n' % (i, ccode(ex)) for i, ex in enumerate(exs)])
#)

kernel = template.format(
    cdef=cdef,
    eq_def=eq_def,
    parameters=parameters,
    equations=equations
)

print kernel


# Of course this is only for one node, no delays or noise or connectivity. 
# Also can do this just with strings:


#system = '''
#dx = tau*(x - x**3/3 + y)
#dy = (a - x)/tau
#'''
#
#ddt2 = {}
#for lhs, rhs in [line.split('=') for line in system.split('\n') if line]:
#    svar = parse_expr(lhs.replace('d', ''))
#    vf = parse_expr(rhs)
#    ddt2[svar] = vf
#    
#ddt2
#
#
## Then we can have just a translating function
#
#
#def translate(system, nstep=100):
#    """
#    Input `system` is either a string with lines containing dx = x - x**3/3 + y
#    or a dictionary like {'x': 'x - x**3/3 + y'}.
#
#    """
#    
#    # parse system definition
#    if type(system) in (str, unicode):
#        ddt = {}
#        for lhs, rhs in [line.split('=') for line in system.split('\n') if line]:
#            svar = parse_expr(lhs.replace('d', ''))
#            vf = parse_expr(rhs)
#            ddt[svar] = vf
#    elif isinstance(system, dict):
#        ddt = {symbols(k): parse_expr(v) for k, v in system.iteritems()}
#    else:
#        raise NotImplementedError("don't know what to do with %r" % (system,))
#    
#    # create Heun discretization
#    half = {u: u + dt*du for u, du in ddt.iteritems()}
#    heun = [u + (du + du.subs(half))*dt/2 for u, du in half.iteritems()]
#    
#    # move common parts to aux variables
#    aux, exs = cse(heun, optimizations='basic', order='none')
#    
#    # compile to kernel source
#    decl = 'float ' + ', '.join(map(ccode, ddt.keys())) + ', '  + ', '.join([ccode(au) for au, _ in aux]) + ';'
#    nstep = str(nstep)
#    loop = ('; '.join(['%s=X[%d]' % (ccode(u), i) for i, u in enumerate(ddt.keys())]) + ';\n\n'
#        + ''.join(['%s = %s;\n' % (l, ccode(r)) for l, r in aux]) + '\n'
#        + ''.join(['DX[%d] = %s;\n' % (i, ccode(ex)) for i, ex in enumerate(exs)])
#    )
#
#    return template.format(
#        decl=decl,
#        nstep=nstep,
#        loop=loop
#    )
#
#
## et voila
#
#
#print translate('''
#dx = tau*(x - x**3/3 + y)
#dy = (a - x + b*y + c*y**2)/tau
#dz = x + y + z + x*y*z
#''')
#
#
## Hm, it automatically changed the order of `x, z, y` but that's because `ddt` is a `dict` which doesn't keep track of order.
#
#
## `Integrator` can inspect the `Model` to find out state variable names, parameter names, and equations. 
## 
## This is sufficient to write a general numexpr evaluator for the dfun (which should be `Model.dfun`), and as done here, sufficient to generate an OpenCL kernel specific to this integration scheme, if so desired.
## 
## However, perhaps the set up is right, if we just take the type(integrator) to indicate which scheme to use. The integrator could be responsible for either evaluating with numexpr the model as would be done now, or providing the correct discretization scheme. After, the simulator is responsible for applying the kernel, etc. 
## 
## OK Not bad.
#
#
#
#