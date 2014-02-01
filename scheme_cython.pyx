cpdef float[:,:,::1] scheme_cython(float[:,:,::1] X, dfun, float[:,:,::1] coupling, float local_coupling, float stimulus):

    cdef:
        float x, y
        float[:,:,::1] DX
        float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        float lc_0, c_0
        float I, a, b, c, d, e, f, g, alpha, beta, tau

    tau = 1.
    I=0.
    a = -2.
    b = -10.0
    c = 0.0
    d = 0.02
    e = 3.0
    f = 1.0
    g = 0.0
    alpha = 1.0
    beta = 1.0
    #DX = np.zeros((2,74,1), dtype=np.float32)
    dt = 0.01
    for i in range(74):
        x=X[0,i,0]; y=X[1,i,0]
        c_0 = coupling[0,i,0]
        lc_0 = local_coupling * x
    
        x0 = x**2;
        x1 = g*x;
        x2 = b*x;
        x3 = d*dt;
        x4 = -f;
        x5 = -beta;
        x6 = e*x0;
        x7 = x5*y;
        x8 = c*x0;
        x9 = x**3*x4;
        x10 = x2 + x7 + x8;
        x11 = y + x3*(a + x10)/tau;
        x12 = alpha*x11;
        x13 = x1 + x12 + x6 + x9;
        x14 = tau*x3*(I + c_0 + lc_0 + x13) + x;
        x15 = x14**2;
        
        X[0,i,0] = tau*x3*(2*I + alpha*y + 2*c_0 + e*x15 + g*x14 + 2*lc_0 + x13 + x14**3*x4)/2 + x
        X[1,i,0] = y + x3*(2*a + b*x14 + c*x15 + x10 + x11*x5)/(2*tau)


    return X
