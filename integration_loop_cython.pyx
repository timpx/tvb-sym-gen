
import numpy as np

def integration_loop_cython(int current_step, float[:,:,:,::1] history,
                            int[:,:,::1] idelays, float[:,:,::1] state, 
                             weights, int horizon, int[:,:,::1] cvar, 
                             int[:,:,::1] node_ids, list monitors,  float stimulus, 
                            int int_steps, int[::1] model_cvar, dfun, float local_coupling,
                            float[:,:,::1] x_i, float[:,:,:,::1] delayed_state,
                            float[:,:,:,::1] coupled_input, float[:,:,::1] node_coupling):
    
    ## dimension and type of inputs
    #current_step: int
    #surface: None
    #history.shape : 4, float64
    #idelays: 3, int32
    #state: 3, float64
    #weights: 4, float64
    #horizon: int
    #cvar:3, int
    #node_ids:3, int
    #int_steps: int
    #local_coupling: float
    #model_cvar: 1,int
    #monitors: list
    #stimulus: float, 

    cdef:
        float x, y
        float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
        float lc_0, c_0
        float I, a, b, c, d, e, f, g, alpha, beta, tau
        float[:,:,::1] X
        float ac,bc, curr
        float[:,:,:,::1] g_ij
        #float[:,:,::1] x_i
        float[:,:,:,::1] x_j#, delayed_state
        int i, j, k, step

    
    for step in range(current_step+1, current_step+int_steps+1):

        ##r regional coupling
        for i in range(cvar.shape[0]):
            for j in range(cvar.shape[1]):
                for k in range(cvar.shape[2]):
                    delayed_state[i,j,k,:] = history[((step-1-idelays[i,j,k]) % horizon), cvar[i,j,k], node_ids[i,j,k], :]

        g_ij = weights
        for i in range(model_cvar.shape[0]):
            x_i[i] = state[model_cvar[i]]
        x_j = delayed_state
        ac= 0.00390625
        bc = 0.0
        
        for j in range(74):
            curr = 0.
            for i in range(74):
                curr += g_ij[i,0,j,0] * x_j[i,0,j,0] 
            node_coupling[0,j,0] = ac*curr+bc

        ## else surface
        
        ## stimulus      

        ## integration

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
        dt = 0.01
        for i in range(74):
            x=state[0,i,0]; y=state[1,i,0]
            c_0 = node_coupling[0,i,0]
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

            state[0,i,0] = tau*x3*(2*I + alpha*y + 2*c_0 + e*x15 + g*x14 + 2*lc_0 + x13 + x14**3*x4)/2 + x
            state[1,i,0] = y + x3*(2*a + b*x14 + c*x15 + x10 + x11*x5)/(2*tau)
        
        ## update history
        history[step % horizon] = state

        ##monitors
        # best way to generate?
        # problem with yield
        time = step * dt
        yield [[time, np.array(state)]]
