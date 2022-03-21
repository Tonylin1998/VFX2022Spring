import numpy as np

def photographic_global_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
    L_white = np.max(Lm) 
    Ld = (Lm * (1 + (Lm/(L_white ** 2)))) / (1 + Lm)
    ldr = np.array(Ld * 255)
    
    return ldr


def photographic_local_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
