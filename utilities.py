import numpy as np

def normalize(epool):
    e_max = np.max(epool)
    e_min = np.min(epool)

    epool_norm = (epool - e_min)/(e_max - e_min)

    return epool_norm

def denormalize(epool_norm, e_min, e_max):
    epool = (e_max - e_min) * epool_norm + e_min

    return epool

def copy_paste(epool_norm, C):
    D = len(epool_norm[0]) # length of transposon
    i, j = np.random.choice(np.arange(D), 2, replace = False)
    epool_norm[C][i] = epool_norm[C][j]

    return epool_norm

def cut_paste(epool_norm, C):
    D = len(epool_norm[0]) 
    i, j = np.sort(np.random.choice(np.arange(D), 2, replace = False))
    temp = np.insert(epool_norm[C], i, epool_norm[C][j])
    epool_norm[C] = np.delete(temp, j+1)

    return epool_norm

def copy_paste_2(epool_norm, C1, C2):
    D = len(epool_norm[0]) # length of transposon
    i, j = np.random.choice(np.arange(D), 2, replace = False)
    temp_c1 = epool_norm[C1][i]
    temp_c2 = epool_norm[C2][j]

    epool_norm[C1][j] = temp_c2
    epool_norm[C2][i] = temp_c1

    return epool_norm


def cut_paste_2(epool_norm, C1, C2):
    D = len(epool_norm[0]) # length of transposon
    i_c1, j_c1 = np.sort(np.random.choice(np.arange(D), 2, replace = False))
    i_c2, j_c2 = np.sort(np.random.choice(np.arange(D), 2, replace = False))

    temp1 = np.insert(epool_norm[C1], j_c1, epool_norm[C2][i_c2])
    temp2 = np.insert(epool_norm[C2], j_c2, epool_norm[C1][i_c1])

    epool_norm[C1] = np.delete(temp1, i_c1)
    epool_norm[C2] = np.delete(temp2, i_c2)

    return epool_norm



def transposon_operator(epool, jrate, L = 1):
    D = len(epool[0]) # length of transposon
    pool_len = len(epool)
    epool_norm = normalize(epool)
    for i in range(pool_len):
        if np.random.random() < jrate:
            C1 = i 
            C2 = np.random.randint(0, pool_len)
            if C1 == C2:
                if np.random.random() > 0.5:
                    # Apply cut and paste operation in epool_norm[C1]
                    epool_norm = cut_paste(epool_norm, C1)
                else:
                    # Apply copy and paste operation in epool_norm[C1]
                    epool_norm = copy_paste(epool_norm, C1)
            else:
                if np.random.random() > 0.5:
                    # Apply cut and paste operation in epool_norm[C1] and epool_norm[C2]
                    epool_norm = cut_paste_2(epool_norm, C1 ,C2)
                else:
                    # Apply copy and paste operation in epool_norm[C1] and epool_norm[C2]
                    epool_norm = copy_paste_2(epool_norm, C1, C2)

    e_min, e_max  = np.min(epool), np.max(epool)
    epool = denormalize(epool_norm, e_min, e_max)
    
    return epool



