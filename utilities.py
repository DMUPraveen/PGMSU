from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat

def correct_permuation(A_pred,A_true,M_pred,M_true):
    '''
    A : (N_end,N_pix)  
    M : (N_channel,N_end)
    Returns Corrected Permuation based on RMSE


    return A_corrected,M_corrected
    '''
    P = A_pred.shape[0]
    best_error = float('inf')
    best_perm = tuple(range(P))
    for perm in permutations(range(P)):
        A_new = A_pred[tuple(perm),:]
        error = np.mean((A_new-A_true)**2)
        if(error < best_error):
            best_error = error
            best_perm = perm

    A_corrected = A_pred[best_perm,:]
    M_corrected = M_pred[:,best_perm]

    return A_corrected,M_corrected

def plot_figures(A_pred,M_pred,A_true,M_true,save_path,H):
    '''
    A : (N_end,N_pix)  
    M : (N_channel,N_end)

    '''
    MM_pred = M_pred/np.max(M_pred,axis=0,keepdims=True)
    MM_true = M_true/np.max(M_true,axis=0,keepdims=True)
    print(MM_pred.shape,MM_true.shape)
    P = A_true.shape[0]
    A_maps = A_pred.reshape(-1,H,H)
    for i in range(P):
        fig,ax = plt.subplots(1,1)
        fig.tight_layout()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        ax.imshow(A_maps[i,:,:])
        fig.savefig(os.path.join(save_path,f"abundance_{i}.png"))

    for i in range(P):
        fig,ax = plt.subplots(1,1)
        fig.tight_layout()
        ax.set_xlabel("Bands")
        ax.set_ylabel("Reflectance")
        ax.plot(MM_pred[:,i].ravel(),label="Predicted")
        ax.plot(MM_true[:,i].ravel(),label="GT")
        ax.legend()
        fig.savefig(os.path.join(save_path,f"endmembers_{i}.png"))

    
    savemat(os.path.join(save_path,"Data.mat"),
            dict(
            A_pred = A_pred,
            A_true = A_true,
            M_true = M_true,
            M_pred = M_pred
            )
            )

    #Create compare abundance compare figure
    fig,ax = plt.subplots(2,P)
    A_true_map = A_true.reshape(-1,H,H)
    for i in range(P):
        ax[0,i].imshow(A_maps[i,:,:])
        ax[1,i].imshow(A_true_map[i,:,:])
    fig.savefig(
        os.path.join(save_path,"compare_fig.png")
    )
    return fig

def calculate_errors(A_pred,M_pred,A_true,M_true,save_path):
    '''
    A : (N_end,N_pix)  
    M : (N_channel,N_end)

    '''
    total_rmse = np.sqrt(np.mean((A_pred-A_true)**2))
    endmember_wise_rmse = np.sqrt(np.mean((A_pred-A_true)**2,axis=1))

    endmember_wise_sad = np.arccos(np.sum(M_pred*M_true,axis=0)/(np.sqrt(np.sum(M_pred**2,axis=0))*np.sqrt(np.sum(M_true**2,axis=0))))

    total_sad = np.mean(endmember_wise_sad)

    print(f"{total_rmse=}")
    print(f"{endmember_wise_rmse=}")
    print(f"{total_sad=}")
    print(f"{endmember_wise_sad=}")

    with open(os.path.join(save_path,"Results.txt"),"w+") as f:
        f.write(f"{total_rmse=}\n")
        f.write(f"{endmember_wise_rmse=}\n")
        f.write(f"{total_sad=}\n")
        f.write(f"{endmember_wise_sad=}\n")

    d = dict(
                total_rmse=total_rmse,
                total_sad = total_sad,
                endmember_wise_sad = endmember_wise_sad,
                endmember_wise_rmse = endmember_wise_rmse
            )
    savemat(os.path.join(save_path,"Results.mat"),
            d
            )

    return d
