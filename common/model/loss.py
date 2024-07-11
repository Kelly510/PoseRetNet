import torch
import numpy as np


def mpjpe(predicted, target, valid, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    predicted & target: B, F, J, 3
    valid: B, F, J
    """
    assert predicted.shape == target.shape
    assert predicted.shape[:-1] == valid.shape
    valid = valid.float()
    errors = torch.norm(predicted - target, dim=-1) * valid
    
    if not return_joints_err:
        return torch.sum(errors) / (torch.sum(valid) + 1e-6)
    else:
        joints_err = torch.sum(errors, dim=(0, 1)) / (torch.sum(valid, dim=(0, 1)) + 1e-6)
        joints_err = joints_err.cpu().numpy().reshape(-1) * 1000
        return torch.sum(errors) / (torch.sum(valid) + 1e-6), joints_err

def weighted_mpjpe(predicted, target, w, valid):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    predicted & target: B, F, J, 3
    w: J
    valid: B, F, J
    """
    assert predicted.shape == target.shape
    assert predicted.shape[:-1] == valid.shape
    valid = valid.float()
    a = torch.sum(w * torch.norm(predicted - target, dim=-1) * valid)
    b = torch.sum(valid) + 1e-6
    return a / b


def TCLoss(predicted_3d_pos, w, valid):
    """
    predicted_3d_pos: B, F, J, 3
    w: F
    valid: B, F, J
    """
    dif_seq = torch.diff(predicted_3d_pos, dim=1)       # B, F - 1, J, 3
    assert w.shape[0] == dif_seq.shape[-2]
    
    dif_valid = valid[:, 1:] & valid[:, :-1]
    dif_valid = dif_valid.float()
    
    a = torch.sum(w.view(1, 1, -1, 1) * torch.square(dif_seq) * dif_valid.unsqueeze(-1))
    b = torch.sum(dif_valid) + 1e-6
    return a / b


def p_mpjpe(predicted, target, valid):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    assert predicted.shape[:-1] == valid.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    valid = valid.astype(np.float32)
    a = np.sum(np.linalg.norm(predicted_aligned - target, axis=-1) * valid)
    b = np.sum(valid) + 1e-6
    return a / b


def n_mpjpe(predicted, target, valid):
    masked_pred, masked_target = predicted * valid[:, :, np.newaxis], target * valid[:, :, np.newaxis]
    nom = np.sum(masked_pred[:, :, 0] * masked_target[:, :, 0], axis=1) + \
          np.sum(masked_pred[:, :, 1] * masked_target[:, :, 1], axis=1) + \
          np.sum(masked_pred[:, :, 2] * masked_target[:, :, 2], axis=1)
    denom = np.sum(masked_pred[:, :, 0] * masked_pred[:, :, 0], axis=1) + \
            np.sum(masked_pred[:, :, 1] * masked_pred[:, :, 1], axis=1) + \
            np.sum(masked_pred[:, :, 2] * masked_pred[:, :, 2], axis=1)
    s_opt = nom / denom
    scaled_pred = predicted * s_opt[:, np.newaxis, np.newaxis]

    a = np.sum(np.linalg.norm(scaled_pred - target, axis=-1) * valid)
    b = np.sum(valid) + 1e-6

    return a / b


def mean_velocity_error_train(predicted, target, valid):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    assert predicted.shape[:-1] == valid.shape
    
    velocity_predicted = torch.diff(predicted, dim=1)
    velocity_target = torch.diff(target, dim=1)
    velocity_valid = valid[:, 1:] & valid[:, :-1]
    
    velocity_valid = velocity_valid.float()
    a = torch.sum(torch.norm(velocity_predicted - velocity_target, dim=-1) * velocity_valid)
    b = torch.sum(velocity_valid) + 1e-6
    return a / b


def mean_velocity_error(predicted, target, valid):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    assert predicted.shape[:-1] == valid.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    velocity_valid = valid[1:] & valid[:-1]
    
    velocity_valid = velocity_valid.astype(np.float32)
    a = np.sum(np.linalg.norm(velocity_predicted - velocity_target, axis=-1) * velocity_valid)
    b = np.sum(velocity_valid) + 1e-6
    return a / b


def PCK(predicted, target, valid, threshold=0.15):
    dists = torch.norm(predicted - target, dim=-1)

    a = torch.sum(((dists < threshold) * valid).float())
    b = torch.sum(valid.float()) + 1e-6
    return a / b


def AUC(predicted, target, valid):
    # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = torch.linspace(0, 150, 31).tolist()
    pck_values = torch.DoubleTensor(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pck_values[i] = PCK(predicted, target, valid, threshold=threshold/1000.0)
    return pck_values.mean().item()
