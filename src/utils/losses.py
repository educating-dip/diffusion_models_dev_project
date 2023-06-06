import torch 

def score_based_loss_fn(x, model, sde, eps=1e-5):

    """
    The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t) # for VESDE the mean is just x
    perturbed_x = mean + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    
    return loss

def epsilon_based_loss_fn(x, model, sde):

    """
    The loss function for training epsilon-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
    """
    random_t = torch.radnint(1, sde.num_steps, (x.shape[0],), device=x.device)
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t)
    # Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0)
    perturbed_x = mean + z * std[:, None, None, None]
    zhat = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((z - zhat)**2, dim=(1,2,3)))

    return loss
