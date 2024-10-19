import torch
from torch import autograd
from torch.autograd import Variable


def gradient_penalty(D, real, fake, labels, c, device):
    """Heuristic for enforcing the 1-Lipshitz Constraint"""
    lam = c["gp_reg"]
    batch_size = int(real.shape[0])
    eta = torch.rand(batch_size, 1, 1, 1, device=device)
    eta = eta.expand_as(real)

    interpolated = eta * real + ((1 - eta) * fake)
    interpolated = Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated, labels)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                              create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - 1) ** 2).mean() * lam
    return grad_penalty


def gradient_penalty_S(D, real, fake, labels, c, device):
    """Heuristic for enforcing the 1-Lipshitz Constraint"""
    lam = c["gp_reg"]
    batch_size = int(real.shape[0])
    eta = torch.rand(batch_size, 1, 1, device=device)
    eta = eta.expand_as(real)

    interpolated = eta * real + ((1 - eta) * fake)
    interpolated = Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated, labels)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                              create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - 1) ** 2).mean() * lam
    return grad_penalty


def wasserstein_gen(D, G, c, device, labels, temporal_gradient_pen=False):
    """This could already work with the temporal loss scheme incorporated in  DVDGAN"""
    # Train Generator to pretend it's genuine
    g_input_z = torch.rand(int(labels.shape[0]), c["g_input_size"], device=device)
    gen_input = Variable(torch.cat((g_input_z, labels), dim=1))
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data, labels)
    g_error = -torch.mean(dg_fake_decision)
    return g_error


def wasserstein_critic(D, G, c, device, real_data, labels, with_grad_pen=True, reweigh_cost=False):
    critic_input_z = torch.rand(int(real_data.shape[0]), c["g_input_size"], device=device, requires_grad=False)
    gen_input = Variable(torch.cat((critic_input_z, labels), axis=1))  # need to add location variable here but we need the time generator
    grad_penalty = 0
    fake_data = G(gen_input).detach()
    critic_real_score = D(real_data, labels)
    critic_fake_score = D(fake_data, labels)
    if reweigh_cost:
        sample_cost_weight = torch.mean(torch.amax(fake_data, dim=(2, 3)), dim=1)
        min_weight = sample_cost_weight.min()
        max_weight = sample_cost_weight.max()
        sample_cost_weight = (sample_cost_weight - min_weight)/(max_weight - min_weight)
        sample_cost_weight = torch.reshape(sample_cost_weight, (sample_cost_weight.shape[0], 1))
        critic_fake_score *= sample_cost_weight  # conditional cost reweighing for imbalanced distribution
    critic_real_err = torch.mean(critic_real_score)
    critic_fake_err = torch.mean(critic_fake_score)

    if with_grad_pen:
        grad_penalty = gradient_penalty(D, real_data, fake_data, labels, c, device)
    total_cost = -1 * (critic_real_err - critic_fake_err) + grad_penalty
    return total_cost, grad_penalty


def temporal_gradient_penalty(fake_data, temporal_grad_factor=1.0):
    """This is used to penalize the gradients in space, forcing it to match the real distribution"""
    penalty = torch.square(fake_data[:, 0:23, :, :] - fake_data[:, 1:24, :, :]).mean()
    return temporal_grad_factor * penalty


def mode_seeking(G, c, device, labels):
    """ This begins the implementation of the mode-seeking GANs (Mao et. Al) """
    # need to find labels with the same value via sorting into unique values
    unique_rows = torch.unique(labels, sorted=False, dim=0)
    g_input_z = torch.rand(int(labels.shape[0]), c["g_input_size"], device=device)
    gen_input = Variable(torch.cat((g_input_z, labels), dim=1))
    g_fake_data = G(gen_input)
    loss = 0
    for row in unique_rows:
        indices = torch.all(labels == row, dim=1)
        X = g_fake_data[indices]

        N = X.shape[0]  # Batch subset size
        X_expanded = X.unsqueeze(4).repeat(1, 1, 1, 1, N)
        X = X.transpose(0, 3).transpose(0, 1).transpose(1, 2).unsqueeze(0)

        z = g_input_z[indices]
        z_expanded = z.unsqueeze(2).repeat(1, 1, N)
        z = z.unsqueeze(0).transpose(1, 2)

        eps = 5e-6  # for numerical stability
        sub_loss = torch.square(X_expanded - X).sum(dim=(0, 1, 2, 3)) / (torch.norm(z_expanded - z, p=2, dim=(0, 1))+eps)
        sub_loss = sub_loss.mean()
        loss += sub_loss

    return 1 / loss
