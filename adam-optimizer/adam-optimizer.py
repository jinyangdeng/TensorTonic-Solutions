import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = beta1 * np.array(m) + (1-beta1) * np.array(grad)
    v = beta2 * np.array(v) + (1-beta2) * np.array(grad) * np.array(grad)
    m_hat = m/(1-beta1**t)
    v_hat = v/(1-beta2**t)
    theta = param - lr * (m_hat / (np.sqrt(v_hat) + eps))

    return (theta, m, v)