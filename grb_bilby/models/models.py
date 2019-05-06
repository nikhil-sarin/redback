import numpy as np
import scipy.special as ss

def magnetar(time, L0, tau, nn, **kwargs):
    """
    :param time:
    :param L0:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    lum = L0 * (1. + time/tau)**((1. + nn)/(1. - nn))
    return lum

def powerlaw(time, A_1, alpha_1, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param kwargs:
    :return:
    """
    return A_1 * time**(alpha_1)

def full_magnetar(time, A_1, alpha_1, L0, tau, nn, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    pl = powerlaw(time=time, A_1=A_1, alpha_1=alpha_1)
    mag = magnetar(time=time, L0=L0, tau=tau, nn=nn)
    return pl + mag

def collapsing_mag(time, A_1, alpha_1, L0, tau, nn, tcol, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param nn:
    :param tcol:
    :param kwargs:
    :return:
    """
    pl = powerlaw(time, A_1, alpha_1)
    mag = np.heaviside(tcol - time,1e-50)*magnetar(time, L0, tau, nn)

    return pl + mag

def integral_general_collapsing(time, T0, kappa, tau, nn, tcol, **kwargs):
    """
    :param time:
    :param T0:
    :param kappa:
    :param tau:
    :param nn:
    :param tcol:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn)/(-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time/tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -T0/tau)
    first_term = (time**(1 + kappa)*pft)/(1 + kappa)
    second_term = (T0**(1 + kappa)*pst)/(1 + kappa)
    integral = np.heaviside(tcol - time,1e-50)* (first_term - second_term)
    return integral

def collapsing_losses(time, A_1, alpha_1, L0, tau, nn, tcol, kappa, T0, E0, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param nn:
    :param tcol:
    :param kappa:
    :param T0:
    :param E0:
    :param kwargs:
    :return:
    """
    pl = powerlaw(time, A_1, alpha_1)
    loss_term = E0 * (T0/time)**(kappa)
    integ = integral_general_collapsing(time, T0, kappa, tau, nn, tcol)
    Energy_loss_total = ((L0/(time**kappa))*integ)  + loss_term
    lum = (kappa*Energy_loss_total/time)
    total = pl + np.heaviside(time - T0, 1)*lum

    return total

def general_magnetar(time, amplitude_one, time_exponent_one,
                           delta_time_one, time_exponent_two, delta_time_two, **kwargs):
    """
    Reparameterized millisecond magnetar model from Sarin et al. (2018b) (piecewise)
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: Reparameterized braking index n
    :param delta_time_two: time between end of prompt emission and end of magnetar model plateau phase, (tau)
    """

    time_one = delta_time_one
    tau = delta_time_one + delta_time_two
    nn = (time_exponent_two - 1.)/(time_exponent_two + 1.)
    gamma = (1. + nn)/(1. - nn)
    num = (amplitude_one * time_one**(time_exponent_one))
    denom = ((1. + (time_one/tau))**(gamma))
    amplitude_two =  num/denom

    w = np.where(time < time_one)
    x = np.where(time > time_one)

    F1 = amplitude_one*time[w]**(time_exponent_one)
    F2 = amplitude_two * (1. + (time[x]/tau))**(gamma)

    total = np.concatenate((F1, F2))

    return total

def two_component_fireball_model(time, amplitude_one, time_exponent_one,
                                 delta_time_one, time_exponent_two):
    """
    Two component fireball model
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: power law decay exponent for the second power law
    """
    time_one = delta_time_one
    amplitude_two = amplitude_one * time_one**(time_exponent_one) / (time_one ** time_exponent_two)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    F1 = amplitude_one*time[w]**(time_exponent_one)
    F2 = amplitude_two*time[x]**(time_exponent_two)

    total = np.concatenate((F1, F2))

    return total


def three_component_fireball(time, amplitude_one, time_exponent_one,
                                   delta_time_one, time_exponent_two,
                                   delta_time_two, time_exponent_three, **kwargs):
    """
    Three component fireball model
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param time_exponent_three: power law decay exponent for third power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    amplitude_two = amplitude_one * time_one**(time_exponent_one) / (time_one ** time_exponent_two)
    amplitude_three = amplitude_two * time_two**(time_exponent_two) / (time_two ** time_exponent_three)


    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where(time > time_two)
    F1 = amplitude_one*time[w]**(time_exponent_one)
    F2 = amplitude_two*time[x]**(time_exponent_two)
    F3 = amplitude_three*time[y]**(time_exponent_three)

    total = np.concatenate((F1, F2, F3))
    return total

def four_component_fireball(time, amplitude_one, time_exponent_one, delta_time_one,
                                  time_exponent_two, delta_time_two,
                                  time_exponent_three, delta_time_three,
                                  time_exponent_four, **kwargs):
    """
    Four component fireball model
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param time_exponent_three: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param time_exponent_four: power law decay exponent for fourth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    amplitude_two = amplitude_one * time_one**(time_exponent_one) / (time_one ** time_exponent_two)
    amplitude_three = amplitude_two * time_two**(time_exponent_two) / (time_two ** time_exponent_three)
    amplitude_four = amplitude_three * time_three**(time_exponent_three) / (time_three ** time_exponent_four)


    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where((time_two < time) & (time < time_three))
    z = np.where(time > time_three)
    F1 = amplitude_one*time[w]**(time_exponent_one)
    F2 = amplitude_two*time[x]**(time_exponent_two)
    F3 = amplitude_three*time[y]**(time_exponent_three)
    F4 = amplitude_four*time[z]**(time_exponent_four)

    total = np.concatenate((F1, F2, F3, F4))

    return total

def five_component_fireball_model(time, amplitude_one, time_exponent_one,
                                  delta_time_one, time_exponent_two,
                                  delta_time_two, time_exponent_three,
                                  delta_time_three, time_exponent_four,
                                  delta_time_four, time_exponent_five):
    """
    Five component fireball model
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param time_exponent_three: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param time_exponent_four: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param time_exponent_five: power law decay exponent for fifth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four

    amplitude_two = amplitude_one * time_one**(time_exponent_one) / (time_one ** time_exponent_two)
    amplitude_three = amplitude_two * time_two**(time_exponent_two) / (time_two ** time_exponent_three)
    amplitude_four = amplitude_three * time_three**(time_exponent_three) / (time_three ** time_exponent_four)
    amplitude_five = amplitude_four * time_four**(time_exponent_four) / (time_four ** time_exponent_five)


    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where(time > time_four)

    F1 = amplitude_one*time[u]**(time_exponent_one)
    F2 = amplitude_two*time[v]**(time_exponent_two)
    F3 = amplitude_three*time[w]**(time_exponent_three)
    F4 = amplitude_four*time[x]**(time_exponent_four)
    F5 = amplitude_five*time[y]**(time_exponent_five)

    total = np.concatenate((F1, F2, F3, F4, F5))

    return total

def six_component_fireball_model(time, amplitude_one, time_exponent_one,
                                  delta_time_one, time_exponent_two,
                                  delta_time_two, time_exponent_three,
                                  delta_time_three, time_exponent_four,
                                  delta_time_four, time_exponent_five,
                                  delta_time_five, time_exponent_six):
    """
    six component fireball model
    :param time: time array for power law
    :param amplitude_one: power law decay amplitude
    :param time_exponent_one: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param time_exponent_two: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param time_exponent_three: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param time_exponent_four: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param time_exponent_five: power law decay exponent for fifth power law
    :param delta_time_five: time between fourth and fifth power laws
    :param time_exponent_six: power law decay exponent for sixth power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four
    time_five = time_four + delta_time_five

    amplitude_two = amplitude_one * time_one**(time_exponent_one) / (time_one ** time_exponent_two)
    amplitude_three = amplitude_two * time_two**(time_exponent_two) / (time_two ** time_exponent_three)
    amplitude_four = amplitude_three * time_three**(time_exponent_three) / (time_three ** time_exponent_four)
    amplitude_five = amplitude_four * time_four**(time_exponent_four) / (time_four ** time_exponent_five)
    amplitude_six = amplitude_five * time_five**(time_exponent_five) / (time_five ** time_exponent_six)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where((time_four < time) & (time < time_five))
    z = np.where(time > time_five)

    F1 = amplitude_one*time[u]**(time_exponent_one)
    F2 = amplitude_two*time[v]**(time_exponent_two)
    F3 = amplitude_three*time[w]**(time_exponent_three)
    F4 = amplitude_four*time[x]**(time_exponent_four)
    F5 = amplitude_five*time[y]**(time_exponent_five)
    F6 = amplitude_six*time[z]**(time_exponent_six)

    total = np.concatenate((F1, F2, F3, F4, F5, F6))

    return total

def integral_general(time, T0, kappa, tau, nn, **kwargs):
    """
    :param time:
    :param T0:
    :param kappa:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn)/(-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time/tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -T0/tau)
    first_term = (time**(1 + kappa)*pft)/(1 + kappa)
    second_term = (T0**(1 + kappa)*pst)/(1 + kappa)
    integral = (first_term - second_term)
    return integral

def piecewise_radiative_losses(time, A_1, alpha_1, L0, tau, nn, kappa, T0, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param nn:
    :param kappa:
    :param T0:
    :param kwargs:
    :return:
    """
    pl_time = np.where(time <= T0)
    magnetar_time = np.where(time > T0)
    E0 = (A_1 * T0**alpha_1 * T0)/kappa
    pl = powerlaw(time[pl_time], A_1, alpha_1)

    loss_term = E0 * (T0/time[magnetar_time])**(kappa)
    integ = integral_general(time[magnetar_time], T0, kappa, tau, nn)
    Energy_loss_total = ((L0/(time[magnetar_time]**kappa))*integ)  + loss_term

    lum = (kappa*Energy_loss_total/time[magnetar_time])

    total = np.concatenate((pl, lum))

    return total

def radiative_losses_full(time, A_1, alpha_1, L0, tau, nn, kappa, T0, E0, **kwargs):
    """
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param nn:
    :param kappa:
    :param T0:
    :param E0:
    :param kwargs:
    :return:
    """
    pl = powerlaw(time, A_1, alpha_1)
    loss_term = E0 * (T0/time)**(kappa)
    integ = integral_general(time, T0, kappa, tau, nn)
    Energy_loss_total = ((L0/(time**kappa))*integ)  + loss_term
    lum = (kappa*Energy_loss_total/time)
    total = pl + np.heaviside(time - T0, 1)*lum

    return total
