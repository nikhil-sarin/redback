import numpy as np
import scipy.special as ss


def magnetar_only(time, L0, tau, nn, **kwargs):
    """
    :param time: time
    :param L0: luminosity parameter
    :param tau: spin-down damping timescale
    :param nn: braking index
    :param kwargs: key word argument for handling plotting
    :return: luminosity or flux (depending on scaling) as a function of time.
    """
    lum = L0 * (1. + time / tau) ** ((1. + nn) / (1. - nn))
    return lum


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


def collapsing_magnetar(time, A_1, alpha_1, L0, tau, nn, tcol, **kwargs):
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
    pl = one_component_fireball_model(time, A_1, alpha_1)
    mag = np.heaviside(tcol - time, 1e-50) * magnetar_only(time, L0, tau, nn)

    return pl + mag


def general_magnetar(time, A_1, alpha_1,
                     delta_time_one, alpha_2, delta_time_two, **kwargs):
    """
    Reparameterized millisecond magnetar model from Sarin et al. (2018b) (piecewise)
    :param time: time array for power law
    :param : power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: Reparameterized braking index n
    :param delta_time_two: time between end of prompt emission and end of magnetar model plateau phase, (tau)
    """

    time_one = delta_time_one
    tau = delta_time_one + delta_time_two
    nn = (alpha_2 - 1.) / (alpha_2 + 1.)
    gamma = (1. + nn) / (1. - nn)
    num = (A_1 * time_one ** (alpha_1))
    denom = ((1. + (time_one / tau)) ** (gamma))
    A_1 = num / denom

    w = np.where(time < time_one)
    x = np.where(time > time_one)

    F1 = A_1 * time[w] ** (alpha_1)
    F2 = amplitude_two * (1. + (time[x] / tau)) ** (gamma)

    total = np.concatenate((F1, F2))

    return total


def one_component_fireball_model(time, A_1, alpha_1, **kwargs):
    """
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param kwargs:
    :return:
    """
    return A_1 * time ** alpha_1


def two_component_fireball_model(time, A_1, alpha_1,
                                 delta_time_one, alpha_2, **kwargs):
    """
    Two component fireball model
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    """
    time_one = delta_time_one
    amplitude_two = A_1 * time_one ** (alpha_1) / (time_one ** alpha_2)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    F1 = A_1 * time[w] ** (alpha_1)
    F2 = amplitude_two * time[x] ** (alpha_2)

    total = np.concatenate((F1, F2))

    return total


def three_component_fireball_model(time, A_1, alpha_1,
                                   delta_time_one, alpha_2,
                                   delta_time_two, alpha_3, **kwargs):
    """
    Three component fireball model
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    amplitude_two = A_1 * time_one ** (alpha_1) / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** (alpha_2) / (time_two ** alpha_3)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where(time > time_two)
    F1 = A_1 * time[w] ** (alpha_1)
    F2 = amplitude_two * time[x] ** (alpha_2)
    F3 = amplitude_three * time[y] ** (alpha_3)

    total = np.concatenate((F1, F2, F3))
    return total


def four_component_fireball_model(time, A_1, alpha_1, delta_time_one,
                                  alpha_2, delta_time_two,
                                  alpha_3, delta_time_three,
                                  alpha_4, **kwargs):
    """
    Four component fireball model
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    amplitude_two = A_1 * time_one ** (alpha_1) / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** (alpha_2) / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** (alpha_3) / (time_three ** alpha_4)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where((time_two < time) & (time < time_three))
    z = np.where(time > time_three)
    F1 = A_1 * time[w] ** (alpha_1)
    F2 = amplitude_two * time[x] ** (alpha_2)
    F3 = amplitude_three * time[y] ** (alpha_3)
    F4 = amplitude_four * time[z] ** (alpha_4)

    total = np.concatenate((F1, F2, F3, F4))

    return total


def five_component_fireball_model(time, A_1, alpha_1,
                                  delta_time_one, alpha_2,
                                  delta_time_two, alpha_3,
                                  delta_time_three, alpha_4,
                                  delta_time_four, alpha_5):
    """
    Five component fireball model
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    """

    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four

    amplitude_two = A_1 * time_one ** (alpha_1) / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** (alpha_2) / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** (alpha_3) / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** (alpha_4) / (time_four ** alpha_5)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where(time > time_four)

    F1 = A_1 * time[u] ** (alpha_1)
    F2 = amplitude_two * time[v] ** (alpha_2)
    F3 = amplitude_three * time[w] ** (alpha_3)
    F4 = amplitude_four * time[x] ** (alpha_4)
    F5 = amplitude_five * time[y] ** (alpha_5)

    total = np.concatenate((F1, F2, F3, F4, F5))

    return total


def six_component_fireball_model(time, A_1, alpha_1,
                                 delta_time_one, alpha_2,
                                 delta_time_two, alpha_3,
                                 delta_time_three, alpha_4,
                                 delta_time_four, alpha_5,
                                 delta_time_five, alpha_6):
    """
    six component fireball model
    :param time: time array for power law
    :param A_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    :param delta_time_three: time between second and third power laws
    :param alpha_4: power law decay exponent for fourth power law
    :param delta_time_four: time between third and fourth power laws
    :param alpha_5: power law decay exponent for fifth power law
    :param delta_time_five: time between fourth and fifth power laws
    :param alpha_6: power law decay exponent for sixth power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    time_three = time_two + delta_time_three
    time_four = time_three + delta_time_four
    time_five = time_four + delta_time_five

    amplitude_two = A_1 * time_one ** (alpha_1) / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** (alpha_2) / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** (alpha_3) / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** (alpha_4) / (time_four ** alpha_5)
    amplitude_six = amplitude_five * time_five ** (alpha_5) / (time_five ** alpha_6)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where((time_four < time) & (time < time_five))
    z = np.where(time > time_five)

    F1 = A_1 * time[u] ** (alpha_1)
    F2 = amplitude_two * time[v] ** (alpha_2)
    F3 = amplitude_three * time[w] ** (alpha_3)
    F4 = amplitude_four * time[x] ** (alpha_4)
    F5 = amplitude_five * time[y] ** (alpha_5)
    F6 = amplitude_six * time[z] ** (alpha_6)

    total = np.concatenate((F1, F2, F3, F4, F5, F6))

    return total


def integral_general(time, T0, kappa, tau, nn, **kwargs):
    """
    General integral for radiative losses model
    :param time:
    :param T0:
    :param kappa:
    :param tau:
    :param nn:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn) / (-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time / tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -T0 / tau)
    first_term = (time ** (1 + kappa) * pft) / (1 + kappa)
    second_term = (T0 ** (1 + kappa) * pst) / (1 + kappa)
    integral = (first_term - second_term)
    return integral


def integral_general_collapsing(time, T0, kappa, tau, nn, tcol, **kwargs):
    """
    General collapsing integral for radiative losses model
    :param time:
    :param T0:
    :param kappa:
    :param tau:
    :param nn:
    :param tcol:
    :param kwargs:
    :return:
    """
    alpha = ((1 + nn) / (-1 + nn))
    pft = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -time / tau)
    pst = ss.hyp2f1(1 + kappa, alpha, 2 + kappa, -T0 / tau)
    first_term = (time ** (1 + kappa) * pft) / (1 + kappa)
    second_term = (T0 ** (1 + kappa) * pst) / (1 + kappa)
    integral = np.heaviside(tcol - time, 1e-50) * (first_term - second_term)
    return integral


def integral_mdr(time, T0, kappa, a, **kwargs):
    z_f = (1 + a * time) ** (-1)
    z_int = (1 + a * T0) ** (-1)
    divisor_i = a ** (1 + kappa) * (kappa - 1) * (1 + a * T0) ** (1 - kappa)
    divisor_f = a ** (1 + kappa) * (kappa - 1) * (1 + a * time) ** (1 - kappa)
    first = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_f) / divisor_f
    second = ss.hyp2f1(1 - kappa, -kappa, 2 - kappa, z_int) / divisor_i
    return first - second


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
    E0 = (A_1 * T0 ** alpha_1 * T0) / kappa
    pl = powerlaw(time[pl_time], A_1, alpha_1)

    loss_term = E0 * (T0 / time[magnetar_time]) ** (kappa)
    integ = integral_general(time[magnetar_time], T0, kappa, tau, nn)
    Energy_loss_total = ((L0 / (time[magnetar_time] ** kappa)) * integ) + loss_term

    lum = (kappa * Energy_loss_total / time[magnetar_time])

    total = np.concatenate((pl, lum))

    return total


def radiative_losses(time, A_1, alpha_1, L0, tau, nn, kappa, T0, E0, **kwargs):
    """
    radiative losses model with a step function
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
    loss_term = E0 * (T0 / time) ** (kappa)
    integ = integral_general(time, T0, kappa, tau, nn)
    Energy_loss_total = ((L0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * Energy_loss_total / time)
    total = pl + np.heaviside(time - T0, 1) * lum

    return total


def radiative_losses_mdr(time, A_1, alpha_1, L0, tau, kappa, E0, T0, **kwargs):
    """
    radiative losses model for vaccum dipole radiation
    :param time:
    :param A_1:
    :param alpha_1:
    :param L0:
    :param tau:
    :param kappa:
    :param T0:
    :param E0:
    :param kwargs:
    :return:
    """
    a = 1. / tau
    pl = one_component_fireball_model(time, A_1, alpha_1)
    loss_term = E0 * (T0 / time) ** (kappa)
    integ = integral_mdr(time, T0, kappa, a)
    Energy_loss_total = ((L0 / (time ** kappa)) * integ) + loss_term

    lightcurve = (kappa * Energy_loss_total / time)

    return lightcurve + pl


def collapsing_radiative_losses(time, A_1, alpha_1, L0, tau, nn, tcol, kappa, T0, E0, **kwargs):
    """
    radiative losses model with collapse time
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
    loss_term = E0 * (T0 / time) ** (kappa)
    integ = integral_general_collapsing(time, T0, kappa, tau, nn, tcol)
    Energy_loss_total = ((L0 / (time ** kappa)) * integ) + loss_term
    lum = (kappa * Energy_loss_total / time)
    total = pl + np.heaviside(time - T0, 1) * lum

    return total
