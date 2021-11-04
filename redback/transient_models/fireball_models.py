import numpy as np


def predeceleration(time, aa, mm, t0, **kwargs):
    """
    :param time:
    :param aa:
    :param mm: deceleration powerlaw gradient; typically 3 but depends on physics
    :param t0: time GRB went off.
    :param kwargs:
    :return: deceleration powerlaw; units are arbitrary and dependent on a_1.
    """
    return aa * (time - t0)**mm


def one_component_fireball_model(time, a_1, alpha_1, **kwargs):
    """
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param kwargs:
    :return:
    """
    return a_1 * time ** alpha_1


def two_component_fireball_model(time, a_1, alpha_1,
                                 delta_time_one, alpha_2, **kwargs):
    """
    Two component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    """
    time_one = delta_time_one
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    w = np.where(time < time_one)
    x = np.where(time > time_one)

    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2

    total = np.concatenate((f1, f2))

    return total


def three_component_fireball_model(time, a_1, alpha_1,
                                   delta_time_one, alpha_2,
                                   delta_time_two, alpha_3, **kwargs):
    """
    Three component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
    :param alpha_1: power law decay exponent
    :param delta_time_one: time between start and end of prompt emission
    :param alpha_2: power law decay exponent for the second power law
    :param delta_time_two: time between first and second power laws
    :param alpha_3: power law decay exponent for third power law
    """
    time_one = delta_time_one
    time_two = time_one + delta_time_two
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where(time > time_two)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3

    total = np.concatenate((f1, f2, f3))
    return total


def four_component_fireball_model(time, a_1, alpha_1, delta_time_one,
                                  alpha_2, delta_time_two,
                                  alpha_3, delta_time_three,
                                  alpha_4, **kwargs):
    """
    Four component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
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
    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)

    w = np.where(time < time_one)
    x = np.where((time_one < time) & (time < time_two))
    y = np.where((time_two < time) & (time < time_three))
    z = np.where(time > time_three)
    f1 = a_1 * time[w] ** alpha_1
    f2 = amplitude_two * time[x] ** alpha_2
    f3 = amplitude_three * time[y] ** alpha_3
    f4 = amplitude_four * time[z] ** alpha_4

    total = np.concatenate((f1, f2, f3, f4))

    return total


def five_component_fireball_model(time, a_1, alpha_1,
                                  delta_time_one, alpha_2,
                                  delta_time_two, alpha_3,
                                  delta_time_three, alpha_4,
                                  delta_time_four, alpha_5):
    """
    Five component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
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

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where(time > time_four)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5

    total = np.concatenate((f1, f2, f3, f4, f5))

    return total


def six_component_fireball_model(time, a_1, alpha_1,
                                 delta_time_one, alpha_2,
                                 delta_time_two, alpha_3,
                                 delta_time_three, alpha_4,
                                 delta_time_four, alpha_5,
                                 delta_time_five, alpha_6):
    """
    six component fireball model
    :param time: time array for power law
    :param a_1: power law decay amplitude
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

    amplitude_two = a_1 * time_one ** alpha_1 / (time_one ** alpha_2)
    amplitude_three = amplitude_two * time_two ** alpha_2 / (time_two ** alpha_3)
    amplitude_four = amplitude_three * time_three ** alpha_3 / (time_three ** alpha_4)
    amplitude_five = amplitude_four * time_four ** alpha_4 / (time_four ** alpha_5)
    amplitude_six = amplitude_five * time_five ** alpha_5 / (time_five ** alpha_6)

    u = np.where(time < time_one)
    v = np.where((time_one < time) & (time < time_two))
    w = np.where((time_two < time) & (time < time_three))
    x = np.where((time_three < time) & (time < time_four))
    y = np.where((time_four < time) & (time < time_five))
    z = np.where(time > time_five)

    f1 = a_1 * time[u] ** alpha_1
    f2 = amplitude_two * time[v] ** alpha_2
    f3 = amplitude_three * time[w] ** alpha_3
    f4 = amplitude_four * time[x] ** alpha_4
    f5 = amplitude_five * time[y] ** alpha_5
    f6 = amplitude_six * time[z] ** alpha_6

    total = np.concatenate((f1, f2, f3, f4, f5, f6))
    return total
