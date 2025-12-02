import numpy as np

# ------------------------------------- #
#              Functions                #
# ------------------------------------- #

def eta_tc_2_O2(eta, tc): # 3Q methyl experiment
    r = 1.813 * (1/ np.power(10,10, dtype=np.longdouble)) 
    hbar = 6.62607004 * (1 / np.power(10, 34, dtype=np.longdouble)) / (2*np.pi)
    mu_0 = 1.25663706 * (1 / np.power(10, 6, dtype=np.longdouble))
    gammaH = 267.52218744 * np.power(10, 6, dtype=np.longdouble) 
    theta = np.pi/2
    P2 = 1/2*(3*np.cos(theta)**2 - 1)
    o2 = (eta * (10/9) * (((4*np.pi)/mu_0)**2) * (r**6)) / ((P2**2) * (gammaH**4) * (hbar**2) * tc)
    return o2

def O2_tc_2_eta(O2, tc): # 3Q methyl experiment
    r = 1.813 * (1/ np.power(10,10, dtype=np.longdouble)) 
    hbar = 6.62607004 * (1 / np.power(10, 34, dtype=np.longdouble)) / (2*np.pi)
    mu_0 = 1.25663706 * (1 / np.power(10, 6, dtype=np.longdouble))
    gammaH = 267.52218744 * np.power(10, 6, dtype=np.longdouble) 
    theta = np.pi/2
    P2 = 1/2*(3*np.cos(theta)**2 - 1)
    eta = (9/10) * ((P2**2) * (O2) * (gammaH**4) * (hbar**2) * tc * ((mu_0/(4*np.pi))**2 ))/ (r**6)
    return eta



def tauc_2_MW(tc):
    if tc < 0.0000001:  # probably units is nanoseconds
        tc = tc * 1000000000
    # return MW guess in kDa units
    return (tc - 0.775137) / 0.433859

def tc_2_MW(tc):
    if tc < 0.0000001:  # probably units is nanoseconds
        tc = tc * 1000000000
    # return MW guess in kDa units
    return (tc - 0.775137) / 0.433859


def MW_2_tc(MW):

    return MW * 0.433859 + 0.775137


def J(w, tc):
    return 0.4 * tc / (1 + (w ** 2 * tc ** 2))


def J_S2(w, tc, S2):
    return S2 * J(w, tc)


def tc_tract_algebraic(Ra, Rb, field, p=None):

    if not p:  # use these defaults
        p = Default_Params()

    B_0 = (
        field * np.power(10, 6, dtype=np.longdouble) * 2 * np.pi / p.gamma_H
    )  # in Tesla
    d = (
        p.mu_0
        * p.gamma_H
        * p.gamma_N
        * p.h
        / (16 * np.pi * np.pi * np.sqrt(2) * np.power(p.r, 3))
    )  # DD 1H-15N bond
    dN = p.gamma_N * B_0 * p.delta_dN / (3 * np.sqrt(2))  # 15N CSA
    w_N = B_0 * p.gamma_N  # 15N frequency (radians/s)

    c = (Rb - Ra) / (2 * dN * d * (3 * np.cos(p.theta) ** 2 - 1))

    t1 = (5 * c) / 24
    t2 = (336 * (w_N ** 2) - 25 * (c ** 2) * (w_N ** 4)) / (
        24
        * (w_N ** 2)
        * (
            1800 * c * (w_N ** 4)
            + 125 * (c ** 3) * (w_N ** 6)
            + 24
            * np.sqrt(3)
            * np.sqrt(
                21952 * (w_N ** 6)
                - 3025 * (c ** 2) * (w_N ** 8)
                + 625 * (c ** 4) * (w_N ** 10)
            )
        )
        ** (1 / 3)
    )
    t3 = (
        1800 * c * (w_N ** 4)
        + 125 * (c ** 3) * (w_N ** 6)
        + 24
        * np.sqrt(3)
        * np.sqrt(
            21952 * (w_N ** 6)
            - 3025 * (c ** 2) * (w_N ** 8)
            + 625 * (c ** 4) * (w_N ** 10)
        )
    ) ** (1 / 3) / (24 * w_N ** 2)

    return t1 - t2 + t3


def tc_tract_algebraic_S2(Ra, Rb, field, S2, p=None):

    if not p:  # use these defaults
        p = Default_Params()

    B_0 = (
        field * np.power(10, 6, dtype=np.longdouble) * 2 * np.pi / p.gamma_H
    )  # in Tesla
    d = (
        p.mu_0
        * p.gamma_H
        * p.gamma_N
        * p.h
        / (16 * np.pi * np.pi * np.sqrt(2) * np.power(p.r, 3))
    )  # DD 1H-15N bond
    dN = p.gamma_N * B_0 * p.delta_dN / (3 * np.sqrt(2))  # 15N CSA
    w_N = B_0 * p.gamma_N  # 15N frequency (radians/s)

    c = (Rb - Ra) / (2 * dN * d * (3 * np.cos(p.theta) ** 2 - 1))
    w = w_N

    t1 = (
        (
            125 * (c ** 3) * (w ** 6)
            + 24
            * np.sqrt(3)
            * np.sqrt(
                625 * (c ** 4) * (S2 ** 2) * (w ** 10)
                - 3025 * (c ** 2) * (S2 ** 4) * (w ** 8)
                + 21952 * (S2 ** 6) * (w ** 6)
            )
            + 1800 * c * (S2 ** 2) * (w ** 4)
        )
        ** (1 / 3)
    ) / (24 * S2 * w ** 2)

    t2 = (
        -1
        * (336 * (S2 ** 2) * (w ** 2) - 25 * (c ** 2) * (w ** 4))
        / (
            (
                24
                * S2
                * (w ** 2)
                * (
                    125 * (c ** 3) * (w ** 6)
                    + 24
                    * np.sqrt(3)
                    * np.sqrt(
                        625 * (c ** 4) * (S2 ** 2) * (w ** 10)
                        - 3025 * (c ** 2) * (S2 ** 4) * (w ** 8)
                        + 21952 * (S2 ** 6) * (w ** 6)
                    )
                    + 1800 * c * (S2 ** 2) * (w ** 4)
                )
                ** (1 / 3)
            )
        )
    )

    t3 = +(5 * c) / (24 * S2)

    return t1 + t2 + t3


def R1_field_tc_S2(field, t_c, S2, p=None):
    if not p:  # use these defaults
        p = Default_Params()

    B_0 = field * np.power(10, 6, dtype=np.longdouble) * 2 * np.pi / p.gamma_H
    d_N = p.gamma_N * B_0 * p.delta_dN / (3 * np.sqrt(2))  # 15N CSA
    w_N = B_0 * p.gamma_N  # 15N frequency (radians/s)
    w_H = B_0 * p.gamma_H

    # equations from Palmer 2001 (Annual Reviews) https://doi.org/10.1146/annurev.biophys.30.1.129
    d = p.h * p.mu_0 * p.gamma_N * p.gamma_H / ((p.r ** 3) * 8 * np.pi * np.pi)
    c = (p.delta_dN * w_N) / np.sqrt(3)

    return ((d ** 2) / 4) * (
        6 * J_S2(w_H + w_N, t_c, S2) + J_S2(w_H - w_N, t_c, S2) + 3 * J_S2(w_N, t_c, S2)
    ) + c ** 2 * J_S2(w_N, t_c, S2)


def R2_field_tc_S2(field, t_c, S2, p=None):
    if not p:  # use these defaults
        p = Default_Params()

    B_0 = field * np.power(10, 6, dtype=np.longdouble) * 2 * np.pi / p.gamma_H
    d_N = p.gamma_N * B_0 * p.delta_dN / (3 * np.sqrt(2))  # 15N CSA
    w_N = B_0 * p.gamma_N  # 15N frequency (radians/s)
    w_H = B_0 * p.gamma_H

    # equations from Palmer 2001 (Annual Reviews) https://doi.org/10.1146/annurev.biophys.30.1.129
    d = p.h * p.mu_0 * p.gamma_N * p.gamma_H / ((p.r ** 3) * 8 * np.pi * np.pi)
    c = (p.delta_dN * w_N) / np.sqrt(3)

    return ((d ** 2) / 8) * (
        6 * J_S2(w_H + w_N, t_c, S2)
        + 6 * J_S2(w_H, t_c, S2)
        + J_S2(w_H - w_N, t_c, S2)
        + 3 * J_S2(w_N, t_c, S2)
        + 4 * J_S2(0, t_c, S2)
    ) + ((c ** 2) / 6) * (3 * J_S2(w_N, t_c, S2) + 4 * J_S2(0, t_c, S2))


# ------------------------------------- #
#               Classes                 #
# ------------------------------------- #


class Default_Params:
    """
    Default numerical parameters for NMR properties and physical constants

    h: Plank's constant
    mu_0: vacuum permeability
    r: bond length. specifically H-N
    delta_dN: CSA value for H-N
    theta: angle between bond and CSA tensor, specifically H-N
    gamma_H: proton gyromagnetic ratio
    gamma_N: 15N gyromagnetic ratio

    You can add further parameters at class instantiation time:
    p = mk.Default_Params(params)
    where params is a dictionary, e.g.
    params = {
        'gamma_C': 67.2828 * np.power(10, 6, dtype=np.float128),
        'gamma_D': 41065000.0,
        }

    parameter values must be floats or np.floating
    """

    def __init__(self, params=None):
        self.h = 6.62607004 * (1 / np.power(10, 34, dtype=np.longdouble))  # Plank's
        self.mu_0 = 1.25663706 * (
            1 / np.power(10, 6, dtype=np.longdouble)
        )  # vacuum permeability
        self.gamma_H = 267.52218744 * np.power(
            10, 6, dtype=np.longdouble
        )  # proton gyromagnetic ratio
        self.gamma_N = -27.116 * np.power(
            10, 6, dtype=np.longdouble
        )  # 15N gyromagnetic ratio
        self.gamma_C = 67.2828 * np.power(
            10, 6, dtype=np.longdouble
        )  # 13C gyromagnetic ratio
        self.gamma_D = 41.065 * np.power(
            10, 6, dtype=np.longdouble
        )  # 2H gyromagnetic ratio
        self.r = 1.02 * (
            1 / np.power(10, 10, dtype=np.longdouble)
        )  # internuclear distance
        self.delta_dN = 160 * (
            1 / np.power(10, 6)
        )  # diff in axially symetric 15N CS tensor
        self.theta = 17 * np.pi / 180  # angle between CSA axis and N-H bond

        if params is not None:
            if not isinstance(params, dict):
                print(
                    "optional params are not in form of dictionary\nparams will be default values only"
                )
            else:
                for key in params.keys():
                    var = params[key]
                    if isinstance(var, (float, np.floating)):
                        setattr(self, key, var)
                    else:
                        print(
                            "param value is not a float or numpy float\nparams will be default values only"
                        )


class Scale:
    """
    A Scale Class
    """
    def __init__(self, domainrange, outrange, strict=False):
        self.d_min = domainrange[0]
        self.d_max = domainrange[1]
        self.d_scope = self.d_max - self.d_min

        self.o_min = outrange[0]
        self.o_max = outrange[1]
        self.o_scope = self.o_max - self.o_min

        self.strict = strict

    def linear(self, indomain):

        if self.strict is True and (indomain < self.d_min or indomain > self.d_max):
            raise Exception(
                f"input value {indomain} is outside the input domain for this scale"
            )

        domainfrac = (indomain - self.d_min) / self.d_scope
        outfrac = domainfrac * self.o_scope
        return self.o_min + outfrac

    def linear_r(self, inrange):

        if self.strict is True and (inrange < self.o_min or inrange > self.o_max):
            raise Exception(
                f"input value {inrange} is outside the input domain for this scale"
            )

        domainfrac = (inrange - self.o_min) / self.o_scope
        outfrac = domainfrac * self.d_scope
        return self.d_min + outfrac