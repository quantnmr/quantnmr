import numpy as np
from scipy.optimize.minpack import curve_fit
import pymc as pm


def fit_power_decay(t, I, param_guess=None):
    # Fit relaxation curves to extract time parameters
    if param_guess is None:
        # make guess
        param_guess = np.zeros(3)
        param_guess[0] = 20
        param_guess[2] = I[0] * 0.01  # 1% of most intense point

    env_model = lambda t, a, b, c, d:  (c-d)*(t+a)**(-b) + d
    fit = curve_fit(env_model, t, I, p0=param_guess, maxfev=200000)
    return fit, env_model(t, fit[0][0], fit[0][1], fit[0][2], fit[0][3])


def fit_exp_decay_I0(t, I, param_guess=None):
    # Fit relaxation curves to extract time parameters
    if param_guess is None:
        # make guess
        param_guess = np.zeros(3)
        param_guess[0] = 20
        param_guess[2] = I[0] * 0.01  # 1% of most intense point

    env_model = lambda t, R, a, b:  (a-b)*np.exp(-R * t) + b
    fit = curve_fit(env_model, t, I, p0=param_guess, maxfev=200000)
    return fit, env_model(t, fit[0][0], fit[0][1], fit[0][2])

def fit_T1_recovery(t, I, param_guess=None):
    # Fit relaxation curves to extract time parameters
    if param_guess is None:
        # make guess
        param_guess = np.zeros(2)
        param_guess[0] = 1
        param_guess[1] = -I[0]
        

    env_model = lambda t, R, a: a*(1-2*np.exp(-R * t))
    fit = curve_fit(env_model, t, I, p0=param_guess, maxfev=200000)
    return fit, env_model(t, fit[0][0], fit[0][1])

def fit_exp_decay(t, I, param_guess=None):
    # Fit relaxation curves to extract time parameters
    if param_guess is None:
        # make guess
        param_guess = np.zeros(3)
        param_guess[0] = 20
        param_guess[1] = I[0]
        param_guess[2] = I[0] * 0.01  # 1% of most intense point

    env_model = lambda t, R, a, b: a * np.exp(-R * t) + b
    fit = curve_fit(env_model, t, I, p0=param_guess, maxfev=200000)
    return fit, env_model(t, fit[0][0], fit[0][1], fit[0][2])


def fit_exp_decay_to_zero(t, I, param_guess=None):
    # Fit relaxation curves to extract time parameters
    if param_guess is None:
        # make guess
        param_guess = np.zeros(2)
        param_guess[0] = 20
        param_guess[1] = I[0]

    env_model = lambda t, R, a: a * np.exp(-R * t)
    fit = curve_fit(env_model, t, I, p0=param_guess, maxfev=200000)
    return fit, env_model(t, fit[0][0], fit[0][1])

def fit_ls_ratio_model(t, R, param_init=None):
    """
    Parameters
    ----------

    Fit ratio of 3Q and SQ data using least squares approach to model. 
    t: numpy array of floats
        Time delays used in the experiment
    R: numpy 1D array of floats with the ratios for a single spin system

    Returns
    -------

    tuple of np.array, np.array
    (the return parameters, eta and delta, error estimates for each parameter)
    """
    
    # Fit relaxation curves to extract relaxation rate parameters
    if param_init is None:
        # make guess
        param_init = np.zeros(2)
        param_init[0] = 300
        param_init[1] = -100
        

    env_model = lambda t, eta, delta: 0.75*eta*np.tanh(np.sqrt(eta**2 + delta**2)*t)/((np.sqrt(eta**2 + delta**2) - delta*np.tanh(np.sqrt(eta**2+delta**2)*t)))
    fit = curve_fit(env_model, t, R, p0=param_init, maxfev=200000, bounds=((0, -1000), (1000, 100)), method='trf')
    return fit[0], np.sqrt(np.diag(fit[1]))

def fit_bayes_ratio_model(t, ratios):


    """
    Fit ratio of 3Q and SQ data to extract \eta and \delta. 

    Parameters
    ----------
    t : numpy array of floats 
        Time delays used in the experiment.
    ratios: numpy 2D array of floats. 
        The ratio of 3Q data to SQ data
        Second dimension is same length as t
        First dimension length is the number of systems being fitted

    Returns
    -------
    tuple of np.array, np.array, np.array, np.array and pymc.model
    (eta_mean, eta_std, delta_mean, delta_std, model)

    
    Example
    --------
    result = fit_bayes_ratio_model(T, ratios)
    
    """


    el = 0 ; eu = 1000
    dl = -1000; du = +100
    with pm.Model() as sys:

        #Priors
        σy = pm.Exponential('σy', 100)#shape=(triQ.shape[0], 1))
        
        ## Params of Scientific Model
        η = pm.Uniform('η', lower=el, upper=eu, shape=(ratios.shape[0], 1))
        δ = pm.Uniform('δ', lower=dl, upper=du, shape=(ratios.shape[0], 1))
        
        ## Model components
        C = 0.75
        a = η*np.tanh(np.sqrt(η**2 + δ**2)*t)
        b = np.sqrt(η**2 + δ**2)
        c = δ*np.tanh(np.sqrt(η**2 + δ**2)*t)
        
        ## Scientific Model
        r = pm.Deterministic('r', (3/4)*a/(b-c))  
        
        Ratio = pm.Normal('Ratio', mu=r, sigma=σy, observed=ratios)
        model = pm.sample( tune=4000, draws=4000, target_accept=0.95, idata_kwargs = {'log_likelihood': True}, nuts_sampler="numpyro", progressbar=False)
        model.extend(pm.sample_posterior_predictive(model))
    ratio_etas_mean = model['posterior']['η'].to_numpy().mean(axis=(0,1,3))
    ratio_deltas_mean = model['posterior']['δ'].to_numpy().mean(axis=(0,1,3))

    ratio_etas_std = model['posterior']['η'].to_numpy().std(axis=(0,1,3))
    ratio_deltas_std = model['posterior']['δ'].to_numpy().std(axis=(0,1,3))
    d = model['sample_stats']['diverging']
    print("Number of Divergent %d" % d.to_numpy().sum())
    return ratio_etas_mean, ratio_etas_std, ratio_deltas_mean, ratio_deltas_std, model


def fit_bayes_biphasic_model(t, sinQ, triQ, R=100, I0=400, fit_ratio=False, calc_ratio=True):
    """
    Fit biphasic 3Q and SQ data to extract \eta and \delta. By default, ratio values and ratio error are calculated,
    but no fitting of ratio data to a ratio model is done in adition to 3Q and SQ fittings. You can opt to fit the 
    ratio of 3Q and SQ as well but this is not the default behaviour.
    

    Parameters
    ----------
    t : numpy array of floats 
        Time delays used in the experiment.
    sinQ : numpy 2D array of floats
        Single Quantum decay values from experiment
        Second dimension is same length as t
        First dimension length is the number of systems being fitted
    triQ : numpy 2D array of floats
        Triple Quantum decay values from experiment
        Second dimension is same length as t
        First dimension length is the number of systems being fitted
    fit_ratio: bool (Default False)
        Should you fit the ratio of sinQ/triQ as well?
    calc_ratio: bool (Default True)
        Should you calculate the ratio values and error?

    Returns
    -------
    tuple of np.array, np.array, np.array, np.array and pymc.model
    (eta_mean, eta_std, delta_mean, delta_std, model)

    
    Example
    --------
    result = fit_bayes_biphasic_model(T, data_SQ, data_3Q)
    
    """
    el = 0 ; eu = 1000
    dl = -500; du = +200
    standardize_max = sinQ.max(axis=1)
    sinQ = (sinQ.T/standardize_max).T
    triQ = (triQ.T/standardize_max).T
    with pm.Model() as sys:

        #Priors
        σn = pm.Exponential('σn', 100)# shape=(triQ.shape[0], 1))
        σd = pm.Exponential('σd', 100)#shape=(triQ.shape[0], 1))
        
        ## Params of Scientific Model
        η = pm.Uniform('η', lower=el, upper=eu, shape=(sinQ.shape[0], 1))
        δ = pm.Uniform('δ', lower=dl, upper=du, shape=(sinQ.shape[0], 1))
        I_0 = pm.Exponential('I_0', I0, shape=(sinQ.shape[0], 1)) # assume all systems have independent I_0 value
        R_Axy = pm.Exponential('R_Axy', 1/R, shape=(sinQ.shape[0], 1)) # There are probably two independent relaxation rates
        R_Bxy = pm.Exponential('R_Bxy', 1/R, shape=(sinQ.shape[0], 1)) # Which we will call R_A and R_B
        
        ## Model components
        C = 0.75
        a = η*np.tanh(np.sqrt(η**2 + δ**2)*t)
        b = np.sqrt(η**2 + δ**2)
        c = δ*np.tanh(np.sqrt(η**2 + δ**2)*t)
        
        ## Scientific Model
        n = pm.Deterministic('n', 3*a*I_0*(np.exp(-R_Axy*t)+np.exp(-R_Bxy*t)))
        d = pm.Deterministic('d', 4*(b - c)*I_0*(np.exp(-R_Axy*t)+np.exp(-R_Bxy*t)))
        
        SQ = pm.Normal('SQ', mu=d, sigma=σd, observed=sinQ)
        MQ = pm.Normal('MQ', mu=n, sigma=σn, observed=triQ)
        
        if calc_ratio:
            r = pm.Deterministic('r', (3/4)*a/(b-c))  
            σr = pm.Deterministic('σr', np.sqrt((σn/n)**2 + (σd/d)**2)*r) #  - 2*(1/(n/d))
        if fit_ratio:      
            Ratio = pm.Normal('Ratio', mu=r, sigma=σr, observed=triQ/sinQ)
        
        
        model = pm.sample(tune=4000, draws=4000, target_accept=0.95, idata_kwargs = {'log_likelihood': True}, nuts_sampler="numpyro", progressbar=False)
        model.extend(pm.sample_posterior_predictive(model))
    
    bpe_etas_mean = model['posterior']['η'].to_numpy().mean(axis=(0,1,3))
    bpe_deltas_mean = model['posterior']['δ'].to_numpy().mean(axis=(0,1,3))

    bpe_etas_std = model['posterior']['η'].to_numpy().std(axis=(0,1,3))
    bpe_deltas_std = model['posterior']['δ'].to_numpy().std(axis=(0,1,3))
    d = model['sample_stats']['diverging']
    print("Number of Divergent %d" % d.to_numpy().sum())
    return bpe_etas_mean, bpe_etas_std, bpe_deltas_mean, bpe_deltas_std, model, sinQ, triQ  