import simsurvey
import sncosmo


class Optical_Transient(object):
    def __init__(name=None, parameters=None, observations=None):

        super.__init__()

    def function_wrapper(model, parameters, function=all_models_dict[model]):
        dense_times = np.linspace(0.0, )
        function()
        return sncosmo.wrapper

    def _get_correct_function(base_model, model_type=None):
        """
        Gets the correct function to use for the base model specified

        :param base_model: string or a function
        :param model_type: type of model, could be None if using a function as input
        :return: function; function to evaluate
        """
        from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
        extinction_base_models = extinction_model_library[model_type]
        module_libary = model_library[model_type]

        if isfunction(base_model):
            function = base_model

        elif base_model not in extinction_base_models:
            logger.warning('{} is not implemented as a base model'.format(base_model))
            raise ValueError('Please choose a different base model')
        elif isinstance(base_model, str):
            function = modules_dict[module_libary][base_model]
        else:
            raise ValueError("Not a valid base model.")

        return function

    def simulate_observations():


    def from_simsurvey():

        return pandas_obs

    def _evaluate_model(time, model_type, **kwargs):
        """
        Generalised evaluate extinction function

        :param time: time in days
        :param av: absolute mag extinction
        :param model_type: None, or one of the types implemented
        :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
            and r_v, default is 3.1
        :return: flux_density or magnitude depending on kwargs['output_format']
        """
        base_model = kwargs['base_model']
        frequency = kwargs['frequency']
        temp_kwargs = kwargs.copy()
        temp_kwargs['output_format'] = 'flux_density'
        function = _get_correct_function(base_model=base_model, model_type=model_type)
        flux_density = function(time, **temp_kwargs)
