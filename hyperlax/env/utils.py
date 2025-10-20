from jumanji.env import Environment
from jumanji.specs import Array, BoundedArray, DiscreteArray


def identify_action_space_type(env: Environment):
    act_spec = env.action_spec()
    if isinstance(act_spec, DiscreteArray):
        return "discrete"
    elif isinstance(act_spec, (BoundedArray, Array)):
        return "continuous"
    else:
        return "unknown"
