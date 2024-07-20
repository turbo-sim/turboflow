

from pydantic import BaseModel
from typing import List, Union, Literal, Dict
from typing_extensions import TypedDict

class Geometry(BaseModel):
    
    area_in : Union[float, list]
    area_out : Union[float, list]

class AxialTurbine(BaseModel):
    
    turbomachinery : str
    operation_points : list
    performance_map : Union[list, Geometry]
    design_optimization : dict
    simulation_options : Dict[str, float]
    cascade_type : Literal["stator", "rotor"]
    geometry : Geometry
    
schema = AxialTurbine.model_json_schema()
print(schema)

# Notes:
    # list gives type 'array'
    # dict gives type 'object'
    # Dict[str, float] gives type 'object' and additionalProperties to assign the type of the value in the dict
    # SubModel (e.g Geometry) gives no type, but a reference to a subschema:
        # subschema gives type 'object'
    # Union gives no type, but an 'anyOf':
        # 'anyOf' is a list, where information on each valid input is given
        # if list, dict, float, str or similar, it gives a type
        # if SubModel it gives a reference to a subschema
    # Literal gives a type, and an enum (a list of valid options)

# Should dict types be included: configuratation file is a yaml file, where its more natural to talk about entries, and not dicts.
# Types can be included when they are lists, floats, strings and enums. 