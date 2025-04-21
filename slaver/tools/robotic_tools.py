import json

from tools.tools import Tool


def arm_class_decorator(cls):
    cls.is_arm_class = True
    return cls


def chasis_class_decorator(cls):
    cls.is_chasis_class = True
    return cls


def camera_class_decorator(cls):
    cls.is_camera_class = True
    return cls


@arm_class_decorator
class PlaceWhere(Tool):
    name = "place_to_where"
    description = "The affordance of target position to place the held object."
    inputs = {
        "affordance": {
            "type": "string",
            "description": "The affordance of target position to place the held object.",
        }
    }
    output_type = "any"
    output = {
        "flag": {
            "type": "bool",
            "description": "The bool flag to show whether the placing action is successful.",
        }
    }
    example = "place_where(table_affordance) -> {'flag': True}"

    @classmethod
    def get_tool_schema(cls):
        return {
            "tool_name": cls.name,
            "input": cls.inputs,
            "output": cls.output,
            "example": cls.example,
        }

    def forward(self, affordance):
        try:
            result = self.robot.perform_place(affordance)
            return {"status": "success", "message": result}
        except Exception as e:
            return {"status": "error", "error_code": "E304", "message": str(e)}


@arm_class_decorator
class GraspObject(Tool):
    name = "grasp_object"
    description = "The affordance of target object to grasp."
    inputs = {
        "affordance": {
            "type": "string",
            "description": "The affordance of target object to grasp.",
        }
    }
    output_type = "any"
    output = {
        "flag": {
            "type": "bool",
            "description": "The bool flag to show whether the grasping action is successful.",
        }
    }
    example = "grasp_object(basket_affordance) -> {'flag': True}"

    @classmethod
    def get_tool_schema(cls):
        return {
            "tool_name": cls.name,
            "input": cls.inputs,
            "output": cls.output,
            "example": cls.example,
        }

    def forward(self, affordance):
        try:
            result = self.robot.perform_grasp(affordance)
            return {"status": "success", "message": result}
        except Exception as e:
            return {"status": "error", "error_code": "E303", "message": str(e)}


@chasis_class_decorator
class Navigate(Tool):
    name = "navigate_to_where"
    description = "The target destination name to navigate to."
    inputs = {
        "target": {
            "type": "string",
            "description": "The target destination name to navigate to.",
        }
    }
    output_type = "any"
    output = {
        "flag": {
            "type": "bool",
            "description": "The bool flag to show whether the navigation action is successful.",
        }
    }
    example = "navigate(servingTable) -> {'flag': True}"

    @classmethod
    def get_tool_schema(cls):
        return {
            "tool_name": cls.name,
            "input": cls.inputs,
            "output": cls.output,
            "example": cls.example,
        }

    def forward(self, target: str):
        try:
            result = self.robot.perform_navigate(target)
            return json.dumps({"status": "success", "message": result})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


@camera_class_decorator
class DetectObject(Tool):
    name = "detect_object"
    description = "The target object name to detect."
    inputs = {
        "target": {
            "type": "string",
            "description": "The target object name to detect.",
        }
    }
    output_type = "any"
    output = {
        "affordance": {
            "type": "list",
            "description": "The affordance of target object to grasp.",
        }
    }
    example = "detect_object(basket) -> {'affordance': basket_affordance}"

    @classmethod
    def get_tool_schema(cls):
        return {
            "tool_name": cls.name,
            "input": cls.inputs,
            "output": cls.output,
            "example": cls.example,
        }

    def forward(self, target: str):
        try:
            result = self.robot.perform_detect(target)
            return json.dumps({"status": "success", "message": result})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
