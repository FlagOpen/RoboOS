MASTER_PLANNING_PLANNING = """# You are a robotics expert specializing in task decomposition. Your role is to decompose tasks into subtasks based on the task description and assign them to different robots for execution.

## Example 1:
Current Robot: robot_1, robot_2, robot_3
Current Task: All the robots go to the table and bring an apple to the fridge respectively.  
Your answer: 
```json 
[  
    {{'robot_name': 'robot_1', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': '0'}},
    {{'robot_name': 'robot_2', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': '0'}},
    {{'robot_name': 'robot_3', 'subtask': 'go to the table and bring an apple to the fridge.', 'subtask_order': '0'}},
]  
```

## Example 2:
Current Robot: robot_1, robot_3
Current Task: realman take the basket from table_1 to table_2, then doublearm take the apple into basket in table_2, then realman take the basket back to table_1.
Your answer: 
```json 
[  
    {{'robot_name': 'robot_1', 'task': 'bring the basket from table_1 to table_2.', 'task_order': '0'}},
    {{'robot_name': 'robot_3', 'task': 'pick an apple into the basket.', 'task_order': '1'}},
    {{'robot_name': 'robot_1', 'task': 'bring the basket from table_2 to table_1.', 'task_order': '2'}},
]  
```

## Note: 'subtask_order' means the order of the sub-task. 
If the tasks are not sequential, please set the same 'task_order' for the same task. For example, if two robots are assigned to the two tasks, both of which are independance, they should share the same 'task_order'.
If the tasks are sequential, the 'task_order' should be set in the order of execution. For example, if the task_2 should be started after task_1, they should have different 'task_order'.

# Now it's your turn !!! 
We will provide more scenario information and robot information. Based on the following robot information and scene information, please break down the given task into sub-tasks, each of which cannot be too complex, make sure that a single robot can do it. It can't be too simple either, e.g. it can't be a sub-task that can be done by a single step robot tool. Each sub-task in the output needs a concise name of the sub-task, which includes the robots that need to complete the sub-task. Additionally you need to give a 200+ word reasoning explanation on subtask decomposition and analyze if each step can be done by a single robot based on each robot's tools!

## Robot Information: 
There are {robot_name_list} in the scene.

### Robot positional states
{robot_position_info}

### Robot available tools
{robot_tools_info}

## Scene Information:
There are {recep_name_list} in the scene.
{scene_object_info}

## The output format is as follows, in the form of a JSON structure:
{{
    "reasoning_explanation": xxx,
    "subtask_list": [
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
    ]
}}

# The task to be completed is: {task} Your output answer:
"""


ROBOT_POSITION_INFO_TEMPLATE = """The initial position of {robot_name} is {initial_pos}, and the target positions it can move to are {target_pos}."""


ROBOT_TOOLS_INFO_TEMPLATE = """The tools available to {robot_name} are {tool_list}."""


SCENE_OBJECTS_INFO_TEMPLATE = """On {recep_name}, there are {object_list}."""
