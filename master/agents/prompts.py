MASTER_PLANNING_PLANNING = """
## CRITICAL RULES:

0. NECESSITY CHECK
- **Only decompose the task if it is executable and necessary based on the scene.**
- **First verify that required objects, locations, and robot state are correct.**
- **A subtask is unnecessary only if its result is already achieved.**
  - *E.g., If the object is already at the target, skip moving it.*
  - *E.g., If the robot is at the destination, skip navigating.*
- **Don't skip a subtask just because another later step mentions the same object/location.**
  - *E.g., After grasping an object, navigation to a new location is still required before placing it.*
- **If the goal is not achieved and the task is executable, fully break it down into atomic, tool-driven steps including all needed navigation.**

### 1. TASK INTEGRITY
- **Never change the intent or swap objects/locations/actions from the original task.**
- **Always preserve the exact objects, locations, and actions.**
- Skip the task if it is impossible or incorrect.

### 2. TOOL-DRIVEN BEHAVIOR
- **All subtasks must be based strictly on available tools.**
- **Every subtask must correspond to at least one tool call.**
- **Only define subtasks that can be executed by robot tools.**
- **Describe subtasks as clear human actions (e.g., 'Navigate to kitchenTable'), not tool names.**
- **Do not create subtasks for reasoning, planning, etc.—only executable actions.**

### 3. SCENE INFORMATION VERIFICATION
- **Before decomposing, check the scene information:**
  - **Target object exists**
  - **Source location is correct**
  - **Destination exists and is accessible**
  - **Robot's current position/holding state**
  - **Required tools are available**
- **If any object, location, or tool is missing, clearly refuse decomposition as not executable.**
- **Do not assume anything not present in the scene.**
- **Do not alter, add, or substitute any object, location, or action from the original.**
- **Each subtask must clearly include the related tool and target(s).**

### 4. ATOMICITY REQUIREMENT
- **Each subtask must be atomic: one tool call per subtask, no combined or macro actions.**
- **Decompose as finely as possible; every action should be a single, indivisible step. All implied (e.g., navigation) steps must be explicit.**

Please break down the given task into sub-tasks, each of which cannot be too complex, make sure that a single robot can do it.
It can't be too simple either, e.g. it can't be a sub-task that can be done by a single step robot tool.
Each sub-task in the output needs a concise name of the sub-task, which includes the robots that need to complete the sub-task.
Additionally you need to give a 200+ word reasoning explanation on subtask decomposition and analyze if each step can be done by a single robot based on each robot's tools!

## The output format is as follows, in the form of a JSON structure:
{{
    "reasoning_explanation": xxx,
    "subtask_list": [
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
        {{"robot_name": xxx, "subtask": xxx, "subtask_order": xxx}},
    ]
}}

## Note: 'subtask_order' means the order of the sub-task.
If the tasks are not sequential, please set the same 'task_order' for the same task. For example, if two robots are assigned to the two tasks, both of which are independance, they should share the same 'task_order'.
If the tasks are sequential, the 'task_order' should be set in the order of execution. For example, if the task_2 should be started after task_1, they should have different 'task_order'.

Please only use {robot_name_list} with skills {robot_tools_info}.
You must also consider the following scene information when decomposing the task:
{scene_info}

**CRITICAL: You MUST verify the scene information before decomposing any task. Check that all target objects exist, source/destination locations are valid, and the robot's current state matches the task requirements. Make sure the entire task is thoroughly decomposed so that all steps required to achieve the goal are explicitly listed; do not omit any necessary atomic subtask.**

# The task to be completed is: {task}. Your output answer:
"""
