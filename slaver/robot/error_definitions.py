ERROR_DEFINITIONS = {
    "navigationErrors": {
        "E101": {
            "code": 101,
            "name": "ObstacleBlockade",
            "description": "Physical obstruction prevents movement to target area",
            "severity": "high",
            "recoveryActions": {
                "Attempt_Other_Path()": "Original A->B navigation path, try A->C->B, C is the closest point to B",
                "Ask_Other_Robot_for_Help()": "Call other idle robots to complete the task instead",
                "Request_Manual_Assistance()": "Voice broadcast obstacle, user assistance required to remove it",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
        "E102": {
            "code": 102,
            "name": "UnreachableCoordinates",
            "description": "Target position outside navigable area",
            "severity": "medium",
            "resolutionChecklist": {
                "Attempt_Other_Position()": "Original target point does not exist, navigate to the semantically closest target point",
                "Use_Vision_Navigation()": "Original target point does not exist, adopt exploratory navigation method, build map/VLN",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
    },
    "visionErrors": {
        "E201": {
            "code": 201,
            "name": "CameraOcclusion",
            "description": "Target view blocked by environmental elements",
            "severity": "medium",
            "resolutionChecklist": {
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Ask_Other_Robot_for_Help()": "Call other idle robots to complete the task instead",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
        "E202": {
            "code": 202,
            "name": "RecognitionEmptyObjects",
            "description": "Target description matches no candidates",
            "severity": "medium",
            "disambiguationProtocol": {
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Recognize_Similar_Object()": "Recognize semantically similar objects, select non-target object based on confidence level",
                "Move_to_Other_Position()": "If still no object, then move to another target point to search again",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
    },
    "graspObjectErrors": {
        "E301": {
            "code": 301,
            "name": "KinematicSingularity",
            "description": "Unresolvable arm configuration near singularities",
            "severity": "medium",
            "resolution": {
                "Adjust_EEF_Orientation()": "Adjust the rotation pose of the end-effector, try again",
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Move_to_Candidate_Position()": "Predefine candidate points for each mark point, try grasping at candidate positions",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
        "E302": {
            "code": 302,
            "name": "ObjectDropping",
            "description": "The target object drops when moving to hang after grasping",
            "severity": "medium",
            "resolution": {
                "grasp()": "Call grasp_object tool defined by the robot to reattempt",
            },
        },
        "E303": {
            "code": 303,
            "name": "InheritedVisionError",
            "description": "Target view blocked by environmental elements",
            "severity": "medium",
            "resolutionChecklist": {
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Ask_Other_Robot_for_Help()": "Call other idle robots to complete the task instead",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
    },
    "placeObjectErrors": {
        "E401": {
            "code": 401,
            "name": "KinematicSingularity",
            "description": "Unresolvable arm configuration near singularities",
            "severity": "medium",
            "resolution": {
                "Adjust_EEF_Orientation()": "Adjust the rotation pose of the end-effector, try again",
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Move_to_Candidate_Position()": "Predefine candidate points for each mark point, try placing at candidate positions",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
        "E402": {
            "code": 402,
            "name": "ObjectDropping",
            "description": "The target object drops when moving to place",
            "severity": "medium",
            "resolution": {
                "grasp_object()": "Call grasp_object tool defined by the robot to reattempt",
                "place_to_where()": "Call place_to_where tool defined by the robot to complete placement",
            },
        },
        "E403": {
            "code": 403,
            "name": "InheritedVisionError",
            "description": "Target view blocked by environmental elements",
            "severity": "medium",
            "resolutionChecklist": {
                "Move_to_Near_Position()": "Move a short distance near the current position, retry multiple times",
                "Ask_Other_Robot_for_Help()": "Call other idle robots to complete the task instead",
                "Ask_RoboBrain_for_Replanning()": "Feedback to RoboBrain for replanning, V2 version only saves the error",
            },
        },
    },
}
