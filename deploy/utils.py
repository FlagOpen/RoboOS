import copy


def split_dot_keys(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = split_dot_keys(value)

        if "." in key:
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            result[key] = value
    return result


def recursive_update(yaml_obj, new_values):
    for key, value in new_values.items():
        if key in yaml_obj:
            if isinstance(value, dict) and isinstance(yaml_obj[key], dict):
                recursive_update(yaml_obj[key], value)
            elif isinstance(value, list) and isinstance(yaml_obj[key], list):
                yaml_obj[key] = copy.deepcopy(value)
            else:
                yaml_obj[key] = value
    return yaml_obj
