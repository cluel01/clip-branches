from jinja2 import Template
import json
import os

cfg_path = 'assets/config'
template_path = 'template.json'

def convert_keys(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k.isdigit():
                new_key = int(k)
            else:
                new_key = k
            new_obj[new_key] = convert_keys(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_keys(i) for i in obj]
    else:
        return obj


# Load the template
with open(template_path, 'r') as file:
    template_content = file.read()

template = Template(template_content)

config_files = [os.path.join(cfg_path,i) for i in os.listdir(cfg_path) if i.endswith('.json')]


datasets_config = {}
for c in config_files:
    with open(c, 'r') as file:
        config = json.load(file)

    # Render the template with the config values
    rendered_content = template.render(config)
    cfg_dict = json.loads(rendered_content)

    cfg_dict = convert_keys(cfg_dict)
    print(cfg_dict)
    datasets_config[cfg_dict["dataset_name"]] = cfg_dict

print(datasets_config)

