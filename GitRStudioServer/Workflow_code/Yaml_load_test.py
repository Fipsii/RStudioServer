
def ConfigImport(PathYaml):
  import yaml
  
  with open(PathYaml, "r") as stream:
      try:
          configurations = yaml.safe_load(stream)
          return configurations
      
      except yaml.YAMLError as exc:
          print(exc)
  




