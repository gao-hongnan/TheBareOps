# DataOps

## Setup Project Structure

```bash
#!/bin/bash

# Create the directories
mkdir -p pipeline_dataops
mkdir -p conf
mkdir -p metadata

# Create inside pipeline_dataops
mkdir -p pipeline_dataops/extract
touch pipeline_dataops/extract/__init__.py

# Create the __init__.py files
touch pipeline_dataops/__init__.py
touch conf/__init__.py
touch metadata/__init__.py

# Create inside conf
mkdir -p conf/environment
mkdir -p conf/extract
mkdir -p conf/load
mkdir -p conf/transform
mkdir -p conf/directory

# Create the __init__.py files
touch conf/environment/__init__.py conf/environment/base.py
touch conf/extract/__init__.py conf/extract/base.py
touch conf/load/__init__.py conf/load/base.py
touch conf/transform/__init__.py conf/transform/base.py

# Create the .gitignore file
touch .gitignore
echo "# Add files and directories to be ignored by git here" > .gitignore

# Create .env file
touch .env
echo "# Add environment variables here" > .env

# Create the pyproject.toml file
touch pyproject.toml
```

## Setup Virtual Environment

```bash
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
rm make_venv.sh && \
source venv/bin/activate
```
