   #!/bin/bash
   
   echo "Conda Environment Sizes:"
   conda env list | grep -v '^#' | while read -r env_path; do
     env_name=$(echo $env_path | awk '{print $1}')
     env_path=$(echo $env_path | awk '{print $2}')
     if [ ! -z "$env_path" ]; then
       size=$(du -sh "$env_path" 2>/dev/null | cut -f1)
       echo "$env_name: $size"
     fi
   done