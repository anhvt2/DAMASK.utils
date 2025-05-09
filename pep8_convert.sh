#!/bin/bash
rm -fv *-pep8.py

cd utils/
# Loop through all Python files in the current directory
for file in *.py; do
    # Skip already converted files
    if [[ "$file" == *-pep8.py ]]; then
        continue
    fi

    # Strip extension and append -pep8.py
    base="${file%.py}"
    output="${base}.py"

    echo "Converting $file -> $output"
    autopep8 "$file" > ../utils-pep8/"$output"
done

echo "All files processed."
cd ..
