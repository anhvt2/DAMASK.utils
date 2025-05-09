#!/bin/bash
rm -fv *-pep8.py

# Loop through all Python files in the current directory
for file in *.py; do
    # Skip already converted files
    if [[ "$file" == *-pep8.py ]]; then
        continue
    fi

    # Strip extension and append -pep8.py
    base="${file%.py}"
    output="${base}-pep8.py"

    echo "Converting $file -> $output"
    autopep8 "$file" > "$output"
done

echo "All files processed."
