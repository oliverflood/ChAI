#!/bin/sh -l

echo "Hello $1"
time=$(date)
echo "time=$time" >> $GITHUB_OUTPUT

stdout=$(which python3)
echo "python3=$stdout" >> $GITHUB_OUTPUT

stdout=$(which chpl)
echo "chapel=$stdout" >> $GITHUB_OUTPUT



stdout=$(ls -la)
echo "ls=$stdout" >> $GITHUB_OUTPUT
