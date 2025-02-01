#!/bin/sh -l

echo "Hello $1"
time=$(date)
echo "time=$time" >> $GITHUB_OUTPUT

stdout=$(which python3)
echo "python3=$stdout" >> $GITHUB_OUTPUT

stdout=$(which chpl)
echo "chapel=$stdout" >> $GITHUB_OUTPUT





echo "Begin test attempt. "
stdout=$(ls -la)
echo "$stdout"

echo "Running tests."

echo "Trying to compile test from entrypoint.sh. " 
chpl test/correspondence/construction/ones/ones.chpl -M lib

echo "Now running correspondence test. "
echo $(cd test/correspondence && python3 correspondence.py)

echo "End test attempt. "



echo $(pwd)

echo $(which python3 || echo "No python3")
echo $(which chpl || echo "No chpl")


echo $(cd / && ls)

