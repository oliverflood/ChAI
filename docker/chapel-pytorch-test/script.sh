#!/usr/bin/env bash

echo "Hello!"

echo $(which chpl)

echo $(which python3 || echo "No 'python3' found!")