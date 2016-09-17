#!/bin/sh

git add -A
echo "crote"
git commit -am "$1"
echo "bu"
git push origin master 
