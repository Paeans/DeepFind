#!/bin/bash

[[ $1 == "" ]] && msg="master update" || msg=$1

remotelist="origin bucket"
branch="master"

for remote in $remotelist
do
    git pull $remote $branch
    [[ $? == 0 ]] || {
        echo "ERROR between local and $remote"
        echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        echo
        exit 1
    }
    echo "**************************************"
    echo
done

git add -A
git commit -m "${msg}" && {
#echo
for remote in $remotelist
do
    echo
    git push $remote $branch
    #echo
done } # || echo
echo
