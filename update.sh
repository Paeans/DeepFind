#!/bin/bash

msg="master update"
[[ $1 == "" ]] || msg=$1

#remotelist="origin bucket"
#remotelist[0]=origin
remotelist[0]=bucket
branch="master"

for remote in ${remotelist[@]}
do
    git pull  --no-edit --no-commit $remote $branch
    [[ $? == 0 ]] || {
        echo "ERROR between local and $remote"
        echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        echo
        exit 1
    }
    echo "**************************************"
done
echo

git add --all
git commit -m "${msg}" #&& {
echo "**************************************"
echo

echotag=0
for remote in ${remotelist[@]}
do
    git push $remote $branch |& grep -v "up-to-date" && {    
      echo "**************************************"
      echotag=1
    }
    #echo
done #} # || echo
[[ $echotag == 1 ]] && echo
